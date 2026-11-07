"""Heuristic for ACO-CVRP: Full-Fidelity Adaptive Parameter Optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_BUDGET = 300.0
_DEADLINE = _T0 + _BUDGET
_TIME_MARGIN = 15.0

def _remaining():
    return _DEADLINE - _TIME_MARGIN - time.time()

def _safe_heuristic(dm):
    """Fallback heuristic: inverse distance."""
    d = np.asarray(dm, dtype=float).copy()
    np.fill_diagonal(d, 1e9)
    return 1.0 / (d + 1e-9)

def _construct_heuristic(dm, alpha, beta):
    """
    Construct the heuristic matrix using the formula:
    H[i,j] = (1/d[i,j])^alpha * (savings[i,j])^beta
    where savings[i,j] = d[i,0] + d[0,j] - d[i,j] (clipped to positive)
    """
    d = np.asarray(dm, dtype=float).copy()
    np.fill_diagonal(d, 1e9)
    
    # Base distance term
    d_inv = 1.0 / (d + 1e-9)
    
    # Savings term
    d_i0 = dm[:, 0]
    d_j0 = dm[0, :]
    savings = d_i0[:, np.newaxis] + d_j0[np.newaxis, :] - d
    savings_pos = np.maximum(savings, 0.0)
    
    max_sav = np.max(savings_pos)
    if max_sav < 1e-9:
        sav_heur = np.ones_like(savings)
    else:
        sav_heur = savings_pos / max_sav
        
    # Combine using logs to prevent overflow/underflow
    log_h = alpha * np.log(d_inv + 1e-9) + beta * np.log(sav_heur + 1e-9)
    h = np.exp(log_h)
    
    # Normalize to keep scale reasonable (max=1)
    max_h = np.max(h)
    if max_h > 1e-9:
        h = h / max_h
    else:
        h = np.ones_like(h)
        
    # Ensure positivity and handle diagonal
    h = np.maximum(h, 1e-9)
    np.fill_diagonal(h, 0.0)
    
    return h

def _fast_colony_sim(dist, D, cap, logH, n_ants=20, n_iter=30):
    """
    High-fidelity simulation of the ACO process to evaluate a heuristic.
    Returns the best cost found.
    """
    n = dist.shape[0]
    P = np.ones((n, n))
    logP = np.zeros((n, n))
    best_cost = float('inf')
    
    for _ in range(n_iter):
        iteration_costs = []
        for _ in range(n_ants):
            node = 0
            remaining_cap = cap
            cost = 0.0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            
            while True:
                # Find feasible neighbors
                feasible = []
                for j in range(1, n):
                    if not visited[j] and D[j] <= remaining_cap + 1e-9:
                        feasible.append(j)
                
                if not feasible:
                    cost += dist[node, 0]
                    break
                
                # Calculate probabilities: P * H
                # Use log-space for stability
                vals = np.array([logP[node, j] + logH[node, j] for j in feasible])
                max_val = np.max(vals)
                weights = np.exp(vals - max_val)
                total_w = np.sum(weights)
                
                if total_w <= 1e-9 or not np.isfinite(total_w):
                    idx = np.random.randint(len(feasible))
                else:
                    probs = weights / total_w
                    r = np.random.random()
                    cumsum = np.cumsum(probs)
                    idx = np.searchsorted(cumsum, r)
                    if idx >= len(feasible):
                        idx = len(feasible) - 1
                
                next_node = feasible[idx]
                cost += dist[node, next_node]
                remaining_cap -= D[next_node]
                visited[next_node] = True
                node = next_node
            
            iteration_costs.append(cost)
            
        best_iter = min(iteration_costs)
        if best_iter < best_cost:
            best_cost = best_iter
            
        # Update Pheromone
        # P = 0.9 * P + 1/best_iter
        P *= 0.9
        P += 1.0 / best_iter
        # Update logP for next iteration
        # Note: P might have 0s if initialized poorly, but here P starts at 1 and adds positive, so P > 0.
        # However, diagonal is not updated in the standard way, but we only use off-diagonal.
        logP = np.log(P + 1e-9)
        
    return best_cost

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO on CVRP.
    Uses adaptive parameter selection via high-fidelity local simulation.
    """
    dm = np.asarray(distance_matrix, dtype=float)
    D = np.asarray(demands, dtype=float)
    cap = float(capacity)
    n = dm.shape[0]
    
    # Basic validation
    if dm.shape != (n, n) or not np.isfinite(dm).all() or not np.isfinite(D).all():
        return _safe_heuristic(dm)
    
    if _remaining() < 20.0:
        return _safe_heuristic(dm)
    
    # Prepare data for simulation
    # We will optimize on the provided instance primarily.
    # If we can identify the other instances, we could include them, but for speed
    # and robustness, we assume the provided instance is representative or
    # we just optimize on it. Given the 300s limit, optimizing on 1 instance
    # with high fidelity is safer than low fidelity on 3.
    
    dist = dm
    D_sim = D
    cap_sim = cap
    
    # Grid search over alpha (distance weight) and beta (savings weight)
    # Based on previous success, alpha ~1-3, beta ~0-2
    alphas = [1.0, 1.5, 2.0, 2.5, 3.0]
    betas = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    best_score = float('inf')
    best_alpha = 2.0
    best_beta = 1.0
    
    # Phase 1: Coarse Grid Search
    # Each sim: 20 ants * 30 iters * ~50 steps * ~50 checks = 1.5M ops.
    # In Python, this might take ~0.5-1s per sim.
    # 25 combos * 1s = 25s. Safe.
    
    for a in alphas:
        if _remaining() < 40.0:
            break
        for b in betas:
            if _remaining() < 40.0:
                break
            try:
                # Construct heuristic for this combo
                h = _construct_heuristic(dm, a, b)
                logH = np.log(h + 1e-9)
                
                # Run simulation
                score = _fast_colony_sim(dist, D_sim, cap_sim, logH, n_ants=20, n_iter=30)
                
                if score < best_score:
                    best_score = score
                    best_alpha = a
                    best_beta = b
            except Exception:
                pass
    
    # Phase 2: Local Refinement
    if _remaining() > 60.0:
        # Refine alpha and beta around the best found
        ref_alphas = np.linspace(best_alpha - 0.5, best_alpha + 0.5, 5)
        ref_betas = np.linspace(best_beta - 0.5, best_beta + 0.5, 5)
        
        for a in ref_alphas:
            if a < 0.1: continue
            if _remaining() < 50.0:
                break
            for b in ref_betas:
                if b < 0.0: continue
                if _remaining() < 50.0:
                    break
                try:
                    h = _construct_heuristic(dm, a, b)
                    logH = np.log(h + 1e-9)
                    score = _fast_colony_sim(dist, D_sim, cap_sim, logH, n_ants=20, n_iter=30)
                    
                    if score < best_score:
                        best_score = score
                        best_alpha = a
                        best_beta = b
                except Exception:
                    pass
                    
    # Final construction with best parameters
    try:
        final_h = _construct_heuristic(dm, best_alpha, best_beta)
        # Final safety checks
        if not np.isfinite(final_h).all() or np.any(final_h <= 0):
            return _safe_heuristic(dm)
        return final_h
    except Exception:
        return _safe_heuristic(dm)

# EVOLVE-BLOCK-END
