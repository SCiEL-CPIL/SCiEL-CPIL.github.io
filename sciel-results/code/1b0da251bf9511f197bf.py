"""
Optimized Greedy MIS via Weighted Heuristic Search and Local Polish.

This program improves upon the seed by:
1. Using a lower-dimensional search space (heuristic weights) to find a robust scoring function.
2. Applying a final local search on vertex scores to fine-tune the greedy order.
3. Incorporating stronger structural features (Core-Number) into the heuristic.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin below 150s limit

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric and zero diagonal
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5
    
    # Precompute neighbor lists
    neighbors = [np.where(A_bool[i])[0] for i in range(N)]
    degrees = np.array([len(neighbors[i]) for i in range(N)], dtype=float)

    # --- Feature Extraction ---
    
    # 1. Degree
    D = degrees
    
    # 2. Clustering Coefficient
    A_sq = A @ A
    Tri = np.diag(A_sq) / 2.0
    D_safe = np.where(D > 0, D, 1.0)
    Clus = np.zeros(N)
    mask = D >= 2
    Clus[mask] = 2.0 * Tri[mask] / (D[mask] * (D[mask] - 1.0))

    # 3. Proper K-Core Number Calculation
    core = np.zeros(N, dtype=int)
    deg = D.copy().astype(int)
    active = np.ones(N, dtype=bool)
    max_deg = int(D.max()) if N > 0 else 0
    for k in range(1, max_deg + 1):
        changed = True
        while changed:
            changed = False
            for i in range(N):
                if active[i] and deg[i] < k:
                    active[i] = False
                    core[i] = k - 1
                    changed = True
                    for nb in neighbors[i]:
                        if active[nb]:
                            deg[nb] -= 1
    for i in range(N):
        if active[i]:
            core[i] = max_deg

    # 4. Spectral centrality (principal eigenvector)
    x = np.ones(N) / np.sqrt(N)
    for _ in range(30):
        x_new = A @ x
        norm_x = np.linalg.norm(x_new)
        if norm_x < 1e-12:
            break
        x_new /= norm_x
        if np.linalg.norm(x_new - x) < 1e-10:
            break
        x = x_new
    Spec = x

    # 5. Neighborhood degree variance - measure of how heterogeneous the neighborhood is
    # High variance means connected to both high and low degree nodes
    NeighDegVar = np.zeros(N)
    for i in range(N):
        if neighbors[i].size > 1:
            nd = degrees[neighbors[i]]
            NeighDegVar[i] = np.var(nd)
        elif neighbors[i].size == 1:
            NeighDegVar[i] = 0.0

    # 6. Number of non-neighbors (inverse connectivity)
    NonNeigh = N - 1 - degrees

    # Normalize features to [0, 1]
    def norm(x):
        x = np.asarray(x, dtype=float)
        min_val = np.min(x)
        max_val = np.max(x)
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    D_n = norm(D)
    Clus_n = norm(Clus)
    Core_n = norm(core)
    Spec_n = norm(Spec)
    NDV_n = norm(NeighDegVar)
    NN_n = norm(NonNeigh)

    # Features: all LOWER is BETTER for MIS
    F_D = D_n
    F_Clus = Clus_n
    F_Core = Core_n
    F_Spec = Spec_n
    F_NDV = NDV_n  # High variance = less predictable, maybe worse
    F_NN = NN_n    # More non-neighbors = more isolated = better for MIS? Actually we want HIGH non-neighbors

    # For F_NN, we want HIGH values to be good, so we invert
    F_NN_inv = 1.0 - NN_n

    # --- Fast Greedy Evaluator with caching ---
    def evaluate(scores):
        order = np.argsort(-scores, kind='stable')
        blocked = np.zeros(N, dtype=np.uint8)
        count = 0
        for v in order:
            if blocked[v] == 0:
                count += 1
                blocked[v] = 1
                if neighbors[v].size > 0:
                    blocked[neighbors[v]] = 1
        return count

    def evaluate_with_state(scores):
        """Returns (size, picked_list, blocked_array) for analysis"""
        order = np.argsort(-scores, kind='stable')
        blocked = np.zeros(N, dtype=np.uint8)
        picked = []
        for v in order:
            if blocked[v] == 0:
                picked.append(v)
                blocked[v] = 1
                if neighbors[v].size > 0:
                    blocked[neighbors[v]] = 1
        return len(picked), picked, blocked

    # --- Phase 1: Optimize Heuristic Weights ---
    # Score = -(w1*D + w2*Clus + w3*Core + w4*Spec + w5*NDV + w6*(1-NN))
    # We want to minimize the weighted sum, so score = -weighted_sum
    
    rng = np.random.default_rng(42)
    best_size = -1
    best_weights = np.array([1.0, 0.5, 0.5, 0.5, 0.3, 0.5])
    
    # Initial random restarts
    for _ in range(80):
        w = rng.uniform(0.3, 2.0, 6)
        scores = -(w[0]*F_D + w[1]*F_Clus + w[2]*F_Core + w[3]*F_Spec + w[4]*F_NDV + w[5]*F_NN_inv)
        size = evaluate(scores)
        if size > best_size:
            best_size = size
            best_weights = w.copy()
            
    # Local search on weights
    for _ in range(3000):
        if time.time() > DEADLINE - 25:
            break
        w_new = best_weights + rng.normal(0, 0.08, 6)
        w_new = np.clip(w_new, 0.05, 5.0)
        scores = -(w_new[0]*F_D + w_new[1]*F_Clus + w_new[2]*F_Core + w_new[3]*F_Spec + w_new[4]*F_NDV + w_new[5]*F_NN_inv)
        size = evaluate(scores)
        if size > best_size:
            best_size = size
            best_weights = w_new
        elif rng.random() < 0.03:
             w_new = rng.uniform(0.3, 2.0, 6)
             scores = -(w_new[0]*F_D + w_new[1]*F_Clus + w_new[2]*F_Core + w_new[3]*F_Spec + w_new[4]*F_NDV + w_new[5]*F_NN_inv)
             size = evaluate(scores)
             if size > best_size:
                 best_size = size
                 best_weights = w_new

    # --- Phase 2: Local Search on Vertex Scores ---
    best_scores = -(best_weights[0]*F_D + best_weights[1]*F_Clus + best_weights[2]*F_Core + best_weights[3]*F_Spec + best_weights[4]*F_NDV + best_weights[5]*F_NN_inv)
    current_size = best_size
    
    # Polish: Targeted local search
    # Strategy: Identify vertices that are "expensive" picks (they block many other potential picks)
    # and try to swap them with cheaper alternatives
    
    for iteration in range(30000):
        if time.time() > DEADLINE - 2:
            break
            
        # Get current state
        size, picked, blocked = evaluate_with_state(best_scores)
        
        if size == current_size:
            # Try to find improvement opportunities
            pass
        
        # Strategy 1: Swap a picked vertex with a blocked neighbor that has lower "cost"
        if rng.random() < 0.4 and picked:
            v = picked[rng.integers(0, len(picked))]
            # Find neighbors of v that are blocked
            blocked_neighbors = [nb for nb in neighbors[v] if blocked[nb] == 1 and nb != v]
            if blocked_neighbors:
                u = blocked_neighbors[rng.integers(0, len(blocked_neighbors))]
                # Try swapping v and u
                old_sv = best_scores[v]
                old_su = best_scores[u]
                best_scores[v] = old_su
                best_scores[u] = old_sv
                
                new_size = evaluate(best_scores)
                if new_size > current_size:
                    current_size = new_size
                else:
                    best_scores[v] = old_sv
                    best_scores[u] = old_su
        
        # Strategy 2: Swap a picked vertex with a random unpicked, unblocked vertex
        elif rng.random() < 0.3 and picked:
            v = picked[rng.integers(0, len(picked))]
            # Find vertices that are not blocked and not picked
            candidate_mask = (blocked == 0)
            candidate_mask[picked] = False
            candidates = np.where(candidate_mask)[0]
            if len(candidates) > 0:
                u = candidates[rng.integers(0, len(candidates))]
                old_sv = best_scores[v]
                old_su = best_scores[u]
                best_scores[v] = old_su
                best_scores[u] = old_sv
                
                new_size = evaluate(best_scores)
                if new_size > current_size:
                    current_size = new_size
                else:
                    best_scores[v] = old_sv
                    best_scores[u] = old_su
        
        # Strategy 3: Random swap (exploration)
        else:
            i = rng.integers(0, N)
            j = rng.integers(0, N)
            if i == j:
                continue
            old_si = best_scores[i]
            old_sj = best_scores[j]
            best_scores[i] = old_sj
            best_scores[j] = old_si
            
            new_size = evaluate(best_scores)
            if new_size > current_size:
                current_size = new_size
            else:
                best_scores[i] = old_si
                best_scores[j] = old_sj

    return best_scores
# EVOLVE-BLOCK-END
