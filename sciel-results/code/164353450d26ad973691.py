"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def _topo_order(n, preds):
    """Kahn's algorithm; returns list of tasks in topological order and successors."""
    indeg = np.zeros(n, dtype=int)
    succs = [[] for _ in range(n)]
    for i, ps in enumerate(preds):
        for p in ps:
            succs[p].append(i)
            indeg[i] += 1
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) < n:
        remaining = [i for i in range(n) if i not in set(order)]
        order.extend(remaining)
    return order, succs

def _simulate_greedy_optimized(times, preds, succs, ct, priorities):
    n = len(times)
    assigned = [False] * n
    unassigned_preds = [len(preds[i]) for i in range(n)]
    
    stations = 0
    current_load = 0.0
    remaining = n
    total_waste = 0.0
    
    while remaining > 0:
        best_task = -1
        best_priority = -np.inf
        
        for i in range(n):
            if not assigned[i] and unassigned_preds[i] == 0:
                if times[i] <= ct - current_load:
                    if priorities[i] > best_priority:
                        best_priority = priorities[i]
                        best_task = i
        
        if best_task == -1:
            if current_load > 0:
                stations += 1
                total_waste += (ct - current_load)
                current_load = 0.0
            else:
                # Safety break for invalid DAGs or deadlock
                break
        else:
            assigned[best_task] = True
            current_load += times[best_task]
            remaining -= 1
            
            for s in succs[best_task]:
                unassigned_preds[s] -= 1
                
    # If there is leftover load in the last station, count it
    if current_load > 0:
        stations += 1
        total_waste += (ct - current_load)

    return stations, total_waste

def _compute_features(instance):
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    order, succs = _topo_order(n, preds)
    
    # 1. Critical Path (Longest path from i to any terminal, including i)
    cp = np.zeros(n, dtype=float)
    for i in reversed(order):
        best = 0.0
        for s in succs[i]:
            if cp[s] > best:
                best = cp[s]
        cp[i] = times[i] + best
        
    # 2. Depth from Start (Longest path from any source to i, including i)
    depth = np.zeros(n, dtype=float)
    for i in order:
        best = 0.0
        for p in preds[i]:
            if depth[p] > best:
                best = depth[p]
        depth[i] = best + times[i]
        
    # 3. Transitive Successor Count
    trans_succ_count = np.zeros(n, dtype=int)
    for i in reversed(order):
        c = 0
        for s in succs[i]:
            c += 1 + trans_succ_count[s]
        trans_succ_count[i] = c
        
    # 4. Direct Successor Count
    direct_succ_count = np.array([len(succs[i]) for i in range(n)], dtype=float)
    
    # 5. Direct Predecessor Count
    direct_pred_count = np.array([len(preds[i]) for i in range(n)], dtype=float)
    
    # 6. Time
    time_feat = times.copy()
    
    features = {
        'cp': cp,
        'depth': depth,
        'trans_succ': trans_succ_count.astype(float),
        'direct_succ': direct_succ_count,
        'direct_pred': direct_pred_count,
        'time': time_feat
    }
    
    return features, order, succs

def _norm(x):
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def priority(instance):
    t0 = time.time()
    deadline = 140.0 # Leave 10s margin for 150s kill
    
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    ct = float(instance["cycle_time"])
    
    features, order, succs = _compute_features(instance)
    
    # Normalize features
    norm_features = {k: _norm(v) for k, v in features.items()}
    
    # Define feature vector for linear combination
    # Order matters for weight vector indexing
    feat_names = ['cp', 'depth', 'trans_succ', 'direct_succ', 'direct_pred', 'time']
    feat_matrix = np.vstack([norm_features[name] for name in feat_names]).T # n x 6
    
    # Initial weights: Heuristic guess
    w0 = np.array([0.4, 0.1, 0.1, 0.1, 0.0, 0.3])
    w0 = w0 / w0.sum()
    
    current_scores = feat_matrix @ w0
    current_stations, current_waste = _simulate_greedy_optimized(times, preds, succs, ct, current_scores)
    
    best_scores = current_scores.copy()
    best_stations = current_stations
    best_waste = current_waste
    
    # Nelder-Mead Simplex Optimization
    # Dimensions: 6 (weights)
    # We optimize a surrogate: stations + 0.01 * waste
    # This helps navigate plateaus in the station count.
    
    def objective(w):
        # Ensure w is non-negative and normalized
        w = np.maximum(w, 0)
        s = w.sum()
        if s < 1e-9:
            return 1e6, 0.0
        w = w / s
        scores = feat_matrix @ w
        stations, waste = _simulate_greedy_optimized(times, preds, succs, ct, scores)
        # Surrogate objective: lower is better
        # We want to minimize stations. Waste is a tie-breaker.
        return stations + 0.01 * waste, (stations, waste)
    
    # Initialize simplex
    # Start with w0 and small perturbations
    simplex = [w0.copy()]
    for i in range(len(feat_names)):
        vertex = w0.copy()
        vertex[i] += 0.1
        vertex = np.maximum(vertex, 0)
        s = vertex.sum()
        if s > 1e-9:
            vertex = vertex / s
        simplex.append(vertex)
    
    # Evaluate initial simplex
    simplex_fvals = []
    simplex_details = []
    for v in simplex:
        f, details = objective(v)
        simplex_fvals.append(f)
        simplex_details.append(details)
        
    # Sort simplex by fval
    order_idx = np.argsort(simplex_fvals)
    simplex = [simplex[i] for i in order_idx]
    simplex_fvals = [simplex_fvals[i] for i in order_idx]
    simplex_details = [simplex_details[i] for i in order_idx]
    
    best_w = simplex[0].copy()
    best_fval = simplex_fvals[0]
    best_stations = simplex_details[0][0]
    best_waste = simplex_details[0][1]
    best_scores = feat_matrix @ best_w
    
    alpha = 1.0  # Reflection
    gamma = 2.0  # Expansion
    rho = 0.5   # Contraction
    sigma = 0.5 # Shrink
    
    max_iter = 2000
    iter_count = 0
    
    while time.time() - t0 < deadline and iter_count < max_iter:
        iter_count += 1
        
        # Worst point is the last one
        x_worst = simplex[-1]
        f_worst = simplex_fvals[-1]
        
        # Centroid of all but worst
        x_centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_ref = x_centroid + alpha * (x_centroid - x_worst)
        x_ref = np.maximum(x_ref, 0)
        s = x_ref.sum()
        if s > 1e-9:
            x_ref = x_ref / s
        f_ref, det_ref = objective(x_ref)
        
        if f_ref < simplex_fvals[0]:
            # Try expansion
            x_exp = x_centroid + gamma * (x_ref - x_centroid)
            x_exp = np.maximum(x_exp, 0)
            s = x_exp.sum()
            if s > 1e-9:
                x_exp = x_exp / s
            f_exp, det_exp = objective(x_exp)
            
            if f_exp < f_ref:
                simplex[-1] = x_exp
                simplex_fvals[-1] = f_exp
                simplex_details[-1] = det_exp
            else:
                simplex[-1] = x_ref
                simplex_fvals[-1] = f_ref
                simplex_details[-1] = det_ref
        elif f_ref < simplex_fvals[-2]:
            # Accept reflection
            simplex[-1] = x_ref
            simplex_fvals[-1] = f_ref
            simplex_details[-1] = det_ref
        else:
            # Contraction
            x_con = x_centroid + rho * (x_worst - x_centroid)
            x_con = np.maximum(x_con, 0)
            s = x_con.sum()
            if s > 1e-9:
                x_con = x_con / s
            f_con, det_con = objective(x_con)
            
            if f_con < f_worst:
                simplex[-1] = x_con
                simplex_fvals[-1] = f_con
                simplex_details[-1] = det_con
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f_sh, det_sh = objective(simplex[i])
                    simplex_fvals[i] = f_sh
                    simplex_details[i] = det_sh
                
        # Re-sort simplex
        order_idx = np.argsort(simplex_fvals)
        simplex = [simplex[i] for i in order_idx]
        simplex_fvals = [simplex_fvals[i] for i in order_idx]
        simplex_details = [simplex_details[i] for i in order_idx]
        
        # Update global best based on actual stations, not surrogate
        # But we track the best surrogate point as the best candidate
        # If the best surrogate point has fewer stations than our recorded best, update
        if simplex_details[0][0] < best_stations or (simplex_details[0][0] == best_stations and simplex_details[0][1] < best_waste):
            best_stations = simplex_details[0][0]
            best_waste = simplex_details[0][1]
            best_w = simplex[0].copy()
            best_scores = feat_matrix @ best_w
            
        # Check for convergence (optional, but good for time management)
        if len(set([round(f, 6) for f in simplex_fvals])) == 1:
            break

    return np.asarray(best_scores, dtype=float)

# EVOLVE-BLOCK-END
