"""Assembly line balancing with model-and-optimize priority scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque
from itertools import product

def _greedy_balance(times, predecessors, cycle_time, priority_scores):
    """
    Greedy station filling: repeatedly pick the highest-priority eligible task that fits.
    Returns number of stations.
    """
    n = len(times)
    if n == 0:
        return 0
    
    successors = defaultdict(list)
    unassigned_preds = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    assigned = [False] * n
    eligible = set(i for i in range(n) if unassigned_preds[i] == 0)
    
    if n > 0 and not eligible:
        return 10**9
    
    station_count = 1
    current_load = 0
    
    while eligible:
        candidates = sorted(eligible, key=lambda i: -priority_scores[i])
        placed = False
        
        for task in candidates:
            if times[task] <= cycle_time - current_load:
                assigned[task] = True
                current_load += times[task]
                eligible.discard(task)
                placed = True
                
                for succ in successors[task]:
                    unassigned_preds[succ] -= 1
                    if unassigned_preds[succ] == 0:
                        eligible.add(succ)
                break
        
        if not placed:
            station_count += 1
            current_load = 0
            if any(times[t] > cycle_time for t in eligible):
                return 10**9
    
    return station_count


def _compute_features(times, predecessors, cycle_time):
    """Compute various features for each task."""
    n = len(times)
    if n == 0:
        return np.zeros((n, 0))
    
    successors = defaultdict(list)
    pred_map = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
            pred_map[i].append(pred)
    
    # Topological sort
    in_degree = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
    
    topo_order = []
    in_deg_copy = in_degree.copy()
    q = deque()
    for i in range(n):
        if in_deg_copy[i] == 0:
            q.append(i)
    
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)
    
    if len(topo_order) != n:
        return None  # Cycle detected
    
    # Critical path (longest path from task to end)
    cp = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        cp[u] = times[u]
        for v in successors[u]:
            cp[u] = max(cp[u], times[u] + cp[v])
    
    # Smooth surrogate: soft-max critical path (log-sum-exp based)
    # This provides a differentiable-like signal for local search
    beta = 5.0  # temperature parameter
    cp_smooth = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if not successors[u]:
            cp_smooth[u] = times[u]
        else:
            # soft-max over (times[u] + cp_smooth[v]) for all successors
            vals = [times[u] + cp_smooth[v] for v in successors[u]]
            m = max(vals)
            # log-sum-exp for stability
            lse = m + np.log(sum(np.exp(x - m) for x in vals) / len(vals))
            cp_smooth[u] = lse / beta * beta  # scaled soft-max
            # Actually let's just use the soft-max value directly
            cp_smooth[u] = m + np.log(np.mean([np.exp(x - m) for x in vals]))
    
    # Position in topological order (0-indexed, earlier = smaller)
    topo_pos = np.zeros(n, dtype=float)
    for idx, u in enumerate(topo_order):
        topo_pos[u] = idx
    
    # Number of direct successors
    num_direct_succ = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Number of direct predecessors
    num_direct_pred = np.array([len(predecessors[i]) for i in range(n)], dtype=float)
    
    # Transitive successor count
    def count_trans(start, adj):
        visited = set()
        stack = [start]
        while stack:
            node = stack.pop()
            for s in adj[node]:
                if s not in visited:
                    visited.add(s)
                    stack.append(s)
        return len(visited)
    
    num_trans_succ = np.array([count_trans(i, successors) for i in range(n)], dtype=float)
    num_trans_pred = np.array([count_trans(i, pred_map) for i in range(n)], dtype=float)
    
    # Slack: cycle_time - time (positive means there's room)
    slack = cycle_time - times
    
    # Bottleneck slack: time / (1 + total transitive successors)
    # Tasks with high time relative to their branching complexity are bottlenecks
    bottleneck_slack = times / (1.0 + num_trans_succ)
    
    # Feature matrix
    features = np.column_stack([
        times,          # 0: processing time
        cp,             # 1: critical path
        topo_pos,       # 2: topological position
        num_direct_succ, # 3: direct successors
        num_direct_pred, # 4: direct predecessors
        num_trans_succ,  # 5: transitive successors
        num_trans_pred,  # 6: transitive predecessors
        slack,          # 7: slack
        times * cp,     # 8: time * critical path
        num_trans_succ / (n + 1),  # 9: normalized transitive successors
        1.0 / (1.0 + num_direct_pred),  # 10: inverse pred factor
        bottleneck_slack,  # 11: bottleneck slack (time / transitive branching)
        cp_smooth,         # 12: smooth surrogate critical path
    ])
    
    return features


def _fit_and_optimize(times, predecessors, cycle_time, deadline):
    """
    Model-and-optimize: parameterize priority as linear combination of features,
    fit coefficients by trying many combinations and selecting the best.
    """
    n = len(times)
    if n == 0:
        return np.array([])
    
    features = _compute_features(times, predecessors, cycle_time)
    if features is None:
        # Cycle in predecessors; return random priorities
        rng = np.random.RandomState(42)
        return rng.rand(n)
    
    num_features = features.shape[1]
    
    # Precompute some candidate weight vectors
    # Strategy: try different weight combinations
    # Also try individual features and simple combinations
    
    candidates = []
    
    # Individual features
    for f in range(num_features):
        candidates.append(np.eye(num_features)[f])
    
    # All ones (average)
    candidates.append(np.ones(num_features) / num_features)
    
    # Random directions
    rng = np.random.RandomState(42)
    for _ in range(50):
        w = rng.randn(num_features)
        candidates.append(w / (np.linalg.norm(w) + 1e-9))
    
    # Grid search over a few key weights with restricted ranges
    # Focus on: time, critical path, transitive successors, topological position, bottleneck slack
    # Weights for [time, cp, topo_pos, num_direct_succ, num_direct_pred, num_trans_succ, num_trans_pred, slack, time*cp, norm_succ, inv_pred, bottleneck_slack, cp_smooth]
    
    time_w = [0.0, 0.5, 1.0, 1.5, 2.0]
    cp_w = [0.0, 0.5, 1.0, 1.5, 2.0]
    succ_w = [0.0, 0.5, 1.0]
    pred_w = [0.0, -1.0, -2.0]
    bottleneck_w = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    for tw in time_w:
        for cw in cp_w:
            for sw in succ_w:
                for pw in pred_w:
                    for bw in bottleneck_w:
                        w = np.zeros(num_features)
                        w[0] = tw      # time
                        w[1] = cw      # cp
                        w[4] = pw      # num_direct_pred (negative to favor fewer)
                        w[5] = sw      # num_trans_succ
                        w[11] = bw     # bottleneck slack
                        # Normalize
                        norm = np.linalg.norm(w)
                        if norm > 1e-9:
                            w = w / norm
                        candidates.append(w)
    
    # Also try: high time, high cp, low topological position (early in chain)
    for tw in [1.0, 2.0]:
        for cw in [1.0, 2.0]:
            for tpw in [-1.0, -2.0]:
                for bw in [0.0, 1.0, 2.0]:
                    w = np.zeros(num_features)
                    w[0] = tw
                    w[1] = cw
                    w[2] = tpw  # topological position (negative to favor earlier)
                    w[11] = bw  # bottleneck slack
                    norm = np.linalg.norm(w)
                    if norm > 1e-9:
                        w = w / norm
                    candidates.append(w)
    
    # Remove duplicates approximately
    unique_candidates = [candidates[0]]
    for c in candidates[1:]:
        is_dup = False
        for uc in unique_candidates:
            if np.allclose(c, uc, atol=1e-6):
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(c)
    
    candidates = unique_candidates
    
    best_score = float('inf')
    best_priority = None
    best_weights = None
    
    start_time = time.time()
    
    for idx, w in enumerate(candidates):
        if time.time() > deadline - 10:
            break
        
        # Compute priority scores
        pri = features @ w
        
        # Evaluate
        n_stations = _greedy_balance(times, predecessors, cycle_time, pri)
        
        if n_stations < best_score:
            best_score = n_stations
            best_priority = pri
            best_weights = w
            if best_score == 1:
                break
    
    if best_priority is None:
        # Fallback: use critical path
        best_priority = features[:, 1]
    
    # Try to improve with local search: perturb best weights
    if best_weights is not None and time.time() < deadline - 15:
        current_best = best_score
        current_weights = best_weights.copy()
        
        # Coordinate descent / perturb
        for iteration in range(200):
            if time.time() > deadline - 10:
                break
            
            improved = False
            for dim in range(num_features):
                for delta in [-0.5, -0.2, 0.2, 0.5]:
                    new_w = current_weights.copy()
                    old_val = new_w[dim]
                    new_w[dim] = old_val + delta
                    
                    pri = features @ new_w
                    n_stations = _greedy_balance(times, predecessors, cycle_time, pri)
                    
                    if n_stations < current_best:
                        current_best = n_stations
                        current_weights = new_w
                        improved = True
                        if current_best == 1:
                            break
                if current_best == 1:
                    break
            if not improved:
                break
            if current_best == 1:
                break
        
        best_priority = features @ current_weights
        best_score = current_best
    
    # Polish pass: use smooth surrogate to break ties and refine
    if time.time() < deadline - 5 and best_weights is not None:
        # Use cp_smooth as a tiebreaker: add small weight to smooth CP
        polish_w = best_weights.copy()
        polish_w[12] += 0.1  # small boost to smooth critical path
        
        pri_polish = features @ polish_w
        n_stations_polish = _greedy_balance(times, predecessors, cycle_time, pri_polish)
        
        if n_stations_polish < best_score:
            best_priority = pri_polish
            best_score = n_stations_polish
    
    return best_priority


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    if n == 0:
        return np.array([])
    
    # Deadline guard: 150s limit, use 140s for safety
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Model-and-optimize approach
    best_priority = _fit_and_optimize(times, predecessors, cycle_time, deadline)
    
    # If time remains, try a few more random perturbations of the best
    if time.time() < deadline - 5:
        rng = np.random.RandomState(42 + int(time.time() % 1000))
        current_best_score = _greedy_balance(times, predecessors, cycle_time, best_priority)
        
        for _ in range(20):
            if time.time() > deadline - 3:
                break
            # Small random perturbation
            perturbation = rng.randn(n) * 0.1
            trial_priority = best_priority + perturbation
            trial_score = _greedy_balance(times, predecessors, cycle_time, trial_priority)
            if trial_score < current_best_score:
                current_best_score = trial_score
                best_priority = trial_priority
                if current_best_score == 1:
                    break
    
    return best_priority
# EVOLVE-BLOCK-END
