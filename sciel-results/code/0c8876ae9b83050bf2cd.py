import numpy as np
import time

def priority(instance):
    start_time = time.time()
    deadline = start_time + 110.0  # 40 second safety margin
    
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]
    
    n = len(durations)
    n_res = len(capacities)
    
    if n == 0:
        return []
        
    # 1. Pre-compute static features
    
    # Successors
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Forward Pass (ES, EF)
    es = np.zeros(n)
    ef = np.zeros(n)
    for i in range(n):
        for p in predecessors[i]:
            es[i] = max(es[i], ef[p])
        ef[i] = es[i] + durations[i]
    
    project_end_static = np.max(ef)
    
    # Backward Pass (LS, LF)
    ls = np.zeros(n)
    lf = np.zeros(n)
    for i in range(n-1, -1, -1):
        if not successors[i]:
            lf[i] = project_end_static
        else:
            lf[i] = min(ls[s] for s in successors[i])
        ls[i] = lf[i] - durations[i]
        
    total_float = ls - es
    tail = project_end_static - ef
    
    # Depth (Longest path from start) - often correlates with criticality
    depth = np.zeros(n)
    for i in range(n):
        if predecessors[i]:
            depth[i] = max(depth[p] for p in predecessors[i])
        depth[i] += durations[i] # This is actually EF if started at 0, but let's use path length in nodes or time
        # Actually, let's use "Level" or "Rank"
    # Rank: 0 for sources, 1 + max(rank(preds)) for others
    rank = np.zeros(n, dtype=int)
    for i in range(n):
        if predecessors[i]:
            rank[i] = 1 + max(rank[p] for p in predecessors[i])
            
    # Resource Features
    # Max demand ratio per activity (0 to 1)
    if n_res > 0:
        max_demand_ratio = np.max(demands / np.maximum(capacities, 1e-9), axis=1)
        # Sum of demand ratios
        sum_demand_ratio = np.sum(demands / np.maximum(capacities, 1e-9), axis=1)
    else:
        max_demand_ratio = np.zeros(n)
        sum_demand_ratio = np.zeros(n)
        
    # Normalization
    # We want features in [0, 1] where 1 is "more important" (should be scheduled earlier)
    
    # Float: Lower is better. So (max_float - float) / max_float
    max_float = np.max(total_float) if np.max(total_float) > 0 else 1.0
    feat_float = (max_float - total_float) / max_float
    
    # Tail: Higher is better.
    max_tail = np.max(tail) if np.max(tail) > 0 else 1.0
    feat_tail = tail / max_tail
    
    # Duration: Higher is better (longer tasks take more time, scheduling them early might help avoid bottlenecks? Or is it neutral? Usually, long tasks are critical).
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    feat_dur = durations / max_dur
    
    # Resource Demand: Higher is better (resource-intensive tasks should be scheduled when resources are likely available? Or avoid peak? 
    # Actually, in RCPSP, scheduling high-demand tasks early can cause congestion. 
    # However, "Critical" tasks often have high demand. 
    # Let's try both: High demand (feat_res) and Low demand (1 - feat_res) might be useful. 
    # Let's stick to High demand for now, as it's often correlated with criticality.
    max_res = np.max(max_demand_ratio) if np.max(max_demand_ratio) > 0 else 1.0
    feat_res = max_demand_ratio / max_res
    
    # Rank: Higher is better (deeper in the network)
    max_rank = np.max(rank) if np.max(rank) > 0 else 1.0
    feat_rank = rank.astype(float) / max_rank
    
    # Successor Count: Higher is better (bottlenecks)
    succ_count = np.array([len(successors[i]) for i in range(n)])
    max_succ = np.max(succ_count) if np.max(succ_count) > 0 else 1.0
    feat_succ = succ_count.astype(float) / max_succ

    # Feature Matrix
    # Columns: float, tail, dur, res, rank, succ
    features = np.column_stack((feat_float, feat_tail, feat_dur, feat_res, feat_rank, feat_succ))
    
    # Simulation Function
    def simulate(scores):
        """
        Simulates serial schedule generation.
        scores: array of priority scores (higher is better).
        Returns makespan.
        """
        finished = [False] * n
        # scheduled: list of (start, end, res_vector)
        scheduled = []
        makespan = 0
        
        for _ in range(n):
            # Find eligible
            eligible = []
            for i in range(n):
                if finished[i]:
                    continue
                if all(finished[p] for p in predecessors[i]):
                    eligible.append(i)
            
            if not eligible:
                break
                
            # Select highest priority
            # Tie-breaker: lower index
            best_idx = -1
            best_score = -np.inf
            for i in eligible:
                if scores[i] > best_score:
                    best_score = scores[i]
                    best_idx = i
                    
            # Find earliest start time
            es_pred = es[best_idx]
            
            # Candidate times: es_pred and end times of scheduled activities > es_pred
            candidate_times = {es_pred}
            for (s, e, d_vec) in scheduled:
                if e > es_pred:
                    candidate_times.add(e)
            
            valid_start = None
            for t in sorted(candidate_times):
                # Check resource usage in [t, t + durations[best_idx]]
                usage = np.zeros(n_res)
                for (s, e, d_vec) in scheduled:
                    if s < t + durations[best_idx] and e > t:
                        usage += d_vec
                if np.all(usage + demands[best_idx] <= capacities):
                    valid_start = t
                    break
            
            if valid_start is None:
                # Should not happen if capacities are sufficient
                valid_start = es_pred
            
            # Schedule activity
            scheduled.append((valid_start, valid_start + durations[best_idx], demands[best_idx]))
            finished[best_idx] = True
            makespan = max(makespan, valid_start + durations[best_idx])
            
            if time.time() > deadline:
                return makespan
                
        return makespan

    # DYNAMIC LOOK-AHEAD SIMULATION
    # Run a fast baseline schedule using different simple rules to estimate
    # actual completion times under resource constraints.
    
    def fast_simulate(scores, time_limit=2.0):
        """Fast serial simulation with early termination."""
        t0 = time.time()
        finished = [False] * n
        scheduled = []
        makespan = 0
        completion_times = np.zeros(n)
        
        for _ in range(n):
            if time.time() - t0 > time_limit:
                break
            
            eligible = []
            for i in range(n):
                if not finished[i] and all(finished[p] for p in predecessors[i]):
                    eligible.append(i)
            
            if not eligible:
                break
            
            best_idx = max(eligible, key=lambda i: scores[i])
            
            es_pred = es[best_idx]
            candidate_times = {es_pred}
            for (s, e, d_vec) in scheduled:
                if e > es_pred:
                    candidate_times.add(e)
            
            valid_start = None
            for t in sorted(candidate_times):
                usage = np.zeros(n_res)
                for (s, e, d_vec) in scheduled:
                    if s < t + durations[best_idx] and e > t:
                        usage += d_vec
                if np.all(usage + demands[best_idx] <= capacities):
                    valid_start = t
                    break
            
            if valid_start is None:
                valid_start = es_pred
            
            scheduled.append((valid_start, valid_start + durations[best_idx], demands[best_idx]))
            finished[best_idx] = True
            completion_times[best_idx] = valid_start + durations[best_idx]
            makespan = max(makespan, valid_start + durations[best_idx])
        
        return makespan, completion_times
    
    # Run multiple baseline schedules to get dynamic completion time estimates
    # These capture resource contention that static CPM misses
    
    # Baseline 1: Longest Duration First (LDF)
    scores_ldf = -durations  # negative so higher = longer duration
    makespan_ldf, comp_ldf = fast_simulate(scores_ldf, time_limit=1.5)
    
    # Baseline 2: Most Successors First (MSF)
    scores_msf = succ_count.astype(float)
    makespan_msf, comp_msf = fast_simulate(scores_msf, time_limit=1.5)
    
    # Baseline 3: Lowest Float First (LF) - static critical path
    scores_lf = feat_float
    makespan_lf, comp_lf = fast_simulate(scores_lf, time_limit=1.5)
    
    # Baseline 4: Combined static (float + tail)
    scores_static = 0.5 * feat_float + 0.5 * feat_tail
    makespan_static, comp_static = fast_simulate(scores_static, time_limit=1.5)
    
    # Normalize completion times: lower completion = higher priority
    # Use min-max normalization across all activities
    def normalize_completion(comp):
        cmin = np.min(comp)
        cmax = np.max(comp)
        if cmax - cmin < 1e-9:
            return np.zeros(n)
        return (cmax - comp) / (cmax - cmin)
    
    # Dynamic features: inverted completion times from different baselines
    dyn_ldf = normalize_completion(comp_ldf)
    dyn_msf = normalize_completion(comp_msf)
    dyn_lf = normalize_completion(comp_lf)
    dyn_static = normalize_completion(comp_static)
    
    # Blend dynamic estimates: average across baselines for robustness
    # Weight baselines by their makespan quality (better baseline = more reliable)
    all_makespans = np.array([makespan_ldf, makespan_msf, makespan_lf, makespan_static])
    min_mk = np.min(all_makespans)
    max_mk = np.max(all_makespans)
    
    if max_mk - min_mk > 1e-9:
        # Weight: better (lower) makespan gets higher weight
        weights_mk = (max_mk - all_makespans) / (max_mk - min_mk)
        weights_mk += 0.1  # avoid zero
        weights_mk /= np.sum(weights_mk)
    else:
        weights_mk = np.ones(4) / 4.0
    
    # Weighted blend of dynamic completion priorities
    dyn_blend = (weights_mk[0] * dyn_ldf + 
                 weights_mk[1] * dyn_msf + 
                 weights_mk[2] * dyn_lf + 
                 weights_mk[3] * dyn_static)
    
    # Add dynamic feature to static features
    # New feature matrix: [float, tail, dur, res, rank, succ, dyn_blend]
    features_ext = np.column_stack((feat_float, feat_tail, feat_dur, feat_res, feat_rank, feat_succ, dyn_blend))
    
    # Search for best weights over extended feature set
    best_makespan = float('inf')
    best_scores = None
    
    # Initial weights: emphasize dynamic blend heavily
    # Weights are for: [float, tail, dur, res, rank, succ, dyn]
    
    # 1. Pure dynamic
    w1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    
    # 2. Dynamic + Float
    w2 = np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7])
    
    # 3. Dynamic + Tail
    w3 = np.array([0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.7])
    
    # 4. Dynamic + Duration
    w4 = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.7])
    
    # 5. Dynamic + Resource
    w5 = np.array([0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.7])
    
    # 6. Dynamic + Successors
    w6 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.7])
    
    # 7. Static critical path (for comparison)
    w7 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # 8. Balanced
    w8 = np.array([0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4])
    
    initial_weights = [w1, w2, w3, w4, w5, w6, w7, w8]
    
    # Add random weights
    np.random.seed(42)
    for _ in range(15):
        w = np.random.rand(7)
        # Bias toward dynamic feature
        w[6] *= 2.0
        w /= np.sum(w)
        initial_weights.append(w)
        
    # Evaluate initial weights
    candidates = []
    for w in initial_weights:
        scores = features_ext @ w
        m = simulate(scores)
        if m < best_makespan:
            best_makespan = m
            best_scores = scores
        candidates.append((m, w))
        
    # Local Search
    candidates.sort(key=lambda x: x[0])
    
    best_w = candidates[0][1].copy()
    
    # Perform hill climbing with adaptive step size
    iterations = 0
    max_iterations = 400
    step_size = 0.15
    
    while iterations < max_iterations:
        if time.time() > deadline:
            break
            
        improved = False
        # Number of perturbations based on remaining time
        remaining = deadline - time.time()
        n_pert = max(5, min(15, int(remaining / 2.0)))
        
        for _ in range(n_pert):
            if time.time() > deadline:
                break
            # Perturb best_w
            w_pert = best_w + np.random.normal(0, step_size, 7)
            w_pert = np.clip(w_pert, 0, None)
            if np.sum(w_pert) < 1e-9:
                continue
            w_pert /= np.sum(w_pert)
            
            scores = features_ext @ w_pert
            m = simulate(scores)
            
            if m < best_makespan:
                best_makespan = m
                best_scores = scores
                best_w = w_pert.copy()
                improved = True
                break
        
        if not improved:
            step_size *= 0.85
            if step_size < 0.01:
                break
                
        iterations += 1

    if best_scores is None:
        # Fallback to dynamic blend
        best_scores = dyn_blend

    return best_scores.tolist()
