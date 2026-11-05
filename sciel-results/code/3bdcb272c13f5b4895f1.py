"""Rewritten program for Resource-Constrained Project Scheduling."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time

_TIME_LIMIT = 148.0
_START_TIME = None

def _build_graph(instance):
    n = len(instance["durations"])
    pred_lists = []
    for p in instance.get("predecessors", [None] * n):
        if p is None:
            pred_lists.append([])
        elif isinstance(p, (list, tuple, np.ndarray)):
            pred_lists.append([x for x in p if x is not None])
        else:
            pred_lists.append([p])
            
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in pred_lists[i]:
            if 0 <= p < n:
                successors[p].append(i)
                in_degree[i] += 1
                
    # Topological Sort
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    top_order = []
    while queue:
        u = queue.popleft()
        top_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(top_order) < n:
        top_order = list(range(n))
        
    # CPM Forward/Backward
    durations = np.asarray(instance["durations"], dtype=float)
    es = np.zeros(n)
    ef = np.zeros(n)
    for u in top_order:
        for v in successors[u]:
            if es[v] < ef[u]:
                es[v] = ef[u]
        ef[u] = es[u] + durations[u]
        
    project_end = np.max(ef) if n > 0 else 0.0
    lf = np.full(n, project_end)
    ls = np.zeros(n)
    for u in reversed(top_order):
        for v in successors[u]:
            if lf[u] > ls[v]:
                lf[u] = ls[v]
        ls[u] = lf[u] - durations[u]
        
    total_float = ls - es
    bottom_level = np.zeros(n)
    for u in reversed(top_order):
        for v in successors[u]:
            candidate = bottom_level[v] + durations[v]
            if candidate > bottom_level[u]:
                bottom_level[u] = candidate
                
    # Resource Metrics
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    if demands is not None and capacities is not None:
        demand_matrix = np.zeros((n, len(capacities)), dtype=float)
        if isinstance(demands, list):
            if len(demands) > 0:
                if isinstance(demands[0], dict):
                    for i, d in enumerate(demands):
                        for r, val in d.items():
                            if r < len(capacities):
                                demand_matrix[i, r] = val
                else:
                    dm = np.asarray(demands, dtype=float)
                    if dm.shape[0] == n:
                        if dm.ndim == 1:
                            demand_matrix[:, 0] = dm
                        else:
                            cols = min(dm.shape[1], len(capacities))
                            demand_matrix[:, :cols] = dm[:, :cols]
        capacities_arr = np.asarray(capacities, dtype=float)
        safe_caps = np.where(capacities_arr > 0, capacities_arr, 1.0)
        res_pressure = np.sum(demand_matrix / safe_caps, axis=1)
        max_res_pressure = np.max(demand_matrix / safe_caps, axis=1)
    else:
        res_pressure = np.zeros(n)
        max_res_pressure = np.zeros(n)

    # Features
    # 1. Neg Float (higher is better)
    float_max = np.max(total_float)
    if float_max > 0:
        f_float = 1.0 - (total_float / float_max)
    else:
        f_float = np.ones(n)
        
    # 2. Bottom Level (higher is better)
    bl_max = np.max(bottom_level)
    if bl_max > 0:
        f_bl = bottom_level / bl_max
    else:
        f_bl = np.ones(n)
        
    # 3. Resource Pressure (higher is better)
    rp_max = np.max(res_pressure)
    if rp_max > 0:
        f_rp = res_pressure / rp_max
    else:
        f_rp = np.zeros(n)
        
    # 4. Successor Count (more successors often means higher priority to avoid bottlenecks)
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
    sc_max = np.max(succ_count) if np.max(succ_count) > 0 else 1.0
    f_sc = succ_count / sc_max

    # 5. Duration (longer activities usually higher priority)
    d_max = np.max(durations) if np.max(durations) > 0 else 1.0
    f_dur = durations / d_max
    
    # 6. Weighted Succ Float (sum of floats of successors, normalized)
    ws_float = np.zeros(n)
    for u in range(n):
        for v in successors[u]:
            ws_float[u] += total_float[v]
    wf_max = np.max(ws_float)
    if wf_max > 0:
        f_wf = 1.0 - (ws_float / wf_max) # Invert so lower succ float = higher priority
    else:
        f_wf = np.ones(n)
    
    # 7. Resource Bottleneck Feature from empirical simulation
    # Run a quick greedy schedule to detect which activities cause resource contention
    f_bottleneck = np.zeros(n)
    if demand_matrix is not None and capacities_arr is not None and len(capacities_arr) > 0:
        # Simple greedy: schedule in topological order, track resource usage
        # Count how often each activity blocks others due to resource limits
        time = 0.0
        active_tasks = []  # (finish_time, task_id)
        res_usage = np.zeros(len(capacities_arr))
        remaining_tasks = set(range(n))
        finished_times = {}
        
        # Pre-compute predecessors for quick check
        pred_sets = []
        for p in instance.get("predecessors", [None] * n):
            if p is None:
                pred_sets.append(set())
            elif isinstance(p, (list, tuple, np.ndarray)):
                pred_sets.append(set(x for x in p if x is not None))
            else:
                pred_sets.append({p})
        
        max_iter = 1000  # Safety
        iter_count = 0
        while remaining_tasks and iter_count < max_iter:
            iter_count += 1
            # Find eligible tasks
            eligible = []
            for i in remaining_tasks:
                all_done = all(p in finished_times for p in pred_sets[i])
                if all_done:
                    eligible.append(i)
            
            if not eligible:
                if active_tasks:
                    next_finish = min(t for t, _ in active_tasks)
                    for ft, tid in list(active_tasks):
                        if ft <= next_finish + 1e-9:
                            active_tasks.remove((ft, tid))
                            res_usage -= demand_matrix[tid]
                            finished_times[tid] = ft
                            remaining_tasks.discard(tid)
                    time = next_finish
                continue
            
            # Try to start eligible tasks (in index order for determinism)
            eligible.sort()
            started_any = False
            for i in eligible:
                if np.all(demand_matrix[i] <= res_usage + capacities_arr - 1e-9):
                    res_usage += demand_matrix[i]
                    ft = time + durations[i]
                    active_tasks.append((ft, i))
                    started_any = True
                else:
                    # This task is blocked by resources - mark as bottleneck contributor
                    # Weight by how many resources are tight
                    tight_count = np.sum(demand_matrix[i] > capacities_arr - res_usage - 1e-9)
                    f_bottleneck[i] += tight_count / max(len(capacities_arr), 1)
            
            if not started_any:
                if active_tasks:
                    next_finish = min(t for t, _ in active_tasks)
                    for ft, tid in list(active_tasks):
                        if ft <= next_finish + 1e-9:
                            active_tasks.remove((ft, tid))
                            res_usage -= demand_matrix[tid]
                            finished_times[tid] = ft
                            remaining_tasks.discard(tid)
                    time = next_finish
                else:
                    break
        
        # Normalize bottleneck feature
        bn_max = np.max(f_bottleneck)
        if bn_max > 0:
            f_bottleneck = f_bottleneck / bn_max
        else:
            f_bottleneck = np.zeros(n)

    features = np.vstack([
        f_float, 
        f_bl, 
        f_rp, 
        f_sc, 
        f_dur, 
        f_wf,
        f_bottleneck
    ]).T # Shape (n, 7)
    
    return features, successors, durations, res_pressure, safe_caps, demand_matrix if 'demand_matrix' in locals() else None, capacities_arr if 'capacities_arr' in locals() else None

def _simulate_serial_schedule(instance, priorities, successors, durations, demand_matrix, capacities_arr, safe_caps):
    n = len(durations)
    # Sort by priority (descending). Stable sort by index for tie-breaking consistency
    # Using np.argsort with kind='stable' isn't directly available in all versions without specifying, 
    # but standard sorts in numpy are stable since 1.15? No, quicksort is not. 
    # We need a stable sort to ensure deterministic behavior and reproducibility.
    # indices = np.argsort(-priorities, kind='stable')
    
    # Manual stable sort for safety
    order = sorted(range(n), key=lambda i: -priorities[i])
    
    start_times = np.full(n, -1)
    current_time = 0.0
    remaining = [True] * n
    finished = [False] * n
    
    # Track resource usage over time
    # Since n is small, we can just iterate through time or events.
    # Event-based simulation is more efficient.
    
    events = []
    # We will simulate step by step.
    # At each step, check if any task can start.
    # If not, jump to the next finish event.
    
    time = 0.0
    active_resources = np.zeros(len(capacities_arr))
    active_tasks = {} # task_id -> finish_time
    
    while any(remaining):
        # Try to start tasks
        started_any = False
        for i in order:
            if not remaining[i]:
                continue
            # Check precedence
            preds_done = True
            # We need pred lists, but we passed successors. Let's assume we can derive or just check if start_time of preds is set.
            # Actually, simpler: check if all predecessors have finished.
            # We don't have pred_lists passed. Let's refactor to pass pred_lists or compute on fly.
            # For speed, let's pre-compute pred_lists in main function and pass.
            pass # Placeholder, see updated function below
            
    # This simple loop is inefficient if we don't manage events well.
    # Better approach: 
    # 1. Identify eligible tasks (all preds finished).
    # 2. Among eligible, sort by priority.
    # 3. Try to start highest priority task if resources allow.
    # 4. If no tasks can start, jump time to min finish time of active tasks.
    
    # Let's rewrite simulation properly
    time = 0.0
    active = [] # list of (finish_time, task_id)
    res_usage = np.zeros(len(capacities_arr))
    
    # We need pred_lists for this simulation
    # I will modify _simulate to take pred_lists
    
    # Re-implementing simulation logic here for clarity in the final code block
    # This inner function is discarded, the real one is below.
    pass

def _simulate_schedule_fast(pred_lists, successors, durations, demand_matrix, capacities_arr, priorities):
    n = len(durations)
    # Stable sort by priority
    order = sorted(range(n), key=lambda i: -priorities[i])
    
    time = 0.0
    remaining = [True] * n
    start_times = np.full(n, np.nan)
    finish_times = np.full(n, np.nan)
    active_tasks = [] # list of [finish_time, task_id]
    res_usage = np.zeros(len(capacities_arr))
    
    # To make it faster, we don't re-scan the whole 'order' list every time if we can avoid it.
    # But n=30 is small. Scanning 30 items is trivial.
    
    while any(remaining):
        # Find eligible tasks
        # Eligible if all preds are finished
        # To avoid O(N^2) per step, we could maintain a set of ready tasks.
        # Given N=30, O(N^2) is 900 ops. Even 10k steps is 9M ops, which is < 0.1s in Python.
        
        candidates = []
        for i in range(n):
            if not remaining[i]:
                continue
            # Check preds
            ready = True
            for p in pred_lists[i]:
                if np.isnan(finish_times[p]) or finish_times[p] > time:
                    ready = False
                    break
            if ready:
                candidates.append(i)
                
        if not candidates:
            # Jump to next event
            if not active_tasks:
                # Deadlock or error?
                break
            next_finish = min(t for t, _ in active_tasks)
            # Process all tasks finishing at next_finish
            for ft, tid in list(active_tasks):
                if ft <= next_finish + 1e-9:
                    active_tasks.remove([ft, tid])
                    res_usage -= demand_matrix[tid]
            time = next_finish
            continue
            
        # Sort candidates by priority (already in 'order' but we filtered)
        # 'candidates' is in index order. We need to sort by priority.
        candidates.sort(key=lambda x: -priorities[x])
        
        started_any = False
        for i in candidates:
            # Check resources
            if np.all(demand_matrix[i] <= res_usage + capacities_arr - 1e-9):
                # Start task i
                res_usage += demand_matrix[i]
                start_times[i] = time
                finish_times[i] = time + durations[i]
                remaining[i] = False
                active_tasks.append([finish_times[i], i])
                started_any = True
                # Do not break, try to start more tasks at same time if possible
                # (Parallel start is allowed if resources allow)
                
        if not started_any:
            # No candidate could start due to resources
            if not active_tasks:
                # Deadlock
                break
            next_finish = min(t for t, _ in active_tasks)
            for ft, tid in list(active_tasks):
                if ft <= next_finish + 1e-9:
                    active_tasks.remove([ft, tid])
                    res_usage -= demand_matrix[tid]
            time = next_finish
            continue
            
    # If any tasks are still remaining (shouldn't happen), assign max time
    max_time = np.nanmax(finish_times) if not np.all(np.isnan(finish_times)) else 0.0
    return max_time

def priority(instance):
    """
    Score each activity using a Model-and-Optimize approach.
    We define a feature space and optimize weights to minimize makespan.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Fallback
        durations = np.asarray(instance["durations"], dtype=float)
        return durations

    n = len(instance["durations"])
    if n == 0:
        return np.array([])
        
    # Build features and data structures
    features, successors, durations, res_pressure, safe_caps, demand_matrix, capacities_arr = _build_graph(instance)
    
    # Reconstruct pred_lists for simulation
    pred_lists = []
    for p in instance.get("predecessors", [None] * n):
        if p is None:
            pred_lists.append([])
        elif isinstance(p, (list, tuple, np.ndarray)):
            pred_lists.append([x for x in p if x is not None])
        else:
            pred_lists.append([p])
            
    # Get number of features
    n_features = features.shape[1]
    
    # Define objective function: Minimize Makespan
    def objective(weights):
        # Weights should be positive? Or can be negative?
        # Usually, higher priority means "do this sooner".
        # Features are normalized to [0,1] roughly.
        # Let's allow weights to be any real number, but normalize them to sum to 1 or max 1?
        # If weights sum to 0, priorities are 0. Tie-breaking by index.
        # Let's constrain weights to be non-negative and sum to 1 to keep the search space bounded.
        # This acts as a simplex optimization.
        
        # Normalize weights
        w_sum = np.sum(weights)
        if w_sum < 1e-9:
            w_norm = np.ones(n_features) / n_features
        else:
            w_norm = weights / w_sum
            
        priorities = features @ w_norm
        makespan = _simulate_schedule_fast(pred_lists, successors, durations, demand_matrix, capacities_arr, priorities)
        return makespan
        
    # Local Search / Coordinate Descent
    # Start with a reasonable heuristic:
    # Features order: [f_float, f_bl, f_rp, f_sc, f_dur, f_wf, f_bottleneck]
    w_current = np.array([0.35, 0.25, 0.10, 0.05, 0.05, 0.05, 0.15])
    w_current = np.clip(w_current, 0, None)
    w_current = w_current / np.sum(w_current)
    
    best_makespan = objective(w_current)
    best_weights = w_current.copy()
    
    # Coordinate Descent with random restarts
    # Budget: ~130s left.
    # Each eval is ~1-5ms. 130s / 5ms = 26000 evals.
    # Let's do 8000 evals to be safe, then a polish pass.
    
    max_evals = 6000
    evals_done = 1
    np.random.seed(42)
    
    while evals_done < max_evals:
        # Check time
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break
            
        # Coordinate Descent step: Perturb one weight
        # Pick a random dimension
        d = np.random.randint(n_features)
        
        # Generate a new weight with adaptive step size
        w_new = best_weights.copy()
        step_size = 0.1 * (1.0 - 0.5 * evals_done / max_evals)  # Decreasing step
        w_new[d] += np.random.uniform(-step_size, step_size)
        w_new = np.clip(w_new, 0, None)
        
        w_sum = np.sum(w_new)
        if w_sum < 1e-9:
            w_new = best_weights.copy()
        else:
            w_new = w_new / w_sum
            
        makespan = objective(w_new)
        evals_done += 1
        
        if makespan < best_makespan - 1e-6:
            best_makespan = makespan
            best_weights = w_new.copy()
    
    # Polish pass: finer-grained search around best_weights
    if time.time() - _START_TIME < _TIME_LIMIT - 3:
        polish_evals = 2000
        polish_done = 0
        while polish_done < polish_evals:
            if time.time() - _START_TIME > _TIME_LIMIT - 2:
                break
                
            # Perturb 1-2 dimensions with smaller steps
            n_perturb = np.random.choice([1, 2], p=[0.7, 0.3])
            dims = np.random.choice(n_features, size=n_perturb, replace=False)
            
            w_new = best_weights.copy()
            for d in dims:
                w_new[d] += np.random.uniform(-0.03, 0.03)
            w_new = np.clip(w_new, 0, None)
            
            w_sum = np.sum(w_new)
            if w_sum < 1e-9:
                w_new = best_weights.copy()
            else:
                w_new = w_new / w_sum
                
            makespan = objective(w_new)
            polish_done += 1
            
            if makespan < best_makespan - 1e-7:
                best_makespan = makespan
                best_weights = w_new.copy()
    
    # Final check
    final_priorities = features @ best_weights
    return final_priorities

# EVOLVE-BLOCK-END
