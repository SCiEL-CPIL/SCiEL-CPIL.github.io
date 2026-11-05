"""Resource-Constrained Project Scheduling with Resource Opportunity Cost."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time

_TIME_LIMIT = 145.0
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
        
        # Resource Pressure: Sum of ratios
        res_pressure = np.sum(demand_matrix / safe_caps, axis=1)
        # Max Resource Pressure: Max of ratios
        max_res_pressure = np.max(demand_matrix / safe_caps, axis=1)
        
        # NEW: Resource Opportunity Cost
        # duration * (1 / remaining_slack)
        # We approximate "remaining slack" for a resource r as (capacity - demand) / capacity
        # If demand >= capacity, slack is 0, cost is infinity (or very high).
        # Actually, the strategy says: "product of its duration and the inverse of the remaining resource slack for its most constrained resource"
        # Let's interpret "most constrained resource" as the one with the highest demand/capacity ratio.
        # Or simply calculate opportunity cost for each resource and take the max?
        # "explicitly penalizing activities that consume scarce resources for long periods"
        
        # Let's calculate: Cost_i = Duration_i * max_r ( 1 / (1 - demand_ir / capacity_r) )
        # This blows up as demand approaches capacity.
        
        ratios = demand_matrix / safe_caps
        # Avoid division by zero if ratio is 1.0
        # If ratio >= 1, it's infeasible or tight. Let's cap it or use a large number.
        # For priority, we want HIGH score for HIGH cost (do it first? or avoid it?)
        # Wait, the strategy says: "priority ... is determined by ... opportunity cost ... thereby explicitly penalizing activities that consume scarce resources for long periods"
        # In SGS, higher priority means started EARLIER.
        # If we penalize scarce resource usage, do we start them EARLIER or LATER?
        # Usually, you want to start critical/bottleneck tasks early to free up resources for others?
        # Or do you delay them to avoid contention?
        # Standard RCPSP heuristics (like MTP, RCP) often prioritize tasks with HIGH resource demand to clear them out.
        # The prompt says: "penalizing activities that consume scarce resources".
        # If I penalize them, their score goes DOWN?
        # "Scored by mean makespan ... lower is better".
        # "priority(instance) -> one score per activity ... starts the highest-priority eligible activity"
        # So HIGH score = START EARLY.
        
        # If I want to minimize makespan, and a task consumes a scarce resource for a long time, it blocks other tasks.
        # Should I start it early to get it out of the way? Or late to avoid blocking?
        # Literature suggests that prioritizing tasks with high resource demand (MTP) is often good.
        # The "opportunity cost" concept: The cost of delaying this task is high because it ties up resources.
        # Therefore, we should start it EARLY to minimize the total resource contention time.
        # So, HIGH Opportunity Cost -> HIGH Priority.
        
        # Let's compute:
        # slack_r = 1 - (demand_ir / capacity_r)
        # If slack_r <= 0, cost is very high.
        
        # We want the "most constrained resource" part.
        # Let's use the max ratio for the task.
        max_ratio = np.max(ratios, axis=1)
        
        # Opportunity Cost = Duration * (1 / (1 - max_ratio))
        # To avoid inf, we add a small epsilon or clip.
        # Let's use: Duration / (1 - max_ratio + 1e-6)
        
        denom = 1.0 - max_ratio + 1e-6
        opp_cost = durations / denom
        
        # Normalize opp_cost
        oc_max = np.max(opp_cost)
        if oc_max > 0:
            f_oc = opp_cost / oc_max
        else:
            f_oc = np.zeros(n)
            
    else:
        res_pressure = np.zeros(n)
        max_res_pressure = np.zeros(n)
        f_oc = np.zeros(n)
        demand_matrix = None
        capacities_arr = None
        safe_caps = None

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
    # Inspiration 1 used MGR (Minimum Number of Successors) which was better?
    # "Switching from ... High Successors ... to ... Low Successors (MGR) ... 1.053 -> 1.057"
    # Wait, the lesson says MGR (Low Successors) improved score (lower is better).
    # So LOW successor count is BETTER priority?
    # Let's invert successor count.
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
    sc_max = np.max(succ_count) if np.max(succ_count) > 0 else 1.0
    # f_sc should be high for LOW successor count.
    f_sc = 1.0 - (succ_count / sc_max)

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

    # 7. Opportunity Cost
    # f_oc is already computed.

    features = np.vstack([
        f_float, 
        f_bl, 
        f_rp, 
        f_sc, 
        f_dur, 
        f_wf,
        f_oc
    ]).T # Shape (n, 7)
    
    return features, successors, durations, pred_lists, demand_matrix, capacities_arr, safe_caps

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
    
    while any(remaining):
        # Find eligible tasks
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
                break
            next_finish = min(t for t, _ in active_tasks)
            for ft, tid in list(active_tasks):
                if ft <= next_finish + 1e-9:
                    active_tasks.remove([ft, tid])
                    res_usage -= demand_matrix[tid]
            time = next_finish
            continue
            
        # Sort candidates by priority
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
                
        if not started_any:
            if not active_tasks:
                break
            next_finish = min(t for t, _ in active_tasks)
            for ft, tid in list(active_tasks):
                if ft <= next_finish + 1e-9:
                    active_tasks.remove([ft, tid])
                    res_usage -= demand_matrix[tid]
            time = next_finish
            continue
            
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
    features, successors, durations, pred_lists, demand_matrix, capacities_arr, safe_caps = _build_graph(instance)
    
    # Get number of features
    n_features = features.shape[1]
    
    # Define objective function: Minimize Makespan
    def objective(weights):
        # Normalize weights to sum to 1
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
    # Features: [f_float, f_bl, f_rp, f_sc, f_dur, f_wf, f_oc]
    # f_sc is now inverted (low succ count is high priority)
    # f_oc is Opportunity Cost (high cost is high priority)
    
    # Initial weights based on importance:
    # Float is usually strong.
    # OC is the new addition, give it significant weight.
    w_current = np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
    w_current = np.clip(w_current, 0, None)
    w_current = w_current / np.sum(w_current)
    
    best_makespan = objective(w_current)
    best_weights = w_current.copy()
    
    # Coordinate Descent with random restarts
    max_evals = 8000
    evals_done = 1
    np.random.seed(42)
    
    while evals_done < max_evals:
        # Check time
        if time.time() - _START_TIME > _TIME_LIMIT - 2:
            break
            
        # Coordinate Descent step: Perturb one weight
        d = np.random.randint(n_features)
        
        w_new = best_weights.copy()
        step_size = 0.1
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
            
    # Final check
    final_priorities = features @ best_weights
    return final_priorities

# EVOLVE-BLOCK-END
