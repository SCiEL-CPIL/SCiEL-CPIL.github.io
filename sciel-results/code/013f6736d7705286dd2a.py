"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
import math
from itertools import combinations

_DEADLINE_SECONDS = 140.0
_START_TIME = time.time()

def _remaining_budget():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)

def _run_serial_sgs(instance, priority_vec):
    """
    Serial Schedule Generation Scheme.
    Optimized for speed using pre-computed structures.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    n = len(durations)
    if n == 0:
        return 0.0
        
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
        
    # Pre-compute resource users
    num_resources = 0
    d_mat = None
    c_vec = None
    if demands is not None and capacities is not None:
        d_mat = np.asarray(demands, dtype=float)
        c_vec = np.asarray(capacities, dtype=float)
        if len(d_mat.shape) == 2:
            num_resources = d_mat.shape[1]
    
    resource_users = [[] for _ in range(num_resources)]
    if demands is not None and d_mat is not None:
        for i in range(n):
            for r in range(num_resources):
                if d_mat[i][r] > 0:
                    resource_users[r].append(i)
                    
    # State
    scheduled = [False] * n
    start_times = np.zeros(n, dtype=float)
    finish_times = np.zeros(n, dtype=float)
    
    scheduled_count = 0
    while scheduled_count < n:
        best_job = -1
        best_prio = -np.inf
        
        # Find eligible job with highest priority
        for i in range(n):
            if not scheduled[i]:
                eligible = True
                for p in predecessors[i]:
                    if not scheduled[p]:
                        eligible = False
                        break
                if eligible:
                    if priority_vec[i] > best_prio:
                        best_prio = priority_vec[i]
                        best_job = i
                        
        if best_job == -1:
            break
            
        # Determine earliest feasible start time
        es_pred = 0.0
        for p in predecessors[best_job]:
            if finish_times[p] > es_pred:
                es_pred = finish_times[p]
                
        # Candidate times: predecessor finish times and finish times of active jobs on shared resources
        candidate_times = {es_pred}
        if d_mat is not None and c_vec is not None:
            for r in range(num_resources):
                if d_mat[best_job][r] > 0:
                    for j in resource_users[r]:
                        if scheduled[j]:
                            candidate_times.add(finish_times[j])
                            
        earliest_start = float('inf')
        # Sort candidate times to find the earliest feasible one
        for t in sorted(candidate_times):
            if t < es_pred:
                continue
            feasible = True
            if d_mat is not None and c_vec is not None:
                for r in range(num_resources):
                    if d_mat[best_job][r] == 0:
                        continue
                    load = 0.0
                    for j in resource_users[r]:
                        if scheduled[j]:
                            if start_times[j] <= t < finish_times[j]:
                                load += d_mat[j][r]
                    if load + d_mat[best_job][r] > c_vec[r]:
                        feasible = False
                        break
            if feasible:
                earliest_start = t
                break
                
        if earliest_start == float('inf'):
            earliest_start = es_pred
            
        start_times[best_job] = earliest_start
        finish_times[best_job] = earliest_start + durations[best_job]
        scheduled[best_job] = True
        scheduled_count += 1
        
    makespan = np.max(finish_times) if n > 0 else 0.0
    return makespan

def _compute_metrics(instance):
    """
    Computes various static priority metrics for the instance.
    Returns a dictionary of metric_name -> numpy array of priorities.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance.get("predecessors", None)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    n = len(durations)
    
    if n == 0:
        return {}
        
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
        
    # Build successor graph
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    # Topological sort
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # Forward pass (Earliest Start/Finish)
    es = np.zeros(n, dtype=float)
    ef = np.zeros(n, dtype=float)
    for u in topo_order:
        if predecessors[u]:
            es[u] = max(ef[p] for p in predecessors[u])
        ef[u] = es[u] + durations[u]
        
    makespan_no_res = np.max(ef)
    
    # Backward pass (Latest Start/Finish)
    ls = np.zeros(n, dtype=float)
    lf = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if successors[u]:
            lf[u] = min(ls[s] for s in successors[u])
        else:
            lf[u] = makespan_no_res
        ls[u] = lf[u] - durations[u]
        
    # Total Float / Slack
    slack = ls - es
    
    # Critical Path Length (Tail) - length of longest path from node to end
    tail = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if successors[u]:
            tail[u] = durations[u] + max(tail[s] for s in successors[u])
        else:
            tail[u] = durations[u]
            
    # Predecessor Critical Path Length (Head)
    head = np.zeros(n, dtype=float)
    for u in topo_order:
        if predecessors[u]:
            head[u] = max(head[p] + durations[p] for p in predecessors[u])
        else:
            head[u] = 0.0
            
    # Resource Weighted Tail
    # Heuristic: Weight tail by resource demand intensity
    res_weighted_tail = tail.copy()
    if demands is not None and capacities is not None:
        d_mat = np.asarray(demands, dtype=float)
        c_vec = np.asarray(capacities, dtype=float)
        # Simple resource intensity: sum of demand/capacity ratios
        intensity = np.zeros(n, dtype=float)
        for r in range(len(c_vec)):
            if c_vec[r] > 0:
                intensity += d_mat[:, r] / c_vec[r]
        res_weighted_tail = tail * (1 + intensity)
        
    # Resource Weighted Slack (Lower slack = higher priority, but penalize resource heavy)
    # We want high priority for low slack.
    # Let's create a composite: -slack + small * resource_intensity?
    # Or just use -slack for now.
    
    metrics = {
        "slack": -slack,  # Lower slack -> higher priority
        "tail": tail,     # Longer tail -> higher priority
        "head": head,     # Longer head -> higher priority (often correlated with criticality)
        "duration": durations, # Longer duration -> higher priority
        "res_tail": res_weighted_tail,
        "latest_start": -ls, # Latest start later -> lower priority? No, usually we want to schedule tight ones early.
        # Actually, -ls is same direction as -slack if slack=ls-es.
        # Let's stick to standard ones.
    }
    
    # Add a combined heuristic: Critical Path based
    # CPL = head + duration + tail
    cpl = head + durations + tail
    metrics["cpl"] = cpl
    
    return metrics

def priority(instance):
    """
    Priority function using a deterministic ensemble of static heuristics.
    Evaluates several standard priority rules and returns the one that yields
    the best makespan in the Serial SGS.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    if n == 0:
        return np.array([])
        
    metrics = _compute_metrics(instance)
    
    # Define a diverse set of candidate priority vectors
    candidates = []
    
    # 1. Total Float (Min Slack)
    if "slack" in metrics:
        candidates.append(metrics["slack"])
        
    # 2. Critical Path Length (Max Tail)
    if "tail" in metrics:
        candidates.append(metrics["tail"])
        
    # 3. Critical Path Length (Max Head)
    if "head" in metrics:
        candidates.append(metrics["head"])
        
    # 4. Duration (Longest Processing Time)
    if "duration" in metrics:
        candidates.append(metrics["duration"])
        
    # 5. Resource Weighted Tail
    if "res_tail" in metrics:
        candidates.append(metrics["res_tail"])
        
    # 6. Combined CPL
    if "cpl" in metrics:
        candidates.append(metrics["cpl"])
        
    # 7. Hybrid: 0.5 * Tail + 0.5 * -Slack
    if "tail" in metrics and "slack" in metrics:
        # Normalize to [0,1] range to combine effectively
        t = metrics["tail"]
        s = metrics["slack"]
        t_norm = (t - np.min(t)) / (np.max(t) - np.min(t) + 1e-9)
        s_norm = (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-9)
        candidates.append(t_norm + s_norm)
        
    # 8. Hybrid: Resource Intensity
    if instance.get("demands", None) is not None and instance.get("capacities", None) is not None:
        demands = instance["demands"]
        capacities = instance["capacities"]
        d_mat = np.asarray(demands, dtype=float)
        c_vec = np.asarray(capacities, dtype=float)
        intensity = np.zeros(n, dtype=float)
        for r in range(len(c_vec)):
            if c_vec[r] > 0:
                intensity += d_mat[:, r] / c_vec[r]
        candidates.append(intensity)
        
    # 9. Reverse Duration (Shortest Processing Time) - sometimes works for resource leveling
    if "duration" in metrics:
        candidates.append(-metrics["duration"])
        
    # 10. Randomized? No, keep deterministic for stability.
    # Add a few more variations
    if "tail" in metrics and "duration" in metrics:
        candidates.append(metrics["tail"] + metrics["duration"])
        
    best_vec = None
    best_score = float('inf')
    
    # Evaluate each candidate
    for vec in candidates:
        # Ensure vec is a numpy array of floats
        vec = np.asarray(vec, dtype=float)
        if vec.shape != (n,):
            continue
            
        # Check time budget
        if _remaining_budget() < 1.0:
            break
            
        score = _run_serial_sgs(instance, vec)
        
        if score < best_score:
            best_score = score
            best_vec = vec
            
    # If no valid vector found or all failed, fallback to slack
    if best_vec is None:
        if "slack" in metrics:
            best_vec = metrics["slack"]
        else:
            best_vec = np.zeros(n)
            
    return best_vec

# EVOLVE-BLOCK-END
