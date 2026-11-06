"""
RCPSP Heuristic: Iterative Priority Refinement
This strategy differs from static heuristics by actually simulating the scheduling process.
It evaluates multiple candidate priority functions (e.g., LPT, Total Float, Resource Pressure,
Random, and Combinations) by running the full Serial Schedule Generation Scheme (SGS) for each.
It returns the priority scores that produced the minimum makespan in the simulation.
This allows the heuristic to adapt to the specific instance's resource bottlenecks and precedence structure.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def priority(instance):
    """
    Returns a priority score for each activity.
    Higher score = higher priority.
    
    Strategy:
    1. Parse instance data.
    2. Define a set of candidate priority strategies (functions that return scores).
    3. For each candidate, run the Serial Schedule Generation Scheme (SGS) to compute the makespan.
    4. Select the candidate that yields the lowest makespan.
    5. Return the priority scores from that best candidate.
    
    This is a "meta-heuristic" approach where we search the space of simple heuristics
    to find the best one for the current instance.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin for 150s limit
    
    # --- 1. Parse Instance ---
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    preds_raw = instance.get("predecessors", instance.get("preds", [[] for _ in range(n)]))
    demands_raw = instance.get("demands", instance.get("resource_demands", None))
    capacities_raw = instance.get("capacities", instance.get("resource_capacities", None))
    
    # Normalize predecessors
    preds = [list(p) if p else [] for p in preds_raw]
    if len(preds) < n:
        preds = [[] for _ in range(n)]
        for i in range(n):
            if i < len(preds_raw) and preds_raw[i]:
                preds[i] = list(preds_raw[i])
                
    # Build successors
    succs = [[] for _ in range(n)]
    for j in range(n):
        for i in preds[j]:
            if 0 <= i < n:
                succs[i].append(j)
                
    # Handle resources
    if demands_raw is not None:
        try:
            demands = np.asarray(demands_raw, dtype=float)
            if demands.ndim == 1:
                demands = demands.reshape(-1, 1)
        except:
            demands = np.zeros((n, 1))
    else:
        demands = np.zeros((n, 1))
        
    if capacities_raw is not None:
        try:
            capacities = np.asarray(capacities_raw, dtype=float)
            if capacities.ndim == 0:
                capacities = np.array([capacities])
        except:
            capacities = np.ones(demands.shape[1])
    else:
        capacities = np.ones(demands.shape[1])
        
    m_resources = demands.shape[1]
    if len(capacities) < m_resources:
        capacities = np.pad(capacities, (0, m_resources - len(capacities)), mode='constant')
        
    # Topological Sort
    in_degree = np.array([len(preds[i]) for i in range(n)], dtype=int)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(topo_order) < n:
        # Cycle detected
        return durations.copy()
        
    # Pre-compute static metrics for use in candidate functions
    # CPM Forward/Backward for Total Float
    es_cpm = np.zeros(n)
    ef_cpm = np.zeros(n)
    for u in topo_order:
        es_cpm[u] = 0
        for p in preds[u]:
            if ef_cpm[p] > es_cpm[u]:
                es_cpm[u] = ef_cpm[p]
        ef_cpm[u] = es_cpm[u] + durations[u]
        
    makespan_cpm = np.max(ef_cpm)
    
    lf_cpm = np.zeros(n)
    ls_cpm = np.zeros(n)
    for u in reversed(topo_order):
        if len(succs[u]) > 0:
            lf_cpm[u] = np.min([ls_cpm[v] for v in succs[u]])
        else:
            lf_cpm[u] = makespan_cpm
        ls_cpm[u] = lf_cpm[u] - durations[u]
        
    tf_cpm = ls_cpm - es_cpm
    
    # Resource Pressure
    if m_resources > 0 and np.all(capacities > 0):
        safe_cap = np.where(capacities > 0, capacities, 1.0)
        pressure = np.sum(demands / safe_cap, axis=1)
        max_pressure = np.max(pressure)
        if max_pressure > 0:
            pressure = pressure / max_pressure
        else:
            pressure = np.zeros(n)
    else:
        pressure = np.zeros(n)
        
    # Successor Count (depth)
    succ_count = np.array([len(succs[i]) for i in range(n)], dtype=float)
    max_succ = np.max(succ_count)
    if max_succ > 0:
        succ_score = succ_count / max_succ
    else:
        succ_score = np.zeros(n)
        
    # --- 2. Define Candidate Strategies ---
    
    def get_scores_lpt():
        return durations.copy()
        
    def get_scores_tf():
        # Higher priority for lower float (critical)
        max_tf = np.max(tf_cpm)
        if max_tf > 0:
            return 1.0 - (tf_cpm / max_tf)
        return np.ones(n)
        
    def get_scores_pressure():
        return pressure.copy()
        
    def get_scores_spt():
        # Shortest Processing Time: lower duration = higher priority? 
        # In SGS, higher score is scheduled first. So for SPT, we want small durations to have high scores.
        max_dur = np.max(durations)
        if max_dur > 0:
            return 1.0 - (durations / max_dur)
        return np.ones(n)
        
    def get_scores_combined_1():
        # 0.7 * TF + 0.3 * Pressure
        max_tf = np.max(tf_cpm)
        if max_tf > 0:
            crit = 1.0 - (tf_cpm / max_tf)
        else:
            crit = np.ones(n)
        return 0.7 * crit + 0.3 * pressure
        
    def get_scores_combined_2():
        # 0.5 * TF + 0.3 * Pressure + 0.2 * Duration
        max_tf = np.max(tf_cpm)
        if max_tf > 0:
            crit = 1.0 - (tf_cpm / max_tf)
        else:
            crit = np.ones(n)
        max_dur = np.max(durations)
        if max_dur > 0:
            dur = durations / max_dur
        else:
            dur = np.ones(n)
        return 0.5 * crit + 0.3 * pressure + 0.2 * dur
        
    def get_scores_random():
        # Random perturbation, might break ties beneficially
        return np.random.rand(n)
        
    def get_scores_successor():
        # Prioritize tasks that unblock many others
        return succ_score.copy()
        
    candidates = [
        get_scores_lpt,
        get_scores_tf,
        get_scores_pressure,
        get_scores_spt,
        get_scores_combined_1,
        get_scores_combined_2,
        get_scores_successor,
    ]
    
    # --- 3. Simulate SGS for each candidate ---
    
    def simulate_sgs(scores):
        """
        Runs Serial Schedule Generation Scheme with given scores.
        Returns makespan.
        """
        # Sort activities by score descending
        # We need to handle ties consistently, maybe by index or random?
        # Standard SGS usually picks the highest priority eligible.
        # If ties, we can break by index or random. Let's use index for determinism in sim, 
        # but the actual scheduler might differ. However, for comparison, consistent tie-breaking is good.
        
        # Efficient way: maintain a list of eligible activities.
        # Since n is small (30), we can just scan.
        
        scheduled = [False] * n
        start_times = np.zeros(n)
        end_times = np.zeros(n)
        
        # Resource usage over time
        # We need to track resource availability.
        # Since n is small, we can simulate time steps or events.
        # Event-based simulation is more efficient.
        
        # Let's use a simple loop:
        # 1. Find all eligible (all preds scheduled).
        # 2. Among eligible, pick the one with highest score.
        # 3. Find the earliest start time for this activity such that resources are available.
        # 4. Schedule it.
        
        # To find earliest start time, we need to know resource usage.
        # We can maintain a list of (start, end, resource_id, amount) or a 2D array of usage per time unit.
        # Given small n, a list of events is fine.
        
        # Let's use a simpler approach for n=30:
        # Maintain current resource usage as a dict or array for "active" tasks.
        # When scheduling a new task, we need to find the earliest time t >= current_time 
        # where for all resources r, capacity[r] - usage[r] >= demand[task][r].
        
        # Since we schedule sequentially, we can keep track of scheduled tasks and their intervals.
        # For a new task, we check potential start times.
        # Potential start times are 0 or the end times of already scheduled tasks.
        
        scheduled_intervals = [] # list of (start, end, demands_vector)
        
        for _ in range(n):
            # Find eligible
            eligible = []
            for i in range(n):
                if not scheduled[i]:
                    if all(scheduled[p] for p in preds[i]):
                        eligible.append(i)
                        
            if not eligible:
                # Should not happen if graph is valid
                break
                
            # Pick best eligible
            best_idx = -1
            best_score = -np.inf
            for i in eligible:
                if scores[i] > best_score:
                    best_score = scores[i]
                    best_idx = i
                    
            if best_idx == -1:
                break
                
            task = best_idx
            d = durations[task]
            dem = demands[task]
            
            # Find earliest start time
            # Candidate start times: 0 and all end times of scheduled tasks
            candidate_starts = [0]
            for (st, en, _) in scheduled_intervals:
                candidate_starts.append(en)
            candidate_starts = sorted(set(candidate_starts))
            
            # Also, we might need to start later if resources are occupied in between?
            # No, if we start at time t, we need resources for [t, t+d].
            # If we pick t from the set of start/end times, we cover all "free" windows boundaries.
            
            found_start = None
            for t in candidate_starts:
                # Check if resources are available in [t, t+d]
                ok = True
                for r in range(m_resources):
                    if dem[r] == 0:
                        continue
                    # Sum demands of all scheduled tasks that overlap [t, t+d]
                    total_demand = 0
                    for (st, en, dem_vec) in scheduled_intervals:
                        # Overlap if st < t+d and en > t
                        if st < t + d and en > t:
                            total_demand += dem_vec[r]
                    if total_demand + dem[r] > capacities[r]:
                        ok = False
                        break
                if ok:
                    found_start = t
                    break
                    
            if found_start is None:
                # This shouldn't happen if capacities are sufficient for single task, 
                # but if a single task demands more than capacity, it's infeasible.
                # Assume feasible.
                # Fallback: just pick the last candidate + d? Or raise error?
                # For robustness, pick the max candidate start.
                found_start = max(candidate_starts) if candidate_starts else 0
                
            # Schedule task
            scheduled[task] = True
            start_times[task] = found_start
            end_times[task] = found_start + d
            scheduled_intervals.append((found_start, found_start + d, dem))
            
        makespan = np.max(end_times) if n > 0 else 0
        return makespan

    best_scores = None
    best_makespan = np.inf
    
    # Evaluate candidates
    for func in candidates:
        if time.time() > DEADLINE:
            break
        try:
            scores = func()
            # Ensure scores are comparable (float)
            scores = np.asarray(scores, dtype=float)
            ms = simulate_sgs(scores)
            if ms < best_makespan:
                best_makespan = ms
                best_scores = scores
        except Exception:
            continue
            
    # Add a few random variations to break potential plateaus
    # If time permits
    if time.time() < DEADLINE - 5:
        for _ in range(5):
            if time.time() > DEADLINE:
                break
            try:
                # Perturb the best found scores slightly? Or just random?
                # Let's try random
                scores = np.random.rand(n)
                ms = simulate_sgs(scores)
                if ms < best_makespan:
                    best_makespan = ms
                    best_scores = scores
            except Exception:
                continue
                
    if best_scores is None:
        # Fallback to LPT
        return durations.copy()
        
    return best_scores

# EVOLVE-BLOCK-END
