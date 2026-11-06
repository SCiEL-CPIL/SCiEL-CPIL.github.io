import numpy as np
import time

def priority(instance):
    """
    Strategy: Deterministic Greedy Criticality Heuristic.
    
    Instead of random local search, we simulate the Serial SGS (SGS) ourselves.
    At each decision point (when we need to pick the next activity to start),
    we evaluate all eligible activities. We assign a dynamic "criticality score"
    to each eligible activity based on:
    1. Remaining Path Length (how much work remains after this activity).
    2. Resource Contention (how much this activity blocks other high-priority activities).
    
    We record the rank of each activity chosen at its start time.
    The final returned priority vector is the rank assigned during this greedy simulation.
    Since the SGS is deterministic given a priority vector, and we are effectively
    constructing a schedule that minimizes makespan step-by-step using a strong
    greedy rule, the resulting priority vector should guide the evaluator's SGS
    to reproduce a near-optimal schedule.
    
    This is structurally different from local search because:
    1. It is constructive (builds the solution) rather than perturbative.
    2. It uses dynamic, context-dependent scoring during the simulation.
    3. It is deterministic (no random restarts/noise).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # 1. Extract and process instance data
    N = len(instance["durations"])
    durations = np.array(instance["durations"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)  # Shape (N, R)
    capacities = np.array(instance["capacities"], dtype=float) # Shape (R,)
    
    # Handle predecessors
    preds = instance.get("predecessors", [[] for _ in range(N)])
    if isinstance(preds, np.ndarray):
        if preds.ndim == 2:
             preds = [np.where(preds[i] > 0)[0].tolist() for i in range(N)]
        else:
             preds = [[] for _ in range(N)]
    elif isinstance(preds, list):
         normalized_preds = []
         for p in preds:
             if isinstance(p, (list, tuple, np.ndarray)):
                 normalized_preds.append(list(map(int, p)))
             else:
                 normalized_preds.append([])
         preds = normalized_preds
    else:
         preds = [[] for _ in range(N)]

    # Build successors
    succs = [[] for _ in range(N)]
    for i in range(N):
        for p in preds[i]:
            succs[p].append(i)
             
    # Pre-compute "Tail Time" (Longest Path from i to end)
    # This is the sum of durations on the longest path starting from i.
    # We need topological sort in reverse (from end to start).
    in_degree = [len(p) for p in preds]
    out_degree = [len(s) for s in succs]
    
    # Reverse topological sort (activities with no successors first)
    q = [i for i in range(N) if out_degree[i] == 0]
    rev_topo_order = []
    outdeg_temp = out_degree[:]
    while q:
        u = q.pop(0)
        rev_topo_order.append(u)
        for p in preds[u]:
            outdeg_temp[p] -= 1
            if outdeg_temp[p] == 0:
                q.append(p)
                
    tail_time = np.zeros(N)
    for u in rev_topo_order:
        max_succ_tail = 0
        for v in succs[u]:
            if tail_time[v] > max_succ_tail:
                max_succ_tail = tail_time[v]
        tail_time[u] = durations[u] + max_succ_tail
        
    # "Earliest Start" (ES) from CPM (no resource constraints)
    in_degree = [len(p) for p in preds]
    q = [i for i in range(N) if in_degree[i] == 0]
    topo_order = []
    indeg_temp = in_degree[:]
    while q:
        u = q.pop(0)
        topo_order.append(u)
        for v in succs[u]:
            indeg_temp[v] -= 1
            if indeg_temp[v] == 0:
                q.append(v)
                
    ES = np.zeros(N)
    for u in topo_order:
        max_pred_es = 0
        for p in preds[u]:
            if ES[p] + durations[p] > max_pred_es:
                max_pred_es = ES[p] + durations[p]
        ES[u] = max_pred_es
        
    # Initial Priority: Tail Time is a strong indicator of criticality
    # High tail time = high priority
    # We will refine this during simulation.
    base_priority = tail_time.copy()

    def simulate_sgs_greedy():
        """
        Runs a greedy SGS. At each step, it picks the best activity based on
        a dynamic score. It returns the priority vector (ranks) that would
        reproduce this schedule.
        """
        scheduled = np.zeros(N, dtype=bool)
        active = []  # List of (finish_time, activity_index)
        remaining_res = capacities.copy()
        time_current = 0.0
        
        unsched_preds = np.array([len(p) for p in preds])
        eligible_set = set(i for i in range(N) if unsched_preds[i] == 0)
        
        # We will store the "rank" or "priority score" assigned to each activity
        # when it was selected.
        # To make it a static vector that works with a standard SGS, we can
        # assign scores such that the order of selection is preserved.
        # A simple way: assign a decreasing score to each activity in the order
        # they were selected? No, that breaks if multiple activities are selected
        # at the same time step.
        
        # Better approach:
        # The SGS picks the highest priority eligible activity that fits.
        # If we want to reproduce a specific schedule, we need to ensure that
        # at each time step, the activities selected are higher priority than
        # those not selected (if they were eligible and fit).
        
        # Let's use a "Dynamic Score" calculation for each eligible activity.
        # Score(i) = w1 * tail_time[i] - w2 * resource_conflict(i)
        
        # We will collect the scores used.
        # Since the SGS is deterministic, if we assign each activity a score
        # that reflects its "desirability" at the moment it was picked,
        # and ensure that at any time t, the picked activity has a higher
        # score than any other eligible activity that could have been picked
        # (but wasn't because it didn't fit or had lower score), we might be able
        # to construct a static vector.
        
        # However, a static vector cannot change over time.
        # So we must find a static vector that approximates the greedy choices.
        
        # Alternative:
        # We run the greedy simulation to find the *optimal* schedule (or near-optimal).
        # Then, we derive a static priority vector that is consistent with this schedule.
        
        # How to derive static vector from a schedule?
        # For each activity i, let S_i be its start time in the greedy schedule.
        # A simple heuristic: Priority(i) = -S_i + tail_time[i]
        # Or even simpler: Just use the tail_time, but adjust for "regret".
        
        # Let's try a different angle.
        # The current program uses local search.
        # The inspiration suggests "Adaptive Resource Weighting".
        # Let's implement the Greedy Simulation to *generate* the best static vector.
        
        # We will simulate the SGS.
        # At each step, for each eligible activity i, we calculate a dynamic score:
        # DScore(i) = tail_time[i] - alpha * sum( demand[i][r] * (1 / (remaining_res[r] + 1)) )
        # The term (1 / (remaining_res[r] + 1)) represents scarcity.
        # High scarcity + high demand = high penalty.
        
        # We pick the activity with the highest DScore.
        # We record this DScore as the "priority" for that activity?
        # No, because DScore changes as resources are consumed.
        
        # If we return the DScore calculated at the moment of selection,
        # will it work?
        # Suppose Activity A is selected at t=0 with DScore=10.
        # Activity B is eligible at t=0 but not selected because DScore=9.
        # Later, at t=5, Activity B is selected.
        # If we return static vector [10, 9], the SGS will pick A first (10>9).
        # This matches.
        # What if Activity C becomes eligible at t=2 with DScore=11?
        # In the greedy sim, C would be picked before B (if B wasn't picked yet).
        # If we return [10, 9, 11], the SGS will pick A (10), then C (11), then B (9).
        # This seems to preserve the order!
        
        # So the strategy is:
        # 1. Run Greedy SGS.
        # 2. For each activity, record the DScore at the moment it was selected.
        # 3. Return these scores as the static priority vector.
        
        scores = np.zeros(N)
        
        while not np.all(scheduled):
            if time.time() > deadline:
                # Fallback to base priority if timeout
                return base_priority
            
            if not eligible_set:
                if not active:
                    break
                min_finish = min(f for f, _ in active)
                time_current = min_finish
                still_active = []
                for f, idx in active:
                    if f > time_current:
                        still_active.append((f, idx))
                    else:
                        remaining_res += demands[idx]
                        for v in succs[idx]:
                            if not scheduled[v]:
                                unsched_preds[v] -= 1
                                if unsched_preds[v] == 0:
                                    eligible_set.add(v)
                active = still_active
                continue
            
            # Calculate dynamic scores for all eligible activities
            best_idx = -1
            best_score = -np.inf
            
            # Pre-compute scarcity factors for speed
            # scarcity_r = 1 / (remaining_res[r] + 1)
            safe_rem = np.where(remaining_res > 0, remaining_res, 1e-9)
            scarcity = 1.0 / (safe_rem + 1.0)
            
            for i in eligible_set:
                # Check if it fits
                if np.any(remaining_res < demands[i]):
                    continue
                
                # Calculate DScore
                # Tail time is the main driver.
                # Penalty for using scarce resources.
                conflict = np.sum(demands[i] * scarcity)
                
                # Weight: Tail time is in time units. Conflict is in "scarcity units".
                # We need to balance them.
                # Let's normalize conflict by max duration? Or just use a small weight.
                # A high conflict means we are using a resource that others need.
                
                dscore = tail_time[i] - 0.5 * conflict
                
                if dscore > best_score:
                    best_score = dscore
                    best_idx = i
                    
            if best_idx == -1:
                # No eligible activity fits. This shouldn't happen if eligible_set is not empty
                # and we handle resource release correctly, but just in case.
                # Force pick one? No, wait for release.
                if not active:
                    break
                min_finish = min(f for f, _ in active)
                time_current = min_finish
                still_active = []
                for f, idx in active:
                    if f > time_current:
                        still_active.append((f, idx))
                    else:
                        remaining_res += demands[idx]
                        for v in succs[idx]:
                            if not scheduled[v]:
                                unsched_preds[v] -= 1
                                if unsched_preds[v] == 0:
                                    eligible_set.add(v)
                active = still_active
                continue
            
            # Select best_idx
            scheduled[best_idx] = True
            scores[best_idx] = best_score
            remaining_res -= demands[best_idx]
            finish_t = time_current + durations[best_idx]
            active.append((finish_t, best_idx))
            eligible_set.remove(best_idx)
            
        return scores

    # Run the greedy simulation
    final_priorities = simulate_sgs_greedy()
    
    # Check if we have valid scores for all activities
    # If some are 0 (because they were never selected? shouldn't happen), fill with base
    if np.any(final_priorities == 0):
        final_priorities = np.where(final_priorities == 0, base_priority, final_priorities)
        
    return final_priorities
