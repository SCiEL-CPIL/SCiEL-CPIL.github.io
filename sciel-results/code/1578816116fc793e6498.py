"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

def dispatch(ready_operations, time):
    """
    Dispatch rule using 2-Step Local Search (Pairwise Swap Evaluation).
    
    Strategy:
    1. For each candidate ready operation 'i', simulate scheduling it first.
    2. Then, using a fast static heuristic (Min Remaining Work / Critical Ratio), 
       determine which operation 'j' would be scheduled second.
    3. Estimate the total time cost for this pair (i, j). 
       - If 'i' and 'j' use the same machine, 'j' is delayed by 'i's duration.
       - The cost is the sum of their completion times (or max completion time, 
         approximated by summing durations + delays).
    4. Select the 'i' that leads to the minimum estimated 2-step cost.
    
    This looks beyond the immediate step, accounting for how the current choice 
    affects the best possible next choice and the resulting machine contention.
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Deadline guard
    start_time = _time.time()
    deadline = 110.0  # Safety margin for 120s limit

    # Pre-compute static scores for the "second step" heuristic.
    # We use a hybrid of Most Work Remaining (MWR) and Shortest Processing Time (SPT).
    # A common effective rule is "Critical Ratio" or just MWR. 
    # Let's use MWR (remaining_work) as the primary tie-breaker for step 2, 
    # but penalize long durations.
    # Actually, a very strong static rule is "Min Remaining Work" (MWR) 
    # or "Max Remaining Work" depending on interpretation. 
    # In job shop, scheduling jobs with more remaining work often helps 
    # clear the pipeline. Let's use a score: remaining_work / min_duration.
    # Higher is better for priority.
    
    static_scores = []
    min_durs = []
    best_machines = []
    
    for op in ready_operations:
        opts = op.get("machine_options", {})
        if not opts:
            static_scores.append(0)
            min_durs.append(float('inf'))
            best_machines.append(None)
            continue
        
        # Find best machine (min duration)
        best_m = min(opts, key=opts.get)
        min_dur = opts[best_m]
        
        min_durs.append(min_dur)
        best_machines.append(best_m)
        
        rem_work = op.get("remaining_work", 0)
        if rem_work is None or rem_work <= 0:
            rem_work = min_dur
            
        # Static priority: High remaining work, low duration is good.
        # Score = rem_work / (1 + min_dur)
        # We will pick the MAX static score for the second step.
        static_scores.append(rem_work / (1.0 + min_dur))

    best_second_index = np.argmax(static_scores)
    
    # Now, for each candidate 'i', estimate the cost if we pick 'i' first.
    # Cost(i) = Duration(i) + Duration(j) + Delay(j)
    # Where j is the best second choice given that i is taken.
    # If i == j (i.e., i was the best static choice), we need the next best j.
    
    # To be efficient, we can pre-sort indices by static score.
    # sorted_indices = np.argsort(static_scores)[::-1]
    
    # However, n is small (10 jobs, so max 10 ready ops). 
    # We can just loop.
    
    costs = np.full(n, np.inf)
    
    for i in range(n):
        # Simulate picking i
        # Find best j != i based on static score
        best_j = -1
        max_score = -1
        for j in range(n):
            if j == i:
                continue
            if static_scores[j] > max_score:
                max_score = static_scores[j]
                best_j = j
        
        if best_j == -1:
            # Only one op ready
            costs[i] = min_durs[i]
            continue
            
        # Calculate cost for pair (i, j)
        dur_i = min_durs[i]
        dur_j = min_durs[best_j]
        m_i = best_machines[i]
        m_j = best_machines[best_j]
        
        # If they share the best machine, j is delayed by dur_i.
        # Note: In a real scheduler, j might switch machines if its best machine is busy.
        # But here we assume the "best machine" is the one it will use if available.
        # If m_i == m_j, then j cannot start until i finishes.
        # So completion of j = time + dur_i + dur_j.
        # Completion of i = time + dur_i.
        # Total "blocked time" or makespan contribution approx:
        # If m_i == m_j: dur_i + dur_j
        # If m_i != m_j: max(dur_i, dur_j) -- but since we are adding to makespan,
        # and they run in parallel, the makespan increases by max(dur_i, dur_j) 
        # ONLY IF the machines were idle. 
        # However, the "cost" to the schedule is the time until both are done.
        # Actually, for makespan minimization, we want to minimize the time 
        # these operations occupy the critical path.
        
        # A better proxy for "badness" is the sum of completion times (mean flow time) 
        # or just the makespan extension.
        # If m_i == m_j: extension = dur_i + dur_j
        # If m_i != m_j: extension = max(dur_i, dur_j)
        
        if m_i == m_j:
            cost = dur_i + dur_j
        else:
            cost = max(dur_i, dur_j)
            
        # Tie-breaking: If costs are equal, prefer the one with higher static score 
        # (i.e., more remaining work) to keep momentum.
        # We can subtract a tiny epsilon * static_score[i] to break ties.
        # Or just use the first one found with min cost.
        
        costs[i] = cost - 0.001 * static_scores[i]

    # Check time
    if _time.time() > deadline:
        # Fallback to simple MWR
        rem_works = [op.get("remaining_work", 0) for op in ready_operations]
        return int(np.argmax(rem_works))

    return int(np.argmin(costs))

# EVOLVE-BLOCK-END
