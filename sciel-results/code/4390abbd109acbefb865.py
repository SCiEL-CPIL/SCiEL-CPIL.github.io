"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Global deadline guard
_START_TIME = _time.time()
_DEADLINE = 145.0  # 5s safety margin before 150s

def dispatch(ready_operations, time):
    # Check deadline
    if _time.time() > _DEADLINE:
        # Fallback: Return first index to prevent crash/timeout score 0
        return 0

    if not ready_operations:
        return 0

    n_ops = len(ready_operations)
    
    # Strategy: Regret-Based Dispatching
    # 
    # The "regret" of scheduling an operation is the difference between the 
    # estimated completion time of its job if we schedule it on its best machine 
    # versus its second-best machine (or if we delay it slightly). 
    # However, a more robust interpretation for FJSP dispatching rules often used 
    # in literature is the "Regret Rule": 
    # Regret(op) = Est_Comp(job, op, best_machine) - Est_Comp(job, op, 2nd_best_machine)
    # But this only measures machine choice regret. 
    # 
    # A better "Regret" for *which operation to pick* is often defined as:
    # How much does the *makespan* (or critical job completion) suffer if we 
    # DON'T pick this operation now?
    # 
    # Let's use a composite score that combines:
    # 1. Criticality: The estimated completion time of the job (higher is worse/urgent).
    # 2. Regret: The difference in completion time between the best and second-best 
    #    machine for that operation. If the regret is high, it means the machine 
    #    choice matters a lot, so we should resolve this ambiguity now.
    #    Alternatively, if we consider "Regret" as "What is the penalty for waiting?",
    #    operations with long durations on critical paths have high regret for waiting.
    #
    # Let's refine the "Regret" concept to be more actionable for a single-step dispatch:
    # For each ready operation `op` belonging to job `J`:
    # 1. Find the best machine `m1` (min duration) and second best `m2`.
    # 2. Calculate `Est_Comp_J_m1 = time + dur(op, m1) + remaining_work(J) - dur(op, m1)` 
    #    Wait, `remaining_work` usually includes the current op? 
    #    The prompt says `remaining_work` is a field. Usually, in these sims, 
    #    `remaining_work` is the sum of processing times of *all* remaining ops 
    #    including the current one? Or excluding? 
    #    Let's assume `remaining_work` is the sum of durations for all *remaining* 
    #    operations in the job sequence, including the current ready one. 
    #    If we pick a machine with duration `d`, the new remaining work for the job 
    #    (after this op) is `remaining_work - d`. 
    #    So, `Est_Comp(J, m) = time + d + (remaining_work - d) = time + remaining_work`. 
    #    This implies the estimated completion time of the job is independent of the 
    #    machine choice for the *current* op if we assume the rest of the ops take 
    #    their minimum durations. This makes "machine regret" zero in this simple model.
    #
    # We need a more nuanced model. 
    # Let's look at "Machine Load" or "Contention". We don't have global state.
    #
    # Let's try a different structural change: **Minimize the Maximum Estimated Completion Time (Min-Max)**
    # 
    # For each candidate operation `i` from job `J_i`:
    # 1. Assume we schedule `i` now on its best machine.
    # 2. The job `J_i` will finish at `T_i = time + min_dur(i) + (remaining_work(J_i) - min_dur(i))`.
    #    Actually, if `remaining_work` includes `i`, then `T_i = time + remaining_work(J_i)`.
    #    This is constant for a job. So all ops from the same job have the same estimated completion.
    #
    # If `remaining_work` does NOT include the current op (i.e., it's the work *after* this op),
    # then `T_i = time + min_dur(i) + remaining_work(J_i)`.
    # In this case, we should pick the op that minimizes `T_i` for the most critical job?
    #
    # Let's assume the standard FJSP dispatching heuristic: **Weighted Shortest Processing Time (WSPT)**
    # combined with **Job Criticality**.
    #
    # Score = (1 / (time + min_dur)) * (1 / (remaining_work + 1)) ? No.
    #
    # Let's go back to the **Regret** idea but defined as:
    # "How unique is this operation's contribution to reducing the job's completion time?"
    # 
    # Actually, a very strong heuristic in FJSP is **Earliest Due Date (EDD)** or **Shortest Processing Time (SPT)**.
    # Since we don't have due dates, SPT is good for throughput.
    # But for makespan, **Longest Processing Time (LPT)** on critical jobs is often better.
    #
    # Let's try a **Hybrid Regret-Criticality** score:
    # 1. Calculate `criticality` = remaining_work of the job.
    # 2. Calculate `regret` = |min_dur - 2nd_min_dur|. If the difference is large, 
    #    the machine choice is critical. We should pick this op to resolve the machine choice 
    #    early? Or avoid it? Usually, if regret is high, it means the "best" machine is much 
    #    better. If we don't pick it, we might be forced to use a worse machine later. 
    #    So high regret = high urgency to pick the best machine now.
    #
    # Score = criticality * (1 + regret)
    #
    # Let's implement this.
    
    best_score = -float('inf')
    best_idx = 0
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        job_id = op["job"]
        remaining_work = op["remaining_work"]
        
        # Get durations for all machines
        durations = list(op["machine_options"].values())
        
        if not durations:
            continue
            
        min_dur = min(durations)
        
        # Calculate regret: difference between best and second-best
        if len(durations) > 1:
            # Sort to find second best
            sorted_durs = sorted(durations)
            second_min_dur = sorted_durs[1]
            regret = second_min_dur - min_dur
        else:
            regret = 0.0
            
        # Heuristic: 
        # High remaining_work = Critical Job (High priority)
        # High regret = Machine choice matters a lot (High priority to resolve now)
        # Short min_dur = Fast completion (Good for clearing the board)
        
        # Let's try: Score = remaining_work / (1 + min_dur) + regret
        # Or: Score = remaining_work * (1 + regret / (min_dur + 1))
        
        # Let's use a weighted sum that emphasizes criticality and regret
        # Normalize regret to 0-1 range roughly? Durations are usually small integers.
        
        score = remaining_work + 0.5 * regret
        
        # Tie-breaker: Shorter min_dur is better (SPT)
        if score > best_score or (score == best_score and min_dur < (best_machine_duration if best_machine else float('inf'))):
             # We need to store best_machine_duration for tie-breaking
             pass
             
    # Let's re-implement the loop properly to track the best machine too.
    
    best_score = -float('inf')
    best_idx = 0
    best_machine = None
    best_min_dur = float('inf')
    
    for i, op in enumerate(ready_operations):
        job_id = op["job"]
        remaining_work = op["remaining_work"]
        
        durations = list(op["machine_options"].values())
        
        if not durations:
            continue
            
        min_dur = min(durations)
        
        # Find the machine with min_dur
        best_m_for_op = None
        for m, d in op["machine_options"].items():
            if d == min_dur:
                best_m_for_op = m
                break
                
        if len(durations) > 1:
            sorted_durs = sorted(durations)
            second_min_dur = sorted_durs[1]
            regret = second_min_dur - min_dur
        else:
            regret = 0.0
            
        # Composite Score:
        # 1. Criticality: remaining_work
        # 2. Regret: How much we lose if we don't pick the best machine? 
        #    Actually, if regret is high, the "best" machine is significantly faster.
        #    If we pick this op, we can assign it to the best machine.
        #    If we don't pick this op, we might be forced to wait for the best machine 
        #    or use a worse one.
        #    So high regret + high criticality = Very Urgent.
        
        score = remaining_work + regret
        
        # Tie-breaker: If scores are equal, prefer shorter processing time (SPT)
        if score > best_score or (score == best_score and min_dur < best_min_dur):
            best_score = score
            best_idx = i
            best_machine = best_m_for_op
            best_min_dur = min_dur
            
    if best_machine is None:
        return 0
        
    return (int(best_idx), best_machine)

# EVOLVE-BLOCK-END
