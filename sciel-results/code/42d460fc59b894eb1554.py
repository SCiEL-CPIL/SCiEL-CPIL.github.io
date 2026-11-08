"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math
import random

_DEADLINE_MARGIN = 5.0
_START_TIME = None


def _now():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    return _time.time() - _START_TIME


def _deadline_exceeded():
    return _now() > (150.0 - _DEADLINE_MARGIN)


def _min_duration(op):
    opts = op.get("machine_options", {})
    if not opts:
        return 0
    return min(opts.values())


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next using a local lookahead heuristic.
    For each candidate, we estimate the resulting machine congestion by
    simulating the immediate placement and resolving conflicts greedily.
    We select the candidate that minimizes the resulting maximum machine load.
    """
    if not ready_operations:
        return 0

    if len(ready_operations) == 1:
        return 0

    if _deadline_exceeded():
        # Fallback: Shortest Processing Time (SPT)
        return int(min(range(len(ready_operations)),
                       key=lambda i: _min_duration(ready_operations[i])))

    n = len(ready_operations)
    
    # 1. Pre-compute the "current" load on each machine based on already scheduled ops
    #    Note: The interface doesn't give us the full schedule, but we can infer
    #    relative urgency. However, a better proxy for "congestion" in a stateless
    #    dispatch rule is to look at the *conflicts* between the ready ops themselves.
    
    # Strategy: 
    # For each candidate op i:
    #   - Assume it is scheduled now.
    #   - Look at the other ready ops.
    #   - If another op j shares a machine with i, it must wait (or i waits, but we assume i goes).
    #   - We want to minimize the "wasted time" or "delay" imposed on the system.
    
    # A more robust heuristic: 
    # Minimize the sum of (processing time of candidate * (1 + number of conflicting ready ops))
    # This favors short ops that don't block others, and short ops that are on unique machines.
    
    # Let's try a "Least Interference" metric:
    # Score = min_duration(op) + sum(min_duration(other) for other in ready if other.machine == op.machine)
    
    best_idx = 0
    best_score = float('inf')
    
    # To make this slightly more sophisticated, we can weight by remaining work
    # to prioritize jobs that are almost done (to reduce makespan of individual jobs)
    # but the primary driver for makespan is usually keeping machines busy with short tasks.
    
    for i in range(n):
        op_i = ready_operations[i]
        opts_i = op_i.get("machine_options", {})
        
        # Pick the fastest machine for this op
        if not opts_i:
            continue
            
        min_dur_i = min(opts_i.values())
        
        # Count how many *other* ready ops are competing for the SAME machine(s)
        # Actually, machine_options is a dict {machine: duration}.
        # If op_i chooses machine M, it conflicts with any op_j that also lists M in its options
        # AND if op_j is forced to use M (or likely to use M).
        
        # Simplest effective metric: 
        # Score = min_dur_i * (1 + conflict_count)
        # where conflict_count is the number of other ready ops that share at least one machine with op_i.
        
        machines_i = set(opts_i.keys())
        conflict_count = 0
        conflict_weight = 0.0
        
        for j in range(n):
            if i == j:
                continue
            op_j = ready_operations[j]
            opts_j = op_j.get("machine_options", {})
            machines_j = set(opts_j.keys())
            
            # If they share any machine, there is potential conflict
            if machines_i.intersection(machines_j):
                conflict_count += 1
                # Weight the conflict by the duration of the other job on the shared machine
                # to penalize blocking a long job or being blocked by a long job?
                # If we pick i, and j is on the same machine, j waits. 
                # The "cost" is the delay imposed on j.
                # Delay imposed on j is roughly min_dur_i (if i goes first).
                # So the total "system delay" caused by picking i is sum(min_dur_i for all conflicting j)?
                # No, min_dur_i is fixed for i.
                # The cost is that j is delayed by min_dur_i.
                # So total cost = min_dur_i * conflict_count.
                
                # However, if j is on a machine that i *could* use but isn't the min for i, 
                # it's still a conflict if they both want that machine.
                # Since we assume i picks its best machine, we only care about conflicts on i's best machine.
                
                # Let's refine: Find i's best machine M*.
                # Count conflicts on M*.
        
        # Let's re-evaluate with specific machine choice
        best_machine_i = min(opts_i.keys(), key=lambda m: opts_i[m])
        dur_i = opts_i[best_machine_i]
        
        conflicts_on_best_machine = 0
        for j in range(n):
            if i == j:
                continue
            op_j = ready_operations[j]
            opts_j = op_j.get("machine_options", {})
            if best_machine_i in opts_j:
                conflicts_on_best_machine += 1
                
        # Metric: 
        # We want to minimize the time the machine is occupied by i, 
        # plus the penalty of blocking others.
        # If we pick i, machine M* is busy for dur_i.
        # The jobs j that want M* are delayed by dur_i.
        # Total "badness" = dur_i * (1 + conflicts_on_best_machine)
        
        score = dur_i * (1.0 + conflicts_on_best_machine)
        
        # Tie-breaker: If scores are close, pick the one with less remaining work (urgency)
        # or just random to break ties consistently? Random can help exploration in some contexts,
        # but for deterministic evaluation, let's use remaining_work as a tie-breaker.
        # Actually, let's add a small epsilon of remaining_work to break ties.
        # Lower remaining work = closer to done = better to finish? 
        # Or higher remaining work = more urgent?
        # Usually, finishing a job early doesn't help makespan unless it unlocks a critical path.
        # Let's stick to the congestion metric primarily.
        
        if score < best_score:
            best_score = score
            best_idx = i
        elif score == best_score:
            # Tie-break: Pick the one with the smaller remaining_work? 
            # Or the one with the larger remaining_work?
            # Let's pick the one with the smaller min_duration (shortest first) if tied.
            # Actually, if score is tied, it means dur_i * (1+c) is same.
            # Let's just pick the first one found to be deterministic and fast.
            pass

    return int(best_idx)
# EVOLVE-BLOCK-END
