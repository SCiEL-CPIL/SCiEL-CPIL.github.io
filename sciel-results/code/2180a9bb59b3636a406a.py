"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random
import math
import heapq

# Deadline guard
_START_TIME = None
_DEADLINE = 145.0  # Safety margin

# State tracking to improve heuristics
# We track estimated load per machine and per job
_machine_load = {}
_job_remaining = {}
_job_completed_ops = {}
_total_ops_per_job = {}

def _init_state(ready_operations, time):
    """Initialize or update state tracking."""
    global _machine_load, _job_remaining, _job_completed_ops, _total_ops_per_job
    
    # This is called at the start of each dispatch. 
    # We can't easily know the total ops per job initially without seeing all ops.
    # However, 'remaining_ops' in the dict might help? The task says 'remaining_ops' is in the dict.
    # Let's assume remaining_ops is an integer count of ops left for that job.
    # And remaining_work is the sum of durations? Or just time?
    # "remaining_work" usually implies time units.
    
    # We will use the current ready_operations to update our view.
    # Since we don't have access to the full simulation state, we have to be careful.
    # We can maintain a global counter for how many ops we've seen for each job?
    # Actually, simpler: just use the info from the current ready ops.
    # The heuristic can be purely local to the current ready set if global state is unreliable.
    # But tracking machine load helps.
    # We can estimate machine load by summing durations of ops assigned to that machine?
    # We don't know which machine was chosen for previous ops unless we return (index, machine).
    # The task says: "optionally (index, machine)".
    # If we return just index, the simulator picks a machine? Or does it expect us to pick?
    # "picks which ready operation to start next and, optionally, which eligible machine runs it"
    # "return index of the ready operation to start (optionally (index, machine))"
    # If we return just index, the simulator likely picks the machine (maybe the one with min duration? or first?).
    # To be safe and potentially better, we should return (index, machine) if we can choose.
    # But the return type in the current program is just `int`.
    # The docstring says: `dispatch(ready_operations, time) -> index of the ready operation to start (optionally (index, machine))`
    # The current code returns `int(best_idx)`.
    # If I change to return a tuple, will the evaluator handle it?
    # "Keep the function names and signatures of the seed program."
    # The seed program returns an int. Changing to tuple might break the evaluator if it expects an int.
    # However, the contract says "optionally (index, machine)".
    # Let's stick to returning an int to be safe, unless the evaluator explicitly supports tuples.
    # Given the risk, I will stick to returning an int, but I can still optimize which operation to pick.
    # If I can't pick the machine, the simulator's choice of machine matters.
    # Usually, simulators pick the machine that minimizes completion time (earliest available) or similar.
    # If the simulator picks the machine, my choice of operation is the only lever.
    
    # Let's refine the heuristic.
    # Current: 0.6 * work + 0.3 * eff + 0.1 * flex
    # This is a linear combination.
    # Let's try a multiplicative or ratio-based approach.
    # Or use a priority rule like "Longest Remaining Processing Time" (LRPT) or "Shortest Processing Time" (SPT).
    # For makespan, LRPT (Longest Remaining Processing Time) is often good because it prioritizes jobs that have a lot of work left, preventing them from getting stuck at the end.
    # SPT is good for flow time, but can hurt makespan if a long job is starved.
    # So, high remaining_work is good.
    # What about the current operation's duration?
    # If an op has a very short duration on its best machine, doing it now frees up the machine quickly for other jobs?
    # Or does it matter less than clearing the job's backlog?
    
    # Let's try a score that emphasizes remaining_work more strongly.
    # Or a score based on the "critical path" estimate.
    # Critical path for a job = sum of remaining_work? No, remaining_work is likely the sum of min durations?
    # If remaining_work is the sum of processing times on the *assigned* machines, it's a lower bound on time to finish.
    
    # Let's test a simple rule: Sort by remaining_work descending.
    # That is essentially what the current code does (0.6 weight).
    # Let's try to incorporate the machine availability.
    # If we can't see machine state, we can't know which machine is free.
    # But we can look at the durations.
    # If an operation has a very long duration on all machines, it's a bottleneck.
    # If an operation has a short duration on at least one machine, it's flexible/quick.
    
    # Let's try a different scoring function:
    # Score = remaining_work / (min_duration + epsilon)
    # This favors ops with high work and low duration.
    # This is similar to the current one but normalized differently.
    
    # Let's also consider the number of remaining ops.
    # If a job has many ops left, it might be better to start it?
    
    # Let's implement a few heuristics and pick the best candidate from a small set using a local search or just a better score.
    
    # Heuristic 1: MWR (Most Work Remaining) - Maximize remaining_work
    # Heuristic 2: SPT (Shortest Processing Time) - Minimize min_duration
    # Heuristic 3: Hybrid - Maximize remaining_work / min_duration
    
    # Let's try Heuristic 3.
    
    pass

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Deadline check
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback: MWR
        if not ready_operations:
            return 0
        return int(max(range(len(ready_operations)), key=lambda i: ready_operations[i].get("remaining_work", 0)))
    
    if not ready_operations:
        return 0
    
    # If only one op, return it
    if len(ready_operations) == 1:
        return 0
    
    # Calculate scores for all ops
    scores = []
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0)
        mo = op.get("machine_options", {})
        if mo:
            durations = list(mo.values())
            min_d = min(durations)
        else:
            min_d = 1.0
        
        # Avoid division by zero
        if min_d == 0:
            min_d = 1e-9
            
        # Score: High rw, Low min_d
        # Option A: rw / min_d
        # Option B: rw * (1 / (min_d + 1))
        # Option C: rw - min_d (subtraction might not scale well)
        
        # Let's try Option A
        score = rw / min_d
        
        # Add a small tie-breaker: prefer higher remaining_work if scores are close
        # Or prefer lower min_d?
        # Let's just use the score.
        scores.append((score, i))
        
    # Sort by score descending
    scores.sort(reverse=True)
    
    # Return the index of the best score
    return int(scores[0][1])

# EVOLVE-BLOCK-END
