"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random

# Deadline guard: keep a safety margin before the 120s evaluation limit.
# We track cumulative dispatch call time to avoid pathological slowdowns.
_CALL_START = None
_TOTAL_CALL_TIME = 0.0
_CALL_COUNT = 0
_TIME_BUDGET = 110.0  # safety margin: 120s limit, reserve 10s


def _elapsed():
    global _TOTAL_CALL_TIME, _CALL_START
    if _CALL_START is None:
        return _TOTAL_CALL_TIME
    return _TOTAL_CALL_TIME + (_time.perf_counter() - _CALL_START)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next and, optionally, which eligible
    machine runs it.
    
    Returns:
        - int: index of the ready operation to start, OR
        - tuple (index, machine): index and chosen machine
    
    Strategy: Look-ahead bottleneck heuristic.
    For each ready operation, estimate the earliest completion time if assigned
    to the least-loaded eligible machine. We combine:
      1. Machine load balancing: prefer operations that can finish earlier
         on the currently least-busy eligible machine.
      2. Remaining work: break ties by remaining work (urgency).
      3. A small look-ahead: consider how many remaining ops each job has,
         to avoid creating bottlenecks downstream.
    """
    global _CALL_START, _TOTAL_CALL_TIME, _CALL_COUNT
    
    # Start timing this call
    if _CALL_START is None:
        _CALL_START = _time.perf_counter()
    
    # Check if we're close to the time budget
    if _elapsed() > _TIME_BUDGET:
        # Safety: return the first ready operation
        return 0
    
    _CALL_COUNT += 1
    
    if not ready_operations:
        return 0
    
    if len(ready_operations) == 1:
        op = ready_operations[0]
        # Choose the shortest-duration machine option
        best_machine = min(op["machine_options"].keys(), key=lambda m: op["machine_options"][m])
        return (0, best_machine)
    
    n = len(ready_operations)
    
    # For each ready operation, compute a score that estimates how "good" it is
    # to schedule now. Lower score = better to schedule first.
    # We want to minimize makespan, so we want to:
    #   - Schedule operations that can complete quickly (short duration on available machine)
    #   - Balance machine loads
    #   - Consider remaining work in the job (urgency)
    
    # Estimate current machine loads from the operation data.
    # We don't have direct access to machine availability, but we can infer
    # relative loads from the durations and the fact that operations are ready.
    # A simpler approach: for each op, find the machine with minimum duration,
    # and use that as a proxy for "how fast this op can finish."
    
    scores = []
    chosen_machines = []
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        remaining_ops = op.get("remaining_ops", 0)
        remaining_work = op.get("remaining_work", sum(machine_options.values()))
        
        # Find the machine with the shortest processing time for this op
        min_duration = min(machine_options.values())
        best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
        
        # Look-ahead: if this job has many remaining ops, scheduling a short
        # op now might not help much if the job will still be long.
        # Conversely, if this is the last op, completing it quickly removes a job.
        
        # Score components:
        # 1. Earliest completion proxy: min_duration (lower is better)
        # 2. Remaining work urgency: remaining_work (lower is better for makespan,
        #    because finishing short-remaining jobs quickly reduces tail)
        # 3. Remaining ops count: fewer remaining ops means this op is closer to
        #    completing the job, which is good for reducing makespan.
        
        # Combine: primary key is min_duration (machine-centric, load balancing),
        # secondary is remaining_work (job-centric urgency),
        # tertiary is remaining_ops (fewer = closer to done).
        
        # We use a weighted combination. The key insight: by preferring ops
        # with short processing times on some machine, we keep machines busy
        # and avoid idle time, which directly reduces makespan.
        
        score = (
            min_duration * 1.0 +
            remaining_work * 0.1 +
            remaining_ops * 0.5
        )
        
        scores.append(score)
        chosen_machines.append(best_machine)
    
    # Pick the operation with the lowest score
    best_idx = min(range(n), key=lambda i: scores[i])
    
    # Stop timing this call
    global _TOTAL_CALL_TIME
    _TOTAL_CALL_TIME += _time.perf_counter() - _CALL_START
    _CALL_START = None
    
    return (best_idx, chosen_machines[best_idx])
# EVOLVE-BLOCK-END
