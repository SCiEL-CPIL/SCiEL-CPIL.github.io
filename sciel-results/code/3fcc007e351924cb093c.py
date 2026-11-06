"""Open-shop dispatch rule: Lookahead simulation with time budget."""
# EVOLVE-BLOCK-START
import time
import random
import math

# Global state for the time budget
_START_TIME = time.time()
# The evaluator kills at 150s. We reserve a safety margin of 5s, so we stop at 145s.
# Since dispatch is called many times, we need to be very fast.
# A total budget of 140s is safer.
_DEADLINE = _START_TIME + 140.0


def _time_left():
    return _DEADLINE - time.time()


def _simulate(ready_operations, time, dispatch_rule_func):
    """
    Simulate the rest of the schedule given the current state.
    Returns the makespan (time when all jobs are done).
    ready_operations: list of dicts
    time: current time
    dispatch_rule_func: function(ready_ops, time) -> index
    """
    # We need to track the state of all jobs and machines.
    # Since we only have ready_operations at this point, we need to reconstruct
    # the full state or simulate forward assuming we know all operations.
    # However, the interface only gives us ready_operations.
    # In an open shop, all operations of a job must be processed, but in any order.
    # The input `ready_operations` contains `remaining_ops` and `remaining_work`.
    # To simulate, we need to know the durations of the remaining operations on each machine.
    # The `machine_options` dict in ready_operations gives the duration of the *current* operation.
    # We don't have the full list of remaining operations' durations for future steps.
    
    # Wait, the prompt says: "each ready operation is a dict with job, op, machine_options {machine: duration}, remaining_ops, remaining_work."
    # It does NOT provide the durations of future operations.
    # This means we CANNOT do a full simulation of the future schedule because we don't know the processing times of the non-ready operations.
    
    # Let's re-read the task carefully.
    # "Write a Python dispatch(ready_operations, time) rule that picks which ready operation to start next in an event-driven open-shop simulation"
    # The evaluator runs the simulation. It calls dispatch() whenever an operation completes (or at t=0).
    # The dispatch function only sees the *currently ready* operations.
    # It does NOT see the future.
    
    # Therefore, any "lookahead simulation" that requires knowing future processing times is impossible unless we make assumptions or the data is hidden in a way I'm missing.
    # Let's look at the fields again:
    # job, op, machine_options, remaining_ops, remaining_work.
    # `remaining_ops` is likely the count of operations left for that job (including the current one? or excluding?).
    # `remaining_work` is the sum of durations of remaining operations for that job.
    
    # Since we don't know the specific durations of future operations on specific machines, we cannot simulate the exact future makespan.
    # We can only use heuristics based on the available information:
    # 1. remaining_work: Total work left for the job.
    # 2. machine_options: Duration of the current operation on each available machine.
    # 3. remaining_ops: Number of operations left.
    
    # The seed program uses `np.argmax([op["remaining_work"] for op in ready_operations])`.
    # This is a "Longest Processing Time" (LPT) variant, but on total job work.
    
    # Can we do better?
    # Common open-shop heuristics:
    # - SPT (Shortest Processing Time) on the current operation.
    # - LPT (Longest Processing Time) on the current operation.
    # - Minimum Slack.
    # - Priority based on machine load.
    
    # Since we don't know machine loads (we don't know when machines become free, only that they are free now because the op is "ready"), we can't use machine load directly unless we track it globally, which we can't easily do in a stateless dispatch function unless we use global state.
    
    # Let's try a few simple heuristics and see which one might be better.
    # 1. Max remaining_work (Seed)
    # 2. Min duration of current op (SPT)
    # 3. Max duration of current op (LPT)
    # 4. Ratio: remaining_work / remaining_ops (Average op length)
    
    # Actually, in open shop scheduling, a good heuristic is often to prioritize jobs with more remaining work to keep them moving, or to prioritize short operations to reduce waiting.
    
    # Let's look at the observed changes.
    # p7 had a score of 1.0021 (slightly better than 1.0). It used "Lookahead Simulated Annealing". But as established, full lookahead is impossible without future data. Maybe it used a simplified lookahead or just random sampling?
    # p6 had a score of 0.9511 (worse).
    
    # Let's try a more sophisticated heuristic that considers the machine options.
    # For each ready operation, we can choose the best machine (shortest duration) or the one that minimizes makespan contribution.
    # But dispatch only returns the index of the operation, not the machine choice.
    # Wait, the interface says: "return index of the ready operation to start (optionally (index, machine))".
    # If we return (index, machine), the simulator will use that machine.
    # If we return just index, the simulator might pick a machine (e.g., the one with the shortest duration, or the first one).
    # To be safe and optimal, we should probably return the machine that minimizes the duration for that operation, as it frees the machine sooner.
    
    # Let's assume the simulator picks the best machine if we don't specify, or we can specify it.
    # Let's try returning the index of the operation with the minimum "effective" duration, where effective duration is the min over machine_options.
    # Or maybe the operation with the maximum remaining work, but on the shortest machine?
    
    # Let's try a combined score:
    # Score = w1 * (min_duration) + w2 * (remaining_work)
    # We want to minimize makespan.
    # SPT (Shortest Processing Time) is often good for flow shop, but for open shop?
    # In open shop, "Most Work Remaining" is a standard heuristic (the seed).
    
    # Let's try "Shortest Processing Time" on the current operation (min duration over machines).
    # Hypothesis: Starting the shortest available operation first reduces the time the machine is busy, allowing other jobs to proceed sooner.
    
    # Let's also consider "Longest Processing Time" on the current operation.
    
    # Since we can't test multiple heuristics easily in one run, let's pick one that is theoretically sound.
    # For makespan minimization in open shop, the problem is NP-hard.
    # A common heuristic is to use the "SPT" rule on the operations.
    
    # Let's implement a dispatch that:
    # 1. For each ready op, finds the minimum duration machine.
    # 2. Sorts by this minimum duration (SPT).
    # 3. Returns the index of the first one.
    
    # Alternatively, let's try a weighted combination:
    # Priority = remaining_work / (1 + min_duration)
    # This favors jobs with lots of work but penalizes long current ops.
    
    # Let's stick to a simple, robust improvement over the seed.
    # The seed uses `argmax(remaining_work)`.
    # Let's try `argmin(min_duration)`.
    
    # To be safe, let's add a tie-breaker: if durations are equal, pick the one with more remaining work.
    
    # Implementation:
    # For each op in ready_operations:
    #   min_dur = min(op["machine_options"].values())
    #   score = min_dur
    #   # We want to minimize score, so we pick argmin.
    #   # But we also want to consider remaining work.
    #   # Let's try: score = min_dur - 0.1 * remaining_work
    #   # This is a guess.
    
    # Let's look at the p7 program diff snippet:
    # "_simulate._best_machine_for_op"
    # This suggests it was trying to simulate.
    
    # Given the constraints, let's try a "Smart SPT":
    # Pick the operation with the shortest processing time on its best machine.
    # If there's a tie, pick the one with the most remaining work.
    
    def _get_min_duration(op):
        return min(op["machine_options"].values())
    
    # We want to minimize the duration.
    # But we also want to balance load.
    # Let's try a simple SPT rule.
    
    best_idx = 0
    best_score = float('inf')
    best_remaining = 0
    
    for i, op in enumerate(ready_operations):
        min_dur = min(op["machine_options"].values())
        remaining_work = op["remaining_work"]
        
        # Primary key: min_duration (lower is better)
        # Secondary key: remaining_work (higher is better)
        # We can combine them into a single score to minimize.
        # score = min_dur - epsilon * remaining_work
        # Let's use epsilon = 0.1 for now.
        score = min_dur - 0.1 * remaining_work
        
        if score < best_score:
            best_score = score
            best_idx = i
            best_remaining = remaining_work
    
    # Return the index
    # We could also return (best_idx, best_machine) to ensure the shortest machine is used.
    # Let's find the best machine for the chosen op.
    best_op = ready_operations[best_idx]
    best_machine = min(best_op["machine_options"], key=lambda m: best_op["machine_options"][m])
    
    # The interface allows returning (index, machine). Let's use that to be explicit.
    return (best_idx, best_machine)

# EVOLVE-BLOCK-END
