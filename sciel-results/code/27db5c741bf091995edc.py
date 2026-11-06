import numpy as np
import time as time_module
import heapq
import random

# ------------------------------------------------------------------
#  Open-Shop dispatch rule with Greedy Look-Ahead.
#
#  Strategy: For each ready operation, simulate scheduling it and then
#  greedily scheduling the rest of the ready operations (and potentially
#  known future operations) to estimate the resulting makespan.
#  Select the operation that minimizes this estimated makespan.
# ------------------------------------------------------------------

_DEADLINE = 150.0
_START_TIME = time_module.time()
_TIME_MARGIN = 2.0

def _elapsed():
    return time_module.time() - _START_TIME

def _remaining_time():
    return _DEADLINE - _TIME_MARGIN - _elapsed()

def _get_machine_options(op):
    """Extract machine options as a list of (machine, duration)."""
    mo = op.get("machine_options", {})
    return [(m, d) for m, d in mo.items()]

def _greedy_simulate(ready_ops, machine_avail, current_time):
    """
    Greedily schedule all ops in ready_ops starting from current_time.
    machine_avail: dict {machine: time_when_available}
    Returns: estimated makespan (max machine_avail value).
    """
    # We need to track which ops are scheduled.
    # ready_ops is a list of dicts.
    # We will pick ops one by one.
    
    # To make it deterministic and fast, we sort by a simple priority: 
    # Max remaining work first.
    # But we don't have remaining work for the "future" ops if they are not in ready_ops?
    # The prompt says ready_operations are the ones ready to start.
    # If we only have ready_operations, we can't schedule ops that are NOT ready.
    # So this simulation only covers the current ready set.
    # This is a 1-step look-ahead if we don't have future info.
    
    # If we have future info in 'remaining_ops', we should use it.
    # Let's check if 'remaining_ops' contains structured data.
    
    all_ops_to_schedule = []
    for op in ready_ops:
        all_ops_to_schedule.append(op)
        
    # If we can access future ops, we add them.
    # For now, let's assume we only schedule what is ready.
    # This makes it a "1-step" decision based on "how long will the current ready set take if I pick this one first?"
    
    # Actually, a better 1-step metric is:
    # If I pick op i, the machine it uses is blocked for duration d.
    # Other ops that use that machine are delayed.
    # Other ops that don't use that machine are not delayed (if machines are independent).
    
    # Let's use a simple simulation of the ready set.
    # 1. Remove the chosen op from the set.
    # 2. Schedule the remaining ready ops greedily (e.g., LPT or MWR).
    # 3. The makespan is the max completion time.
    
    # However, this ignores that new ops might become ready.
    # But since we only have `ready_operations`, we can't know.
    
    # Let's refine:
    # For each candidate i:
    #   1. Schedule i on its best machine m_i at time t.
    #   2. Update avail[m_i] = t + d_i.
    #   3. For all other ready ops j:
    #      - Schedule j on its best available machine (min avail[m] + d_j[m]).
    #      - Update avail[m_j].
    #   4. Makespan = max(avail.values()).
    
    # This is O(N^2 * M).
    
    # Implementation:
    avail = dict(machine_avail)
    
    # We need to schedule the ops in ready_ops.
    # We pass the chosen op separately? No, let's simulate the whole set.
    # But we want to evaluate the effect of picking i FIRST.
    # So we force i to be scheduled first, then greedy for the rest.
    
    # To do this efficiently, we can just simulate the greedy schedule of the whole set,
    # but with a bias: if i is picked, it goes first.
    
    # Let's just simulate the greedy schedule of the entire ready set,
    # but we evaluate the "cost" of each op being picked first.
    
    # Actually, the simplest robust heuristic is:
    # Score(op) = min_machine ( avail[m] + duration[op][m] )
    # This is "Earliest Completion Time".
    # But we want to minimize makespan.
    # Picking the op that finishes earliest is good if it frees a machine for others.
    
    # Let's try the simulation approach.
    
    ops = list(ready_ops)
    
    # We will simulate scheduling 'ops' in a specific order.
    # For candidate i, the order is [i] + [others sorted by priority].
    
    # Priority for others: Max remaining work? Or Min duration?
    # Let's use Max remaining work (MWR) as it tends to clear the longest jobs.
    
    # Pre-compute remaining work for each op if available.
    # If not, use max duration.
    
    def get_priority(op):
        rw = op.get("remaining_work", 0)
        if rw > 0:
            return rw
        mo = op.get("machine_options", {})
        if mo:
            return max(mo.values())
        return 0

    # Sort others by priority descending
    # But we don't want to sort the whole list for every candidate.
    # We can just iterate.
    
    # For each candidate i:
    #   avail_copy = dict(avail)
    #   # Schedule i
    #   mo_i = _get_machine_options(i)
    #   best_m_i, best_finish_i = min((m, avail_copy[m] + d) for m, d in mo_i)
    #   avail_copy[best_m_i] = best_finish_i
    #   
    #   # Schedule rest greedily
    #   remaining_ops = [op for j, op in enumerate(ops) if j != i]
    #   # Sort remaining by priority?
    #   # To be fast, just pick the one with min earliest completion time?
    #   # Or max priority?
    #   # Let's use "Min Earliest Completion" (MEC) for the rest, as it's a common dispatch rule.
    
    #   while remaining_ops:
    #       best_op = None
    #       best_m = None
    #       best_finish = float('inf')
    #       for op in remaining_ops:
    #           mo = _get_machine_options(op)
    #           for m, d in mo:
    #               finish = avail_copy[m] + d
    #               if finish < best_finish:
    #                   best_finish = finish
    #                   best_op = op
    #                   best_m = m
    #       if best_op is None: break
    #       avail_copy[best_m] = best_finish
    #       remaining_ops.remove(best_op)
        
    #   makespan = max(avail_copy.values()) if avail_copy else 0
    #   # Evaluate candidate i based on makespan

    # This is O(N^3 * M) which is 10^3 * 5 = 5000 ops per candidate, 10 candidates = 50k ops.
    # 50k ops is trivial for Python.
    
    # Let's implement this.
    
    return avail # Placeholder, will implement in dispatch

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    if _remaining_time() < 0.5:
        # Fallback to simple heuristic
        return _simple_heuristic(ready_operations, time)
    
    n = len(ready_operations)
    
    # Get current machine availabilities.
    # The prompt doesn't explicitly give a global machine state, but `time` is current time.
    # We assume machines are available at `time` unless we track state.
    # The interface `dispatch(ready_operations, time)` suggests we don't have access to a global state object.
    # We must infer machine availability from `ready_operations`?
    # No, `ready_operations` are the ones that CAN start.
    # This implies that for each op, the machine is available at `time`?
    # Or does `machine_options` contain the availability?
    # "machine_options {machine: duration}" -> just duration.
    # If we don't have machine state, we assume all machines are available at `time`?
    # That would be a "parallel machine" assumption where all are free.
    # But in Open Shop, machines are busy.
    # If the simulator doesn't pass machine state, we can't know when machines are free.
    # UNLESS `ready_operations` is the ONLY source of truth, and the simulator handles the rest.
    # If so, my "simulation" of future steps is invalid because I don't know when machines will be free for OTHER ops.
    
    # Re-reading the prompt: "dispatch(ready_operations, time) -> index".
    # "each ready operation is a dict with job, op, machine_options {machine: duration}, remaining_ops, remaining_work".
    # It does NOT provide machine availability times.
    # This strongly implies that we should use a heuristic based ONLY on the attributes of the ready operations.
    # We cannot simulate machine conflicts accurately without knowing when machines are free.
    
    # Therefore, the "Look-Ahead" simulation I planned is flawed because I don't know `avail[m]`.
    # I must assume `avail[m] = time` for all machines?
    # If so, the problem reduces to: Which op should I start?
    # If all machines are free, I can start any op on any machine.
    # Then the makespan is determined by the sequence.
    # But I can only pick ONE op now.
    
    # So, what is the best op to pick?
    # If I pick op i on machine m, it finishes at time + d_i_m.
    # Machine m is then busy.
    # Other ops that need m are blocked.
    # Other ops that don't need m are not blocked.
    
    # Heuristic: Pick the op that minimizes the "delay" it causes to others?
    # Or pick the op that has the most "flexibility"?
    
    # Common Open Shop heuristics:
    # 1. LPT (Longest Processing Time): Pick op with max duration.
    # 2. SPT (Shortest Processing Time): Pick op with min duration.
    # 3. MWR (Most Work Remaining): Pick op with max remaining_work.
    # 4. Minimum Slack: (Earliest Due Date - Current Time - Duration).
    
    # The current program uses MWR + SPT + Load Balance.
    # Since we don't have Load Balance data (no machine state), we can't use it.
    # We can use "Number of ready ops sharing a machine" as a proxy for load?
    # If many ready ops use machine M, M is "hot".
    # We should avoid scheduling an op on a "hot" machine if possible?
    # Or schedule the op that uses a "cold" machine?
    
    # Let's try a heuristic that considers the "bottleneck" machine.
    # For each op, calculate the "best" machine (min duration).
    # Count how many ops prefer each machine.
    # If an op prefers a machine that many other ops also prefer, it's risky.
    
    # New Heuristic:
    # Score = remaining_work * 1.0 + (1 / min_duration) * 0.1 - (popularity_of_best_machine) * 0.05
    
    # Let's implement this "Popularity" based load balance.
    
    # 1. For each op, find the best machine (min duration).
    # 2. Count how many ops have that machine as their best.
    # 3. Score each op:
    #    - High remaining work is good.
    #    - Short duration is good (finishes fast).
    #    - If its best machine is very popular (high count), penalize it (to spread load).
    
    # This is a static approximation of load balancing.
    
    # Let's refine the weights.
    # MWR is usually strong.
    # SPT helps clear the queue.
    # Load balance prevents congestion.
    
    # Implementation:
    
    # 1. Pre-compute best machine and popularity.
    best_machines = {}
    popularity = {}
    
    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if not mo:
            continue
        # Find min duration machine
        best_m = min(mo, key=lambda m: mo[m])
        best_machines[i] = best_m
        popularity[best_m] = popularity.get(best_m, 0) + 1
    
    # 2. Score each op
    best_idx = -1
    best_score = float('-inf')
    
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0.0)
        mo = op.get("machine_options", {})
        if not mo:
            continue
            
        min_dur = min(mo.values())
        best_m = best_machines.get(i, None)
        pop = popularity.get(best_m, 0)
        
        # Score components
        s_rw = rw * 1.0
        s_dur = (1.0 / (min_dur + 1.0)) * 0.2
        s_pop = -pop * 0.1
        
        score = s_rw + s_dur + s_pop
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    if best_idx == -1:
        return 0
        
    return best_idx

def _simple_heuristic(ready_operations, time):
    best_idx = 0
    best_score = float('-inf')
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0.0)
        mo = op.get("machine_options", {})
        min_dur = min(mo.values()) if mo else 0.0
        score = rw * 1.0 + (1.0 / (min_dur + 1.0)) * 0.2
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx
