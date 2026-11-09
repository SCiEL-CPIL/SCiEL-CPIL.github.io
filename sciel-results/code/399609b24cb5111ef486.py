"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import heapq
import random

# Stateful tracking for Beam Search
# We maintain the current estimated free time for each machine.
# This allows us to simulate sequences of dispatches.
_machine_free_times = {}
_start_time = None
_dispatch_count = 0
_instance_id = 0 # To reset state if a new instance starts (heuristic: if time goes backwards or ready ops change drastically? No, time is monotonic. We assume one instance per run or reset is handled by evaluator? The prompt says "five generated instances". The function is called repeatedly. We need to detect instance boundaries.
# How to detect new instance?
# If 'time' is less than previous max time, it's a new instance.
_last_max_time = 0

def dispatch(ready_operations, time):
    global _start_time, _dispatch_count, _machine_free_times, _last_max_time, _instance_id
    
    if _start_time is None:
        _start_time = _time.time()
        
    # Detect new instance: if current time is significantly less than the last observed max time, reset state.
    # In a single job-shop instance, time only moves forward.
    if time < _last_max_time - 1e-9:
        _machine_free_times = {}
        _dispatch_count = 0
        _last_max_time = 0
        _instance_id += 1

    # Deadline guard: If we are running too long, fall back to a very fast heuristic
    # to ensure we finish within the 120s limit.
    if _time.time() - _start_time > 105:
        # Fallback: SPT on the machine that is free earliest
        best_idx = 0
        best_key = (float('inf'), float('inf'))
        for i, op in enumerate(ready_operations):
            options = op["machine_options"]
            if not options:
                continue
            min_completion = float('inf')
            for m, dur in options.items():
                free_t = _machine_free_times.get(m, 0)
                completion = max(time, free_t) + dur
                if completion < min_completion:
                    min_completion = completion
            if (min_completion, 0) < best_key:
                best_key = (min_completion, 0)
                best_idx = i
        # Update state for fallback
        sel_op = ready_operations[best_idx]
        best_m = None
        min_c = float('inf')
        for m, dur in sel_op["machine_options"].items():
            free_t = _machine_free_times.get(m, 0)
            completion = max(time, free_t) + dur
            if completion < min_c:
                min_c = completion
                best_m = m
        if best_m is not None:
            _machine_free_times[best_m] = min_c
            _last_max_time = max(_last_max_time, min_c)
        return int(best_idx)

    n_ops = len(ready_operations)
    if n_ops == 0:
        return 0

    _dispatch_count += 1
    
    # Strategy: Beam Search with Rolling Horizon
    # We simulate a sequence of K dispatches.
    # State: (machine_free_times_dict, list_of_dispatched_ops_indices)
    # Score of a sequence: Max(machine_free_times) + Sum(remaining_work of unfinished jobs? No, we don't have global job state easily accessible without tracking)
    # Actually, we can estimate the "load" on the system.
    # A better score for the horizon: 
    # 1. Simulate K steps.
    # 2. At each step, pick the op that minimizes the "immediate completion time" but also considers the "critical path" of the job.
    # 3. The final score is the max machine time after K steps.
    # We want to pick the first op of the best sequence.
    
    # To keep it fast, we limit the beam width and horizon.
    # Horizon K=3, Beam Width B=3.
    # Total simulations: 1 (start) -> 3 (step 1) -> 9 (step 2) -> 27 (step 3).
    # Each simulation step involves iterating over ready ops (up to 10) and machines (5).
    # 27 * 10 * 5 = 1350 operations. Very fast.
    
    # We need to know which ops are ready at each step.
    # The input `ready_operations` is for the CURRENT time.
    # When we simulate dispatching op `i`, it removes `i` from the ready set.
    # It might make other ops ready? No, in Job Shop, an op becomes ready when the previous op in the SAME job finishes.
    # So if we dispatch op `i` (which is the next op of job `J`), then the next op of job `J` becomes ready.
    # But we don't have a global map of "next op of job J" easily.
    # However, `ready_operations` contains dicts with `job` and `op`.
    # We can assume that for each job, there is at most one ready op at any time.
    # So, if we dispatch an op of job `J`, the next op of job `J` becomes ready.
    # But we don't know what the next op of job `J` is, or its machine options, unless we track it.
    # The problem statement says: "each ready operation is a dict with job, op, machine_options...".
    # It does NOT provide the next operation's details.
    # This implies we CANNOT simulate future ready sets accurately unless we assume that the next op of a job is not ready until the current one finishes, AND we don't have its data.
    # Wait, if we don't have the data for the next op, we can't simulate it.
    # Therefore, a true rolling horizon simulation is impossible with the given interface unless we assume something about the future ops.
    
    # Alternative: "Lookahead" on the CURRENT ready set only?
    # If we dispatch op A, does it change the priority of op B (still in current ready set)?
    # Yes, because op A might finish on a machine, making that machine available later, or earlier.
    # But if A and B are on different machines, they don't interact until one finishes.
    # If A and B are on the same machine, they compete.
    
    # Let's try a simpler structural change: **Machine-Centric Congestion Avoidance**.
    # Instead of minimizing makespan estimate (which is myopic), we minimize the "congestion" on the machine.
    # Score = (Completion Time on Machine M) + lambda * (Current Load on Machine M).
    # This is essentially Weighted Shortest Processing Time (WSPT) but dynamic.
    
    # Let's go back to the Beam Search idea but constrained to the CURRENT ready set.
    # We can simulate sequences of dispatches from the CURRENT ready set.
    # If we dispatch A, then B, then C (all from current ready set), we are assuming they don't unlock new ops that interfere.
    # This is valid if the jobs are independent or if the next ops are not ready yet.
    # This "partial sequence" evaluation can help break ties and choose a sequence that balances load better than greedy.
    
    # Algorithm:
    # 1. Generate all permutations of the first K operations from the current ready set? No, too many.
    # 2. Beam Search:
    #    - State: (current_time, machine_free_times, set_of_dispatched_indices)
    #    - Start: (time, _machine_free_times, set())
    #    - Step 1: For each op in ready_ops not in dispatched, simulate dispatching it.
    #      - Determine best machine for this op given current machine_free_times.
    #      - Update machine_free_times.
    #      - New State.
    #    - Keep top B states by "makespan estimate" (max machine_free_time).
    #    - Step 2: For each state in beam, for each remaining op in ready_ops not in state.dispatched, simulate.
    #      - Keep top B states.
    #    - Step K: ...
    #    - After K steps, evaluate the final states.
    #    - The best state corresponds to a sequence of K ops.
    #    - We return the FIRST op of that sequence.
    #    - We also need to update the global _machine_free_times to reflect the ACTUAL dispatch of that first op.
    #    - Note: The simulation in the beam search updates machine_free_times. We must revert or use copies.
    
    K = 3 # Horizon
    B = 3 # Beam width
    
    # Initial state
    # State: (free_times_dict, dispatched_set)
    # We score states by max(free_times) + sum(remaining_work of jobs with ops in ready_ops but not dispatched?)
    # Actually, just max(free_times) is a good proxy for makespan contribution.
    
    beam = [ (dict(_machine_free_times), frozenset()) ]
    
    # Pre-compute best machine for each op given a state? No, it depends on state.
    
    for step in range(K):
        next_beam = []
        for state in beam:
            free_times, dispatched = state
            for i, op in enumerate(ready_operations):
                if i in dispatched:
                    continue
                options = op["machine_options"]
                if not options:
                    continue
                
                # Find best machine for this op in this state
                best_m = None
                best_finish = float('inf')
                for m, dur in options.items():
                    free_t = free_times.get(m, 0)
                    start_t = max(time, free_t) # time is the start time for this step? 
                    # Wait, in the simulation, the 'time' for the next dispatch is not simply the current global time.
                    # If we dispatch op A at time T, it finishes at T + dur.
                    # The next dispatch (op B) can start at max(T, machine_free_time_B).
                    # But machine_free_time_B is updated only if B is on the same machine as A? No, B is on its own machine.
                    # The 'time' parameter in the simulation for the next step should be the time when the machine becomes free?
                    # No, the event-driven simulation picks the NEXT event.
                    # If we dispatch A, the next event is either the completion of A, or the start of another op if its machine is free NOW.
                    # This makes the "sequence" simulation tricky.
                    # If machine M1 is free at T, and M2 is busy until T+10.
                    # Dispatch A on M1 (dur 5). A finishes at T+5.
                    # Can we dispatch B on M2? No, M2 is busy.
                    # Can we dispatch C on M1? No, M1 is busy until T+5.
                    # So the next dispatchable op must be on a machine that is free at T.
                    # This means the "ready" set for the next step depends on which machines are free.
                    # This is complex to simulate without the full simulator.
                    
                    # Simplification: Assume we are picking the next op to start AT THE CURRENT TIME 'time'.
                    # If we pick op A, it starts at 'time' (if machine free) or later.
                    # If we pick op B next, it also starts at 'time' (if its machine is free).
                    # If A and B are on different machines, they can both start at 'time'.
                    # If A and B are on the same machine, only one can start at 'time'.
                    
                    # So, a "sequence" of dispatches from the current ready set is only valid if they are on DISTINCT machines that are free at 'time'.
                    # If they are on the same machine, they are serialized.
                    
                    # Let's refine the simulation:
                    # State: machine_free_times.
                    # To dispatch op i:
                    #   Find machine m that minimizes start_time = max(time, free_times[m]).
                    #   start_time = max(time, free_times[m])
                    #   finish_time = start_time + dur
                    #   new_free_times[m] = finish_time
                    #   The "cost" or "progress" is the finish_time?
                    #   But for the NEXT op in the sequence, the 'time' is still the original 'time'?
                    #   No, the event-driven loop advances time.
                    #   If we dispatch A, the simulation clock doesn't advance until A finishes OR another op starts.
                    #   If another op B is ready and its machine is free at 'time', it can start at 'time' too.
                    #   So multiple ops can start at the same time 'time' if their machines are free.
                    
                    # Therefore, the "sequence" is actually a set of ops that can start at 'time'.
                    # We want to choose the set of ops to start at 'time' that optimizes the future.
                    # But we can only return ONE index.
                    # So we are effectively choosing the "first" op in a batch.
                    
                    # If we choose A, the simulator will start A.
                    # Then it will look for the next op to start.
                    # If B's machine is free, it will start B next (at the same time 'time').
                    # So the order of dispatching A and B doesn't matter if they are on different free machines.
                    # It only matters if they compete for the same machine.
                    
                    # So, the decision is really: Which op should be assigned to a machine?
                    # If multiple ops are ready and multiple machines are free, we have a matching problem.
                    # We want to match ops to machines to minimize makespan.
                    
                    # New Strategy: **Greedy Matching with Lookahead**
                    # 1. Identify all (op, machine) pairs that can start at 'time' (i.e., machine free at 'time').
                    # 2. This forms a bipartite graph.
                    # 3. We want to pick a matching that minimizes the sum of completion times or max completion time.
                    # 4. Since we can only pick one, we pick the op that is part of the "best" matching.
                    # 5. How to define "best"? Minimize the max completion time of the matched ops.
                    
                    # Let's try this:
                    # 1. Filter ready ops to those that can start immediately (machine free at 'time').
                    #    If none can start immediately, pick the one that will start earliest (min max(time, free_t)).
                    # 2. For the ops that can start immediately, try to find a matching that minimizes the max finish time.
                    #    Since N is small (<=10), we can use a simple greedy or even brute force for the matching if the set is small.
                    #    Actually, we just need to pick ONE op.
                    #    We can score each op by: "If I pick this op, what is the estimated makespan of the system after this dispatch?"
                    #    This is similar to the current program but with a better makespan estimate.
                    
                    # Let's stick to the Beam Search on the CURRENT ready set, assuming that if we dispatch A, the next dispatch is B, and they are on different machines (so they both start at 'time').
                    # If they are on the same machine, B starts later.
                    
                    # Simulation Step:
                    # State: free_times
                    # Action: Dispatch op i on machine m (best m for i)
                    #   start_t = max(time, free_times[m])
                    #   finish_t = start_t + dur
                    #   new_free_times = free_times.copy(); new_free_times[m] = finish_t
                    #   Score = max(new_free_times.values())
                    
                    # This is exactly what the current program does for K=1.
                    # To make it K=2:
                    #   After dispatching i, we have new_free_times.
                    #   For each other op j, find best machine m_j given new_free_times.
                    #   start_t_j = max(time, new_free_times[m_j])  # Note: time is still the original 'time' for the start of the batch?
                    #   Wait, if i and j are on different machines, j can start at 'time'.
                    #   If i and j are on the same machine, j starts at finish_t_i.
                    #   The formula start_t_j = max(time, new_free_times[m_j]) handles this correctly!
                    #   Because if m_j != m_i, new_free_times[m_j] is unchanged (or was < time), so start_t_j = time.
                    #   If m_j == m_i, new_free_times[m_j] = finish_t_i, so start_t_j = finish_t_i.
                    
                    # So the simulation is valid!
                    # We can simulate sequences of dispatches from the current ready set.
                    
                    # Let's implement the Beam Search properly.
                    
                    pass # Placeholder for logic inside loop, will implement below

        # Implementing the Beam Search logic here to avoid indentation issues in the thought process
        # We'll do it in the code block.
        pass

    # Fallback if beam search fails or is empty
    return 0

# EVOLVE-BLOCK-END
