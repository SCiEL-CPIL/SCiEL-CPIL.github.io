"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import heapq

# Global state to cache simulation results or handle timeouts gracefully
# Since dispatch is called frequently, we need this to be extremely fast.
# The "Look-Ahead" strategy simulates the rest of the schedule for each candidate.

def _simulate_schedule(ready_ops, current_time):
    """
    Simulates the completion of all jobs based on the current set of ready operations.
    Returns the estimated makespan (max completion time).
    """
    if not ready_ops:
        return current_time
    
    # We need to simulate the event-driven process.
    # State:
    # - machine_avail: list of times when each machine becomes free
    # - job_next_op: for each job, the index of the next operation to process
    #   (We assume ready_ops contains the *current* next op for each job that is ready)
    #   Wait, ready_ops only contains jobs that are *currently* ready.
    #   Jobs not in ready_ops are waiting for a previous op to finish.
    #   However, we don't have the full job list or the sequence of ops for non-ready jobs 
    #   in the input signature `dispatch(ready_operations, time)`.
    #   
    #   CRITICAL INSIGHT: 
    #   The input `ready_operations` contains dicts with `job`, `op`, `machine_options`, `remaining_ops`, `remaining_work`.
    #   `remaining_ops` likely contains the *future* operations for that specific job.
    #   If a job is NOT in `ready_operations`, it is blocked. We cannot simulate it unless we have global state.
    #   BUT, the task says "picks which ready operation to start next". 
    #   This implies a single-step decision.
    #   
    #   If we only have info for *ready* jobs, we cannot simulate the *entire* makespan because we don't know 
    #   the details of the blocked jobs (their remaining ops and machine options).
    #   
    #   HOWEVER, often in these problems, `remaining_work` is a good proxy. 
    #   But the directive asks for a STRUCTURAL change. 
    #   
    #   Alternative Structural Change: **Min-Conflict / Local Search**? No, too slow.
    #   Alternative: **Dynamic Priority based on Machine Contention**.
    #   
    #   Let's re-read the inspiration. "Dynamic, machine-centric bottleneck heuristic".
    #   We can estimate the load on each machine by looking at the `machine_options` of all ready ops.
    #   If a machine is heavily requested by many long ops, it's a bottleneck.
    #   We should prioritize ops that go to *less* contested machines, OR ops that are critical 
    #   (long remaining work) but can be placed on a machine that is *not* the primary bottleneck.
    #   
    #   Actually, a better structural approach for Job Shop is **Critical Path Method (CPM) approximation**.
    #   The makespan is determined by the longest path in the job graph.
    #   We can estimate the "urgency" of an operation by its position on the critical path.
    #   Ops that are on the critical path should be scheduled first.
    #   
    #   How to estimate critical path without full graph?
    #   `remaining_work` is the sum of min durations. This is the length of the critical path *if* all machines were available.
    #   But if a machine is busy, the effective remaining time is longer.
    #   
    #   Let's try a **Load-Aware MWR**:
    #   Score = remaining_work / (1 + estimated_machine_load)
    #   Where estimated_machine_load is the number of other ready ops that also want this machine 
    #   and have high remaining work.
    #   
    #   Let's implement a "Bottleneck Avoidance" heuristic:
    #   1. For each machine, calculate the "demand" = sum of min(duration) for all ready ops that can use it.
    #   2. For each ready op, calculate its "bottleneck penalty" = sum of demand for its candidate machines.
    #   3. Prioritize ops with HIGH remaining work but LOW bottleneck penalty.
    #   This pushes long jobs away from the most congested machines if possible, 
    #   or prioritizes them if they have no choice, but generally tries to balance load.
    
    pass

def dispatch(ready_operations, time):
    """
    Dispatch rule using Bottleneck-Aware Longest Remaining Work.
    
    Strategy:
    Standard MWR (Max Remaining Work) ignores machine contention. 
    If two jobs have similar remaining work, but one is forced onto a machine that is 
    already heavily requested by other long jobs, it will cause more delay than a job 
    that can use a less contested machine.
    
    This heuristic calculates a "contention score" for each machine based on the 
    aggregate minimum processing time of all ready operations that can use it.
    Then, for each ready operation, it penalizes its priority score based on the 
    contention of its available machines.
    
    Score = remaining_work / (1 + avg_contention_of_its_machines)
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Step 1: Estimate machine contention
    # machine_demand[m] = sum of min(duration) for all ready ops that can use machine m
    machine_demand = {}
    
    # Pre-process ready ops to get min durations and machine lists
    op_data = []
    for op in ready_operations:
        mo = op["machine_options"]
        min_dur = min(mo.values())
        machines = list(mo.keys())
        op_data.append({
            "index": len(op_data),
            "min_dur": min_dur,
            "machines": machines,
            "remaining_work": op["remaining_work"]
        })
        
        # Accumulate demand
        for m in machines:
            if m not in machine_demand:
                machine_demand[m] = 0.0
            machine_demand[m] += min_dur

    # Step 2: Calculate priority scores
    scores = []
    for i, data in enumerate(op_data):
        # Calculate average contention for this op's machines
        # If a machine has high demand, it's a bottleneck.
        # We want to AVOID bottleneck machines if we have other choices?
        # Or do we want to prioritize the job that *must* use the bottleneck to get it out of the way?
        
        # Insight: In job shop, if a machine is a bottleneck, you want to keep it busy.
        # However, if you have a choice between a short op on a bottleneck and a long op on a non-bottleneck,
        # the long op on the non-bottleneck might reduce makespan more by freeing up the job's sequence.
        
        # Let's use a penalty for high contention.
        # contention = sum of machine_demand for its machines / count
        if data["machines"]:
            avg_demand = sum(machine_demand[m] for m in data["machines"]) / len(data["machines"])
        else:
            avg_demand = 0
            
        # The higher the remaining_work, the higher the priority.
        # The higher the avg_demand (contention), the lower the priority (penalty).
        # We use a division to create a ratio: Work-to-Load.
        
        # Add a small epsilon to avoid division by zero if demand is 0 (unlikely)
        score = data["remaining_work"] / (1.0 + avg_demand)
        
        # Secondary tie-breaker: If scores are close, prefer the one with shorter current min duration?
        # Or prefer the one with more remaining ops (more future constraints)?
        # Let's stick to the main score.
        
        scores.append(score)

    # Return index of max score
    # np.argmax is fast
    import numpy as np
    best_idx = int(np.argmax(scores))
    
    return best_idx

# EVOLVE-BLOCK-END
