"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We track machine availability locally to estimate congestion.
# Since the simulator calls dispatch repeatedly, we can maintain a simple view of machine states
# based on the 'time' parameter and previous decisions if we could see them, but we only see current state.
# However, 'time' is passed. If we assume the simulation is event-driven, 'time' is the current simulation clock.
# We don't have explicit machine availability times in the input.
# But we can estimate "congestion" by looking at the durations of operations that *would* be on that machine.
# Since we don't see non-ready operations, we have to rely on the ready operations' characteristics.
# 
# A common heuristic for FJSP is "Regret". 
# Regret = (2nd best option cost) - (best option cost).
# High regret means we lose a lot if we don't pick this specific machine/job combo now.
#
# In our context, "cost" can be approximated by the completion time of the operation on that machine.
# Since we don't know exact machine availability, we assume that if we pick a machine, the operation starts at 'time' 
# (or shortly after if the machine is busy, but we don't know busy time).
# Actually, the prompt says "optionally, which eligible machine runs it". 
# If we return just an index, the simulator likely picks a machine (maybe random or first available).
# If we return (index, machine), we control the assignment.
# 
# Let's try returning (index, machine) to maximize control.
# 
# Strategy:
# 1. For each ready operation, calculate the "best" and "second best" potential completion times.
#    Since we don't know machine load, we approximate completion time as `time + duration`.
#    This makes all machines equivalent for a single operation unless we have load info.
#    
#    Wait, if we don't have load info, "Regret" based on completion time alone will be 0 for all (since time+duration is the same for all machines if we assume start at 'time').
#    
#    We need a proxy for machine load. 
#    One idea: Operations with longer durations are "heavier". If we assign a long op to a machine, it blocks it longer.
#    Maybe we should balance load.
#    
#    Let's look at the "remaining_work" of the job. 
#    If a job has high remaining work, it's critical.
#    If an operation has high duration, it's a bottleneck.
#    
#    Let's try a hybrid:
#    Priority = (Job Criticality) * (Operation Regret/Impact)
#    
#    Actually, let's look at the inspiration 2 which scored 0.976 (worse than 1.0). It used Regret.
#    The current program (1.0) uses Critical Job + SPT.
#    
#    Let's try a different structural change: **Machine-Load Balancing with Critical Path Awareness**.
#    
#    We will estimate the "load" on each machine by summing the durations of all ready operations that *could* go to that machine, weighted by the probability we might pick them? No, that's weak.
#    
#    Let's try **Shortest Remaining Processing Time (SRPT) for the Job**, but combined with **Machine Affinity**.
#    
#    Actually, a very effective heuristic for FJSP is to minimize the "critical path" of the job.
#    The current program picks the job with max remaining work.
#    
#    Let's try **Minimizing the Maximum Completion Time of the Job's Critical Path**.
#    
#    Alternative Strategy: **Dynamic Priority based on "Time Pressure"**.
#    Time Pressure = (Remaining Work) / (Time Left until Deadline?) No deadline given.
#    
#    Let's go back to **Regret**, but define it differently.
#    Regret of choosing Operation i on Machine m is the difference between the best available slot and the chosen slot.
#    Since we don't know slots, let's assume that picking a machine with a shorter duration for the *current* operation is better, BUT we also want to avoid bottlenecking a machine that has many other options.
#    
#    Let's try a **Greedy Assignment with Lookahead on Machine Congestion**.
#    
#    1. Identify the "Most Critical Job": The job with the highest `remaining_work`.
#    2. For that job's ready operation, select the machine that minimizes `duration`. (SPT)
#    3. BUT, if there are multiple jobs, maybe we should check if another job's operation has a very high "regret" (e.g., it can only be done on one machine, which is currently "busy" in a proxy sense).
#    
#    Let's implement a **Regret-Based** approach where "Cost" is the duration, and we assume all machines are available at `time`.
#    If Cost is just duration, Regret = 2nd_min_duration - min_duration.
#    We pick the operation with the HIGHEST Regret. Why? Because if we don't do it now, we might be forced to use the slower machine later? No, machines are interchangeable for the operation.
#    
#    Actually, in FJSP, machines are NOT interchangeable in terms of future load.
#    
#    Let's try a **Load-Balancing Heuristic**:
#    We maintain a global dictionary `machine_load` that accumulates the durations of operations assigned to each machine.
#    Since we don't have access to previous assignments, we can't do this across calls unless we use a global variable that persists?
#    The problem says "Write a Python dispatch... rule". It doesn't forbid global state, but it's risky if the simulator resets state.
#    However, `time` is increasing. We can infer that if `time` jumps, new operations might be ready.
#    
#    Let's assume we CAN use a global variable to track estimated machine loads.
#    We will add the duration of the chosen operation to the estimated load of the chosen machine.
#    This allows us to balance load over time.
    
# Global state for estimated machine loads
_estimated_machine_loads = {}

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Update estimated loads if time has advanced significantly? 
    # We can't really update loads accurately without knowing start times.
    # But we can assume that operations assigned in the past are "done" or "running".
    # This is tricky. Let's stick to a stateless heuristic first, or a very simple stateful one.
    
    # Let's try a **Stateless Regret** heuristic, but with a twist:
    # Score = (Remaining Work of Job) * (1 + Regret)
    # Regret = (2nd best duration) - (best duration)
    # This favors jobs with lots of work, and operations where the machine choice matters (high variance in duration).
    
    best_index = 0
    best_machine = None
    best_score = -float('inf')
    
    for i, op in enumerate(ready_operations):
        job_id = op["job"]
        rem_work = op["remaining_work"]
        durations = list(op["machine_options"].values())
        
        if len(durations) == 0:
            continue
            
        durations.sort(reverse=True)
        best_dur = durations[0]
        second_best_dur = durations[1] if len(durations) > 1 else best_dur
        
        regret = second_best_dur - best_dur
        
        # Score: We want to maximize this.
        # High remaining work is good.
        # High regret is good (because we have a strong preference for a specific machine/duration combo).
        # Short best duration is good (SPT).
        
        # Let's combine:
        # Priority = rem_work * (1.0 + regret / (best_dur + 1e-6))
        # This normalizes regret by the duration.
        
        priority = rem_work * (1.0 + regret / (best_dur + 1e-6))
        
        if priority > best_score:
            best_score = priority
            best_index = i
            # Find the machine with the best duration
            min_dur = float('inf')
            for m, d in op["machine_options"].items():
                if d < min_dur:
                    min_dur = d
                    best_machine = m
    
    # If we have a best machine, return (index, machine) to ensure the best choice is made.
    # If not, return index.
    if best_machine is not None:
        return (int(best_index), best_machine)
    else:
        return int(best_index)

# EVOLVE-BLOCK-END
