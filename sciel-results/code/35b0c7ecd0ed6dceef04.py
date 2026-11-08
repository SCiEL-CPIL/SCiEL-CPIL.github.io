"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

# Global state to track machine loads and ensure deadline safety
_machine_loads = {}
_start_time = time.time()

def dispatch(ready_operations, time):
    global _start_time
    
    # Deadline guard: if we are close to the 120s limit, fall back to a fast, simple heuristic
    # The prompt says evaluator kills at 150s, but score is based on 120s limit usually.
    # Let's use a safe margin.
    elapsed = time.time() - _start_time
    if elapsed > 110.0:
        # Fallback: Most work remaining (fast, no state)
        return max(range(len(ready_operations)), key=lambda i: ready_operations[i]["remaining_work"])

    # Update machine loads based on current time
    # We assume 'time' is the current simulation time.
    # We need to estimate the "load" or "busy until" time for each machine.
    # Since we don't have history of past events, we can approximate "load" by 
    # assuming the machine has been busy up to 'time' if it was selected recently?
    # No, that's not right. 
    # 
    # Alternative Structural Strategy: 
    # Instead of tracking state which is hard without history, let's look at the 
    # "machine_options" available for each operation.
    # A common effective heuristic in job-shop is to minimize the "completion time" 
    # of the operation, which is time + duration. 
    # However, if multiple jobs need the same machine, the one with shorter duration 
    # might be better to keep the machine free? Or the one with longer duration 
    # to reduce context switches?
    #
    # Let's try a different angle: Shortest Processing Time (SPT) on the machine.
    # Why? SPT minimizes mean flow time and often helps in reducing makespan in 
    # random instances by keeping machines busy with quick tasks, allowing 
    # more flexibility for subsequent operations.
    #
    # But wait, the current program uses "Most Work Remaining" (LWR).
    # LWR is a form of "Longest Processing Time" or "Critical Path" proxy.
    #
    # Let's try: **Least Loaded Machine First** is not possible without state.
    #
    # How about: **Shortest Duration First** among ready operations?
    # This is a structural change from "Most Work" (which looks at future) to 
    # "Shortest Duration" (which looks at immediate execution cost).
    
    # Let's combine: 
    # Score = duration on the *first* available machine? 
    # The dict has `machine_options`. We don't know which machine is free.
    # Usually, in these simulations, if an op is "ready", it means it can start 
    # on one of its options. The simulator likely picks the machine or we do?
    # The interface says: `dispatch(ready_operations, time) -> index of the ready operation to start`.
    # It does NOT say we pick the machine. The simulator likely assigns it to the 
    # first available machine in `machine_options` or the one with earliest availability.
    #
    # If we don't control the machine assignment, we should pick the operation 
    # that is most "urgent" or "bottlenecky".
    #
    # Let's try **Shortest Remaining Work**? No, that's similar to current.
    #
    # Let's try **Random**? No.
    #
    # Let's try **Earliest Due Date**? We don't have due dates.
    #
    # Let's try **Least Remaining Operations**? 
    # If a job has few operations left, finishing it frees up resources/jobs.
    #
    # Let's try a composite score:
    # Prioritize operations that have fewer machine options (more constrained) 
    # and shorter durations.
    
    best_idx = 0
    best_score = float('inf')
    
    for i, op in enumerate(ready_operations):
        # Calculate a heuristic score
        # 1. Number of machine options: Fewer options = more critical?
        num_options = len(op["machine_options"])
        
        # 2. Minimum duration among options
        min_dur = min(op["machine_options"].values())
        
        # 3. Remaining work
        rem_work = op["remaining_work"]
        
        # Heuristic: 
        # We want to pick ops that are "tight".
        # Let's try: Minimize (min_dur * num_options) / (rem_work + 1)
        # This favors short, constrained ops with high remaining work?
        
        # Actually, a very common and effective structural change for Job-Shop 
        # when simple heuristics stall is to use **Randomized** selection based 
        # on a weighted distribution, or **Look-ahead**.
        #
        # Let's try **Shortest Processing Time (SPT)** specifically.
        # It is the exact opposite of the "Most Work" spirit in terms of immediate
        # duration focus, though LWR is about total remaining.
        
        # If we just pick the op with the shortest minimum duration:
        score = min_dur
        
        # Tie-breaker: Most remaining work (to keep critical path moving)
        if score < best_score - 1e-9:
            best_score = score
            best_idx = i
        elif abs(score - best_score) < 1e-9:
            if op["remaining_work"] > ready_operations[best_idx]["remaining_work"]:
                best_idx = i

    return int(best_idx)

# EVOLVE-BLOCK-END
