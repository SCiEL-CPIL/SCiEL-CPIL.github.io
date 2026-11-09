"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Track start time for deadline guard
_t0 = _time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

# Global state to track cumulative load on each machine
# This allows for load balancing across the simulation run
_machine_load = {}


def dispatch(ready_operations, time):
    """
    Dispatch rule that picks which ready operation to start next and,
    optionally, which eligible machine runs it.
    
    Strategy: 
    1. Estimate the "cost" of processing each ready operation on each machine,
       defined as processing duration + current accumulated load on that machine.
    2. Select the machine that minimizes this cost (Load-Balanced SPT).
    3. Prioritize operations based on a composite score:
       - High remaining work (MRW)
       - Low estimated completion cost (SPT/Balancing)
       - Low flexibility (fewer machine options = more urgent/constrained)
    4. Return (index, machine) for the highest priority operation.
    """
    # Deadline guard
    if _time.time() - _t0 > _DEADLINE:
        return 0
    
    if not ready_operations:
        return 0
    
    best_idx = -1
    best_score = -float('inf')
    best_machine = None
    best_cost = 0
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        machine_options = op["machine_options"]
        
        # Find the machine that minimizes (duration + current_load)
        min_cost = float('inf')
        best_m = None
        for m, d in machine_options.items():
            current_load = _machine_load.get(m, 0)
            cost = d + current_load
            if cost < min_cost:
                min_cost = cost
                best_m = m
        
        # Heuristic Score:
        # 1. Ratio of remaining work to estimated step cost. 
        #    Higher ratio means "more work to do per unit of time/cost", 
        #    which often indicates critical path items.
        # 2. Flexibility penalty: Jobs with fewer machine options are more 
        #    constrained and should be scheduled earlier to avoid blocking.
        
        # Avoid division by zero
        step_cost = min_cost + 1e-9
        ratio_score = remaining_work / step_cost
        
        # Flexibility factor: 1.0 if only 1 option, 0.5 if 2, etc.
        # We add this to boost priority of constrained jobs.
        flexibility_score = 1.0 / len(machine_options)
        
        # Combined score
        # Weighting: The ratio is the primary driver. Flexibility is a tiebreaker/boost.
        score = ratio_score + flexibility_score * 0.5
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_m
            best_cost = min_cost
    
    # If no operation was selected (should not happen if ready_operations is not empty)
    if best_idx == -1:
        return 0
    
    # Update global machine load for the chosen machine
    # We use the duration of the operation on the selected machine
    chosen_op = ready_operations[best_idx]
    duration = chosen_op["machine_options"][best_machine]
    _machine_load[best_machine] = _machine_load.get(best_machine, 0) + duration
    
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
