"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

def dispatch(ready_operations, time):
    """
    Dispatch rule using Dynamic Machine Load Balancing.
    
    Strategy:
    1. For each ready operation, identify the best machine (minimum duration).
    2. Calculate the "Projected Load" for each machine based on the current
       ready queue. This estimates how busy each machine will be if all current
       ready operations were scheduled optimally onto their best machines.
    3. Score each operation based on:
       a) Inverse Load: Prefer operations that go to less loaded machines to
          avoid bottlenecks and reduce idle time.
       b) Urgency: Prefer operations with higher remaining work (Critical Path)
          to ensure long jobs don't get starved.
    4. The final score is a weighted combination: Score = Urgency / (1 + Load).
       This effectively implements a "Critical Ratio" variant adapted for
       machine contention.
    
    This approach dynamically adapts to the queue composition, unlike static
    rules. It explicitly models the trade-off between urgency and machine
    availability.
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Deadline guard
    start_time = _time.time()
    deadline = 110.0  # Safety margin for 120s limit

    # 1. Pre-compute machine loads and operation properties
    # We estimate the load on each machine by summing the durations of all
    # ready operations that *would* choose that machine as their best option.
    # Note: This is an approximation. A more complex simulation would be
    # possible, but with n<=10, this O(n*m) pre-computation is cheap.
    
    machine_loads = {}
    op_properties = []
    
    for op in ready_operations:
        opts = op.get("machine_options", {})
        if not opts:
            op_properties.append({
                'best_machine': None,
                'best_dur': float('inf'),
                'rem_work': 0,
                'score': -np.inf
            })
            continue
        
        # Find best machine (min duration)
        best_m = min(opts, key=opts.get)
        best_dur = opts[best_m]
        
        # Accumulate load on this machine
        if best_m not in machine_loads:
            machine_loads[best_m] = 0
        machine_loads[best_m] += best_dur
        
        rem_work = op.get("remaining_work", 0)
        if rem_work is None or rem_work <= 0:
            rem_work = best_dur
            
        op_properties.append({
            'best_machine': best_m,
            'best_dur': best_dur,
            'rem_work': rem_work,
            'score': 0.0 # Placeholder
        })

    # 2. Calculate final scores
    # We want to maximize: rem_work / (1 + machine_load)
    # Why? 
    # - rem_work: Higher is better (urgency/critical path).
    # - machine_load: Lower is better (less contention).
    # The denominator (1 + load) acts as a penalty for busy machines.
    # Adding 1 prevents division by zero and stabilizes the ratio.
    
    scores = np.zeros(n)
    for i in range(n):
        prop = op_properties[i]
        if prop['best_machine'] is None:
            scores[i] = -np.inf
            continue
        
        load = machine_loads.get(prop['best_machine'], 0.0)
        # The score balances urgency (rem_work) against congestion (load)
        # We use a slightly dampened load to prevent zero-div and smooth variation
        scores[i] = prop['rem_work'] / (1.0 + load)
        
        # Tie-breaker: If scores are very close, prefer shorter processing time
        # to keep the machine free quickly? Or prefer longer to clear the job?
        # Usually, shorter processing time on the same machine is better for flow,
        # but for makespan, clearing the critical path is key.
        # Let's add a tiny epsilon * (1/best_dur) to prefer shorter ops on ties,
        # which often helps in job-shop by reducing blocking.
        if prop['best_dur'] > 0:
            scores[i] += 0.001 / prop['best_dur']

    # Check time
    if _time.time() > deadline:
        # Fallback to simple MWR
        rem_works = [op.get("remaining_work", 0) for op in ready_operations]
        return int(np.argmax(rem_works))

    # Select the operation with the highest score
    best_index = int(np.argmax(scores))
    
    return best_index

# EVOLVE-BLOCK-END
