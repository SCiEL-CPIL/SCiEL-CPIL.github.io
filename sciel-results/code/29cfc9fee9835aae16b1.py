# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dynamic bottleneck identification strategy.
    
    For each ready operation, we consider assigning it to its least-loaded available machine
    (estimated via duration). We then compute the minimum slack time across all currently ready
    operations after hypothetically assigning that operation. The goal is to select the operation
    whose assignment maximizes the minimum slack time, thereby preventing machine starvation
    and reducing idle gaps.
    
    Slack time for a ready operation is estimated as:
      slack = remaining_work - estimated_time_to_complete_if_started_now
    
    We approximate the impact of starting an operation on other operations' slack by considering
    that starting an operation on a machine reduces the availability of that machine, which could
    increase the effective remaining work for other operations that need that machine.
    
    A simpler, more computationally tractable approach:
    - For each ready operation i, find the machine m_i that minimizes duration (least loaded / fastest).
    - Estimate the "load" added to machine m_i by this operation.
    - For each other ready operation j, estimate how much its slack is affected if it also wants m_i.
    - Compute the minimum slack across all ready operations under this hypothetical assignment.
    - Select the operation i that maximizes this minimum slack.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    
    if n == 1:
        return 0
    
    # Precompute for each operation:
    # - best machine (min duration)
    # - best duration
    # - second best machine and duration
    # - remaining_work (total remaining effort)
    best_machines = []
    best_durations = []
    second_best_machines = []
    second_best_durations = []
    remaining_works = []
    remaining_ops_counts = []
    
    for op in ready_operations:
        mo = op["machine_options"]
        machines_sorted = sorted(mo.keys(), key=lambda m: mo[m])
        best_m = machines_sorted[0]
        best_machines.append(best_m)
        best_durations.append(mo[best_m])
        remaining_works.append(op["remaining_work"])
        remaining_ops_counts.append(op["remaining_ops"])
        
        if len(machines_sorted) > 1:
            second_best_machines.append(machines_sorted[1])
            second_best_durations.append(mo[machines_sorted[1]])
        else:
            second_best_machines.append(None)
            second_best_durations.append(None)
    
    best_score = -float('inf')
    best_index = 0
    
    for i in range(n):
        m_i = best_machines[i]
        d_i = best_durations[i]
        
        min_slack = float('inf')
        for j in range(n):
            if j == i:
                # Operation i is being started. Its remaining work decreases by d_i.
                # We use total remaining work as the slack metric (proxy for critical path length).
                slack_j = remaining_works[i] - d_i
            else:
                # Operation j is waiting.
                # Base slack is total remaining work.
                base_slack = remaining_works[j]
                
                # Calculate penalty due to machine m_i being occupied by i for d_i time.
                penalty = 0.0
                if best_machines[j] == m_i:
                    # j's preferred machine is m_i.
                    # If j has a second option, it might switch.
                    # Cost of switching = d_2nd - d_1st.
                    # If d_i (wait time) > cost of switching, j switches.
                    # If d_i < cost of switching, j waits d_i.
                    # Effective delay = min(d_i, switch_cost).
                    if second_best_durations[j] is not None:
                        switch_cost = second_best_durations[j] - best_durations[j]
                        # If switch cost is negative (shouldn't happen with sorted), it's 0.
                        if switch_cost < 0:
                            switch_cost = 0
                        penalty = min(d_i, switch_cost)
                    else:
                        # Only one machine, must wait full d_i
                        penalty = d_i
                elif second_best_machines[j] == m_i:
                    # j's second choice is m_i.
                    # j prefers its best machine. If best is free, no penalty.
                    # If best is busy (not modeled), penalty would occur.
                    # We assume best is free for now, so penalty is 0.
                    # However, if best machine is *also* m_i, it's handled above.
                    pass
                
                slack_j = base_slack - penalty
            
            if slack_j < min_slack:
                min_slack = slack_j
        
        if min_slack > best_score:
            best_score = min_slack
            best_index = i
    
    return int(best_index)
# EVOLVE-BLOCK-END
