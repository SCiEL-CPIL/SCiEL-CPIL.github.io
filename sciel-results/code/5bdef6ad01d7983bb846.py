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
    # - remaining_work
    best_machines = []
    best_durations = []
    remaining_works = []
    all_machines = set()
    
    for op in ready_operations:
        mo = op["machine_options"]
        best_m = min(mo.keys(), key=lambda m: mo[m])
        best_machines.append(best_m)
        best_durations.append(mo[best_m])
        remaining_works.append(op["remaining_work"])
        all_machines.update(mo.keys())
    
    # For each candidate operation i, simulate assigning it to its best machine
    # and compute the minimum slack across all ready operations.
    
    # Slack estimation:
    # For operation j, if we haven't started it yet, its slack is roughly:
    #   remaining_work[j] - (estimated additional delay due to machine contention)
    #
    # If operation i is assigned to machine m_i, then any other operation j that also prefers m_i
    # might face a delay of best_durations[i] (since it would have to wait for m_i to be free).
    #
    # So for operation j:
    #   if best_machines[j] == m_i:
    #       effective_remaining = remaining_works[j] + best_durations[i]
    #   else:
    #       effective_remaining = remaining_works[j]
    #
    # Slack for j = remaining_works[j] - (some baseline) ... this isn't quite right.
    #
    # Let's think differently. The "slack" of an operation is how much time it has before it
    # becomes a bottleneck. A common proxy is:
    #   slack_j = remaining_works[j] - best_durations[j]  # how much work remains beyond the first op
    #
    # But that doesn't capture machine contention.
    #
    # Alternative approach: For each candidate i, compute a "penalty" for each other operation j
    # based on machine conflict, and define the minimum slack as the minimum over j of:
    #   remaining_works[j] - penalty_j
    # where penalty_j = best_durations[i] if best_machines[j] == best_machines[i] else 0
    #
    # We want to maximize the minimum slack.
    
    best_score = -float('inf')
    best_index = 0
    
    for i in range(n):
        m_i = best_machines[i]
        d_i = best_durations[i]
        
        min_slack = float('inf')
        for j in range(n):
            if j == i:
                # Operation i is being started now, so its remaining work after this op is:
                # remaining_works[i] - d_i
                slack_j = remaining_works[i] - d_i
            else:
                # Operation j is not started yet. If it wants the same machine as i,
                # it may have to wait d_i time units.
                if best_machines[j] == m_i:
                    # Effective remaining work increases by d_i due to waiting
                    slack_j = remaining_works[j] - d_i
                else:
                    slack_j = remaining_works[j]
            
            if slack_j < min_slack:
                min_slack = slack_j
        
        if min_slack > best_score:
            best_score = min_slack
            best_index = i
    
    return int(best_index)
# EVOLVE-BLOCK-END
