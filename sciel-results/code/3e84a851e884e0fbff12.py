"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Global state to track machine availability if needed, but we rely on local simulation
# We need to be careful not to mutate the passed-in state.

def _estimate_makespan(ready_operations, time, chosen_index, chosen_machine=None):
    """
    Placeholder for structural change.
    The new strategy uses Regret-based scoring instead of simple simulation or static rules.
    """
    pass

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Strategy: Regret-based Dispatching
    # For each ready operation, we calculate a "Regret" score.
    # Regret = (Score of 2nd best machine) - (Score of best machine)
    # Higher regret means the choice of machine is more critical, so we prioritize it.
    # If regret is tied, we prioritize by urgency (remaining work) and efficiency (duration).
    
    # First, infer machine load.
    # We don't have explicit machine busy times, but we can estimate "congestion"
    # by counting how many ready operations are targeting each machine.
    # A machine with more ready ops is likely more congested (or will be soon).
    
    machine_load = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            if m not in machine_load:
                machine_load[m] = 0
            machine_load[m] += 1
    
    # Calculate scores for each operation
    # Score for a specific (op, machine) pair:
    # We want to minimize the time this op takes to complete, considering:
    # 1. Duration on that machine
    # 2. Congestion on that machine (more ops = more waiting)
    # 3. Urgency of the job (more remaining work = more urgent)
    
    # Let's define a priority function for assigning an op to a machine.
    # Lower is better.
    # priority = duration * (1 + alpha * congestion) + beta * (total_remaining_work - op_remaining_work)
    # Actually, remaining_work is per job. Higher remaining work means more urgent.
    # So we want to prioritize jobs with high remaining work.
    # Let's use: score = duration * congestion_factor - urgency_weight * remaining_work
    
    # To make it comparable, let's normalize or just use raw values with tuned weights.
    # Since we don't know the scale, let's use a multiplicative factor for congestion.
    
    # For each op, find the best and second best machine based on a local cost function.
    # Cost(m) = duration[m] * (1 + 0.5 * machine_load[m])
    # We subtract a urgency term? No, let's keep the machine assignment cost separate from job priority.
    # The Regret is about the *machine choice* cost difference.
    # Then we combine Regret with Job Urgency to get the final dispatch score.
    
    best_idx = -1
    best_regret = -1
    best_urgency = 0
    best_min_cost = float('inf')
    
    for i, op in enumerate(ready_operations):
        options = op["machine_options"]
        if not options:
            continue
            
        # Calculate cost for each machine
        costs = []
        for m, dur in options.items():
            # Congestion factor: more ready ops on this machine increases effective cost
            load = machine_load.get(m, 0)
            # Simple model: cost = duration * (1 + load * 0.2)
            # This penalizes busy machines.
            cost = dur * (1 + load * 0.2)
            costs.append(cost)
        
        if not costs:
            continue
            
        # Sort costs to find best and second best
        sorted_costs = sorted(costs)
        min_cost = sorted_costs[0]
        
        # Calculate Regret
        # If only one machine, regret is 0 (no choice)
        if len(sorted_costs) > 1:
            second_best_cost = sorted_costs[1]
            regret = second_best_cost - min_cost
        else:
            regret = 0
            
        # Urgency: remaining_work
        # Higher remaining work means the job has more to do, so it's more critical.
        urgency = op["remaining_work"]
        
        # Combine Regret and Urgency
        # We want to pick the op with highest Regret.
        # If Regret is similar, pick the one with higher Urgency.
        # If both are similar, pick the one with lower Min Cost (efficiency).
        
        # Let's create a composite score.
        # Primary: Regret
        # Secondary: Urgency
        # Tertiary: Min Cost (lower is better)
        
        # To compare, we can use a tuple or a weighted sum.
        # Weighted sum is sensitive to scales. Let's use lexicographic order via sorting key.
        # We want to MAXIMIZE regret, then MAXIMIZE urgency, then MINIMIZE min_cost.
        
        # Current best comparison:
        # 1. If regret > best_regret, update.
        # 2. If regret == best_regret:
        #    a. If urgency > best_urgency, update.
        #    b. If urgency == best_urgency:
        #       i. If min_cost < best_min_cost, update.
        
        if best_idx == -1:
            best_idx = i
            best_regret = regret
            best_urgency = urgency
            best_min_cost = min_cost
        else:
            if regret > best_regret:
                best_idx = i
                best_regret = regret
                best_urgency = urgency
                best_min_cost = min_cost
            elif regret == best_regret:
                if urgency > best_urgency:
                    best_idx = i
                    best_regret = regret
                    best_urgency = urgency
                    best_min_cost = min_cost
                elif urgency == best_urgency:
                    if min_cost < best_min_cost:
                        best_idx = i
                        best_regret = regret
                        best_urgency = urgency
                        best_min_cost = min_cost
                        
    if best_idx == -1:
        return 0
        
    return int(best_idx)

# EVOLVE-BLOCK-END
