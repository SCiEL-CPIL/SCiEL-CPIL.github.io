"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    """
    Score activities for serial SGS: higher score = scheduled earlier.
    
    Heuristic: combine duration (critical path length), resource scarcity
    (demand/capacity ratio), and successor count (critical path position).
    
    This is a refinement of longest-duration-first that accounts for:
    1. Resource contention: activities using scarce resources should go first
    2. Critical path position: activities with more successors are more urgent
    """
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Get resource demands and capacities
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    # Compute resource scarcity score per activity
    # Higher = more resource-intensive relative to capacity
    if demands is not None and capacities is not None:
        demands_arr = np.asarray(demands, dtype=float)  # shape (n, num_resources)
        capacities_arr = np.asarray(capacities, dtype=float)  # shape (num_resources,)
        
        # For each activity, compute max demand/capacity ratio across resources
        # This captures which activity is most "resource-constrained"
        ratios = demands_arr / np.maximum(capacities_arr, 1e-9)  # (n, num_resources)
        scarcity = np.max(ratios, axis=1)  # (n,)
        
        # Normalize scarcity to [0, 1]
        scarcity_max = np.max(scarcity)
        if scarcity_max > 0:
            scarcity = scarcity / scarcity_max
    else:
        scarcity = np.zeros(n)
    
    # Compute successor count (how many activities depend on this one)
    # More successors = more critical path downstream
    if predecessors is not None:
        # Build successor counts
        successor_counts = np.zeros(n, dtype=float)
        for i in range(n):
            preds = predecessors[i]
            if preds is not None:
                for pred in preds:
                    successor_counts[pred] += 1
        
        # Normalize to [0, 1]
        succ_max = np.max(successor_counts)
        if succ_max > 0:
            successor_counts = successor_counts / succ_max
    else:
        successor_counts = np.zeros(n)
    
    # Combine: duration (primary) + scarcity (secondary) + successor count (tertiary)
    # Weights: duration should dominate, but resource contention and critical path
    # position can break ties and improve scheduling when resources are tight
    score = durations + 0.5 * scarcity + 0.3 * successor_counts
    
    return score
# EVOLVE-BLOCK-END
