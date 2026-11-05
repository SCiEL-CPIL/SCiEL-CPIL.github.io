"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(instance):
    """
    Compute priority scores using a dynamic resource-aware heuristic.
    Primary: Total resource demand (higher demand = higher priority to avoid idle resources)
    Secondary: Number of successors (higher = more downstream impact)
    Tertiary: Duration (longer first to reduce tail)
    """
    start_time = time.time()
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance.get("predecessors", [[] for _ in range(n)])
    
    # Extract resource data
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    # Fallback if no resource data
    if demands is None or capacities is None:
        return durations.copy()
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    
    # Calculate total resource demand per activity (sum across all resources)
    # This acts as a proxy for "resource pressure"
    if demands.ndim == 2:
        total_demand = np.sum(demands, axis=1)
    else:
        total_demand = demands
    
    # Normalize total demand to [0, 1] to combine with other factors
    max_demand = np.max(total_demand) if np.max(total_demand) > 0 else 1.0
    normalized_demand = total_demand / max_demand
    
    # Build successor list to count successors
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Count successors for each activity
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
    max_succ = np.max(succ_count) if np.max(succ_count) > 0 else 1.0
    normalized_succ = succ_count / max_succ
    
    # Normalize duration
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    normalized_dur = durations / max_dur
    
    # Combine: 70% resource demand, 20% successor count, 10% duration
    # This prioritizes activities that consume scarce resources and unblock many tasks
    priority_scores = 0.7 * normalized_demand + 0.2 * normalized_succ + 0.1 * normalized_dur
    
    # Ensure we return a numpy array of floats
    return priority_scores
# EVOLVE-BLOCK-END
