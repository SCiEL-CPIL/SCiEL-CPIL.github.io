"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    """
    Compute priority scores for activities using a composite heuristic that considers:
    1. Longest duration (critical path tendency)
    2. Number of successors (topological urgency)
    3. Resource demand relative to capacity (bottleneck penalty)
    
    The idea is to rank activities that are on the critical path, have many successors,
    and consume scarce resources with higher priority.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # shape: (n_activities, n_resources)
    capacities = np.asarray(instance["capacities"], dtype=float)  # shape: (n_resources,)
    predecessors = instance["predecessors"]  # list of lists, predecessors[i] = list of pred indices
    
    n = len(durations)
    n_res = len(capacities)
    
    # Build successor counts
    succ_count = np.zeros(n, dtype=int)
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        for p in preds:
            succ_count[p] += 1
    
    # Normalize durations
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    norm_dur = durations / max_dur
    
    # Normalize successor counts
    max_succ = np.max(succ_count) if np.max(succ_count) > 0 else 1.0
    norm_succ = succ_count.astype(float) / max_succ
    
    # Resource scarcity score: for each activity, compute how much it strains each resource
    # demand[i, r] / capacity[r] gives utilization fraction
    if n_res > 0:
        # Avoid division by zero
        safe_cap = np.where(capacities > 0, capacities, 1.0)
        utilization = demands / safe_cap  # shape: (n, n_res)
        # Take the max utilization across resources as a bottleneck indicator
        max_util = np.max(utilization, axis=1)  # shape: (n,)
        # Normalize
        max_mu = np.max(max_util) if np.max(max_util) > 0 else 1.0
        norm_util = max_util / max_mu
    else:
        norm_util = np.zeros(n, dtype=float)
    
    # Composite score: higher is more urgent
    # We want activities with long durations, many successors, and high resource strain to be prioritized
    # But we also want to avoid over-prioritizing activities that will block others due to resource scarcity
    # So we use a weighted combination
    w_dur = 1.0
    w_succ = 0.8
    w_util = 0.5
    
    score = w_dur * norm_dur + w_succ * norm_succ + w_util * norm_util
    
    # Return as a numpy array of scores (higher = higher priority)
    return score
# EVOLVE-BLOCK-END
