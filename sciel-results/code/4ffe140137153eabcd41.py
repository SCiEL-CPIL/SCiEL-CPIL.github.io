"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill


def priority(instance):
    """
    Compute priority scores for each activity in a resource-constrained project scheduling problem.
    Uses a combination of:
    1. Longest path to end (critical path based)
    2. Resource demand intensity
    3. Number of successors (bottleneck potential)
    4. Duration
    """
    if time.time() - _START_TIME > _DEADLINE:
        return np.asarray(instance["durations"], dtype=float)
    
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]
    
    n = len(durations)
    n_resources = len(capacities)
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Compute longest path from each activity to any sink (backward pass)
    # longest_to_end[i] = max over all paths from i to any sink of sum of durations
    longest_to_end = np.zeros(n)
    
    # Topological order (reverse)
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for pred in predecessors[i]:
            in_degree[i] += 1
    
    # Process in reverse topological order
    # Use a simple approach: repeatedly find nodes with no unprocessed successors
    processed = np.zeros(n, dtype=bool)
    for _ in range(n):
        progress = False
        for i in range(n):
            if processed[i]:
                continue
            # Check if all successors are processed
            all_succ_processed = all(processed[s] for s in successors[i])
            if all_succ_processed:
                # longest_to_end[i] = duration[i] + max(longest_to_end[s] for s in successors[i])
                if successors[i]:
                    longest_to_end[i] = durations[i] + max(longest_to_end[s] for s in successors[i])
                else:
                    longest_to_end[i] = durations[i]
                processed[i] = True
                progress = True
        if not progress:
            break
    
    # Compute number of successors (direct and indirect)
    num_successors = np.zeros(n, dtype=int)
    for _ in range(n):
        progress = False
        for i in range(n):
            if processed[i]:
                continue
            all_succ_processed = all(processed[s] for s in successors[i])
            if all_succ_processed:
                if successors[i]:
                    num_successors[i] = 1 + max(num_successors[s] for s in successors[i])
                else:
                    num_successors[i] = 0
                processed[i] = True
                progress = True
        if not progress:
            break
    
    # Resource demand intensity: sum of demand / capacity for each resource
    # Higher intensity means more constraining
    if n_resources > 0:
        demand_intensity = np.zeros(n)
        for r in range(n_resources):
            cap_r = capacities[r] if capacities[r] > 0 else 1.0
            demand_intensity += demands[:, r] / cap_r
    else:
        demand_intensity = np.zeros(n)
    
    # Combine scores:
    # Primary: longest path to end (critical path)
    # Secondary: resource demand intensity
    # Tertiary: number of successors
    # Quaternary: duration
    
    # Normalize each component to [0, 1]
    def normalize(arr):
        if arr.max() - arr.min() < 1e-10:
            return np.zeros_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())
    
    lp_norm = normalize(longest_to_end)
    di_norm = normalize(demand_intensity)
    ns_norm = normalize(num_successors.astype(float))
    dur_norm = normalize(durations)
    
    # Weighted combination
    # Emphasize critical path most, then resource constraints
    score = (
        1.0 * lp_norm +
        0.3 * di_norm +
        0.15 * ns_norm +
        0.1 * dur_norm
    )
    
    return score
# EVOLVE-BLOCK-END
