"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Compute priority scores using a critical-path-aware, resource-sensitive heuristic.
    Primary: Critical path length (longest path from activity to project end)
    Secondary: Resource scarcity ratio (demand/capacity, weighted by criticality)
    Tertiary: Successor count (downstream impact)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance.get("predecessors", [[] for _ in range(n)])

    # Extract resource data
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)

    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)

    # Compute critical path: longest path from each node to the project end
    # Using topological order (reverse BFS from sinks)
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for pred in predecessors[i]:
            in_degree[i] += 1

    # Find nodes with no successors (sinks)
    sinks = [i for i in range(n) if len(successors[i]) == 0]

    # Longest path from each node to any sink
    # Process in reverse topological order
    # First, get topological order
    from collections import defaultdict
    adj = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            adj[pred].append(i)

    # Kahn's algorithm for topological sort
    topo_order = []
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    temp_in_degree = in_degree.copy()
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            temp_in_degree[succ] -= 1
            if temp_in_degree[succ] == 0:
                queue.append(succ)

    # If not all nodes processed (cycle), use fallback
    if len(topo_order) != n:
        topo_order = list(range(n))

    # Compute longest path to end (reverse topological order)
    longest_to_end = np.zeros(n, dtype=float)
    # Process in reverse topological order
    for node in reversed(topo_order):
        if len(successors[node]) == 0:
            longest_to_end[node] = durations[node]
        else:
            max_succ_path = 0.0
            for succ in successors[node]:
                max_succ_path = max(max_succ_path, longest_to_end[succ])
            longest_to_end[node] = durations[node] + max_succ_path

    # Normalize critical path
    max_cp = np.max(longest_to_end) if np.max(longest_to_end) > 0 else 1.0
    normalized_cp = longest_to_end / max_cp

    # Resource scarcity: demand/capacity ratio per resource, then aggregate
    if demands is not None and capacities is not None:
        demands_arr = np.asarray(demands, dtype=float)
        capacities_arr = np.asarray(capacities, dtype=float)

        if demands_arr.ndim == 2:
            # Compute per-resource scarcity: demand / capacity (higher = more scarce)
            # Avoid division by zero
            safe_capacities = np.where(capacities_arr > 0, capacities_arr, 1.0)
            scarcity = demands_arr / safe_capacities
            # Weight scarcity by critical path importance
            # Activities on the critical path should prioritize scarce resources more
            scarcity_weighted = scarcity * normalized_cp[:, np.newaxis]
            resource_score = np.sum(scarcity_weighted, axis=1)
        else:
            safe_capacities = np.where(capacities_arr > 0, capacities_arr, 1.0)
            scarcity = demands_arr / safe_capacities
            resource_score = scarcity * normalized_cp

        max_res = np.max(resource_score) if np.max(resource_score) > 0 else 1.0
        normalized_resource = resource_score / max_res
    else:
        normalized_resource = np.zeros(n, dtype=float)

    # Successor count (immediate + transitive via longest path already captures transitive)
    # Use immediate successor count as tie-breaker
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
    max_succ = np.max(succ_count) if np.max(succ_count) > 0 else 1.0
    normalized_succ = succ_count / max_succ

    # Duration normalization
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    normalized_dur = durations / max_dur

    # Combine: 50% critical path, 30% resource scarcity, 15% successors, 5% duration
    # Critical path is the most important for makespan reduction
    priority_scores = (
        0.50 * normalized_cp +
        0.30 * normalized_resource +
        0.15 * normalized_succ +
        0.05 * normalized_dur
    )

    # Check deadline
    if time.time() > deadline:
        # Return best available
        pass

    return priority_scores
# EVOLVE-BLOCK-END
