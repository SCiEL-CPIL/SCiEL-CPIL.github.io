"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _compute_priorities(instance):
    """
    Compute a priority score for each task using a combination of:
    1. Longest path down (successor depth)
    2. Longest path up (predecessor depth)
    3. Task duration
    4. Number of successors (to break ties and favor tasks with more downstream work)
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    
    # Build predecessor and successor lists
    preds = instance.get("predecessors", None)
    
    if preds is None:
        # No predecessors info, just use duration
        return times.copy()
    
    # Build adjacency lists
    # preds[i] is a list of indices of predecessors of task i
    succs = [[] for _ in range(n)]
    pred_lists = [[] for _ in range(n)]
    
    for i in range(n):
        if i in preds:
            for p in preds[i]:
                pred_lists[i].append(p)
                succs[p].append(i)
        else:
            # Task i has no predecessors
            pass
    
    # Compute longest path down (from each node to any sink)
    # Using topological order
    # First, compute in-degrees
    in_degree = [0] * n
    for i in range(n):
        for p in pred_lists[i]:
            in_degree[i] += 1
    
    # Topological sort (Kahn's algorithm)
    from collections import deque
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in succs[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # If topo_order doesn't contain all nodes, there's a cycle; handle gracefully
    if len(topo_order) < n:
        # Cycle detected, fall back to duration-based
        return times.copy()
    
    # Longest path down: max sum of times from node to any sink
    # Process in reverse topological order
    longest_down = np.zeros(n)
    for node in reversed(topo_order):
        max_succ = 0
        for s in succs[node]:
            if longest_down[s] > max_succ:
                max_succ = longest_down[s]
        longest_down[node] = times[node] + max_succ
    
    # Longest path up: max sum of times from any source to node
    # Process in topological order
    longest_up = np.zeros(n)
    for node in topo_order:
        max_pred = 0
        for p in pred_lists[node]:
            if longest_up[p] > max_pred:
                max_pred = longest_up[p]
        longest_up[node] = times[node] + max_pred
    
    # Number of successors (direct)
    num_succs = np.array([len(s) for s in succs], dtype=float)
    
    # Priority: weighted combination
    # Emphasize longest path down (critical path length from this task)
    # Add a small bonus for tasks with many successors to encourage parallelism
    # Normalize to avoid scale issues
    max_down = np.max(longest_down) if n > 0 else 1
    max_up = np.max(longest_up) if n > 0 else 1
    max_time = np.max(times) if n > 0 else 1
    
    # Combine: primarily longest_down, then longest_up, then time
    priority_scores = (
        1.0 * longest_down / (max_down + 1e-9) +
        0.5 * longest_up / (max_up + 1e-9) +
        0.3 * times / (max_time + 1e-9)
    )
    
    # Add small tie-breaker based on number of successors
    priority_scores += 0.01 * num_succs / (max(n, 1))
    
    return priority_scores


def priority(instance):
    """
    Returns a priority score for each task.
    Higher score = higher priority.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    try:
        scores = _compute_priorities(instance)
    except Exception as e:
        # Fallback to duration-based
        try:
            scores = np.asarray(instance["times"], dtype=float)
        except Exception:
            scores = np.zeros(1)
    
    return scores
# EVOLVE-BLOCK-END
