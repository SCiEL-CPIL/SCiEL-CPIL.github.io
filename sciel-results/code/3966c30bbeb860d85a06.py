"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def _build_graph(instance):
    """
    Builds the successor adjacency list and in-degrees from the instance.
    Returns:
        successors: list of lists, where successors[i] contains tasks that depend on i.
        in_degree: list of ints, where in_degree[i] is the number of predecessors of i.
        n: number of tasks.
    """
    times = instance["times"]
    predecessors = instance["predecessors"]
    n = len(times)
    
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    return successors, in_degree, n


def _topological_order(n, in_degree, successors):
    """
    Computes topological order using Kahn's algorithm.
    Returns a list of nodes in topological order.
    """
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
            
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
                
    return order


def _compute_metrics(instance):
    """
    Computes time, longest downstream path, and transitive successor count for each task.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    successors, in_degree, _ = _build_graph(instance)
    
    # We need a fresh copy of in_degree for topological sort if we modify it, 
    # but _topological_order modifies the passed list. 
    # Let's pass a copy.
    in_degree_copy = in_degree[:]
    topo_order = _topological_order(n, in_degree_copy, successors)
    
    # 1. Longest downstream path (including own time)
    # Process in reverse topological order
    longest_down = np.zeros(n)
    for u in reversed(topo_order):
        max_path = 0.0
        for v in successors[u]:
            # Path from u to sink via v is time[u] + (path from v to sink)
            # But we define longest_down[v] as the longest path starting at v.
            # So longest_down[u] = time[u] + max(longest_down[v])
            if longest_down[v] > max_path:
                max_path = longest_down[v]
        longest_down[u] = times[u] + max_path
        
    # 2. Transitive successor count (number of descendants)
    # Process in reverse topological order
    # count_desc[u] = sum(1 + count_desc[v]) for all v in successors[u]
    count_desc = np.zeros(n)
    for u in reversed(topo_order):
        total = 0
        for v in successors[u]:
            total += 1 + count_desc[v]
        count_desc[u] = total
        
    return times, longest_down, count_desc


def priority(instance):
    """
    Score each assembly task for priority-based station assignment.
    Higher score = higher priority.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    try:
        times, longest_down, count_desc = _compute_metrics(instance)
        n = len(times)
        
        # Normalize components to [0, 1]
        max_time = np.max(times) if np.max(times) > 0 else 1.0
        max_ld = np.max(longest_down) if np.max(longest_down) > 0 else 1.0
        max_cd = np.max(count_desc) if np.max(count_desc) > 0 else 1.0
        
        norm_time = times / max_time
        norm_ld = longest_down / max_ld
        norm_cd = count_desc / max_cd
        
        # Combine heuristics
        # Insight: Transitive successors (bottleneck) is very important.
        # Longest downstream path is also a strong critical path indicator.
        # Time is important for packing efficiency.
        
        # Weights:
        # Current program used: 0.5 * time + 0.3 * ld + 0.2 * direct_succ
        # New program uses: transitive_succ instead of direct_succ, and we weight it higher.
        # Let's try: 0.4 * time + 0.3 * ld + 0.3 * transitive_succ
        # Or even more emphasis on structure: 0.3 * time + 0.4 * ld + 0.3 * transitive_succ
        
        # Let's stick to a balanced approach that highlights the structural bottleneck.
        w_time = 0.4
        w_ld = 0.3
        w_cd = 0.3
        
        priority_scores = w_time * norm_time + w_ld * norm_ld + w_cd * norm_cd
        
        # Ensure positive scores for heap if needed, though max() works with any comparable
        # Adding a small epsilon to avoid ties? No, ties are broken arbitrarily by max().
        # But to be safe and distinct, we can add a tiny bit of noise based on index? 
        # No, deterministic is better. The evaluator handles ties.
        
        return priority_scores
        
    except Exception:
        # Fallback to simple time-based priority if something goes wrong
        return np.asarray(instance["times"], dtype=float)

# EVOLVE-BLOCK-END
