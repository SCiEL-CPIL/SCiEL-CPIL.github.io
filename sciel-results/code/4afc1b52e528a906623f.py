"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque


def priority(instance):
    """
    Score each assembly task for a greedy line-balancing heuristic.
    
    The heuristic used by the evaluator:
      - Stations are filled with the highest-priority eligible task that fits the cycle time.
      - Eligible = all predecessors already assigned to previous stations (or same station if allowed, but typically previous).
      - We want to minimize the number of stations.
    
    Strategy for priority:
      1. Compute the longest path from each task to the end (successor depth / "rank").
         Tasks with more successors (longer remaining chain) should be scheduled early
         to avoid bottlenecks later.
      2. Also consider the task duration: longer tasks are harder to place, so prioritize them.
      3. A common effective heuristic is: priority = longest path to end (successor rank),
         with ties broken by duration (longer first).
      
    Let's compute:
      - succ_rank[i] = longest path length (in terms of number of tasks or total time?) from i to the end.
        Using total time along the longest path to the end gives a "remaining work" measure.
      
    Actually, a well-known heuristic for assembly line balancing is the "largest number of immediate successors" or "largest number of total successors" (positional weight).
    Another is the "largest processing time".
    
    Let's combine: 
      - Compute the longest path (in time) from each node to the end (i.e., the critical path remaining after this task).
      - priority[i] = longest_path_to_end[i] + times[i]  (total critical path through this task)
      Or simply: priority[i] = longest_path_to_end[i]
    
    Let me think about what helps. If a task has a long chain of successors, it must be placed early enough so that its successors can also fit. So higher priority for tasks with longer remaining chains.
    
    I'll compute:
      1. Topological order.
      2. For each task, compute the longest path (sum of times) from that task to any end task (including itself).
      3. Use that as the primary priority.
      4. Break ties by duration (longer duration first) to handle equal-length chains.
    
    This should be better than just duration.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    predecessors = instance["predecessors"]
    
    # Build adjacency: successors[i] = list of tasks that depend on i
    successors = defaultdict(list)
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1
    
    # Topological sort (Kahn's algorithm)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    temp_in_degree = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            temp_in_degree[v] -= 1
            if temp_in_degree[v] == 0:
                queue.append(v)
    
    # If topological sort didn't include all nodes (cycle), handle gracefully
    if len(topo_order) < n:
        # Fallback: just use times
        return times.copy()
    
    # Compute longest path to end (in time) for each node
    # longest_to_end[i] = times[i] + max(longest_to_end[j] for j in successors[i], default=0)
    longest_to_end = np.zeros(n, dtype=float)
    # Process in reverse topological order
    for i in reversed(topo_order):
        max_succ = 0.0
        for j in successors[i]:
            if longest_to_end[j] > max_succ:
                max_succ = longest_to_end[j]
        longest_to_end[i] = times[i] + max_succ
    
    # Priority: primarily longest_to_end, secondarily times (longer first)
    # To make it a single score, we can use:
    # priority[i] = longest_to_end[i] + times[i] * epsilon
    # But better to just use longest_to_end as the main key, and the evaluator likely uses
    # argmax or sorts by this value. Since we return a single score per task, we need to
    # combine them.
    
    # A common approach: priority = longest_to_end[i]
    # Ties are broken by the evaluator's sort (stable or not). To be safe, let's add a
    # small tie-breaker based on times.
    
    # Normalize times to a small range to use as tie-breaker
    max_time = np.max(times) if n > 0 else 1.0
    if max_time > 0:
        tie_breaker = times / max_time * 0.01
    else:
        tie_breaker = np.zeros(n, dtype=float)
    
    priority_scores = longest_to_end + tie_breaker
    
    return priority_scores
# EVOLVE-BLOCK-END
