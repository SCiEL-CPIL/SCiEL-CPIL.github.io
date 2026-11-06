"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each assembly task for the greedy priority-based station filling.
    We want to minimize the number of stations, so we want to pack tasks tightly.
    
    Strategy:
    1. Compute the "critical path" length for each task (longest path from task to any successor, including itself).
       This is the "bottom-level" in assembly line balancing literature.
    2. Combine with the task's own time and potentially the number of successors.
    
    The classic heuristic for ALB (Assembly Line Balancing) is to use the bottom-level (or rank) which is the
    longest path from the task to the end of the line. Tasks with higher bottom-level should be scheduled first
    because they are on the critical path and delaying them delays the whole line.
    
    However, we also need to consider that tasks with many successors might be more "important" to schedule early
    to allow their successors to be scheduled.
    
    Let's compute:
    - time[i]: the processing time of task i
    - succ_count[i]: number of immediate successors
    - bottom_level[i]: the longest path length from task i to any terminal task (including i's own time)
    
    A good priority is often: bottom_level[i] + time[i] or just bottom_level[i].
    Since bottom_level already includes time[i], let's use bottom_level.
    
    But wait, the greedy algorithm picks the HIGHEST priority eligible task.
    If we use bottom_level, tasks on the critical path get high priority.
    
    Let's also consider the "rank" which is sometimes defined as the sum of times of all successors plus own time.
    Actually, bottom_level is the max, not the sum.
    
    Let's try a combination:
    priority[i] = bottom_level[i] * 1000 + time[i]
    Or simply bottom_level[i].
    
    Let's compute bottom_level properly.
    bottom_level[i] = time[i] + max(bottom_level[j] for j in successors of i)
    If no successors, bottom_level[i] = time[i].
    
    This is a standard heuristic.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Compute bottom_level using topological order (reverse)
    # We can use a simple iterative approach or DFS.
    # Since n is small (30), we can do a simple loop until convergence or use DFS.
    
    # Let's use a DFS-based approach with memoization to avoid recursion depth issues (though 30 is small).
    # Or simply iterative relaxation.
    
    bottom_level = np.zeros(n)
    
    # Topological sort to process in reverse order
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for pred in predecessors[i]:
            in_degree[i] += 1
    
    # Actually, predecessors[i] are the tasks that must come before i.
    # So edges go from pred -> i.
    # In-degree of i is the number of predecessors.
    
    # Let's do a Kahn's algorithm to get topological order.
    from collections import deque
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Process in reverse topological order
    # bottom_level[i] = time[i] + max(bottom_level[j] for j in successors[i])
    for i in reversed(topo_order):
        if successors[i]:
            max_succ = max(bottom_level[j] for j in successors[i])
            bottom_level[i] = times[i] + max_succ
        else:
            bottom_level[i] = times[i]
    
    # The priority should be high for tasks that should be scheduled first.
    # In ALB, we typically schedule tasks with the highest bottom-level first.
    return bottom_level
# EVOLVE-BLOCK-END
