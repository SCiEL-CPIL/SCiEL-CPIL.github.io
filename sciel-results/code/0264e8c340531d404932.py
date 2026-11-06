"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each task for the greedy line-balancing heuristic.
    
    The heuristic fills stations with the highest-priority eligible task that fits.
    We want to minimize the number of stations.
    
    Strategy:
    1. Compute the longest path from any source to each task (earliest start time).
    2. Compute the longest path from each task to any sink (latest finish time / slack).
    3. A good priority for minimizing stations is often 'critical path' based.
       However, the standard 'Longest Task First' is the seed.
       Let's try a combination: prioritize tasks that are on the critical path.
       Specifically, a task's criticality can be measured by the length of the
       longest path that includes it.
       
       Another strong heuristic is 'Minimum Slack' (tasks with less slack first),
       which is equivalent to prioritizing tasks on the critical path.
       
       Let's compute:
       - `rank[i]`: longest path from start to i (including i's time).
       - `slack[i]`: total time of longest path from i to end (including i's time) 
                    - but actually, we need to be careful.
       
       Let's use the following metric:
       `priority[i] = longest_path_from_start[i] + longest_path_to_end[i]`
       This is the length of the longest path that goes through task i.
       Tasks with higher values are more 'critical' and should be placed earlier
       to allow their successors to be placed in subsequent stations as early as possible.
       
       Wait, the greedy algorithm places tasks in stations. If we place a task with
       high 'longest path through it' first, we ensure that the critical chain is
       established early.
       
       Let's refine:
       `rank_start[i]`: longest path from any source to i (sum of times).
       `rank_end[i]`: longest path from i to any sink (sum of times).
       
       If we prioritize by `rank_start[i] + rank_end[i]`, we prioritize the most central tasks.
       
       Alternatively, a common heuristic is 'Most Successors' or 'Longest Processing Time'.
       The seed is 'Longest Processing Time'.
       
       Let's try 'Most Immediate Successors' + 'Longest Processing Time'.
       Or better: 'Minimum Slack'.
       Slack = (Cycle Time - rank_start[i] - rank_end[i])? No, that's not quite right for line balancing.
       
       Let's stick to a robust critical path metric.
       `criticality[i] = rank_start[i] + rank_end[i]`
       
       Let's implement this.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build adjacency list for successors
    successors = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i]
        if isinstance(preds, list):
            for p in preds:
                successors[p].append(i)
        elif isinstance(preds, (set, tuple)):
            for p in preds:
                successors[p].append(i)
        elif preds is not None and not isinstance(preds, (list, set, tuple)):
            # Maybe it's a single int?
            try:
                successors[int(preds)].append(i)
            except:
                pass

    # Compute rank_start (longest path from start to i)
    # Topological sort
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        preds = predecessors[i]
        if isinstance(preds, list):
            in_degree[i] = len(preds)
        elif isinstance(preds, (set, tuple)):
            in_degree[i] = len(preds)
        elif preds is not None and not isinstance(preds, (list, set, tuple)):
            in_degree[i] = 1

    from collections import deque
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    rank_start = np.zeros(n, dtype=float)
    processed = 0
    while queue:
        u = queue.popleft()
        processed += 1
        for v in successors[u]:
            if rank_start[u] + times[u] > rank_start[v]:
                rank_start[v] = rank_start[u] + times[u]
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If there's a cycle or some nodes not processed, this might fail.
    # But the problem implies a valid DAG.
    
    # Compute rank_end (longest path from i to end)
    # Reverse topological sort
    out_degree = np.zeros(n, dtype=int)
    for i in range(n):
        out_degree[i] = len(successors[i])
    
    queue = deque()
    for i in range(n):
        if out_degree[i] == 0:
            queue.append(i)
            
    rank_end = np.zeros(n, dtype=float)
    processed = 0
    while queue:
        u = queue.popleft()
        processed += 1
        # Find predecessors of u to update their rank_end
        # We need a reverse adjacency list or just iterate all nodes.
        # Since n=30, iterating all nodes is fine.
        for i in range(n):
            if u in successors[i]:
                if rank_end[u] + times[u] > rank_end[i]:
                    rank_end[i] = rank_end[u] + times[u]
                out_degree[i] -= 1
                if out_degree[i] == 0:
                    queue.append(i)

    # Priority: highest criticality first
    # Criticality = rank_start + rank_end
    # But wait, rank_start includes times[i]?
    # My rank_start definition: rank_start[v] = max(rank_start[u] + times[u]).
    # So rank_start[i] is the time accumulated BEFORE task i starts.
    # rank_end[i] is the time accumulated AFTER task i finishes?
    # My rank_end definition: rank_end[i] = max(rank_end[v] + times[v]).
    # So rank_end[i] is the time accumulated AFTER task i.
    
    # Total path length through i = rank_start[i] + times[i] + rank_end[i]
    
    criticality = rank_start + times + rank_end
    
    # We want to place high criticality tasks first.
    # So return criticality.
    
    return criticality

# EVOLVE-BLOCK-END
