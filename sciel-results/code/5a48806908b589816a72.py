import numpy as np
import time

def priority(instance):
    """
    Computes priority scores for assembly line balancing tasks.
    Uses a combination of Longest Path to End (Rank) and Successor Count.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    times = np.asarray(instance["times"], dtype=float)
    n_tasks = len(times)
    
    # Parse predecessors
    # instance["predecessors"] is usually a list of lists or a dict
    # Standard format in these tasks: list of lists of predecessor indices
    pred_list = instance.get("predecessors", [])
    
    # Build successor list
    successors = [[] for _ in range(n_tasks)]
    in_degree = [0] * n_tasks
    
    # If pred_list is a list of lists
    if isinstance(pred_list, list):
        for i in range(n_tasks):
            if i < len(pred_list):
                preds = pred_list[i]
                if isinstance(preds, (list, tuple, np.ndarray)):
                    for p in preds:
                        if 0 <= p < n_tasks:
                            successors[p].append(i)
                            in_degree[i] += 1
                elif isinstance(preds, int):
                    if 0 <= preds < n_tasks:
                        successors[preds].append(i)
                        in_degree[i] += 1
    elif isinstance(pred_list, dict):
        for i, preds in pred_list.items():
            i = int(i)
            if isinstance(preds, (list, tuple, np.ndarray)):
                for p in preds:
                    p = int(p)
                    if 0 <= p < n_tasks:
                        successors[p].append(i)
                        in_degree[i] += 1
            elif isinstance(preds, int):
                if 0 <= preds < n_tasks:
                    successors[preds].append(i)
                    in_degree[i] += 1

    # Calculate Longest Path to End (Rank)
    # We need the longest path from each node to any terminal node (sink).
    # This can be computed via reverse topological sort or simply by dynamic programming 
    # if we process nodes in reverse topological order.
    
    # First, get topological order
    # For small N, we can just use a simple DFS or Kahn's algorithm.
    # Let's use a simple memoized DFS for longest path to end.
    
    import sys
    sys.setrecursionlimit(10000)
    
    rank = [-1.0] * n_tasks
    visited = [False] * n_tasks
    
    def compute_rank(i):
        if visited[i]:
            return rank[i]
        if time.time() > DEADLINE:
            return times[i] # Fallback to just time if deadline approached
        visited[i] = True
        max_succ_rank = 0.0
        for s in successors[i]:
            # Recursively get rank of successor
            # Note: We are calculating path length INCLUDING current node's time
            # So rank[i] = times[i] + max(rank[s] for s in successors[i])
            # Wait, if rank[s] includes times[s], then rank[i] = times[i] + max(rank[s]) is correct.
            s_rank = compute_rank(s)
            if s_rank > max_succ_rank:
                max_succ_rank = s_rank
        rank[i] = times[i] + max_succ_rank
        return rank[i]

    for i in range(n_tasks):
        compute_rank(i)
        
    if time.time() > DEADLINE:
        return np.asarray(rank, dtype=float)

    # Calculate Successor Counts
    succ_counts = [len(succs) for succs in successors]
    
    # Combine Metrics
    # Priority = Rank + small weight * Successor Count
    # This prioritizes tasks on the critical path, with a slight bias towards tasks that unlock many others.
    weights = np.ones(n_tasks)
    # Optional: Normalize or just use raw values. Raw values are fine for comparison.
    priorities = np.array(rank, dtype=float) + 0.1 * np.array(succ_counts, dtype=float)
    
    return priorities
