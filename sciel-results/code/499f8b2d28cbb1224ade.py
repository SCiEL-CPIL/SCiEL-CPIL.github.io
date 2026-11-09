"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque

def priority(instance):
    """
    Score each assembly task for the greedy station-filling heuristic.
    Uses a combination of:
      1. Longest task time (base priority)
      2. Downstream critical path length (tasks that gate many successors)
      3. Number of successors (fan-out)
      4. Number of predecessors (fan-in) - tasks with many preds are 'harder' to place early
    
    The idea: prioritize tasks that are both long AND on critical paths,
    so the greedy filler reserves tight slots for them early.
    """
    deadline = time.time() + 145  # safety margin before 150s kill
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    
    # Build adjacency structures
    # predecessors[i] = list of tasks that must come before task i
    preds = instance.get("predecessors", [])
    # Ensure preds is a list of lists
    if isinstance(preds, (np.ndarray, list)):
        if len(preds) > 0 and isinstance(preds[0], (list, np.ndarray, tuple)):
            pred_lists = [list(p) if not isinstance(p, (int, float)) else [] for p in preds]
        else:
            # Maybe it's a flat structure? Assume it's per-task lists
            pred_lists = [list(p) for p in preds]
    else:
        pred_lists = [[] for _ in range(n)]
    
    # Build successors
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in pred_lists[i]:
            if p < n:
                succs[p].append(i)
    
    # Compute longest path from each task to any sink (downstream critical path)
    # Use topological order
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in pred_lists[i]:
            if p < n:
                in_degree[i] += 1
    
    # Kahn's algorithm for topological sort
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    temp_in_deg = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            temp_in_deg[v] -= 1
            if temp_in_deg[v] == 0:
                queue.append(v)
    
    # If not all tasks in topo order, there's a cycle; handle gracefully
    if len(topo_order) < n:
        # Fallback: just use task times
        return times
    
    # Compute downstream longest path (longest path from task to any terminal)
    # Process in reverse topological order
    down_path = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if not succs[u]:
            down_path[u] = times[u]
        else:
            max_succ = max(down_path[v] for v in succs[u])
            down_path[u] = times[u] + max_succ
    
    # Compute upstream longest path (longest path from any source to task)
    up_path = np.zeros(n, dtype=float)
    for u in topo_order:
        if not pred_lists[u]:
            up_path[u] = times[u]
        else:
            max_pred = max(up_path[p] for p in pred_lists[u])
            up_path[u] = times[u] + max_pred
    
    # Fan-out: number of direct successors
    fan_out = np.array([len(succs[i]) for i in range(n)], dtype=float)
    
    # Fan-in: number of direct predecessors
    fan_in = np.array([len(pred_lists[i]) for i in range(n)], dtype=float)
    
    # Normalize components to [0, 1]
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)
    
    norm_time = normalize(times)
    norm_down = normalize(down_path)
    norm_up = normalize(up_path)
    norm_fanout = normalize(fan_out)
    norm_fanin = normalize(fan_in)
    
    # Bottleneck density: how much of the cycle time is consumed by this task
    # relative to its downstream requirement
    bottleneck = np.ones(n, dtype=float)
    for i in range(n):
        if down_path[i] > 0:
            bottleneck[i] = times[i] / down_path[i]
    norm_bottleneck = normalize(bottleneck)
    
    # Criticality: tasks on the longest overall path
    overall_critical = up_path + down_path - times
    norm_critical = normalize(overall_critical)
    
    # Combine priorities
    # Weighted sum emphasizing long tasks on critical paths with high fan-out
    priority_scores = (
        1.0 * norm_time +
        0.8 * norm_down +
        0.6 * norm_up +
        0.5 * norm_critical +
        0.4 * norm_fanout +
        0.3 * norm_bottleneck +
        0.2 * norm_fanin
    )
    
    # Add a small bonus for tasks that are 'tight' in the cycle time
    # (i.e., their time is a significant fraction of cycle time)
    tightness = np.minimum(times / cycle_time, 1.0) if cycle_time > 0 else np.zeros(n)
    norm_tightness = normalize(tightness)
    priority_scores += 0.3 * norm_tightness
    
    # Ensure no negative values and proper scaling
    priority_scores = priority_scores - priority_scores.min()
    
    return priority_scores
# EVOLVE-BLOCK-END
