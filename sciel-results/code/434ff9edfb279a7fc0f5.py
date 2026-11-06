"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from collections import deque

def _build_graph(instance):
    """Build predecessor and successor lists from the instance."""
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    preds = instance.get("predecessors", None)
    
    if preds is None:
        return times, n, [[] for _ in range(n)], [[] for _ in range(n)]
    
    succs = [[] for _ in range(n)]
    pred_lists = [[] for _ in range(n)]
    
    # Handle both dict and list formats for predecessors
    if isinstance(preds, dict):
        for i in range(n):
            if i in preds:
                for p in preds[i]:
                    pred_lists[i].append(p)
                    succs[p].append(i)
    else:
        # Assume preds is a list of lists
        for i in range(n):
            if preds[i]:
                for p in preds[i]:
                    pred_lists[i].append(p)
                    succs[p].append(i)
    
    return times, n, pred_lists, succs

def _topological_order(n, pred_lists, succs):
    """Return topological order using Kahn's algorithm."""
    in_degree = [len(pred_lists[i]) for i in range(n)]
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in succs[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    return topo_order

def _compute_priorities(instance):
    """
    Compute priority scores using downstream slack deficit and related metrics.
    The core idea: tasks with less temporal buffer before violating station limits
    should be prioritized. This is captured by 'downstream slack deficit'.
    """
    times, n, pred_lists, succs = _build_graph(instance)
    cycle_time = instance["cycle_time"]
    
    # Get topological order
    topo_order = _topological_order(n, pred_lists, succs)
    
    # If cycle detected or incomplete topo order, fall back to duration
    if len(topo_order) < n:
        return times.copy()
    
    # 1. Longest path down (from each node to any sink, including own time)
    longest_down = np.zeros(n)
    for node in reversed(topo_order):
        max_succ = 0.0
        for s in succs[node]:
            if longest_down[s] > max_succ:
                max_succ = longest_down[s]
        longest_down[node] = times[node] + max_succ
    
    # 2. Longest path up (from any source to node, including own time)
    longest_up = np.zeros(n)
    for node in topo_order:
        max_pred = 0.0
        for p in pred_lists[node]:
            if longest_up[p] > max_pred:
                max_pred = longest_up[p]
        longest_up[node] = times[node] + max_pred
    
    # 3. Downstream slack deficit: how much "buffer" is left after this task's downstream work
    #    A task with long downstream path has little slack -> higher priority
    #    slack_deficit[i] = max(0, longest_down[i] - (cycle_time - times[i]))
    #    But actually, we want to prioritize tasks that are "critical" in terms of
    #    consuming the cycle time budget. Let's define:
    #    downstream_pressure = longest_down[i] / cycle_time
    #    Higher pressure = less slack = higher priority
    
    # 4. Total transitive successor time (sum of times of all transitive successors)
    total_succ_time = np.zeros(n)
    for node in reversed(topo_order):
        total_succ_time[node] = times[node] + sum(total_succ_time[s] for s in succs[node])
    
    # 5. Number of direct successors
    num_succs = np.array([len(succs[i]) for i in range(n)], dtype=float)
    
    # 6. Number of direct predecessors
    num_preds = np.array([len(pred_lists[i]) for i in range(n)], dtype=float)
    
    # Normalize features
    max_down = np.max(longest_down) if n > 0 and np.max(longest_down) > 0 else 1.0
    max_up = np.max(longest_up) if n > 0 and np.max(longest_up) > 0 else 1.0
    max_time = np.max(times) if n > 0 and np.max(times) > 0 else 1.0
    max_succ_time = np.max(total_succ_time) if n > 0 and np.max(total_succ_time) > 0 else 1.0
    
    # Downstream slack deficit: tasks with long downstream paths have less slack
    # We normalize by cycle_time to make it relative
    downstream_pressure = longest_down / (cycle_time + 1e-9)
    max_pressure = np.max(downstream_pressure) if n > 0 and np.max(downstream_pressure) > 0 else 1.0
    norm_pressure = downstream_pressure / max_pressure
    
    # Combined priority:
    # - downstream_pressure: most important, captures the "slack deficit" idea
    # - longest_up: tasks early in the chain
    # - total_succ_time: tasks that block a lot of work
    # - time: longer tasks
    # - num_succs: bottleneck tasks with many successors
    
    priority_scores = (
        0.35 * norm_pressure +
        0.25 * (longest_down / max_down) +
        0.15 * (total_succ_time / max_succ_time) +
        0.15 * (times / max_time) +
        0.10 * (num_succs / max(n, 1))
    )
    
    # Add small deterministic tie-breaker
    priority_scores = priority_scores + np.arange(n) * 1e-10
    
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
    
    # Deadline guard
    if time.time() - start_time > deadline:
        try:
            scores = np.asarray(instance["times"], dtype=float)
        except Exception:
            scores = np.zeros(1)
    
    return scores
# EVOLVE-BLOCK-END
