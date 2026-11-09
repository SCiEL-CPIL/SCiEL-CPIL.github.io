"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_successors_and_depths(times, predecessors, cycle_time):
    """
    Compute for each task:
      - successors: list of tasks that depend on it
      - longest_path_down: the longest path (in time) from this task to any sink
        (i.e., the minimum number of stations needed if we could pack perfectly,
        approximated by longest path in time / cycle_time, but we use raw time sum
        along the longest path)
      - total_downstream_time: sum of times of all successors (including self)
    """
    n = len(times)
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Topological order (Kahn's algorithm)
    in_degree = [len(predecessors[i]) for i in range(n)]
    from collections import deque
    q = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    
    # If not all nodes are in topo, there's a cycle; handle gracefully
    if len(topo) < n:
        # Fallback: just return times as priority
        return np.asarray(times, dtype=float), successors, np.zeros(n), np.zeros(n)
    
    # Process in reverse topological order to compute longest path down and total downstream time
    longest_path_down = np.zeros(n, dtype=float)
    total_downstream_time = np.zeros(n, dtype=float)
    
    for u in reversed(topo):
        t_u = times[u]
        if not successors[u]:
            longest_path_down[u] = t_u
            total_downstream_time[u] = t_u
        else:
            max_succ = max(longest_path_down[v] for v in successors[u])
            longest_path_down[u] = t_u + max_succ
            total_downstream_time[u] = t_u + sum(total_downstream_time[v] for v in successors[u])
    
    return np.asarray(times, dtype=float), successors, longest_path_down, total_downstream_time


def priority(instance):
    """
    Score each assembly task using a 'bottleneck density' heuristic:
    - Combine longest-path depth (criticality) with processing time
    - Penalize tasks that are on long critical paths and have large processing times
    - This encourages the greedy filler to place high-impact, long-duration tasks early
      when station capacity is most available
    """
    start_time = time.time()
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Compute graph properties
    times_arr, successors, longest_path_down, total_downstream_time = _compute_successors_and_depths(
        times, predecessors, cycle_time
    )
    
    # Check time budget
    if time.time() - start_time > 100:
        return times_arr
    
    # Normalize values for combination
    max_lpd = np.max(longest_path_down) if n > 0 else 1.0
    if max_lpd == 0:
        max_lpd = 1.0
    
    max_tdt = np.max(total_downstream_time) if n > 0 else 1.0
    if max_tdt == 0:
        max_tdt = 1.0
    
    # Bottleneck density: ratio of task time to "remaining stations needed"
    # Approximate remaining stations needed as longest_path_down / cycle_time
    # So density = time / (longest_path_down / cycle_time) = time * cycle_time / longest_path_down
    # But for tasks with no successors, longest_path_down = time, so density = cycle_time (constant)
    
    # Alternative: use a weighted combination
    # 1. Processing time (longer tasks first)
    # 2. Longest path depth (critical tasks first)
    # 3. Number of successors (tasks with many successors first - they unlock more work)
    
    num_successors = np.array([len(successors[i]) for i in range(n)], dtype=float)
    max_ns = np.max(num_successors) if n > 0 else 1.0
    if max_ns == 0:
        max_ns = 1.0
    
    # Normalize each component
    norm_time = times_arr / (np.max(times_arr) if np.max(times_arr) > 0 else 1.0)
    norm_lpd = longest_path_down / max_lpd
    norm_ns = num_successors / max_ns
    
    # Combine: emphasize time and criticality, with a smaller weight on successor count
    # The idea: tasks that are both long AND on critical paths should have highest priority
    # Use a multiplicative-like combination via sum of normalized components with weights
    priority_scores = 0.5 * norm_time + 0.4 * norm_lpd + 0.1 * norm_ns
    
    # Additionally, add a "bottleneck" term: tasks with high time relative to their
    # downstream slack get extra boost
    # Slack = cycle_time - time (how much room is left in a station)
    # If a task takes up most of the cycle time, it's a bottleneck
    slack_ratio = times_arr / cycle_time  # fraction of cycle time used
    priority_scores += 0.2 * slack_ratio
    
    # Final check on time
    if time.time() - start_time > 120:
        return times_arr
    
    return priority_scores
# EVOLVE-BLOCK-END
