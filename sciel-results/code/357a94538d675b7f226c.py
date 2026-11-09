"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _longest_downstream(n, times, successors):
    """Compute longest path from each node to any sink, including own time."""
    longest = np.zeros(n, dtype=float)
    # Process in reverse topological order (sinks first)
    # Use iterative DFS/BFS approach
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        for i in successors[j]:
            in_degree[i] += 1
    
    # Topological order via Kahn's algorithm
    from collections import deque
    queue = deque()
    for j in range(n):
        if in_degree[j] == 0:
            queue.append(j)
    
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Reverse topological: sinks have no successors
    # longest[j] = times[j] + max(longest[v] for v in successors[j])
    for j in reversed(topo):
        max_down = 0.0
        for v in successors[j]:
            if longest[v] > max_down:
                max_down = longest[v]
        longest[j] = times[j] + max_down
    
    return longest


def _longest_upstream(n, times, predecessors):
    """Compute longest path from any source to each node, including own time."""
    longest = np.zeros(n, dtype=float)
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        for i in predecessors[j]:
            in_degree[j] += 1
    
    from collections import deque
    queue = deque()
    for j in range(n):
        if in_degree[j] == 0:
            queue.append(j)
    
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in range(n):
            if u in predecessors[v]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
    
    # Forward topological: longest[j] = times[j] + max(longest[i] for i in predecessors[j])
    for j in topo:
        max_up = 0.0
        for i in predecessors[j]:
            if longest[i] > max_up:
                max_up = longest[i]
        longest[j] = times[j] + max_up
    
    return longest


def _successors_from_preds(n, predecessors):
    """Build successor lists from predecessor lists."""
    successors = [set() for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            successors[i].add(j)
    return [list(s) for s in successors]


def priority(instance):
    t_start = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Build predecessor and successor lists
    pred_lists = [list(p) for p in preds]
    succ_lists = _successors_from_preds(n, pred_lists)
    
    # Compute longest downstream and upstream paths
    downstream = _longest_downstream(n, times, succ_lists)
    upstream = _longest_upstream(n, times, pred_lists)
    
    # Slack: how much time is available before this task's chain exceeds cycle_time
    # slack[i] = cycle_time - downstream[i] (if downstream[i] > cycle_time, slack is negative -> very urgent)
    slack = cycle_time - downstream
    
    # Critical path length through task i
    critical_path = downstream + upstream - times
    
    # Number of successors (fewer = more constrained)
    num_succ = np.array([len(s) for s in succ_lists], dtype=float)
    num_pred = np.array([len(p) for p in pred_lists], dtype=float)
    
    # Build composite priority
    # We want: high time, low slack (urgent), high criticality, few successors
    # Use a weighted combination with normalization
    
    # Normalize components to [0, 1] range for stable weighting
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)
    
    time_norm = normalize(times)
    # Invert slack: low slack -> high urgency
    slack_inv = normalize(-slack)
    crit_norm = normalize(critical_path)
    # Invert num_succ: few successors -> high priority
    succ_inv = normalize(-num_succ)
    # Invert num_pred: few predecessors -> high priority (sources are flexible, but we want to place constrained ones first)
    pred_inv = normalize(-num_pred)
    
    # Downstream time as a proxy for "how much work remains after this task"
    down_norm = normalize(downstream)
    
    # Composite: weighted sum
    w_time = 0.30
    w_slack = 0.25
    w_crit = 0.20
    w_succ = 0.10
    w_pred = 0.05
    w_down = 0.10
    
    score = (w_time * time_norm +
             w_slack * slack_inv +
             w_crit * crit_norm +
             w_succ * succ_inv +
             w_pred * pred_inv +
             w_down * down_norm)
    
    # Add a small deterministic tie-breaker based on index to ensure reproducibility
    # but keep it very small so it doesn't dominate
    tie_break = np.arange(n, dtype=float) / (n + 1) * 0.001
    score = score + tie_break
    
    if time.time() - t_start > DEADLINE:
        return score
    
    return score
# EVOLVE-BLOCK-END
