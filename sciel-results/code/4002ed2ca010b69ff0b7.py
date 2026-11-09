"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def _compute_flow_criticality(n, successors, times, cycle_time):
    """
    Compute flow-network-based criticality scores.
    
    Models the DAG as a flow network where each task has capacity = its time.
    The 'criticality' of a task is how much it constrains the total flow
    (total work) that can pass through its downstream subgraph.
    
    We compute:
    1. downstream_work[u] = sum of times of all transitive successors of u (including u)
    2. bottleneck_score[u] = 1 / max(1, downstream_work[u]) - this gives higher score
       to tasks that are bottlenecks (small downstream work relative to their position)
    
    Actually, let's think differently: a task is a bottleneck if removing it
    would disconnect much of the remaining work. We approximate this by:
    - For each task, compute the fraction of total work that MUST go through it
      (i.e., work that has no alternative path around it)
    
    Simpler approach: compute for each task the "minimum cut" value of its
    downstream subgraph. We approximate by:
    - downstream_sum[u] = sum of times of u and all transitive successors
    - The criticality is how "concentrated" the work is: tasks with high 
      downstream_sum relative to the number of parallel paths are more critical.
    
    Let's use: criticality[u] = downstream_sum[u] / (1 + num_parallel_paths[u])
    where num_parallel_paths approximates the width of the downstream DAG.
    
    Even simpler and more direct: use the concept from max-flow in series-parallel
    graphs. For a DAG, the "flow capacity" through a node is limited by the
    minimum of (its own time, sum of capacities of successors). This is like
    computing the max flow from each node to sinks.
    
    Let's compute: flow_cap[u] = min(times[u], sum(flow_cap[v] for v in successors[u]))
    But this doesn't quite work either.
    
    Let me try a different angle: the task's importance in constraining cycle time
    is related to how many tasks depend on it and the total work they represent.
    
    I'll compute:
    - transitive_succ_work[u] = sum of times of all transitive successors of u
    - The "bottleneck factor" = transitive_succ_work[u] / (1 + num_immediate_succs[u])
      This gives higher score to tasks that have lots of downstream work but
      few parallel paths (i.e., they are true bottlenecks).
    """
    if n == 0:
        return np.array([])
    
    # Compute transitive successor work sum using reverse topological order
    # First, get topological order
    pred_counts = [len(successors[i]) for i in range(n)]  # This is actually out-degree
    # We need in-degree for topo sort
    in_deg = [0] * n
    for u in range(n):
        for v in successors[u]:
            in_deg[v] += 1
    
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in successors[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    
    if len(topo) < n:
        return np.ones(n)
    
    # Compute transitive successor work: work_after[u] = sum of times of all 
    # transitive successors of u (not including u itself)
    work_after = [0.0] * n
    for u in reversed(topo):
        total = 0.0
        for v in successors[u]:
            total += times[v] + work_after[v]
        work_after[u] = total
    
    # Compute number of immediate successors
    num_succ = [len(successors[u]) for u in range(n)]
    
    # Bottleneck score: high when lots of downstream work but few parallel paths
    # This identifies tasks that are true serial bottlenecks
    bottleneck = np.zeros(n)
    for u in range(n):
        if num_succ[u] == 0:
            bottleneck[u] = 0.0  # Sink nodes have no downstream constraint
        else:
            # Normalize: more downstream work / fewer paths = more critical
            bottleneck[u] = work_after[u] / (1.0 + num_succ[u])
    
    return bottleneck


def _compute_parallel_width(n, successors):
    """
    Compute the 'parallel width' of the downstream subgraph for each node.
    This is an approximation of how many independent paths exist downstream.
    
    We compute the longest path length (in terms of number of nodes) from 
    each node to any sink, which gives a measure of serial depth.
    The 'width' is approximated by total_downstream_nodes / serial_depth.
    """
    if n == 0:
        return np.array([])
    
    # Get topological order
    in_deg = [0] * n
    for u in range(n):
        for v in successors[u]:
            in_deg[v] += 1
    
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in successors[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    
    if len(topo) < n:
        return np.ones(n)
    
    # Compute longest path (in nodes) from each node to sink
    depth_to_sink = [1] * n  # At least 1 (itself)
    for u in reversed(topo):
        max_d = 0
        for v in successors[u]:
            max_d = max(max_d, depth_to_sink[v])
        depth_to_sink[u] = 1 + max_d
    
    # Compute total number of transitive successors
    trans_count = [0] * n
    for u in reversed(topo):
        total = 0
        for v in successors[u]:
            total += 1 + trans_count[v]
        trans_count[u] = total
    
    # Width approximation: total downstream nodes / serial depth
    width = np.ones(n)
    for u in range(n):
        if trans_count[u] > 0 and depth_to_sink[u] > 1:
            width[u] = trans_count[u] / depth_to_sink[u]
        else:
            width[u] = 1.0
    
    return width


def priority(instance):
    """
    Score each assembly task for the priority-based list scheduling.
    
    Strategy: Flow-network criticality scoring.
    
    1. Compute downstream work and bottleneck scores from the flow network model
    2. Combine with critical path height and task time
    3. Use parallel width to identify true serial bottlenecks
    
    The key insight: tasks that act as bottlenecks in the flow network
    (high downstream work, few parallel paths) should be prioritized
    to prevent them from becoming the limiting factor in station packing.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    # Build successor map
    successors = [[] for _ in range(n)]
    pred_count = [0] * n
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        pred_count[i] = len(preds)
        for p in preds:
            successors[p].append(i)
    
    # Topological order
    in_deg = [0] * n
    for u in range(n):
        for v in successors[u]:
            in_deg[v] += 1
    
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in successors[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    
    if len(topo) < n:
        return times.copy()
    
    # Compute height (longest path from node to end in time units)
    height = np.zeros(n)
    for u in reversed(topo):
        max_h = 0.0
        for v in successors[u]:
            max_h = max(max_h, height[v])
        height[u] = times[u] + max_h
    
    # Compute depth (longest path from start to node)
    depth = np.zeros(n)
    for u in topo:
        for v in successors[u]:
            if depth[v] < depth[u] + times[u]:
                depth[v] = depth[u] + times[u]
    
    # Compute flow-network criticality scores
    bottleneck = _compute_flow_criticality(n, successors, times, cycle_time)
    width = _compute_parallel_width(n, successors)
    
    # Transitive successor count
    trans_count = [0] * n
    for u in reversed(topo):
        total = 0
        for v in successors[u]:
            total += 1 + trans_count[v]
        trans_count[u] = total
    trans_count = np.array(trans_count, dtype=float)
    
    # Normalize all components
    max_time = np.max(times) if n > 0 else 1.0
    max_height = np.max(height) if n > 0 else 1.0
    max_depth = np.max(depth) if n > 0 else 1.0
    max_bottleneck = np.max(bottleneck) if n > 0 else 1.0
    max_width = np.max(width) if n > 0 else 1.0
    max_trans = np.max(trans_count) if n > 0 else 1.0
    
    if max_time == 0: max_time = 1.0
    if max_height == 0: max_height = 1.0
    if max_depth == 0: max_depth = 1.0
    if max_bottleneck == 0: max_bottleneck = 1.0
    if max_width == 0: max_width = 1.0
    if max_trans == 0: max_trans = 1.0
    
    # Composite score: higher is better priority
    # 
    # Key components:
    # 1. height: critical path remaining - tasks on the critical path first
    # 2. bottleneck: flow-network criticality - tasks that are true bottlenecks
    # 3. time: longer tasks first to reduce fragmentation
    # 4. trans_count: tasks with more downstream work first
    # 5. width (inverted): tasks with LESS parallel width (more serial) get higher priority
    #    because they are harder to parallelize and must be placed carefully
    #
    # The bottleneck score captures the "minimum cut" intuition: a task is
    # more critical if removing it would disconnect a large portion of the
    # downstream work. This is approximated by downstream_work / parallel_paths.
    
    # Invert width: lower width (more serial) = higher priority
    inv_width = 1.0 / (width / max_width + 0.1)  # Regularize to avoid division by zero
    
    score = (
        1.2 * (height / max_height) +
        0.8 * (bottleneck / max_bottleneck) +
        0.5 * (times / max_time) +
        0.3 * (trans_count / max_trans) +
        0.2 * (inv_width / np.max(inv_width))
    )
    
    # Check deadline
    if time.time() > deadline:
        return score
    
    return score
# EVOLVE-BLOCK-END
