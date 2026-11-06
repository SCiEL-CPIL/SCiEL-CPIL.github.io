"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Parse predecessors
    preds = instance.get("predecessors", [])
    if not preds:
        preds = [[] for _ in range(n)]
    
    # Ensure preds are lists of indices
    parsed_preds = []
    for i in range(n):
        p = preds[i]
        if isinstance(p, (list, tuple, np.ndarray)):
            parsed_preds.append(list(p))
        elif isinstance(p, int):
            parsed_preds.append([p])
        else:
            parsed_preds.append([])

    # Build successors
    succs = [[] for _ in range(n)]
    indegree = [0] * n
    for i in range(n):
        for p in parsed_preds[i]:
            succs[p].append(i)
            indegree[i] += 1

    # Topological Sort (Kahn's Algorithm)
    in_deg = indegree[:]
    queue = [i for i in range(n) if in_deg[i] == 0]
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in succs[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
        
    if len(topo_order) != n:
        # Fallback for cycles (should not happen in valid DAGs)
        return times

    # Compute Forward Rank (Earliest Start Time / Longest Path from Source)
    # start_rank[i] = time[i] + max(start_rank[p] for p in preds[i])
    start_rank = np.zeros(n, dtype=float)
    for u in topo_order:
        max_pred_rank = 0.0
        for p in parsed_preds[u]:
            if start_rank[p] > max_pred_rank:
                max_pred_rank = start_rank[p]
        start_rank[u] = times[u] + max_pred_rank

    # Compute Backward Rank (Longest Path to Sink / Tail Length)
    # end_rank[i] = time[i] + max(end_rank[s] for s in succs[i])
    end_rank = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        max_succ_rank = 0.0
        for s in succs[u]:
            if end_rank[s] > max_succ_rank:
                max_succ_rank = end_rank[s]
        end_rank[u] = times[u] + max_succ_rank

    # Compute Transitive Successor Count (Bottleneck Metric)
    # Tasks with many transitive successors are bottlenecks that unlock many tasks.
    trans_succ = [set() for _ in range(n)]
    # Process in reverse topological order to ensure successors are processed before predecessors
    for u in reversed(topo_order):
        for v in succs[u]:
            trans_succ[u].add(v)
            trans_succ[u].update(trans_succ[v])
    
    tsc = np.array([len(s) for s in trans_succ], dtype=float)

    # Check deadline before final scoring
    if time.time() > deadline - 5.0:
        return times

    # --- Composite Priority Strategy ---
    # 
    # Components:
    # 1. end_rank (Tail): Prioritizes tasks that lead to long chains (Critical Path).
    #    This is the "Highest Positional Weight" heuristic, known to be strong for ALB.
    # 2. start_rank (Head): Prioritizes tasks that are late in the dependency chain.
    #    The directive suggests prioritizing based on "positional criticality" or temporal bottlenecks.
    #    However, placing tasks with high start_rank (late in the graph) early might violate precedence?
    #    No, the greedy filler only picks *eligible* tasks. 
    #    If a task has high start_rank, it has many predecessors. It becomes eligible late.
    #    Prioritizing it *highly* means that *when* it becomes eligible, it is picked immediately.
    #    This is good for "tail" packing.
    #    
    # 3. tsc (Bottleneck Density): Prioritizes tasks that unlock many others.
    #    Placing these early (when eligible) maximizes the pool of eligible tasks for subsequent stations.
    #
    # 4. times (LPT): Prioritizes long tasks to avoid wasting cycle time space.
    #
    # Proposed Formula:
    # We want to combine these into a single score.
    # The current program (score 1.0263) used: tsc * 1.0 + times * 0.1
    # Inspiration 1 (score 1.0263) used: end_rank + times
    # 
    # Let's try a blend that respects the "flow" (TSC) and "criticality" (end_rank).
    # end_rank is a time-weighted depth. TSC is a count-weighted breadth.
    # 
    # Hypothesis: A task is "critical" if it is on a long path (high end_rank) AND it unlocks many tasks (high TSC).
    # 
    # Let's normalize the components to have similar scales?
    # n=30, times ~ O(1-10), cycle_time=40.
    # end_rank can be up to ~40-50.
    # TSC can be up to ~29.
    # times is ~1-10.
    #
    # Let's try:
    # score = end_rank + 0.5 * tsc + 0.1 * times
    #
    # Alternatively, the directive mentions "ratio of their earliest possible start time... to cycle time".
    # start_rank / cycle_time is a normalized "depth".
    # If we prioritize high start_rank, we are prioritizing tasks that are deep in the dependency tree.
    # This ensures that when the "front" of the line is cleared, the "back" (deep tasks) are ready to be packed efficiently.
    #
    # Let's combine:
    # 1. Tail (end_rank): Most important for ensuring we don't strand short chains.
    # 2. Bottleneck (tsc): Important for opening up the graph.
    # 3. Time (times): Important for packing.
    #
    # Let's try a weighted sum:
    # score = 1.0 * end_rank + 0.5 * tsc + 0.2 * times
    #
    # Let's also consider the "slack" idea from Inspiration 3.
    # Slack = cycle_time - (time[i] + ...)? Not directly applicable to static priority.
    #
    # Let's stick to the structural metrics which have shown improvement over pure LPT.
    # The current best (1.0263) used TSC. Inspiration 1 (1.0263) used End_Rank.
    # They scored the same. Combining them might be better.
    
    scores = 1.0 * end_rank + 0.5 * tsc + 0.2 * times
    
    return scores

# EVOLVE-BLOCK-END
