"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from collections import defaultdict

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

    # Build successors for topological sort / eligibility checking
    succs = [[] for _ in range(n)]
    indegree = [0] * n
    for i in range(n):
        for p in parsed_preds[i]:
            succs[p].append(i)
            indegree[i] += 1

    # --- Strategy: "Bottleneck Density" Priority ---
    # Instead of static Rank (Longest Tail) or Time, we use a structural metric
    # that identifies tasks which are critical to the *flow* of the line.
    # 
    # Metric: Transitive Successor Count (TSC)
    # Tasks with many transitive successors are "bottlenecks". If a bottleneck
    # task is placed in a station, it unlocks many other tasks for subsequent
    # stations. Placing them early (high priority) ensures that the "fan-out"
    # happens as soon as possible, allowing the greedy algorithm to have more
    # eligible tasks to choose from in later stations, potentially leading to
    # tighter packing.
    #
    # Additionally, we combine this with "Time" to break ties or add weight.
    # A high-priority task should be one that is both structurally important
    # (high TSC) and substantial (high time), but TSC is the primary driver
    # for eligibility flow.
    
    # Compute Transitive Successor Count for each node
    # This is the number of nodes reachable from i via directed paths.
    # We can compute this using DFS/BFS from each node, or by processing
    # in reverse topological order and using bitsets or sets.
    # For n=30, sets are fine.
    
    # First, get a topological order (sinks to sources or sources to sinks?)
    # Let's do a standard Kahn's algorithm to get a valid topological order (sources to sinks).
    # Then process in reverse to accumulate successor sets.
    
    in_deg = indegree[:]
    queue = [i for i in range(n) if in_deg[i] == 0]
    topo_order = []
    while queue:
        # Pick a node, prefer those with higher indegree? No, just any.
        # For determinism, sort by index.
        u = queue.pop(0)
        topo_order.append(u)
        for v in succs[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
        
    # If there's a cycle, some nodes won't be in topo_order. 
    # The problem states "generated instances", likely DAGs.
    # If a node is not in topo_order, it's part of a cycle. We handle it by 
    # just giving it a low priority or default.
    
    # Compute transitive successors using sets
    # trans_succ[i] is a set of all nodes reachable from i
    trans_succ = [set() for _ in range(n)]
    
    # Process in reverse topological order (sinks first)
    # Wait, if we process sinks first, we don't have their successors yet?
    # No, trans_succ[u] = union(trans_succ[v] for v in succs[u]) + {v for v in succs[u]}
    # So we need to process nodes AFTER their successors.
    # Reverse topological order (sinks to sources) allows this.
    
    # Let's get reverse topological order
    rev_topo_order = list(reversed(topo_order))
    
    for u in rev_topo_order:
        for v in succs[u]:
            # Add v and all of v's transitive successors
            trans_succ[u].add(v)
            trans_succ[u].update(trans_succ[v])
    
    # Now, trans_succ[i] contains all transitive successors of i.
    # Score = len(trans_succ[i])
    
    # Let's refine the score.
    # Pure TSC might favor "hub" nodes early.
    # What if we combine TSC with Time?
    # score[i] = TSC[i] * (1 + times[i])?
    # Or score[i] = TSC[i] + times[i]?
    
    # Let's try: Score = TSC[i] * 10 + times[i]
    # This heavily weights the structural bottleneck aspect, but uses time as a tie-breaker/secondary factor.
    # Actually, if TSC is 0 for all (no successors), it falls back to time.
    # If TSC varies, it dominates.
    
    # Let's check the deadline.
    if time.time() > deadline - 5.0:
        # Fallback to simple time if running out of time
        return times
    
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        tsc = len(trans_succ[i])
        # Weighted combination
        # Let's try TSC as the primary, and add a small fraction of time
        # to break ties and account for the actual load.
        scores[i] = tsc * 1.0 + times[i] * 0.1
        
    return scores

# EVOLVE-BLOCK-END
