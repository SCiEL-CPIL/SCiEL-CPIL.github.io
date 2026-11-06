"""Resource-Constrained Project Scheduling: priority function for serial SGS."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict


def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Returns an array of scores (higher priority = started earlier).
    
    We use a composite heuristic:
    1. Longest path to end (critical path length) - activities on critical path first
    2. Resource contention - activities that demand scarce resources first
    3. Duration - longer activities first (to avoid fragmentation)
    4. Successor count - activities with more successors first (to unblock more work)
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    n = len(durations)
    
    # Build successor list
    if predecessors is not None:
        # predecessors[i] = list of predecessor activities for activity i
        successors = defaultdict(list)
        for i in range(n):
            preds = predecessors[i] if isinstance(predecessors, (list, tuple, np.ndarray)) else []
            for p in preds:
                successors[p].append(i)
    else:
        successors = defaultdict(list)
    
    # Compute longest path from each activity to the end (reverse topological)
    # This is the "critical path length" from each node
    # Use topological sort
    in_degree = np.zeros(n, dtype=int)
    if predecessors is not None:
        for i in range(n):
            preds = predecessors[i] if isinstance(predecessors, (list, tuple, np.ndarray)) else []
            in_degree[i] = len(preds)
    
    # Topological order (Kahn's algorithm)
    from queue import Queue
    q = Queue()
    for i in range(n):
        if in_degree[i] == 0:
            q.put(i)
    
    topo_order = []
    while not q.empty():
        node = q.get()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                q.put(succ)
    
    # If not all nodes in topo_order, there's a cycle; handle gracefully
    if len(topo_order) < n:
        # Add remaining nodes
        remaining = set(range(n)) - set(topo_order)
        topo_order.extend(list(remaining))
    
    # Longest path from each node to any sink (reverse DP)
    longest_path = np.zeros(n, dtype=float)
    # Process in reverse topological order
    for node in reversed(topo_order):
        max_succ = 0.0
        for succ in successors[node]:
            max_succ = max(max_succ, longest_path[succ])
        longest_path[node] = durations[node] + max_succ
    
    # Resource contention: sum of demand/capacity ratio for each activity
    # Higher ratio means more scarce resource usage
    resource_pressure = np.zeros(n, dtype=float)
    if demands is not None and capacities is not None:
        # demands[i] could be a list/array of resource demands for activity i
        # capacities could be a list/array of resource capacities
        if isinstance(demands, np.ndarray):
            d = demands
        else:
            d = np.array(demands, dtype=float)
        
        if isinstance(capacities, np.ndarray):
            c = capacities
        else:
            c = np.array(capacities, dtype=float)
        
        # d shape: (n, num_resources) or (n,)
        # c shape: (num_resources,) or (num_resources, 1)
        if d.ndim == 1:
            # Single resource
            if c.ndim == 1:
                cap = c[0] if len(c) > 0 else 1.0
                if cap > 0:
                    resource_pressure = d / cap
            else:
                cap = c.flat[0]
                if cap > 0:
                    resource_pressure = d / cap
        else:
            # Multiple resources: d is (n, R), c is (R,)
            if c.ndim == 1:
                # Avoid division by zero
                safe_c = np.where(c > 0, c, 1.0)
                ratios = d / safe_c
                resource_pressure = np.sum(ratios, axis=1)
            else:
                safe_c = np.where(c > 0, c, 1.0)
                ratios = d / safe_c
                resource_pressure = np.sum(ratios, axis=-1)
    
    # Successor count
    succ_count = np.zeros(n, dtype=float)
    for i in range(n):
        succ_count[i] = len(successors[i])
    
    # Normalize each component to [0, 1] range for combination
    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    lp_norm = normalize(longest_path)
    rp_norm = normalize(resource_pressure)
    dur_norm = normalize(durations)
    sc_norm = normalize(succ_count)
    
    # Composite score: weighted sum
    # Weights: critical path most important, then resource pressure, duration, successors
    w_lp = 0.45
    w_rp = 0.25
    w_dur = 0.20
    w_sc = 0.10
    
    scores = w_lp * lp_norm + w_rp * rp_norm + w_dur * dur_norm + w_sc * sc_norm
    
    return scores
# EVOLVE-BLOCK-END
