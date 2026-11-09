"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from collections import defaultdict, deque


def _build_graph(n, predecessors):
    """
    Build adjacency structures.
    predecessors: list where predecessors[i] is list of task indices that must precede i.
    """
    succ = [[] for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        preds = predecessors[i] if predecessors[i] is not None else []
        for p in preds:
            succ[p].append(i)
            indeg[i] += 1
    return succ, indeg


def _compute_rank(n, succ, indeg):
    """
    Compute a topological rank / position.
    We'll use a simple BFS-based longest path from sources.
    Actually, let's compute:
      - rank[i] = longest path length (in edges) from any source to i.
    This gives a good ordering: tasks with higher rank are "later" in the chain.
    """
    rank = [0] * n
    # Use Kahn's algorithm
    q = deque()
    in_deg = indeg[:]
    for i in range(n):
        if in_deg[i] == 0:
            q.append(i)
    while q:
        u = q.popleft()
        for v in succ[u]:
            if rank[u] + 1 > rank[v]:
                rank[v] = rank[u] + 1
            in_deg[v] -= 1
            if in_deg[v] == 0:
                q.append(v)
    return rank


def _greedy_balance(n, times, succ, indeg, cycle_time):
    """
    Standard greedy: repeatedly pick the eligible task with highest priority that fits.
    Here we use a simple heuristic: among eligible, pick the one with largest time.
    This is the baseline. We'll use this to get a baseline station count.
    """
    # We'll simulate the greedy process to count stations.
    # But this is O(n^2) which is fine for n=30.
    remaining = set(range(n))
    completed = set()
    stations = 0
    station_time = 0.0
    
    # Precompute successors for quick lookup
    while remaining:
        # Find eligible tasks
        eligible = []
        for i in remaining:
            # Check if all predecessors are completed
            ok = True
            for p in predecessors_global[i]:
                if p not in completed:
                    ok = False
                    break
            if ok and station_time + times[i] <= cycle_time + 1e-9:
                eligible.append(i)
        
        if not eligible:
            # Start new station
            stations += 1
            station_time = 0.0
            # Re-find eligible (now no time constraint issue for first task)
            for i in remaining:
                ok = True
                for p in predecessors_global[i]:
                    if p not in completed:
                        ok = False
                        break
                if ok and times[i] <= cycle_time + 1e-9:
                    eligible.append(i)
            if not eligible:
                # Shouldn't happen if instance is valid
                break
        
        # Pick the one with largest time
        best = max(eligible, key=lambda i: times[i])
        station_time += times[best]
        remaining.discard(best)
        completed.add(best)
    
    if station_time > 0:
        stations += 1
    
    return stations


# Global to store predecessors for the greedy function
predecessors_global = None


def priority(instance):
    global predecessors_global
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Store for greedy
    predecessors_global = preds
    
    # Build graph
    succ, indeg = _build_graph(n, preds)
    
    # Compute rank (longest path position)
    rank = _compute_rank(n, succ, indeg)
    
    # Compute number of successors (downstream tasks)
    # This is a good indicator of "criticality" - tasks with many successors are important
    # Let's compute the number of tasks that depend on each task (transitively)
    # For n=30, we can do a BFS from each node.
    dependents_count = [0] * n
    for i in range(n):
        # BFS from i to count all nodes reachable from i
        visited = [False] * n
        q = deque([i])
        visited[i] = True
        count = 0
        while q:
            u = q.popleft()
            for v in succ[u]:
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    q.append(v)
        dependents_count[i] = count
    
    # Also compute the number of predecessors (transitively)
    predecessors_count = [0] * n
    # Build reverse graph
    pred_succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            pred_succ[p].append(i)
    # Actually, let's just use the direct predecessors count from indeg
    # But we want transitive. Let's do BFS on reverse graph.
    # Reverse graph: edge from i to p if p is predecessor of i
    rev_succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            rev_succ[i].append(p)
    
    for i in range(n):
        visited = [False] * n
        q = deque([i])
        visited[i] = True
        count = 0
        while q:
            u = q.popleft()
            for v in rev_succ[u]:
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    q.append(v)
        predecessors_count[i] = count
    
    # Priority score: combine time, rank, and dependents count
    # We want higher priority for:
    # 1. Longer tasks (to fill stations efficiently)
    # 2. Tasks with more dependents (critical path)
    # 3. Tasks with higher rank (later in sequence, so they must be scheduled)
    
    # Normalize each component
    max_time = np.max(times) if n > 0 else 1.0
    max_rank = np.max(rank) if n > 0 else 1.0
    max_dep = np.max(dependents_count) if n > 0 else 1.0
    
    # Scale to [0, 1]
    time_norm = times / max_time if max_time > 0 else times
    rank_norm = np.array(rank) / max_rank if max_rank > 0 else np.array(rank, dtype=float)
    dep_norm = np.array(dependents_count) / max_dep if max_dep > 0 else np.array(dependents_count, dtype=float)
    
    # Weighted sum
    # Emphasize time and dependents
    priority_scores = 0.5 * time_norm + 0.3 * dep_norm + 0.2 * rank_norm
    
    return priority_scores
# EVOLVE-BLOCK-END
