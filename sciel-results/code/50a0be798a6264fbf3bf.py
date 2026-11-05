"""
K4-free graphs without independent 6-sets (Ramsey number R(4,6)).
Goal: maximize n such that a simple graph on n vertices has no K4 and no independent set of size 6.

Strategy:
1. Start from known good constructions (Paley graph on 17, or random graphs).
2. Use local search with simulated annealing to fix violations (K4s and independent 6-sets).
3. Use bitmask-based fast checks for K4 and independent set detection.
4. Try to grow the graph by adding vertices while maintaining constraints.

Time budget: 150 seconds wall-clock, with safety margin.
"""

import time
import random
import itertools
from copy import deepcopy

# Deadline guard
_DEADLINE = 150.0  # seconds
_start_time = None

def _elapsed():
    return time.time() - _start_time

def _time_left():
    return _DEADLINE - _elapsed()

def _is_valid_graph(n, adj):
    """Check if graph has no K4 and no independent set of 6."""
    # Check for K4: for each triple, see if they share a common neighbor
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                continue
            # Find common neighbors of i and j
            common = []
            for k in range(j+1, n):
                if adj[i][k] and adj[j][k]:
                    common.append(k)
            # For each pair in common, check if they are adjacent
            for a_idx in range(len(common)):
                for b_idx in range(a_idx+1, len(common)):
                    a = common[a_idx]
                    b = common[b_idx]
                    if adj[a][b]:
                        return False  # Found K4: i, j, a, b
    # Check for independent set of 6
    # Use a simple recursive search with pruning
    return not _has_independent_set_6(n, adj)

def _has_independent_set_6(n, adj):
    """Check if there's an independent set of size 6."""
    # Use backtracking with pruning
    # Independent set: no two vertices are adjacent
    # We can use the complement: clique of size 6 in complement
    # But let's just do direct backtracking
    
    def backtrack(start, current_set):
        if len(current_set) == 6:
            return True
        # Pruning: not enough vertices left
        if start >= n:
            return False
        if len(current_set) + (n - start) < 6:
            return False
        
        for v in range(start, n):
            # Check if v is non-adjacent to all in current_set
            ok = True
            for u in current_set:
                if adj[u][v]:
                    ok = False
                    break
            if not ok:
                continue
            if backtrack(v+1, current_set + [v]):
                return True
        return False
    
    return backtrack(0, [])

def _count_k4(n, adj):
    """Count number of K4 subgraphs."""
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                continue
            common = []
            for k in range(j+1, n):
                if adj[i][k] and adj[j][k]:
                    common.append(k)
            for a_idx in range(len(common)):
                for b_idx in range(a_idx+1, len(common)):
                    if adj[common[a_idx]][common[b_idx]]:
                        count += 1
    return count

def _count_indep6(n, adj):
    """Count number of independent sets of size 6."""
    count = 0
    def backtrack(start, current_set):
        nonlocal count
        if len(current_set) == 6:
            count += 1
            return
        if start >= n:
            return
        if len(current_set) + (n - start) < 6:
            return
        for v in range(start, n):
            ok = True
            for u in current_set:
                if adj[u][v]:
                    ok = False
                    break
            if not ok:
                continue
            backtrack(v+1, current_set + [v])
    backtrack(0, [])
    return count

def _local_search(n, adj, max_iter=5000):
    """
    Local search to fix violations.
    Returns (n, adj) with no K4 and no independent 6-set, or None if failed.
    """
    adj = [row[:] for row in adj]
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Check for K4
        k4_count = _count_k4(n, adj)
        if k4_count > 0:
            # Find a K4 and try to break it by removing an edge
            # Find all K4s
            k4s = []
            for i in range(n):
                for j in range(i+1, n):
                    if not adj[i][j]:
                        continue
                    common = []
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            common.append(k)
                    for a_idx in range(len(common)):
                        for b_idx in range(a_idx+1, len(common)):
                            if adj[common[a_idx]][common[b_idx]]:
                                k4s.append((i, j, common[a_idx], common[b_idx]))
            if k4s:
                # Pick a random K4 and remove a random edge from it
                k4 = random.choice(k4s)
                edge = random.choice([(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]),
                                       (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])])
                u, v = edge
                adj[u][v] = 0
                adj[v][u] = 0
                continue
        
        # Check for independent set of 6
        if _has_independent_set_6(n, adj):
            # Find an independent set of 6 and try to add an edge
            # Find all independent sets of 6
            indep6s = []
            def find_indep6(start, current_set, results):
                if len(current_set) == 6:
                    results.append(current_set[:])
                    return
                if start >= n:
                    return
                if len(current_set) + (n - start) < 6:
                    return
                for v in range(start, n):
                    ok = True
                    for u in current_set:
                        if adj[u][v]:
                            ok = False
                            break
                    if not ok:
                        continue
                    current_set.append(v)
                    find_indep6(v+1, current_set, results)
                    current_set.pop()
            find_indep6(0, [], indep6s)
            if indep6s:
                # Pick a random independent set and add a random edge within it
                indep = random.choice(indep6s)
                # Try all pairs in indep, pick one that's currently non-adjacent
                candidates = []
                for a_idx in range(len(indep)):
                    for b_idx in range(a_idx+1, len(indep)):
                        u, v = indep[a_idx], indep[b_idx]
                        if not adj[u][v]:
                            candidates.append((u, v))
                if candidates:
                    u, v = random.choice(candidates)
                    # Before adding, check if it creates a K4
                    # Check if u and v have a common neighbor that is adjacent to both
                    # Actually, adding edge (u,v) creates K4 iff there exist a,b such that
                    # u-a, u-b, v-a, v-b, a-b all exist
                    # Simpler: just add and check
                    adj[u][v] = 1
                    adj[v][u] = 1
                    if _count_k4(n, adj) > 0:
                        # Revert
                        adj[u][v] = 0
                        adj[v][u] = 0
                        # Try another candidate
                        if len(candidates) > 1:
                            # Just continue to next iteration
                            pass
                        continue
                else:
                    # All pairs in indep are already adjacent? That can't be if it's independent
                    pass
            continue
        
        # No violations
        return n, adj
    
    # Check final state
    if _count_k4(n, adj) == 0 and not _has_independent_set_6(n, adj):
        return n, adj
    return None

def _try_grow(n, adj):
    """
    Try to add a new vertex (n) to the graph while maintaining constraints.
    Returns (n+1, new_adj) if successful, else None.
    """
    if _time_left() < 10.0:
        return None
    
    new_n = n + 1
    new_adj = [row[:] + [0] for row in adj]
    new_adj.append([0] * new_n)
    
    # Try different connection patterns for the new vertex
    # The new vertex should be connected to some subset of existing vertices
    # We need to ensure no K4 is created and no independent set of 6 is created
    
    # Strategy: try random subsets of neighbors
    best_result = None
    
    for attempt in range(200):
        if _time_left() < 5.0:
            break
        
        # Randomly decide which vertices to connect to
        neighbors = set()
        for v in range(n):
            if random.random() < 0.5:
                neighbors.add(v)
        
        # Apply
        for v in neighbors:
            new_adj[n][v] = 1
            new_adj[v][n] = 1
        
        # Check validity
        if _count_k4(new_n, new_adj) == 0 and not _has_independent_set_6(new_n, new_adj):
            return new_n, new_adj
    
    return None

def construct():
    global _start_time
    _start_time = time.time()
    
    # Start with Paley graph on 17 vertices
    n = 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if (j - i) % 17 in squares:
                adj[i][j] = 1
                adj[j][i] = 1
    
    # Verify initial graph is valid
    if _count_k4(n, adj) != 0 or _has_independent_set_6(n, adj):
        # Shouldn't happen, but handle gracefully
        pass
    
    best_n = n
    best_adj = [row[:] for row in adj]
    
    # Try to grow the graph
    while _time_left() > 20.0:
        result = _try_grow(n, adj)
        if result is None:
            break
        n, adj = result
        best_n = n
        best_adj = [row[:] for row in adj]
    
    # Try local search on random graphs to see if we can find better
    for trial in range(10):
        if _time_left() < 30.0:
            break
        
        # Try different starting sizes
        for start_n in [20, 25, 30, 35]:
            if _time_left() < 20.0:
                break
            
            # Create random graph
            adj_rand = [[0] * start_n for _ in range(start_n)]
            for i in range(start_n):
                for j in range(i+1, start_n):
                    if random.random() < 0.4:
                        adj_rand[i][j] = 1
                        adj_rand[j][i] = 1
            
            # Local search
            result = _local_search(start_n, adj_rand, max_iter=2000)
            if result is not None:
                rn, r_adj = result
                if rn > best_n:
                    best_n = rn
                    best_adj = [row[:] for row in r_adj]
    
    # Build edge list from best_adj
    edges = []
    for i in range(best_n):
        for j in range(i+1, best_n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
