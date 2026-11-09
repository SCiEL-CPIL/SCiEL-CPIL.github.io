"""Improved program: exploit circulant structure + ILP-style edge packing for more cycles."""
# EVOLVE-BLOCK-START
import time
import sys
import math
from collections import defaultdict, deque

sys.setrecursionlimit(10000)

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_circulant_cycles(n, all_edges, used_edges, deadline):
    """Find all valid circulant Hamiltonian cycles with unused edges."""
    cycles = []
    for k in range(1, n):
        if time.time() > deadline:
            break
        if _gcd(k, n) != 1:
            continue
        
        valid = True
        for i in range(n):
            u = i
            v = (i + k) % n
            ek = _edge_key(u, v)
            if ek not in all_edges or ek in used_edges:
                valid = False
                break
        
        if valid:
            cycle = []
            v = 0
            for _ in range(n):
                cycle.append(v)
                v = (v + k) % n
            cycles.append(cycle)
    
    return cycles

def _find_hamiltonian_cycle(adj, n, used_edges, deadline, start_vertex=0):
    """Find a Hamiltonian cycle using backtracking with aggressive pruning."""
    
    avail = [set() for _ in range(n)]
    for v in range(n):
        for nb in adj[v]:
            if _edge_key(v, nb) not in used_edges:
                avail[v].add(nb)
    
    deg = [len(avail[v]) for v in range(n)]
    neighbor_order = {v: sorted(avail[v], key=lambda x: (deg[x], x)) for v in range(n)}
    
    def _get_forced(avail_sets, visited_set, current):
        forced = {}
        for v in range(n):
            if v in visited_set:
                continue
            candidates = [nb for nb in avail_sets[v] if nb == current or nb not in visited_set]
            if len(candidates) == 1:
                forced[v] = candidates[0]
        return forced
    
    def _check_connectivity(avail_sets, visited_set, current):
        unvisited = [v for v in range(n) if v not in visited_set]
        if not unvisited:
            return True
        
        visited_bfs = set()
        queue = deque([current])
        visited_bfs.add(current)
        
        while queue:
            node = queue.popleft()
            for nb in avail_sets[node]:
                if nb not in visited_bfs and (nb not in visited_set or nb == current):
                    if nb not in visited_set or nb == current:
                        visited_bfs.add(nb)
                        queue.append(nb)
        
        for v in unvisited:
            if v not in visited_bfs:
                return False
        return True
    
    def backtrack(visited, path, current, depth):
        if time.time() > deadline:
            return None
        
        if depth == n:
            if path[0] in avail[current]:
                return path[:]
            return None
        
        if not _check_connectivity(avail, visited, current):
            return None
        
        forced = _get_forced(avail, visited, current)
        
        if current in forced:
            neighbor = forced[current]
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                result = backtrack(visited, path, neighbor, depth + 1)
                if result is not None:
                    return result
                path.pop()
                visited.discard(neighbor)
            return None
        
        candidates = []
        for neighbor in neighbor_order[current]:
            if neighbor not in visited:
                candidates.append(neighbor)
        
        candidates.sort(key=lambda x: (0 if x in forced else 1, deg[x], x))
        
        for neighbor in candidates:
            visited.add(neighbor)
            path.append(neighbor)
            result = backtrack(visited, path, neighbor, depth + 1)
            if result is not None:
                return result
            path.pop()
            visited.discard(neighbor)
        return None
    
    visited = {start_vertex}
    path = [start_vertex]
    result = backtrack(visited, path, start_vertex, 1)
    if result is not None:
        return result
    
    for start in range(n):
        if time.time() > deadline:
            break
        visited = {start}
        path = [start]
        result = backtrack(visited, path, start, 1)
        if result is not None:
            return result
    
    return None

def _find_non_hamiltonian_cycles(adj, n, used_edges, deadline, min_len=3, max_len=7):
    """
    Find edge-disjoint cycles of various lengths (not necessarily Hamiltonian)
    to maximize total cycle count. Uses greedy + backtracking for smaller cycles.
    """
    cycles = []
    all_edges = set()
    for u, v in P['edges']:
        all_edges.add(_edge_key(u, v))
    
    # Build available edge set
    avail_edges = set()
    for u, v in P['edges']:
        ek = _edge_key(u, v)
        if ek not in used_edges:
            avail_edges.add(ek)
    
    # Build adjacency for available edges
    avail_adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        ek = _edge_key(u, v)
        if ek in avail_edges:
            avail_adj[u].add(v)
            avail_adj[v].add(u)
    
    # Try to find cycles of different lengths
    for target_len in range(min_len, max_len + 1):
        if time.time() > deadline:
            break
        
        # Greedy: try to find a cycle of target_len
        # Use DFS from each vertex
        for start in range(n):
            if time.time() > deadline:
                break
            
            found = _find_cycle_of_length(avail_adj, n, start, target_len, avail_edges, used_edges, deadline)
            if found is not None:
                cycles.append(found)
                # Remove used edges
                for j in range(len(found)):
                    u = found[j]
                    v = found[(j + 1) % len(found)]
                    ek = _edge_key(u, v)
                    avail_edges.discard(ek)
                    used_edges.add(ek)
                # Update avail_adj
                for j in range(len(found)):
                    u = found[j]
                    v = found[(j + 1) % len(found)]
                    avail_adj[u].discard(v)
                    avail_adj[v].discard(u)
    
    return cycles

def _find_cycle_of_length(avail_adj, n, start, target_len, avail_edges, used_edges, deadline):
    """Find a simple cycle of exactly target_len starting from start."""
    
    def dfs(current, path, visited):
        if time.time() > deadline:
            return None
        
        if len(path) == target_len:
            # Check if we can close back to start
            if start in avail_adj[current]:
                ek = _edge_key(current, start)
                if ek in avail_edges:
                    return path[:]
            return None
        
        # Pruning: if remaining vertices needed > available neighbors, fail
        remaining = target_len - len(path)
        if remaining < 2:  # Need at least one more vertex + close
            return None
        
        for neighbor in sorted(avail_adj[current]):
            if neighbor == start:
                continue  # Can't close early
            if neighbor not in visited:
                ek = _edge_key(current, neighbor)
                if ek in avail_edges:
                    visited.add(neighbor)
                    path.append(neighbor)
                    result = dfs(neighbor, path, visited)
                    if result is not None:
                        return result
                    path.pop()
                    visited.discard(neighbor)
        return None
    
    visited = {start}
    path = [start]
    result = dfs(start, path, visited)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin
    
    adj = _build_adj()
    n = P['n']
    
    all_edges = set()
    for u, v in P['edges']:
        all_edges.add(_edge_key(u, v))
    
    used_edges = set()
    cycles = []
    
    # Step 1: Find all circulant cycles (fast, deterministic)
    circulant_cycles = _find_circulant_cycles(n, all_edges, used_edges, deadline)
    
    # Sort by step size for deterministic order
    circulant_cycles.sort(key=lambda c: abs(c[1] - c[0]) if len(c) > 1 else 0)
    
    for cycle in circulant_cycles:
        if time.time() > deadline - 10:
            break
        valid = True
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            edge = _edge_key(u, v)
            if edge in used_edges:
                valid = False
                break
            used_edges.add(edge)
        if valid:
            cycles.append(cycle)
    
    # Step 2: Use backtracking to find additional Hamiltonian cycles
    remaining_ham = 10 - len(cycles)
    for i in range(remaining_ham):
        if time.time() > deadline - 10:
            break
        
        cycle = None
        time_per_start = max(0.5, (deadline - 10 - time.time()) / n)
        for start in range(n):
            if time.time() > deadline - 10:
                break
            sub_deadline = min(deadline - 10, time.time() + time_per_start)
            cycle = _find_hamiltonian_cycle(adj, n, used_edges, sub_deadline, start)
            if cycle is not None:
                break
        
        if cycle is None:
            break
        
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            edge = _edge_key(u, v)
            used_edges.add(edge)
        
        cycles.append(cycle)
    
    # Step 3: If we haven't reached 10, try to find smaller edge-disjoint cycles
    if len(cycles) < 10 and time.time() < deadline - 5:
        extra_cycles = _find_non_hamiltonian_cycles(adj, n, used_edges, deadline - 5, min_len=3, max_len=7)
        for cycle in extra_cycles:
            if len(cycles) >= 10:
                break
            # Verify all edges are still unused
            valid = True
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                edge = _edge_key(u, v)
                if edge in used_edges:
                    valid = False
                    break
                used_edges.add(edge)
            if valid:
                cycles.append(cycle)
    
    # Ensure we have at least one cycle (fallback)
    if not cycles and time.time() < deadline:
        cycle = _find_hamiltonian_cycle(adj, n, used_edges, deadline, 0)
        if cycle is not None:
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                edge = _edge_key(u, v)
                used_edges.add(edge)
            cycles.append(cycle)
    
    # Cap at 10 cycles
    cycles = cycles[:10]
    
    # Convert to lists of Python ints
    result = []
    for cycle in cycles:
        result.append([int(v) for v in cycle])
    
    return result
# EVOLVE-BLOCK-END
