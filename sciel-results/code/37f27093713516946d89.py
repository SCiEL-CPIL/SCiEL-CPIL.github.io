"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict
from itertools import combinations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

_START_TIME = time.time()
DEADLINE = _START_TIME + 80.0  # safety margin before 90s kill

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def build_adjacency(edges):
    n = P['n']
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def check_hamiltonian_necessary_conditions(adj, n):
    """Check necessary conditions for Hamiltonian cycle existence."""
    for v in range(n):
        if len(adj[v]) < 2:
            return False
    # Check connectivity
    visited = [False] * n
    visited[0] = True
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                stack.append(v)
    if not all(visited):
        return False
    return True

def find_hamiltonian_cycles(adj, n, start=0, deadline=None, edge_set=None, max_cycles=20):
    """
    Find multiple Hamiltonian cycles using Warnsdorff's rule + connectivity pruning.
    Returns a list of cycles (each is a list of vertices).
    """
    if deadline is not None and time.time() > deadline:
        return []
    
    if not check_hamiltonian_necessary_conditions(adj, n):
        return []
    
    if edge_set is None:
        edge_set = set()
        for u in range(n):
            for v in adj[u]:
                edge_set.add(_edge_key(u, v))
    
    sys.setrecursionlimit(10000)
    adj_list = {v: sorted(adj[v]) for v in range(n)}
    
    found_cycles = []
    used_edge_sets = []  # To avoid duplicate cycles
    
    def get_unvisited_neighbors(v):
        """Get unvisited neighbors of v, prioritized by Warnsdorff's rule."""
        candidates = []
        for nb in adj_list[v]:
            if not visited[nb]:
                # Count unvisited neighbors of nb (Warnsdorff)
                cnt = 0
                for nb2 in adj_list[nb]:
                    if not visited[nb2]:
                        cnt += 1
                candidates.append((cnt, nb))
        candidates.sort(key=lambda x: x[0])
        return [nb for _, nb in candidates]
    
    def is_connected_unvisited(unvisited_set, current):
        """Check if unvisited vertices (plus current) form a connected subgraph."""
        if not unvisited_set:
            return True
        visited_local = set()
        stack = [current]
        visited_local.add(current)
        while stack:
            u = stack.pop()
            for v in adj_list[u]:
                if v in unvisited_set and v not in visited_local:
                    visited_local.add(v)
                    stack.append(v)
        return visited_local >= (unvisited_set | {current})
    
    def dfs(current, path_len, path):
        if deadline is not None and time.time() > deadline:
            return False
        if path_len == n:
            if _edge_key(current, start) in edge_set:
                cycle = list(path)
                # Check for duplicates
                cycle_edges = frozenset(_edge_key(cycle[i], cycle[(i+1) % n]) for i in range(n))
                for existing in used_edge_sets:
                    if cycle_edges == existing:
                        return False
                found_cycles.append(cycle)
                used_edge_sets.append(cycle_edges)
                return len(found_cycles) >= max_cycles
            return False
        
        # Connectivity pruning
        unvisited = set()
        for v in range(n):
            if not visited[v]:
                unvisited.add(v)
        if unvisited:
            if not is_connected_unvisited(unvisited, current):
                return False
        
        # Degree pruning: each unvisited vertex must have >= 2 available neighbors
        for v in unvisited:
            avail = 0
            for nb in adj_list[v]:
                if not visited[nb] or nb == current:
                    avail += 1
            if avail < 2:
                return False
        
        # Get next candidates with Warnsdorff ordering
        candidates = get_unvisited_neighbors(current)
        for nxt in candidates:
            visited[nxt] = True
            path.append(nxt)
            result = dfs(nxt, path_len + 1, path)
            path.pop()
            visited[nxt] = False
            if result:
                return True
        return False
    
    # Try multiple starting vertices to find diverse cycles
    for start_v in range(n):
        if deadline is not None and time.time() > deadline:
            break
        if len(found_cycles) >= max_cycles:
            break
        
        visited = [False] * n
        visited[start_v] = True
        path = [start_v]
        dfs(start_v, 1, path)
    
    return found_cycles

def select_edge_disjoint_cycles(cycles, n, max_count=3):
    """
    Select up to max_count edge-disjoint cycles from the list.
    Uses greedy selection with conflict checking.
    """
    selected = []
    used_edges = set()
    
    # Sort cycles by some heuristic (e.g., edge overlap with already selected)
    # Try all permutations for small sets, or use greedy
    for cycle in cycles:
        if len(selected) >= max_count:
            break
        cycle_edges = set()
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            key = _edge_key(u, v)
            cycle_edges.add(key)
        
        # Check if this cycle conflicts with already selected ones
        if cycle_edges & used_edges:
            continue
        
        selected.append(cycle)
        used_edges |= cycle_edges
    
    return selected

def find_sequential_cycles(adj, n, deadline=None, max_cycles=3):
    """
    Sequentially find edge-disjoint Hamiltonian cycles.
    """
    if deadline is not None and time.time() > deadline:
        return []
    
    cycles = []
    used_edges = set()
    
    for cycle_idx in range(max_cycles):
        if deadline is not None and time.time() > deadline:
            break
        
        # Build subgraph of unused edges
        remaining_edges = []
        for u in range(n):
            for v in adj[u]:
                key = _edge_key(u, v)
                if key not in used_edges:
                    remaining_edges.append([u, v])
        
        # Check if remaining graph can support a Hamiltonian cycle
        rem_adj = build_adjacency(remaining_edges)
        if not check_hamiltonian_necessary_conditions(rem_adj, n):
            break
        
        # Try to find Hamiltonian cycle
        cycle = None
        for start_v in range(n):
            if deadline is not None and time.time() > deadline:
                break
            
            # Build edge set for this subgraph
            rem_edge_set = set()
            for e in remaining_edges:
                rem_edge_set.add(_edge_key(e[0], e[1]))
            
            # Use a simple backtracking search
            sys.setrecursionlimit(10000)
            rem_adj_list = {v: sorted(rem_adj[v]) for v in range(n)}
            visited = [False] * n
            visited[start_v] = True
            path = [start_v]
            
            def dfs_simple(current, path_len):
                if deadline is not None and time.time() > deadline:
                    return None
                if path_len == n:
                    if _edge_key(current, start_v) in rem_edge_set:
                        return list(path)
                    return None
                
                # Simple pruning
                unvisited = set()
                for v in range(n):
                    if not visited[v]:
                        unvisited.add(v)
                
                for v in unvisited:
                    avail = 0
                    for nb in rem_adj_list[v]:
                        if not visited[nb] or nb == current:
                            avail += 1
                    if avail < 2:
                        return None
                
                # Warnsdorff ordering
                candidates = []
                for nb in rem_adj_list[current]:
                    if not visited[nb]:
                        cnt = 0
                        for nb2 in rem_adj_list[nb]:
                            if not visited[nb2]:
                                cnt += 1
                        candidates.append((cnt, nb))
                candidates.sort(key=lambda x: x[0])
                
                for _, nxt in candidates:
                    visited[nxt] = True
                    path.append(nxt)
                    result = dfs_simple(nxt, path_len + 1)
                    path.pop()
                    visited[nxt] = False
                    if result is not None:
                        return result
                return None
            
            cycle = dfs_simple(start_v, 1)
            if cycle is not None:
                break
        
        if cycle is None:
            break
        
        # Mark edges as used
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            key = _edge_key(u, v)
            used_edges.add(key)
        
        cycles.append(cycle)
    
    return cycles

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build initial adjacency
    adj = build_adjacency(edges)
    
    # Build edge set
    all_edge_keys = set()
    for u, v in edges:
        all_edge_keys.add(_edge_key(u, v))
    
    # Strategy 1: Find multiple cycles and select edge-disjoint ones
    # This is more robust than sequential extraction
    
    # First, try to find a good set of cycles
    found_cycles = find_hamiltonian_cycles(adj, n, start=0, deadline=DEADLINE - 10.0, 
                                            edge_set=all_edge_keys, max_cycles=15)
    
    # Select edge-disjoint cycles
    selected = select_edge_disjoint_cycles(found_cycles, n, max_count=3)
    
    # If we got 3 cycles, we're done
    if len(selected) == 3:
        result = [[int(v) for v in c] for c in selected]
        return result
    
    # Strategy 2: Sequential extraction for remaining cycles
    if len(selected) < 3:
        # Build used edges from selected cycles
        used_edges = set()
        for cycle in selected:
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                key = _edge_key(u, v)
                used_edges.add(key)
        
        # Find additional cycles sequentially
        additional = find_sequential_cycles(adj, n, deadline=DEADLINE - 5.0, 
                                             max_cycles=3 - len(selected))
        
        # Filter out any that conflict
        for cycle in additional:
            if len(selected) >= 3:
                break
            cycle_edges = set()
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                key = _edge_key(u, v)
                cycle_edges.add(key)
            
            if cycle_edges & used_edges:
                continue
            
            selected.append(cycle)
            used_edges |= cycle_edges
    
    # Ensure we have at least 1 cycle
    if not selected:
        # Fallback: try to find any single Hamiltonian cycle
        cycle = find_hamiltonian_cycles(adj, n, start=0, deadline=DEADLINE - 2.0,
                                         edge_set=all_edge_keys, max_cycles=1)
        if cycle:
            selected = cycle
        else:
            # Last resort: return a trivial cycle (this should never happen)
            return [list(range(n))]
    
    # Convert to proper format
    result = [[int(v) for v in c] for c in selected]
    return result
# EVOLVE-BLOCK-END
