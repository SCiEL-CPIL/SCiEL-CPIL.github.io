"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_cycles_on_vertex_set(vertices, adj, used_edges, max_cycles, deadline):
    """
    Find up to max_cycles edge-disjoint cycles all using exactly the vertex set 'vertices'.
    Returns a list of cycles (each a list of vertices) or None if fewer than max_cycles found.
    Uses backtracking.
    """
    vertex_set = set(vertices)
    n = len(vertices)
    # Map vertex indices to local indices for speed
    vlist = sorted(vertices)
    v_to_local = {v: i for i, v in enumerate(vlist)}
    
    # Local adjacency
    local_adj = [set() for _ in range(n)]
    for u in vlist:
        for v in adj[u]:
            if v in vertex_set:
                local_adj[v_to_local[u]].add(v_to_local[v])
    
    # We need to find max_cycles edge-disjoint cycles on n vertices
    # Each cycle uses n edges, all distinct
    # Total edges available on this vertex set:
    available_edges = set()
    for i in range(n):
        for j in local_adj[i]:
            if i < j:
                available_edges.add((i, j))
    
    # A Hamiltonian cycle on n vertices uses n edges.
    # We need k disjoint Hamiltonian cycles.
    # Try to find them greedily with backtracking.
    
    found_cycles = []
    used_local_edges = set()
    
    def _find_one_cycle(visited, current, start, path, remaining_edges):
        """Find a Hamiltonian cycle starting from 'start', current node is 'current'."""
        if len(path) == n:
            # Check if we can close the cycle
            if start in local_adj[current]:
                edge = (min(start, current), max(start, current))
                if edge not in used_local_edges:
                    return path[:]
            return None
        
        for nxt in local_adj[current]:
            if nxt not in visited:
                edge = (min(current, nxt), max(current, nxt))
                if edge not in used_local_edges:
                    visited.add(nxt)
                    path.append(nxt)
                    result = _find_one_cycle(visited, nxt, start, path, remaining_edges)
                    if result is not None:
                        return result
                    path.pop()
                    visited.remove(nxt)
        return None
    
    def _try_find_cycles(depth, target):
        if depth == target:
            return True
        if time.time() > deadline:
            return False
        
        # Try all possible starting vertices (use symmetry: fix start to 0)
        start = 0
        visited = {start}
        path = [start]
        
        result = _find_one_cycle(visited, start, start, path, n)
        if result is None:
            # Try other orderings - actually let's try different permutations of the remaining vertices
            # This is expensive. Let's try a different approach.
            return False
        
        # Convert local cycle back to original
        cycle = [vlist[v] for v in result]
        # Mark edges as used
        for i in range(n):
            u, v = result[i], result[(i+1) % n]
            used_local_edges.add((min(u, v), max(u, v)))
        
        found_cycles.append(cycle)
        
        if _try_find_cycles(depth + 1, target):
            return True
        
        # Backtrack
        found_cycles.pop()
        for i in range(n):
            u, v = result[i], result[(i+1) % n]
            used_local_edges.discard((min(u, v), max(u, v)))
        
        return False
    
    # Try to find 'max_cycles' disjoint cycles
    if _try_find_cycles(0, max_cycles):
        return found_cycles
    return None

def _find_hamiltonian_cycles(vertices, adj, k, deadline):
    """
    Find k edge-disjoint Hamiltonian cycles on the given vertex set.
    Uses backtracking search.
    """
    vertex_set = set(vertices)
    n = len(vertices)
    vlist = sorted(vertices)
    v_to_local = {v: i for i, v in enumerate(vlist)}
    
    # Build local adjacency
    local_adj = [set() for _ in range(n)]
    for u in vlist:
        for v in adj[u]:
            if v in vertex_set:
                local_adj[v_to_local[u]].add(v_to_local[v])
    
    # Total edges on this vertex set
    all_edges_local = set()
    for i in range(n):
        for j in local_adj[i]:
            if i < j:
                all_edges_local.add((i, j))
    
    # We need k edge-disjoint Hamiltonian cycles
    # Each uses n edges, so we need k*n distinct edges
    if len(all_edges_local) < k * n:
        return None
    
    used_edges = set()
    found = []
    
    def find_cycle(visited, current, start, path):
        if len(path) == n:
            edge = (min(start, current), max(start, current))
            if start in local_adj[current] and edge not in used_edges:
                return path[:]
            return None
        for nxt in local_adj[current]:
            if nxt not in visited:
                edge = (min(current, nxt), max(current, nxt))
                if edge not in used_edges:
                    visited.add(nxt)
                    path.append(nxt)
                    res = find_cycle(visited, nxt, start, path)
                    if res is not None:
                        return res
                    path.pop()
                    visited.remove(nxt)
        return None
    
    def backtrack(idx, target):
        if idx == target:
            return True
        if time.time() > deadline:
            return False
        
        # Try to find a cycle. To reduce symmetry, fix the first vertex to 0.
        start = 0
        visited = {start}
        path = [start]
        
        # Try all permutations via DFS
        cyc = find_cycle(visited, start, start, path)
        if cyc is None:
            return False
        
        # Record edges
        for i in range(n):
            u, v = cyc[i], cyc[(i+1) % n]
            used_edges.add((min(u, v), max(u, v)))
        found.append([vlist[v] for v in cyc])
        
        if backtrack(idx + 1, target):
            return True
        
        # Undo
        found.pop()
        for i in range(n):
            u, v = cyc[i], cyc[(i+1) % n]
            used_edges.discard((min(u, v), max(u, v)))
        return False
    
    if backtrack(0, k):
        return found
    return None

def construct():
    deadline = time.time() + 80  # 80s deadline with margin
    
    adj = _build_adj()
    n = P['n']
    
    # The full vertex set
    full_set = list(range(n))
    
    # Try to find maximum number of edge-disjoint Hamiltonian cycles on full set
    # Max possible: floor(total_edges / n)
    total_edges = len(P['edges'])
    max_possible = total_edges // n  # 75 // 21 = 3
    
    # Try from max_possible down
    for k in range(max_possible, 0, -1):
        result = _find_hamiltonian_cycles(full_set, adj, k, deadline)
        if result is not None:
            # Convert to proper format
            cycles = []
            for cyc in result:
                cycles.append([int(v) for v in cyc])
            return cycles
    
    # Fallback: try smaller vertex sets
    # Try subsets of size 3, 4, etc.
    # For size 3, a cycle is just a triangle
    for size in range(3, min(n, 8)):
        for vertices in combinations(range(n), size):
            if time.time() > deadline:
                break
            vset = set(vertices)
            # Check if all consecutive pairs are edges (for a cycle on these vertices)
            # Actually we need to find ANY cycle on this vertex set
            # For size 3: just check if it's a triangle
            if size == 3:
                a, b, c = vertices
                if (a, b) in [(min(u,v), max(u,v)) for u, v in P['edges']] or (b, a) in [(min(u,v), max(u,v)) for u, v in P['edges']]:
                    pass  # This is getting complicated
                # Simpler: check edges
                edge_set = set()
                for u, v in P['edges']:
                    edge_set.add((min(u,v), max(u,v)))
                if (min(a,b), max(a,b)) in edge_set and (min(b,c), max(b,c)) in edge_set and (min(a,c), max(a,c)) in edge_set:
                    # It's a triangle, so we can have 1 cycle
                    return [[a, b, c]]
            elif size == 4:
                # Check if there's a 4-cycle
                a, b, c, d = vertices
                edge_set = set()
                for u, v in P['edges']:
                    edge_set.add((min(u,v), max(u,v)))
                # Try all 3 pairings for 4-cycles: (a-b-c-d-a), (a-b-d-c-a), (a-c-b-d-a)
                for perm in [(a,b,c,d), (a,b,d,c), (a,c,b,d)]:
                    if all((min(perm[i], perm[(i+1)%4]), max(perm[i], perm[(i+1)%4])) in edge_set for i in range(4)):
                        return [list(perm)]
    
    # Absolute fallback: return the full vertex set as one cycle if it's a Hamiltonian cycle
    # Check if 0,1,2,...,20 is a cycle
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u,v), max(u,v)))
    full_cycle = list(range(n))
    valid = True
    for i in range(n):
        u, v = full_cycle[i], full_cycle[(i+1)%n]
        if (min(u,v), max(u,v)) not in edge_set:
            valid = False
            break
    if valid:
        return [full_cycle]
    
    # Last resort: find any cycle
    # Just return a triangle if one exists
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u,v), max(u,v)))
    for a in range(n):
        for b in range(a+1, n):
            if (a, b) not in edge_set:
                continue
            for c in range(b+1, n):
                if (b, c) in edge_set and (a, c) in edge_set:
                    return [[a, b, c]]
    
    # Should never get here
    return [[0, 1, 2]]
# EVOLVE-BLOCK-END
