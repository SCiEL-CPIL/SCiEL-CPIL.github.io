"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

# Build adjacency structure
def _build_graph():
    n = P['n']
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    return n, adj, edge_set

N, ADJ, EDGE_SET = _build_graph()
ALL_VERTICES = frozenset(range(N))

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _canonical_cycle(cycle):
    """Return canonical form of a cycle for deduplication."""
    if len(cycle) < 3:
        return None
    # Try all rotations and both directions, pick lexicographically smallest
    n = len(cycle)
    best = None
    for i in range(n):
        rot = cycle[i:] + cycle[:i]
        if best is None or rot < best:
            best = rot
        rot_rev = list(reversed(cycle[i:] + cycle[:i]))
        if best is None or rot_rev < best:
            best = rot_rev
    return tuple(best)

def _find_all_cycles_on_vertex_set(vertex_set, adj):
    """Find all simple cycles that use exactly the given vertex set."""
    if len(vertex_set) < 3:
        return []
    
    vs = sorted(vertex_set)
    n_vs = len(vs)
    vs_set = set(vs)
    
    cycles = []
    seen = set()
    
    # We'll do DFS from the smallest vertex to avoid duplicates
    start = vs[0]
    
    # Path so far
    path = [start]
    visited = {start}
    
    def dfs(current, path, visited):
        if len(path) == n_vs:
            # Check if we can close the cycle
            if current in adj[start]:
                # Found a cycle
                canon = _canonical_cycle(path)
                if canon and canon not in seen:
                    seen.add(canon)
                    cycles.append(list(canon))
            return
        
        # Only continue if remaining vertices can be reached
        remaining = vs_set - visited
        if not remaining:
            return
        
        for nxt in sorted(adj[current]):
            if nxt in visited:
                # Only allow closing to start if we have all vertices
                if nxt == start and len(path) == n_vs:
                    continue  # handled above
                continue
            if nxt not in vs_set:
                continue
            # Heuristic: check connectivity of remaining vertices
            dfs(nxt, path + [nxt], visited | {nxt})
    
    dfs(start, path, visited)
    return cycles

def _find_cycles_with_edges(edges_available, adj, vertex_set):
    """Find cycles on vertex_set using only edges in edges_available."""
    vs = sorted(vertex_set)
    vs_set = set(vs)
    n_vs = len(vs)
    
    if n_vs < 3:
        return []
    
    # Filter adjacency to only available edges within vertex_set
    filtered_adj = {v: set() for v in vs}
    for u, v in edges_available:
        u2, v2 = min(u, v), max(u, v)
        if u2 in vs_set and v2 in vs_set:
            filtered_adj[u2].add(v2)
            filtered_adj[v2].add(u2)
    
    # Check each vertex has degree >= 2
    for v in vs:
        if len(filtered_adj[v]) < 2:
            return []
    
    cycles = []
    seen = set()
    start = vs[0]
    
    # DFS to find Hamiltonian cycles on this vertex set
    def dfs(current, path, visited):
        if len(path) == n_vs:
            if current in filtered_adj[start]:
                canon = _canonical_cycle(path)
                if canon and canon not in seen:
                    seen.add(canon)
                    cycles.append(list(canon))
            return
        
        for nxt in sorted(filtered_adj[current]):
            if nxt == start and len(path) < n_vs:
                continue
            if nxt in visited:
                continue
            if nxt not in vs_set:
                continue
            dfs(nxt, path + [nxt], visited | {nxt})
    
    dfs(start, [start], {start})
    return cycles

def _greedy_cycle_packing(adj, edge_set):
    """Greedy approach: repeatedly find a cycle and remove its edges."""
    cycles = []
    used_edges = set()
    available_edges = set(edge_set)
    
    max_cycles = 10
    deadline = time.time() + 55  # Leave 5s margin
    
    while len(cycles) < max_cycles and time.time() < deadline:
        # Try to find a cycle using only available edges
        # Start with small cycles (triangles, 4-cycles) for efficiency
        found = False
        
        # Try triangles first
        for i in range(N):
            if time.time() > deadline:
                break
            for j in adj[i]:
                if j <= i:
                    continue
                if _edge_key(i, j) not in available_edges:
                    continue
                for k in adj[j]:
                    if k <= j:
                        continue
                    if _edge_key(j, k) not in available_edges:
                        continue
                    if _edge_key(i, k) not in available_edges:
                        continue
                    # Triangle i-j-k
                    cycle = [i, j, k]
                    cycles.append(cycle)
                    used_edges.add(_edge_key(i, j))
                    used_edges.add(_edge_key(j, k))
                    used_edges.add(_edge_key(i, k))
                    available_edges -= used_edges & {_edge_key(i, j), _edge_key(j, k), _edge_key(i, k)}
                    found = True
                    break
                if found:
                    break
            if found:
                break
        
        if found:
            continue
        
        # Try 4-cycles
        found = False
        for i in range(N):
            if time.time() > deadline:
                break
            for j in sorted(adj[i]):
                if j <= i:
                    continue
                if _edge_key(i, j) not in available_edges:
                    continue
                for k in sorted(adj[j]):
                    if k <= j:
                        continue
                    if _edge_key(j, k) not in available_edges:
                        continue
                    for l in sorted(adj[k]):
                        if l <= k:
                            continue
                        if _edge_key(k, l) not in available_edges:
                            continue
                        if _edge_key(i, l) not in available_edges:
                            continue
                        # Check no repeated vertices
                        if len({i, j, k, l}) < 4:
                            continue
                        cycle = [i, j, k, l]
                        cycles.append(cycle)
                        e1 = _edge_key(i, j)
                        e2 = _edge_key(j, k)
                        e3 = _edge_key(k, l)
                        e4 = _edge_key(i, l)
                        used_edges.add(e1)
                        used_edges.add(e2)
                        used_edges.add(e3)
                        used_edges.add(e4)
                        available_edges -= {e1, e2, e3, e4}
                        found = True
                        break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        
        if found:
            continue
        
        # Try 5-cycles
        found = False
        for i in range(N):
            if time.time() > deadline:
                break
            for j in sorted(adj[i]):
                if j <= i:
                    continue
                if _edge_key(i, j) not in available_edges:
                    continue
                for k in sorted(adj[j]):
                    if k <= j:
                        continue
                    if _edge_key(j, k) not in available_edges:
                        continue
                    for l in sorted(adj[k]):
                        if l <= k:
                            continue
                        if _edge_key(k, l) not in available_edges:
                            continue
                        for m in sorted(adj[l]):
                            if m <= l:
                                continue
                            if _edge_key(l, m) not in available_edges:
                                continue
                            if _edge_key(i, m) not in available_edges:
                                continue
                            if len({i, j, k, l, m}) < 5:
                                continue
                            cycle = [i, j, k, l, m]
                            cycles.append(cycle)
                            e1 = _edge_key(i, j)
                            e2 = _edge_key(j, k)
                            e3 = _edge_key(k, l)
                            e4 = _edge_key(l, m)
                            e5 = _edge_key(i, m)
                            used_edges.update([e1, e2, e3, e4, e5])
                            available_edges -= {e1, e2, e3, e4, e5}
                            found = True
                            break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        
        if not found:
            break
    
    return cycles

def _validate_result(cycles):
    """Check all constraints are satisfied."""
    if not cycles:
        return False
    if len(cycles) > 10:
        return False
    
    edge_usage = defaultdict(int)
    vertex_sets = []
    
    for cycle in cycles:
        if not isinstance(cycle, (list, tuple)):
            return False
        if len(cycle) < 3:
            return False
        # Check distinct vertices
        if len(set(cycle)) != len(cycle):
            return False
        # Check all are ints
        for v in cycle:
            if not isinstance(v, int) or isinstance(v, bool):
                return False
        if not all(0 <= v < N for v in cycle):
            return False
        
        vs = frozenset(cycle)
        vertex_sets.append(vs)
        
        # Check edges exist
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            ek = _edge_key(u, v)
            if ek not in EDGE_SET:
                return False
            edge_usage[ek] += 1
    
    # All cycles must use same vertex set
    if len(set(vertex_sets)) != 1:
        return False
    
    # No edge used more than once
    for ek, count in edge_usage.items():
        if count > 1:
            return False
    
    return True

def construct():
    deadline = time.time() + 58
    
    # Strategy: We need cycles all on the same vertex set.
    # The full vertex set is 21 vertices. Let's try to find edge-disjoint Hamiltonian cycles.
    # But Hamiltonian cycles on 21 vertices are expensive. Let's try smaller subsets.
    
    # First, let's try the full 21-vertex set with a smarter approach
    # Actually, let's think about this differently.
    
    # For a set of k vertices, the number of edges in a cycle is k.
    # With 21 vertices, we have 78 edges total.
    # Max cycles if all on 21 vertices: 78/21 = 3 cycles (using 63 edges)
    # If on smaller subsets, we can have more cycles but they must all use the same vertex set.
    
    # Wait - "All cycles must use exactly the same vertex set of size at least three"
    # So all cycles must be on the SAME vertex set. Let's say that set has size k.
    # Each cycle uses k edges. Total edges available in the induced subgraph on those k vertices.
    # Max cycles = floor(number of edges in induced subgraph / k)
    
    # To maximize, we want to find a vertex set where the induced subgraph has many edges
    # relative to its size.
    
    # Let's compute for various vertex set sizes
    best_cycles = []
    
    # Try full vertex set first
    full_vs = frozenset(range(N))
    # Find edge-disjoint Hamiltonian cycles
    # This is hard. Let's use a different approach.
    
    # Alternative: pick a specific vertex set and find many edge-disjoint cycles on it.
    # The key insight: if we use a small vertex set (like 3 or 4 vertices), we can pack more cycles
    # if the induced subgraph is dense.
    
    # For a triangle (3 vertices): 3 edges, 1 cycle per triangle
    # For a 4-clique: 6 edges, 4-cycles use 4 edges, so at most 1 cycle... 
    # Wait, we need edge-disjoint cycles ALL on the same vertex set.
    
    # Let me reconsider. If vertex set is {a,b,c}, we can only form cycles that are exactly
    # the triangle a-b-c-a. That's 1 cycle using all 3 edges.
    
    # If vertex set is {a,b,c,d} and it's a K4 (6 edges), we can form:
    # - One 4-cycle using 4 edges
    # - Or two triangles? No, triangles use 3 vertices, not all 4.
    # The cycle must use ALL vertices in the set.
    
    # So for a k-vertex set, each cycle is a Hamiltonian cycle on those k vertices, using k edges.
    # Max cycles = floor(E(k) / k) where E(k) is the number of edges in the induced subgraph.
    
    # For the full 21-vertex set with 78 edges: max 3 cycles (78/21 = 3.71)
    # For a 14-vertex set with, say, 50 edges: max 3 cycles
    # For a 7-vertex set with 15 edges: max 2 cycles
    # For a 6-vertex set with 12 edges (K6): max 2 cycles
    # For a 5-vertex set with 10 edges (K5): max 2 cycles
    # For a 4-vertex set with 6 edges (K4): max 1 cycle
    # For a 3-vertex set with 3 edges: max 1 cycle
    
    # So the maximum possible is limited. Let's find the best vertex set.
    
    # Actually wait - I need to re-read: "each represented by a list of distinct vertices"
    # and "All cycles must use exactly the same vertex set"
    
    # So yes, each cycle must be a Hamiltonian cycle on the chosen vertex set.
    
    # Let's search for vertex sets that allow 3+ edge-disjoint Hamiltonian cycles.
    
    # Start with the full vertex set
    n = N
    vs = list(range(n))
    vs_set = set(vs)
    
    # Find Hamiltonian cycles on the full vertex set that are edge-disjoint
    # Use backtracking with pruning
    
    def find_edge_disjoint_hamiltonian_cycles(vertex_list, max_cycles=10):
        """Find edge-disjoint Hamiltonian cycles on vertex_list."""
        vl = sorted(vertex_list)
        vl_set = set(vl)
        k = len(vl)
        if k < 3:
            return []
        
        # Build filtered adjacency
        fadj = {v: set() for v in vl}
        for u, v in EDGE_SET:
            if u in vl_set and v in vl_set:
                fadj[u].add(v)
                fadj[v].add(u)
        
        # Check min degree
        for v in vl:
            if len(fadj[v]) < 2:
                return []
        
        cycles = []
        used_edges = set()
        deadline_local = time.time() + 50
        
        for target_cycle in range(max_cycles):
            if time.time() > deadline_local:
                break
            
            # Find one more Hamiltonian cycle avoiding used_edges
            found = None
            start = vl[0]
            
            def dfs_ham(current, path, visited):
                if time.time() > deadline_local:
                    return None
                if len(path) == k:
                    # Check closing edge
                    if start in fadj[current] and _edge_key(current, start) not in used_edges:
                        return list(path)
                    return None
                
                # Pruning: check if remaining vertices can form a path
                remaining = vl_set - visited
                for nxt in sorted(fadj[current]):
                    if nxt in visited:
                        if nxt == start and len(path) == k:
                            continue
                        continue
                    ek = _edge_key(current, nxt)
                    if ek in used_edges:
                        continue
                    result = dfs_ham(nxt, path + [nxt], visited | {nxt})
                    if result is not None:
                        return result
                return None
            
            found = dfs_ham(start, [start], {start})
            if found:
                cycles.append(found)
                for i in range(k):
                    ek = _edge_key(found[i], found[(i + 1) % k])
                    used_edges.add(ek)
            else:
                break
        
        return cycles
    
    # Try full vertex set
    result = find_edge_disjoint_hamiltonian_cycles(list(range(N)), 3)
    if len(result) >= 1:
        best_cycles = result
    
    if time.time() > deadline:
        # Return what we have
        if best_cycles:
            return [list(c) for c in best_cycles]
        return [list(range(N))]
    
    # If full set didn't work well, try subsets
    # Try all subsets of size 14 down to 7
    # This is expensive, so be selective
    
    # Actually, let's try a random search approach for better vertex sets
    random.seed(42)
    
    for trial in range(100):
        if time.time() > deadline:
            break
        
        # Try random vertex sets of various sizes
        for k in [14, 12, 10, 9, 8, 7, 6, 5]:
            if time.time() > deadline:
                break
            
            # Pick random subset
            subset = random.sample(range(N), k)
            subset_set = set(subset)
            
            # Count edges in induced subgraph
            edge_count = sum(1 for u, v in EDGE_SET if u in subset_set and v in subset_set)
            
            # Max possible cycles
            max_possible = edge_count // k
            if max_possible <= len(best_cycles):
                continue
            
            # Try to find edge-disjoint Hamiltonian cycles on this subset
            result = find_edge_disjoint_hamiltonian_cycles(subset, min(max_possible, 10))
            if len(result) > len(best_cycles):
                best_cycles = result
    
    # Validate and return
    if best_cycles and _validate_result(best_cycles):
        return [list(c) for c in best_cycles]
    
    # Fallback: return a single valid cycle (the full vertex set as a cycle if valid, or find one)
    # The seed returns list(range(21)) which is a 21-cycle. Let's verify it's valid.
    fallback = list(range(N))
    if _validate_result([fallback]):
        return [fallback]
    
    # Find any valid cycle
    for i in range(N):
        for j in ADJ[i]:
            if j > i:
                for k in ADJ[j]:
                    if k > j and _edge_key(i, k) in EDGE_SET:
                        return [[i, j, k]]
    
    return [list(range(N))]

# EVOLVE-BLOCK-END
