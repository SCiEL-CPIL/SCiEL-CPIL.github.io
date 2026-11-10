"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, product

def _cycle_lengths(adj, n):
    """Find all distinct simple cycle lengths in graph."""
    lengths = set()
    
    def dfs(start, node, visited, path_len):
        if path_len >= 4 and node == start:
            lengths.add(path_len)
            return
        if path_len > 11:  # max cycle length in 12-vertex graph
            return
        for nb in adj[node]:
            if nb == start and path_len >= 3:
                lengths.add(path_len)
                return
            if nb not in visited and nb > start:  # avoid duplicates by ordering
                visited.add(nb)
                dfs(start, nb, visited, path_len + 1)
                visited.remove(nb)
    
    for s in range(n):
        dfs(s, s, {s}, 0)
    
    return lengths

def _validate(edges, n=12):
    """Check all constraints."""
    if not edges:
        return False
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if not (isinstance(u, int) and isinstance(v, int) and not isinstance(u, bool) and not isinstance(v, bool)):
            return False
        if u == v:
            return False
        key = (min(u,v), max(u,v))
        if key in edge_set:
            return False
        edge_set.add(key)
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        adj[u].add(v)
        adj[v].add(u)
    
    # Minimum degree >= 3
    for i in range(n):
        if len(adj[i]) < 3:
            return False
    
    # Must contain a cycle (check if graph has more edges than n-1 in any component, 
    # but simpler: just check girth >= 4 implies no triangles, and we need at least one cycle)
    # Triangle-free check
    for i in range(n):
        for j in adj[i]:
            if j > i:
                for k in adj[j]:
                    if k != i and k in adj[i]:
                        return False
    
    # Must contain at least one cycle
    # Use DFS to detect cycle
    visited = [False] * n
    def has_cycle(u, parent):
        visited[u] = True
        for v in adj[u]:
            if not visited[v]:
                if has_cycle(v, u):
                    return True
            elif v != parent:
                return True
        return False
    
    for i in range(n):
        if not visited[i]:
            if has_cycle(i, -1):
                break
    else:
        return False  # No cycle found
    
    return True

def _distinct_cycle_count(edges, n=12):
    """Count distinct cycle lengths."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    lengths = _cycle_lengths(adj, n)
    return len(lengths)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    
    n = 12
    best_edges = None
    best_count = 100
    
    # Pre-computed good candidate: try to find graph with 1 distinct cycle length
    # A graph where all cycles are 4-cycles would be ideal.
    # Consider a graph that's a union of disjoint 4-cycles? But min degree 3 requires more.
    
    # Let's try: two copies of K_{3,3} connected appropriately? No, K_{3,3} has 6 vertices.
    # K_{3,3} has cycles of length 4 and 6.
    
    # Try: a 4-regular bipartite graph on 12 vertices (6+6)
    # If we can make it so all cycles are length 4, that's 1 distinct length.
    # But in a bipartite graph with min degree 3 on 6+6, there will be longer cycles.
    
    # Let's do a targeted search with time limit
    random.seed(42)
    
    # Strategy: start from known good graphs and modify
    # Known: prism graph gives {4,6,12}
    
    # Try to construct a graph with only 4-cycles and 6-cycles, or just 4-cycles.
    
    # Approach: random bipartite graphs with parts of size 6, min degree 3
    # Check distinct cycle lengths
    
    all_vertices = list(range(n))
    
    # First try some structured candidates
    candidates = []
    
    # Candidate 1: Two K_{3,3} connected by a perfect matching (bipartite 6+6)
    # Part A: 0,1,2,3,4,5; Part B: 6,7,8,9,10,11
    # K_{3,3} on {0,1,2} x {6,7,8} and K_{3,3} on {3,4,5} x {9,10,11}
    # Plus matching 0-9, 1-10, 2-11
    edges1 = []
    for a in [0,1,2]:
        for b in [6,7,8]:
            edges1.append((a,b))
    for a in [3,4,5]:
        for b in [9,10,11]:
            edges1.append((a,b))
    for a, b in [(0,9),(1,10),(2,11)]:
        edges1.append((a,b))
    candidates.append(edges1)
    
    # Candidate 2: Complete bipartite K_{3,3} + K_{3,3} with different matching
    edges2 = []
    for a in [0,1,2]:
        for b in [6,7,8]:
            edges2.append((a,b))
    for a in [3,4,5]:
        for b in [9,10,11]:
            edges2.append((a,b))
    for a, b in [(0,10),(1,11),(2,9)]:
        edges2.append((a,b))
    candidates.append(edges2)
    
    # Candidate 3: A 4-regular bipartite graph - use a circulant-like structure
    # Part A: 0-5, Part B: 6-11
    # Connect i in A to (i+0)%6, (i+1)%6, (i+2)%6, (i+3)%6 in B
    edges3 = []
    for i in range(6):
        for offset in [0, 1, 2, 3]:
            j = (i + offset) % 6
            edges3.append((i, 6+j))
    candidates.append(edges3)
    
    # Candidate 4: Different 4-regular bipartite
    edges4 = []
    for i in range(6):
        for offset in [0, 1, 2, 4]:
            j = (i + offset) % 6
            edges4.append((i, 6+j))
    candidates.append(edges4)
    
    # Evaluate candidates
    for edges in candidates:
        if time.time() > deadline:
            break
        if _validate(edges, n):
            cnt = _distinct_cycle_count(edges, n)
            if cnt < best_count:
                best_count = cnt
                best_edges = [list(e) for e in sorted(edges)]
    
    # Random search
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        # Random bipartite graph with parts 0-5 and 6-11
        # Each edge included with some probability, then ensure min degree 3
        edges = set()
        
        # Start with a random 3-regular bipartite base
        # Use random permutation-based construction
        perm1 = list(range(6))
        random.shuffle(perm1)
        perm2 = list(range(6))
        random.shuffle(perm2)
        perm3 = list(range(6))
        random.shuffle(perm3)
        
        for i in range(6):
            edges.add((i, 6 + perm1[i]))
            edges.add((i, 6 + perm2[i]))
            edges.add((i, 6 + perm3[i]))
        
        # Add random extra edges to make it more connected
        extra = random.randint(0, 12)
        all_possible = [(i, 6+j) for i in range(6) for j in range(6)]
        remaining = [e for e in all_possible if e not in edges]
        random.shuffle(remaining)
        for e in remaining[:extra]:
            edges.add(e)
        
        edge_list = [list(e) for e in sorted(edges)]
        
        if not _validate(edge_list, n):
            continue
        
        cnt = _distinct_cycle_count(edge_list, n)
        if cnt < best_count:
            best_count = cnt
            best_edges = edge_list
            if best_count <= 1:
                break
        
        # Occasionally try non-bipartite (but triangle-free) graphs
        if attempts % 10 == 0:
            # Random triangle-free graph: use a random 3-regular graph and check
            # This is harder, so skip for now
            pass
    
    if best_edges is None:
        # Fallback: return the original prism graph
        es = set()
        for i in range(6):
            for u, v in [(i, (i+1)%6), (i+6, (i+1)%6+6), (i, i+6)]:
                es.add(tuple(sorted((u,v))))
        best_edges = [list(e) for e in sorted(es)]
    
    # Ensure proper types
    result = []
    for e in best_edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END
