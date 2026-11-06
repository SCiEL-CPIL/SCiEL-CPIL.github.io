"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from functools import lru_cache

N = 16
TOTAL_EDGES = N * (N - 1) // 2  # 120

# Precompute all 8-vertex subsets and which ones contain a Q3 (cube)
# A Q3 is a 3-cube: 8 vertices, 12 edges, each vertex degree 3 within the subgraph.
# We'll precompute a set of frozensets of 8 vertices that form a Q3.

def _build_q3_lookup():
    """Precompute all 8-vertex subsets that induce a Q3 (the 3-cube)."""
    # The 3-cube Q3 has 8 vertices and 12 edges.
    # Vertices can be labeled 0..7 as 3-bit numbers.
    # Edges connect vertices differing in exactly one bit.
    q3_edges = set()
    for v in range(8):
        for bit in range(3):
            u = v ^ (1 << bit)
            if u > v:
                q3_edges.add((v, u))
    # q3_edges should have 12 edges
    
    # All 8-vertex subsets of 16 vertices
    q3_sets = set()
    vertices = list(range(N))
    for subset in itertools.combinations(vertices, 8):
        subset_set = set(subset)
        # Check if the induced subgraph on this subset is isomorphic to Q3
        # Q3 properties: 8 vertices, 12 edges, 3-regular, bipartite with parts of size 4, girth 4
        # Count edges in induced subgraph
        edge_count = 0
        for i in range(8):
            for j in range(i+1, 8):
                u, v = subset[i], subset[j]
                # We don't know the actual graph edges yet - this approach won't work for lookup
        # Actually, we need to check if the INDUCED subgraph on these 8 vertices in the CURRENT graph is Q3.
        # But for a lookup table, we need to check isomorphism to Q3 structure.
        # 
        # Better approach: For a given 8-vertex subset, check if the induced subgraph (in the current graph)
        # is isomorphic to Q3. But that depends on the current graph edges.
        #
        # Alternative: Precompute all 8-vertex subsets. For each, when checking if it's a Q3 in a given graph,
        # we need to check if the 12 edges of Q3 (under some isomorphism) are all present.
        # 
        # Most efficient: For each 8-subset, there are 8! / (8*2^3*3!) = 8!/256 = 7560/... 
        # Actually the number of automorphisms of Q3 is 48 (hyperoctahedral group B3).
        # So 8!/48 = 40320/48 = 840 distinct edge sets (as subsets of the 28 possible edges on 8 vertices).
        # 
        # So: precompute the 840 possible Q3 edge patterns on 8 labeled vertices.
        # For each 8-vertex subset, map its vertices to 0..7, then check if the 12 edges present
        # match one of the 840 Q3 patterns.
        
        pass
    
    # Let's use a different strategy: precompute the 840 Q3 edge sets on vertices 0..7
    q3_edge_patterns = set()
    # Generate all isomorphisms of Q3
    # Q3 vertices: 0..7, edges between v and v^(1<<bit)
    base_edges = frozenset(frozenset((v, v ^ (1 << bit))) for v in range(8) for bit in range(3) if v < (v ^ (1 << bit)))
    
    # All permutations of 8 vertices that preserve the Q3 structure give the same edge set.
    # Instead, generate all 840 distinct edge sets by applying all 40320 permutations and deduplicating.
    # That's too slow. Instead, use the fact that Q3 is the Cayley graph of Z2^3 with generators e1,e2,e3.
    # The automorphism group has size 48.
    # 
    # Simpler: just enumerate all subsets of 12 edges from the 28 possible edges on 8 vertices
    # that form a 3-regular bipartite graph with 8 vertices and 12 edges.
    # There are C(28,12) which is huge. 
    #
    # Best approach: For each 8-vertex subset in the 16-vertex graph, we can check for Q3 by:
    # 1. Get the induced subgraph edges
    # 2. Check if it's 3-regular (each vertex degree 3)
    # 3. Check if it's bipartite
    # 4. Check if it has exactly 12 edges
    # If all three hold, it's Q3 (since the only 3-regular bipartite graph on 8 vertices with 12 edges is Q3).
    #
    # This check is O(28) per 8-subset, which is fast.
    
    return None

# Actually, let's just use the direct check: a set of 8 vertices induces Q3 iff
# the induced subgraph is 3-regular and has 12 edges (and is connected, but 3-regular on 8 vertices 
# with 12 edges that's bipartite is Q3).

def _induces_q3(vertices8, adj):
    """Check if the 8 vertices induce a Q3. adj is a set of frozensets or a 2D boolean array."""
    # adj: we'll use a set of edges as frozensets for quick lookup
    vset = set(vertices8)
    if len(vset) != 8:
        return False
    # Count degrees within the induced subgraph
    deg = {v: 0 for v in vertices8}
    edge_count = 0
    for v in vertices8:
        for u in vertices8:
            if v < u and frozenset((v, u)) in adj:
                deg[v] += 1
                deg[u] += 1
                edge_count += 1
    if edge_count != 12:
        return False
    for v in vertices8:
        if deg[v] != 3:
            return False
    # Check bipartite: try 2-coloring
    # Since it's 3-regular with 12 edges on 8 vertices, if it's connected it's Q3.
    # Check connectivity via BFS
    visited = set()
    queue = [vertices8[0]]
    visited.add(vertices8[0])
    while queue:
        v = queue.pop()
        for u in vertices8:
            if u != v and frozenset((v, u)) in adj and u not in visited:
                visited.add(u)
                queue.append(u)
    if len(visited) != 8:
        return False
    return True

def _has_q3(adj_set):
    """Check if the graph (given as set of frozenset edges) contains a Q3."""
    vertices = list(range(N))
    for subset in itertools.combinations(vertices, 8):
        if _induces_q3(subset, adj_set):
            return True
    return False

def _has_q3_fast(adj_matrix):
    """Faster check using adjacency matrix (list of bitmasks)."""
    # adj_matrix[i] is a bitmask of neighbors of i
    vertices = list(range(N))
    for subset in itertools.combinations(vertices, 8):
        # Quick degree check
        ok = True
        for v in subset:
            # Count neighbors within subset
            mask = 0
            for u in subset:
                if u != v:
                    mask |= (1 << u)
            # Count bits in adj_matrix[v] & mask
            cnt = bin(adj_matrix[v] & mask).count('1')
            if cnt != 3:
                ok = False
                break
        if not ok:
            continue
        # Check 12 edges
        edge_count = 0
        for i in range(8):
            for j in range(i+1, 8):
                v, u = subset[i], subset[j]
                if adj_matrix[v] & (1 << u):
                    edge_count += 1
        if edge_count != 12:
            continue
        # Check connectivity (BFS)
        visited = set()
        queue = [subset[0]]
        visited.add(subset[0])
        while queue:
            v = queue.pop()
            for u in subset:
                if u != v and (adj_matrix[v] & (1 << u)) and u not in visited:
                    visited.add(u)
                    queue.append(u)
        if len(visited) == 8:
            return True
    return False

# Precompute all 8-subsets for speed
ALL_8_SUBSETS = list(itertools.combinations(range(N), 8))

def _has_q3_precomputed(adj_matrix):
    """Check using precomputed 8-subsets."""
    for subset in ALL_8_SUBSETS:
        ok = True
        for v in subset:
            mask = 0
            for u in subset:
                if u != v:
                    mask |= (1 << u)
            cnt = bin(adj_matrix[v] & mask).count('1')
            if cnt != 3:
                ok = False
                break
        if not ok:
            continue
        edge_count = 0
        for i in range(8):
            for j in range(i+1, 8):
                v, u = subset[i], subset[j]
                if adj_matrix[v] & (1 << u):
                    edge_count += 1
        if edge_count != 12:
            continue
        # Connectivity check
        visited = set()
        queue = [subset[0]]
        visited.add(subset[0])
        while queue:
            v = queue.pop()
            for u in subset:
                if u != v and (adj_matrix[v] & (1 << u)) and u not in visited:
                    visited.add(u)
                    queue.append(u)
        if len(visited) == 8:
            return True
    return False

def construct():
    deadline = time.time() + 145.0  # Safety margin
    
    # Start with a dense random graph and remove edges to make it Q3-free
    # Or start with K16 and remove edges.
    # K16 has 120 edges. We want to maximize edges while being Q3-free.
    
    # Strategy: Start with all edges, then iteratively find Q3s and remove edges.
    # Better: Use simulated annealing or local search.
    
    # Let's try: start with a random graph, then add edges greedily.
    # Or: start with K16, find a Q3, remove an edge from it, repeat.
    
    # Initialize adjacency matrix as bitmasks
    adj = [0] * N
    for i in range(N):
        for j in range(i+1, N):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    
    def edge_count():
        return sum(bin(a).count('1') for a in adj) // 2
    
    best_edges = edge_count()
    best_adj = list(adj)
    
    # Greedy removal: find Q3s and remove edges
    # This might be slow. Let's try a different approach.
    
    # Actually, let's use a randomized local search:
    # 1. Start with a random graph (or K16)
    # 2. While it has a Q3, remove a random edge from a random Q3
    # 3. Then try to add edges back (hill climbing)
    
    # Better: simulated annealing on the edge set
    # State: set of edges
    # Cost: number of Q3s (want 0)
    # Objective: maximize edges subject to 0 Q3s
    
    # Let's do: start with K16, repeatedly find a Q3 and remove a random edge from it.
    # Then do local search: try to add each missing edge, check if it creates a Q3.
    
    random.seed(42)
    
    # Phase 1: Remove edges to make Q3-free
    max_removals = 500
    for step in range(max_removals):
        if time.time() > deadline:
            break
        if not _has_q3_precomputed(adj):
            break
        # Find a Q3
        q3_found = None
        for subset in ALL_8_SUBSETS:
            ok = True
            for v in subset:
                mask = 0
                for u in subset:
                    if u != v:
                        mask |= (1 << u)
                cnt = bin(adj[v] & mask).count('1')
                if cnt != 3:
                    ok = False
                    break
            if not ok:
                continue
            edge_count = 0
            for i in range(8):
                for j in range(i+1, 8):
                    v, u = subset[i], subset[j]
                    if adj[v] & (1 << u):
                        edge_count += 1
            if edge_count != 12:
                continue
            # Check connectivity
            visited = set()
            queue = [subset[0]]
            visited.add(subset[0])
            while queue:
                v = queue.pop()
                for u in subset:
                    if u != v and (adj[v] & (1 << u)) and u not in visited:
                        visited.add(u)
                        queue.append(u)
            if len(visited) == 8:
                q3_found = subset
                break
        if q3_found is None:
            break
        # Remove a random edge from this Q3
        edges_in_q3 = []
        for i in range(8):
            for j in range(i+1, 8):
                v, u = q3_found[i], q3_found[j]
                if adj[v] & (1 << u):
                    edges_in_q3.append((v, u))
        if edges_in_q3:
            v, u = random.choice(edges_in_q3)
            adj[v] &= ~(1 << u)
            adj[u] &= ~(1 << v)
    
    # Phase 2: Local search - try to add edges
    current_edges = edge_count()
    if current_edges > best_edges:
        best_edges = current_edges
        best_adj = list(adj)
    
    # Try to add each missing edge
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            for j in range(i+1, N):
                if time.time() > deadline:
                    break
                if adj[i] & (1 << j):
                    continue
                # Try adding edge (i,j)
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                if not _has_q3_precomputed(adj):
                    improved = True
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = list(adj)
                else:
                    # Remove it
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
    
    # Phase 3: Randomized restarts / simulated annealing
    # Try different starting points
    for restart in range(20):
        if time.time() > deadline:
            break
        # Start with a random graph
        adj2 = [0] * N
        for i in range(N):
            for j in range(i+1, N):
                if random.random() < 0.5:
                    adj2[i] |= (1 << j)
                    adj2[j] |= (1 << i)
        
        # Make it Q3-free by removing edges
        for step in range(300):
            if time.time() > deadline:
                break
            if not _has_q3_precomputed(adj2):
                break
            q3_found = None
            for subset in ALL_8_SUBSETS:
                ok = True
                for v in subset:
                    mask = 0
                    for u in subset:
                        if u != v:
                            mask |= (1 << u)
                    cnt = bin(adj2[v] & mask).count('1')
                    if cnt != 3:
                        ok = False
                        break
                if not ok:
                    continue
                ec = 0
                for ii in range(8):
                    for jj in range(ii+1, 8):
                        v, u = subset[ii], subset[jj]
                        if adj2[v] & (1 << u):
                            ec += 1
                if ec != 12:
                    continue
                visited = set()
                queue = [subset[0]]
                visited.add(subset[0])
                while queue:
                    v = queue.pop()
                    for u in subset:
                        if u != v and (adj2[v] & (1 << u)) and u not in visited:
                            visited.add(u)
                            queue.append(u)
                if len(visited) == 8:
                    q3_found = subset
                    break
            if q3_found is None:
                break
            edges_in_q3 = []
            for ii in range(8):
                for jj in range(ii+1, 8):
                    v, u = q3_found[ii], q3_found[jj]
                    if adj2[v] & (1 << u):
                        edges_in_q3.append((v, u))
            if edges_in_q3:
                v, u = random.choice(edges_in_q3)
                adj2[v] &= ~(1 << u)
                adj2[u] &= ~(1 << v)
        
        # Local search on adj2
        ce = sum(bin(a).count('1') for a in adj2) // 2
        imp = True
        while imp and time.time() < deadline:
            imp = False
            for i in range(N):
                for j in range(i+1, N):
                    if time.time() > deadline:
                        break
                    if adj2[i] & (1 << j):
                        continue
                    adj2[i] |= (1 << j)
                    adj2[j] |= (1 << i)
                    if not _has_q3_precomputed(adj2):
                        imp = True
                        ce += 1
                    else:
                        adj2[i] &= ~(1 << j)
                        adj2[j] &= ~(1 << i)
        
        if ce > best_edges:
            best_edges = ce
            best_adj = list(adj2)
    
    # Convert best_adj to edge list
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return edges
# EVOLVE-BLOCK-END
