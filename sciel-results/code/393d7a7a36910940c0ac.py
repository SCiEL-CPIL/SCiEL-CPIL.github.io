import time
import random
from itertools import combinations

# Precompute canonical Q3 patterns on vertices 0..7
# A Q3 has vertices 0..7. Edges connect vertices differing by 1 bit.
# We need all distinct subgraphs isomorphic to Q3.
# Since Q3 is rigid (no non-trivial automorphisms that map edges to different edges? No, it has automorphisms).
# Actually, we just need the set of edge sets.
# There are 840 such sets.

def get_canon_q3_patterns():
    patterns = []
    seen = set()
    # Vertices 0..7
    # Edges in Q3:
    # 0-1, 0-2, 0-4
    # 1-3, 1-5
    # 2-3, 2-6
    # 3-7
    # 4-5, 4-6
    # 5-7
    # 6-7
    # Total 12 edges.
    
    # We generate all permutations of 0..7 and check if the induced subgraph on the permuted vertices
    # (mapped back to 0..7) is isomorphic to Q3?
    # No, simpler:
    # Iterate all 8! permutations of 0..7.
    # For each perm p, define a mapping f: i -> p[i].
    # The edges of Q3 in the permuted space are { (f(u), f(v)) | (u,v) in Q3_edges }.
    # Normalize the edge set (sort edges, sort vertices in edges).
    # Add to seen set.
    
    q3_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    
    import itertools
    for p in itertools.permutations(range(8)):
        edges = set()
        for u, v in q3_edges:
            eu, ev = p[u], p[v]
            if eu > ev: eu, ev = ev, eu
            edges.add((eu, ev))
        # Canonical form: tuple of sorted edges
        key = tuple(sorted(edges))
        if key not in seen:
            seen.add(key)
            patterns.append(key)
            
    return patterns

CANON_PATTERNS = get_canon_q3_patterns()

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initialize K16
    adj = [0] * 16
    for i in range(16):
        for j in range(i+1, 16):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    
    def has_q3():
        # Check if current graph contains Q3
        # Iterate subsets of 8 vertices
        for comb in combinations(range(16), 8):
            # Map comb indices to 0..7
            # Check each pattern
            for pattern in CANON_PATTERNS:
                # Check if all edges in pattern exist in adj
                valid = True
                for u, v in pattern:
                    # u, v are indices in 0..7
                    # Actual vertices are comb[u], comb[v]
                    vu = comb[u]
                    vv = comb[v]
                    if not (adj[vu] & (1 << vv)):
                        valid = False
                        break
                if valid:
                    # Return the edges of this Q3
                    return [(comb[u], comb[v]) for u, v in pattern]
        return None

    # Phase 1: Remove edges to eliminate Q3s
    while time.time() < deadline - 20:
        q3_edges = has_q3()
        if q3_edges is None:
            break
        # Remove a random edge from the found Q3
        e = random.choice(q3_edges)
        u, v = e
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    # Phase 2: Local Search - try to add edges back
    # Create list of all possible edges
    all_edges = [(i, j) for i in range(16) for j in range(i+1, 16)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() > deadline - 5:
            break
        # Check if edge is already present
        if adj[u] & (1 << v):
            continue
            
        # Tentatively add
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Check for Q3
        q3_edges = has_q3()
        if q3_edges is not None:
            # Revert
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        # Else keep

    # Count edges
    edge_count = sum(bin(adj[i]).count('1') for i in range(16)) // 2
    edges = []
    for i in range(16):
        for j in range(i+1, 16):
            if adj[i] & (1 << j):
                edges.append([i, j])
                
    return edges
