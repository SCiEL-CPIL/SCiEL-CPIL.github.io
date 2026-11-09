"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START_TIME = None
_DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _has_q3(edges_set, vertices):
    """Check if the subgraph induced by 'vertices' (a set of 8 vertices) contains Q3 as a subgraph.
    edges_set is a set of frozensets or sorted tuples for fast lookup."""
    if len(vertices) != 8:
        return False
    vlist = list(vertices)
    # Q3 has 12 edges. We need to check if there's an injective mapping from Q3's 8 vertices
    # to vlist such that all 12 Q3 edges map to edges in edges_set.
    # Q3 vertices: 0..7, edges:
    q3_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    # Try all permutations is too expensive (8! = 40320). 
    # Instead, use the structure: pick which vertex maps to Q3 vertex 0, then its neighbors must map to Q3's neighbors of 0.
    # Q3 vertex 0 has neighbors {1,2,4}. 
    # This is still complex. Let's use a simpler but slower check for small graphs.
    # Actually, for 8 vertices, we can check: does the induced subgraph have at least 12 edges?
    # If not, it can't contain Q3. If yes, we need to verify.
    induced_edges = 0
    for i in range(8):
        for j in range(i+1, 8):
            if (vlist[i], vlist[j]) in edges_set or (vlist[j], vlist[i]) in edges_set:
                induced_edges += 1
    if induced_edges < 12:
        return False
    # Now check if Q3 is a subgraph. Use backtracking with pruning.
    # Map Q3 vertex 0 to each of the 8 candidates
    # Q3 adjacency:
    q3_adj = {
        0: {1,2,4},
        1: {0,3,5},
        2: {0,3,6},
        3: {1,2,7},
        4: {0,5,6},
        5: {1,4,7},
        6: {2,4,7},
        7: {3,5,6}
    }
    
    def backtrack(q3_v, mapping, used):
        if q3_v == 8:
            return True
        for cand in vlist:
            if cand in used:
                continue
            # Check if cand is adjacent to all already-mapped neighbors of q3_v
            ok = True
            for q3_u in q3_adj[q3_v]:
                if q3_u < q3_v and q3_u in mapping:
                    if (mapping[q3_u], cand) not in edges_set and (cand, mapping[q3_u]) not in edges_set:
                        ok = False
                        break
            if ok:
                mapping[q3_v] = cand
                used.add(cand)
                if backtrack(q3_v + 1, mapping, used):
                    return True
                del mapping[q3_v]
                used.remove(cand)
        return False
    
    mapping = {}
    return backtrack(0, mapping, set())

def _add_edge_creates_q3(edges_set, u, v):
    """Check if adding edge (u,v) to the graph creates a Q3 subgraph."""
    # A new Q3 must include edge (u,v).
    # In Q3, each edge is in exactly 4 different 4-cycles, but for subgraph containment:
    # We need 6 other vertices such that {u,v}+those 6 contain Q3 with (u,v) as an edge.
    # 
    # In Q3, fix edge (0,1). The other 6 vertices are {2,3,4,5,6,7}.
    # The edges involving 0 or 1: 0-2, 0-4, 1-3, 1-5.
    # So vertex 0 has neighbors {1,2,4} and vertex 1 has neighbors {0,3,5}.
    # 
    # So we need: 2 vertices a,b adjacent to u (not v) that will map to {2,4}
    #             2 vertices c,d adjacent to v (not u) that will map to {3,5}
    #             2 vertices e,f that complete the Q3 structure
    #
    # This is complex. Let's just check all 6-subsets of the other 14 vertices.
    other_vertices = [x for x in range(16) if x != u and x != v]
    for subset in combinations(other_vertices, 6):
        vertices = set(subset) | {u, v}
        # Temporarily add the edge and check
        edges_set.add((u, v))
        if _has_q3(edges_set, vertices):
            edges_set.discard((u, v))
            return True
        edges_set.discard((u, v))
    return False

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with K3,13
    edges = set()
    for a in range(3):
        for b in range(3, 16):
            edges.add((a, b))
    
    best_edges = set(edges)
    best_count = len(edges)
    
    # Try to add edges greedily
    # Candidate edges: all pairs not already in edges
    all_pairs = [(i, j) for i in range(16) for j in range(i+1, 16)]
    remaining = [p for p in all_pairs if p not in edges]
    
    random.seed(42)
    random.shuffle(remaining)
    
    for (u, v) in remaining:
        if _elapsed() > _DEADLINE:
            break
        if _add_edge_creates_q3(edges, u, v):
            continue
        edges.add((u, v))
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)
    
    return [[u, v] for (u, v) in best_edges]
# EVOLVE-BLOCK-END
