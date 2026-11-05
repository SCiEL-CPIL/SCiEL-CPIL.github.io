"""Q3-free dense graph on 16 vertices: randomized greedy + 2-opt local search."""
import time
import random
from itertools import combinations

N = 16

def _build_adj(edge_set):
    """Build adjacency as list of 16-bit integers."""
    adj = [0] * N
    for u, v in edge_set:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _has_q3(adj, u, v):
    """Check if adding edge (u,v) to graph with adjacency `adj` creates a Q3.
    Assumes (u,v) is NOT yet in adj. We check all 6 ways to complete a cube
    containing edge (u,v).
    
    A cube Q3 has 8 vertices, 12 edges. Each edge in Q3 belongs to exactly
    two 4-cycles (faces). If we add edge (u,v), we need to find two 4-cycles
    sharing that edge, where the remaining 4 vertices are distinct and the
    6 "side" edges exist.
    
    For edge (u,v) to be in a Q3, we need:
    - A 4-cycle: u-v-a-b-u (or u-v-b-a-u) where edges (u,a),(v,b),(a,b),(b,u) exist
      Actually, let me think more carefully.
    
    In Q3, an edge connects two vertices that differ in exactly one bit. The two
    4-cycles (faces) containing this edge are determined by choosing which of the
    other 2 bit positions to "move" in. So for edge (u,v), we need two distinct
    vertices a1, a2 such that:
      - u connects to a1, v connects to a2, and a1-a2 is an edge (forming cycle u-v-a2-a1-u)
      - OR u connects to a2, v connects to a1, and a1-a2 is an edge (cycle u-v-a1-a2-u)
    
    More precisely: for the edge (u,v) to be part of a Q3, we need two "parallel"
    edges (a,b) and (c,d) such that {u,v,a,b} and {u,v,c,d} form 4-cycles with
    appropriate cross edges, and all 8 vertices are distinct, plus the 4 "long" edges
    connecting the two 4-cycles exist.
    
    Let me use a cleaner approach: enumerate all pairs (a,b) where a is adjacent to u
    and b is adjacent to v, a != b, a != v, b != u, and (a,b) is an edge. This gives
    a 4-cycle u-v-b-a-u. Then for a Q3, we need another such pair (c,d) with
    c != a, c != b, d != a, d != b, and edges (a,c), (b,d), (a,d), (b,c) -- wait,
    no. In a Q3, if we have two faces sharing edge (u,v), the other 4 vertices form
    a matching with the first 4. Let me just do:
    
    Find all 4-cycles containing (u,v): these are of form u-a-v-b-u where
    (u,a),(a,v),(v,b),(b,u) are edges. Equivalently: a in N(u), b in N(v),
    (a,v) and (b,u) are edges, and a != b, a != v, b != u.
    
    For each pair of such 4-cycles with distinct other 4 vertices, check if the
    4 cross-edges exist to complete the cube.
    """
    nu = [x for x in range(N) if x != u and x != v and (adj[u] >> x) & 1]
    nv = [x for x in range(N) if x != u and x != v and (adj[v] >> x) & 1]
    
    # Find all 4-cycles through (u,v): u-a-v-b-u
    cycles = []
    for a in nu:
        if (adj[a] >> v) & 1:  # a-v edge exists
            for b in nv:
                if b == a:
                    continue
                if (adj[b] >> u) & 1:  # b-u edge exists
                    cycles.append((a, b))
    
    if len(cycles) < 2:
        return False
    
    # For each pair of cycles, check if they form a cube
    # Cube: vertices {u, v, a1, b1, a2, b2, c, d} -- wait, 4-cycles share (u,v)
    # Cycle 1: u-a1-v-b1-u, Cycle 2: u-a2-v-b2-u
    # For these to be two faces of a cube, we need:
    # - a1, b1, a2, b2 all distinct and different from u, v
    # - Edges: a1-a2, b1-b2, a1-b2, b1-a2 (the 4 edges connecting the two faces)
    #   Wait, let me verify the cube structure.
    
    # Q3 vertices can be labeled 000,001,010,011,100,101,110,111
    # Edge (000,001). Faces containing it:
    #   Face 1: 000-001-101-100-000 (bits: 0,1,1,0 in last two positions... let me redo)
    #   Actually: 000-001 (diff bit 0), 001-101 (diff bit 2), 101-100 (diff bit 0), 100-000 (diff bit 2)
    #   Face 2: 000-001-011-010-000
    #   Other edges in cube: 010-110, 011-111, 100-110, 101-111, 010-011, 100-101, 110-111, 010-100? No.
    #   Let me list all 12 edges of Q3:
    #   000-001, 000-010, 000-100, 001-011, 001-101, 010-011, 010-110, 100-101, 100-110, 011-111, 101-111, 110-111
    #   Edge (000,001). Faces:
    #     000-001-101-100-000: edges 000-001, 001-101, 101-100, 100-000
    #     000-001-011-010-000: edges 000-001, 001-011, 011-010, 010-000
    #   The remaining 4 edges: 010-110, 011-111, 100-110, 101-111
    #   So connecting vertices: {010,011} to {100,101}: edges 010-110, 011-111, 100-110, 101-111
    #   And the 4th vertex is 111.
    
    # So for two 4-cycles u-a1-v-b1-u and u-a2-v-b2-u:
    # The 8 vertices are {u, v, a1, b1, a2, b2, x, y} where x,y are the "opposite" vertices.
    # Actually, in the cube, the 4 non-shared vertices of the two faces (a1,b1,a2,b2) 
    # plus 2 more (x,y) form the rest. Let me re-examine.
    #
    # Face 1: u, a1, v, b1 (4 vertices)
    # Face 2: u, a2, v, b2 (4 vertices)
    # Shared: u, v. Total distinct so far: u, v, a1, b1, a2, b2 (6 vertices)
    # The cube has 8 vertices, so 2 more: let's call them x, y.
    # In the example: u=000, v=001, a1=100, b1=101, a2=010, b2=011
    # x=110, y=111
    # Edges to check: 
    #   a1-x (100-110): yes
    #   b1-y (101-111): yes  
    #   a2-x (010-110): yes
    #   b2-y (011-111): yes
    #   x-y (110-111): yes
    #   And a1-b2? 100-011: no. b1-a2? 101-010: no.
    #
    # So the pattern: from {a1,b1,a2,b2}, we need two more vertices x,y such that:
    #   x connects to a1 and a2, y connects to b1 and b2, and x-y is an edge.
    #   OR x connects to a1 and b1, y connects to a2 and b2... let me check.
    #   In example: x=110 connects to a1=100 and a2=010. y=111 connects to b1=101 and b2=011. x-y edge exists.
    #   Alternatively: x connects to b1 and b2, y connects to a1 and a2.
    #
    # So: find x in N(a1) ∩ N(a2) that is not in {u,v,a1,b1,a2,b2}, and y in N(b1) ∩ N(b2) 
    # that is not in {u,v,a1,b1,a2,b2}, x != y, and (x,y) is an edge.
    # OR: x in N(b1) ∩ N(b2), y in N(a1) ∩ N(a2), x != y, (x,y) edge.
    
    used = {u, v}
    for i in range(len(cycles)):
        for j in range(i + 1, len(cycles)):
            a1, b1 = cycles[i]
            a2, b2 = cycles[j]
            cand = {a1, b1, a2, b2}
            if len(cand) != 4:
                continue
            used = {u, v, a1, b1, a2, b2}
            # Option 1: x ~ a1,a2 and y ~ b1,b2
            for x in range(N):
                if x in used:
                    continue
                if (adj[x] >> a1) & 1 and (adj[x] >> a2) & 1:
                    for y in range(N):
                        if y in used or y == x:
                            continue
                        if (adj[y] >> b1) & 1 and (adj[y] >> b2) & 1 and (adj[x] >> y) & 1:
                            return True
            # Option 2: x ~ b1,b2 and y ~ a1,a2
            for x in range(N):
                if x in used:
                    continue
                if (adj[x] >> b1) & 1 and (adj[x] >> b2) & 1:
                    for y in range(N):
                        if y in used or y == x:
                            continue
                        if (adj[y] >> a1) & 1 and (adj[y] >> a2) & 1 and (adj[x] >> y) & 1:
                            return True
    return False

def _try_add(adj, es, u, v):
    """Try to add edge (u,v). Returns True if added."""
    if (adj[u] >> v) & 1:
        return True  # already present
    # Test if adding creates Q3
    # Temporarily add
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    ok = not _has_q3(adj, u, v)
    if not ok:
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return False
    es.add((min(u,v), max(u,v)))
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Precompute all pairs
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    best_edges = set()
    best_count = 0
    
    # Initial: K3,13
    base = set((a, b) for a in range(3) for b in range(3, N))
    if len(base) > best_count:
        best_count = len(base)
        best_edges = set(base)
    
    random.seed(42)
    
    restart = 0
    while time.time() < deadline:
        restart += 1
        
        if restart == 1:
            es = set(base)
        else:
            es = set()
            order = all_pairs[:]
            random.shuffle(order)
            adj = _build_adj(es)
            for (u, v) in order:
                if (adj[u] >> v) & 1:
                    continue
                if _try_add(adj, es, u, v):
                    pass
        adj = _build_adj(es)
        
        # Local search: 1-for-1 swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to remove one edge and add one or two others
            for (ru, rv) in list(es):
                if time.time() > deadline:
                    break
                # Remove (ru, rv)
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                es.discard((min(ru,rv), max(ru,rv)))
                
                # Try adding random edges
                added_any = False
                candidates = [p for p in all_pairs if (min(p), max(p)) not in es]
                random.shuffle(candidates)
                for (u, v) in candidates[:20]:
                    if _try_add(adj, es, u, v):
                        added_any = True
                
                if not added_any:
                    # Put back
                    _try_add(adj, es, ru, rv)
                
                if len(es) > best_count:
                    best_count = len(es)
                    best_edges = set(es)
                    improved = True
        
        if best_count >= 110:
            break
    
    if best_edges is None:
        best_edges = set(base)
    
    return [[u, v] for (u, v) in best_edges]
