"""C6-free graph on 40 vertices: maximize edges via randomized greedy insertion."""
import time
import random

N = 40
TOTAL_EDGES = N * (N - 1) // 2  # 780

def _init_adj():
    return [0] * N

def _add_edge(adj, u, v):
    adj[u] |= 1 << v
    adj[v] |= 1 << u

def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def _has_c6_after_add(adj, u, v):
    """Check if adding edge (u,v) creates a C6. Returns True if C6 exists."""
    # After adding u-v, we need to check all length-3 paths from u to some c
    # and length-3 paths from v to c, with c != u, c != v, and the paths
    # using distinct intermediate vertices (and not using the new edge u-v
    # in a way that creates a shorter cycle).
    #
    # A C6 containing the new edge (u,v): u - a - b - c - d - e - v - u
    # where a,b,c,d,e are distinct and none equal u or v.
    # Equivalently: there exist a, b, c, d, e all in {0..39}\{u,v}, all distinct,
    # such that:
    #   (u,a), (a,b), (b,c), (c,d), (d,e), (e,v) are edges in adj (before adding u-v)
    #   and none of a,b,c,d,e equals u or v.
    #
    # We can enumerate over the middle vertex c (1..39 choices, minus u,v).
    # For each c, find all length-2 paths from u to c: u-a-b-c where a,b != u,v,c.
    # And all length-2 paths from v to c: v-d-e-c where d,e != u,v,c.
    # Then check if there's a pair (a,b) and (d,e) with {a,b} ∩ {d,e} = ∅.
    
    # Actually let's be more careful. A 6-cycle through edge (u,v):
    # u - x1 - x2 - x3 - x4 - x5 - v - u
    # x1..x5 all distinct, all not in {u,v}.
    # Edges needed (in current adj, before adding u-v):
    #   (u,x1), (x1,x2), (x2,x3), (x3,x4), (x4,x5), (x5,v)
    
    # Enumerate x3 (the "middle" vertex opposite the edge):
    # For each candidate x3 (not u, not v):
    #   Find all (x1,x2) such that (u,x1)∈adj, (x1,x2)∈adj, (x2,x3)∈adj, x1≠x2, x1≠x3, x2≠u
    #   Find all (x4,x5) such that (x3,x4)∈adj, (x4,x5)∈adj, (x5,v)∈adj, x4≠x5, x4≠x3, x5≠u
    #   Check if there exists a pair with x1,x2,x4,x5 all distinct (and not u,v,x3).
    
    # For efficiency, precompute for each vertex w the set of vertices at distance 2 from w.
    # dist2[w] = {z : ∃ y, (w,y)∈adj and (y,z)∈adj}
    # But we need the intermediate to be tracked for the distinctness constraint.
    
    # Let's just enumerate. N=40, so at most 38 choices for x3, and for each
    # we enumerate pairs. In the worst case this could be slow, but in practice
    # the degree is moderate.
    
    # Actually, a cleaner approach: enumerate (x1, x2) pairs for paths u-x1-x2-x3
    # and (x4, x5) for paths x3-x4-x5-v, and check disjointness.
    
    # Let's precompute for each vertex w, the list of (prev, next) pairs for
    # length-2 paths through w. Actually, let me just do a direct enumeration.
    
    # For a given x3, the set of valid (x1,x2) pairs:
    #   x1 in N(u) \ {v, x3}, x2 in N(x1) ∩ N(x3) \ {u, v, x3}
    # The set of valid (x4,x5) pairs:
    #   x5 in N(v) \ {u, x3}, x4 in N(x5) ∩ N(x3) \ {u, v, x3}
    
    # We need x1,x2,x4,x5 all distinct.
    
    # For speed, let's represent the set of (x2, x4) or use bitmasks.
    # For each x3, compute:
    #   S_left = set of (x1, x2) valid
    #   S_right = set of (x4, x5) valid
    # Then check if ∃ (x1,x2) in S_left, (x4,x5) in S_right with all four distinct.
    
    # Since degrees are small in practice, this should be fast enough.
    
    for x3 in range(N):
        if x3 == u or x3 == v:
            continue
        # Find all (x1, x2) for path u-x1-x2-x3
        # x1 must be adjacent to u, x1 != v, x1 != x3
        # x2 must be adjacent to x1 and x3, x2 != u, x2 != v, x2 != x3, x2 != x1
        left_pairs = []
        for x1 in range(N):
            if x1 == u or x1 == v or x1 == x3:
                continue
            if not (adj[u] >> x1) & 1:
                continue
            # x1 is adjacent to u. Now find x2.
            # x2 must be in N(x1) ∩ N(x3), and x2 not in {u, v, x3, x1}
            common = adj[x1] & adj[x3]
            common &= ~((1 << u) | (1 << v) | (1 << x3) | (1 << x1))
            while common:
                lb = common & (-common)
                x2 = lb.bit_length() - 1
                left_pairs.append((x1, x2))
                common &= common - 1
        
        if not left_pairs:
            continue
            
        # Find all (x4, x5) for path x3-x4-x5-v
        # x5 must be adjacent to v, x5 != u, x5 != x3
        # x4 must be adjacent to x5 and x3, x4 not in {u, v, x3, x5}
        right_pairs = []
        for x5 in range(N):
            if x5 == u or x5 == v or x5 == x3:
                continue
            if not (adj[v] >> x5) & 1:
                continue
            common = adj[x5] & adj[x3]
            common &= ~((1 << u) | (1 << v) | (1 << x3) | (1 << x5))
            while common:
                lb = common & (-common)
                x4 = lb.bit_length() - 1
                right_pairs.append((x4, x5))
                common &= common - 1
        
        if not right_pairs:
            continue
        
        # Now check if there exist (x1,x2) in left_pairs and (x4,x5) in right_pairs
        # such that x1, x2, x4, x5 are all distinct.
        # For each left pair, check against each right pair.
        # To speed up: for each (x1,x2), the forbidden set for right is {x1,x2}.
        # We need (x4,x5) with x4 not in {x1,x2} and x5 not in {x1,x2} and x4 != x5
        # (already guaranteed by construction).
        
        # Build a quick lookup: for right_pairs, group by x4 or use a set.
        # Since the lists are small, just do the double loop.
        for x1, x2 in left_pairs:
            forbidden = (1 << x1) | (1 << x2)
            for x4, x5 in right_pairs:
                if x4 & forbidden or x5 & forbidden:
                    continue
                return True
    return False

def _count_edges(adj):
    return sum(bin(adj[i]).count('1') for i in range(N)) // 2

def _greedy_randomized(seed):
    """Greedy insertion with random edge ordering. Returns (edge_count, adj)."""
    rng = random.Random(seed)
    adj = _init_adj()
    
    # Generate all possible edges
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    rng.shuffle(all_edges)
    
    for u, v in all_edges:
        if (adj[u] >> v) & 1:
            continue  # already present
        if _has_c6_after_add(adj, u, v):
            continue
        _add_edge(adj, u, v)
    
    # Local improvement: try to add any missing edge
    improved = True
    while improved:
        improved = False
        for u in range(N):
            for v in range(u + 1, N):
                if (adj[u] >> v) & 1:
                    continue
                if _has_c6_after_add(adj, u, v):
                    continue
                _add_edge(adj, u, v)
                improved = True
    
    return _count_edges(adj), adj

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    best_count = 0
    best_adj = None
    
    seed = 42
    while time.time() < deadline:
        count, adj = _greedy_randomized(seed)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        seed += 1
    
    if best_adj is None:
        # Fallback: return the seed program's answer (8 copies of K5 = 80 edges)
        edges = []
        for k in range(8):
            edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
        return edges
    
    # Convert best_adj to edge list
    edges = []
    for u in range(N):
        mask = best_adj[u]
        for v in range(u + 1, N):
            if (mask >> v) & 1:
                edges.append([u, v])
    return edges
