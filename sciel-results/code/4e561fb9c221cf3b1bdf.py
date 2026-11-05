import time
import itertools

# Q3 edges (canonical, vertices 0..7)
_Q3 = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]

def _q3_check_full(adj, n):
    """Return True if graph (bitmask adj list) contains a Q3 subgraph."""
    # For each edge (u,v) in graph, check if it can be an edge of a Q3.
    for u in range(n):
        au = adj[u]
        # iterate neighbors v of u
        v = au
        while v:
            lsb = v & (-v)
            vv = lsb.bit_length() - 1
            v ^= lsb
            if vv <= u:
                continue
            # check if (u,v) completes a Q3, treating u as Q3-vertex 0, v as 1 (and also swapped)
            if _edge_makes_q3(adj, n, u, v):
                return True
    return False

def _edge_makes_q3(adj, n, u, v):
    """Check if edge (u,v) is part of some Q3. u plays role 0, v role 1 (symmetric enough
    because we test both (u,v) and (v,u) callers? We test here with u=0,v=1; the caller
    ensures we also consider u=1,v=0 by swapping. Actually we call for each unordered edge
    once with (u,v); to cover all roles we need both assignments. We'll handle both here.)"""
    # Assignment A: u->0, v->1
    if _check_role(adj, n, u, v):
        return True
    # Assignment B: u->1, v->0
    if _check_role(adj, n, v, u):
        return True
    return False

def _check_role(adj, n, a0, a1):
    """a0 is Q3-vertex 0, a1 is Q3-vertex 1 (edge (0,1))."""
    # 0's other Q3-neighbors: 2 and 4 (both neighbors of a0, not a1)
    # 1's other Q3-neighbors: 3 and 5 (both neighbors of a1, not a0)
    n0 = adj[a0] & ~ (1 << a1)
    n1 = adj[a1] & ~ (1 << a0)
    # enumerate ordered pairs (a2,a4) from n0 and (a3,a5) from n1
    # 6 = common neighbor of a2,a4 ; 7 = common neighbor of a3,a5
    for a2 in _bits(n0):
        for a4 in _bits(n0):
            if a4 == a2:
                continue
            # 6 must be common neighbor of a2 and a4
            common6 = adj[a2] & adj[a4]
            common6 &= ~ (1 << a0)
            common6 &= ~ (1 << a1)
            for a6 in _bits(common6):
                if a6 == a2 or a6 == a4:
                    continue
                # 7 must be common neighbor of a3,a5, and adjacent to a6
                # also a6 adjacent to a0? No: 6 adjacent to 2,4,7 only (not 0). Check: Q3 edges with 6: (2,6),(4,6),(6,7). Yes 6 not adj to 0.
                # a6 must NOT be adjacent to a0 in Q3? In Q3, 6's neighbors: 2,4,7. So 6 not adj to 0 or 1. We have a6 not a0,a1 already.
                for a3 in _bits(n1):
                    for a5 in _bits(n1):
                        if a5 == a3:
                            continue
                        common7 = adj[a3] & adj[a5]
                        common7 &= ~ (1 << a0)
                        common7 &= ~ (1 << a1)
                        common7 &= ~ (1 << a6)
                        for a7 in _bits(common7):
                            if a7 == a3 or a7 == a5 or a7 == a6:
                                continue
                            # verify all required edges exist
                            # edges: (0,1)=a0-a1 (given), (0,2)=a0-a2, (0,4)=a0-a4,
                            # (1,3)=a1-a3, (1,5)=a1-a5, (2,3)=a2-a3, (2,6)=a2-a6,
                            # (3,7)=a3-a7, (4,5)=a4-a5, (4,6)=a4-a6, (5,7)=a5-a7, (6,7)=a6-a7
                            if not (adj[a0] >> a2) & 1: continue
                            if not (adj[a0] >> a4) & 1: continue
                            if not (adj[a1] >> a3) & 1: continue
                            if not (adj[a1] >> a5) & 1: continue
                            if not (adj[a2] >> a3) & 1: continue
                            if not (adj[a2] >> a6) & 1: continue
                            if not (adj[a3] >> a7) & 1: continue
                            if not (adj[a4] >> a5) & 1: continue
                            if not (adj[a4] >> a6) & 1: continue
                            if not (adj[a5] >> a7) & 1: continue
                            if not (adj[a6] >> a7) & 1: continue
                            # all distinct?
                            verts = [a0,a1,a2,a4,a3,a5,a6,a7]
                            if len(set(verts)) == 8:
                                return True
    return False

def _bits(mask):
    while mask:
        lsb = mask & (-mask)
        yield lsb.bit_length() - 1
        mask ^= lsb


def construct():
    n = 16
    t_start = time.time()
    deadline = t_start + 130.0  # safety margin before 150s kill

    # Start from a reasonably dense seed: complete bipartite K8,8 is Q3-free? 
    # Actually K8,8 contains Q3? Q3 is bipartite with parts of size 4,4. K8,8 contains
    # any bipartite graph with parts <=8, so it CONTAINS Q3. Bad seed.
    # Use K3,13 seed (39 edges) which is Q3-free (each part <4).
    # Then greedily add edges.
    best_edges = None
    best_count = -1

    # adjacency as bitmask
    adj = [0] * n
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def edge_list():
        es = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    es.append([u, v])
        return es

    def try_add(u, v):
        nonlocal best_edges, best_count
        if (adj[u] >> v) & 1:
            return False
        # temporarily add
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if _edge_makes_q3(adj, n, u, v):
            # remove
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return False
        # keep
        cnt = sum(bin(adj[i]).count('1') for i in range(n)) // 2
        if cnt > best_count:
            best_count = cnt
            best_edges = edge_list()
        return True

    # seed K3,13
    for a in range(3):
        for b in range(3, 16):
            add_edge(a, b)
    cnt = sum(bin(adj[i]).count('1') for i in range(n)) // 2
    best_count = cnt
    best_edges = edge_list()

    # greedy: repeatedly try to add a missing edge
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        for u in range(n):
            for v in range(u+1, n):
                if time.time() > deadline - 5.0:
                    break
                if not ((adj[u] >> v) & 1):
                    if try_add(u, v):
                        changed = True
            if time.time() > deadline - 5.0:
                break

    return best_edges
