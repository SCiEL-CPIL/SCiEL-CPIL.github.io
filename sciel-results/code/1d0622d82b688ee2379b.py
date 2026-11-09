"""Hexagon-free graph construction on 40 vertices via MIS on C6-conflict graph + local search."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

N = 40
TOTAL_PAIRS = N * (N - 1) // 2  # 780


def _pair_index(u, v):
    """Map (u,v) with u<v to a unique index in [0, TOTAL_PAIRS)."""
    if u > v:
        u, v = v, u
    # Triangular number: index = u*(2*N - u - 1)//2 + (v - u - 1)
    return u * (2 * N - u - 1) // 2 + (v - u - 1)


def _index_to_pair(idx):
    """Map index back to (u, v) with u < v."""
    # Solve: idx = u*(2N - u - 1)//2 + (v - u - 1)
    # For given u, base = u*(2N - u - 1)//2
    # v - u - 1 = idx - base, so v = u + 1 + idx - base
    u = 0
    while u < N:
        base = u * (2 * N - u - 1) // 2
        if idx >= base:
            v = u + 1 + (idx - base)
            if v < N:
                return (u, v)
        u += 1
    return (0, 1)  # fallback


def _build_conflict_graph():
    """
    Build the C6-conflict graph:
    - Nodes: all C(40,2)=780 possible edges
    - Edge between node i and node j if adding both edges to a graph
      containing the "support" path would create a C6.
    
    Actually, the conflict is: two edges e1=(a,b) and e2=(c,d) conflict if
    there exists a path of length 4 between {a,b} and {c,d} such that
    adding both creates a C6.
    
    More precisely: e1=(u,v) and e2=(w,x) conflict if there's a path
    u - p1 - p2 - p3 - w (length 4) such that v is not on that path and
    x is not on that path, AND the cycle u-v-...-x-w-p3-p2-p1-u would be
    a valid C6. But this is complex.
    
    Simpler formulation: e1=(u,v) and e2=(w,x) conflict if there exists
    a simple path of length 4 from u to w (or v to w, u to x, v to x)
    that doesn't use e1 or e2, such that adding both edges closes a C6.
    
    For a C6: u-v-w-x-y-z-u. If we fix two edges, say (u,v) and (w,x),
    they conflict if there's a path v-...-w of length 3 (3 edges) that
    doesn't go through u or x, OR path u-...-x of length 3 not through v or w,
    etc. Actually for a 6-cycle with edges (u,v) and (w,x) both present:
    The cycle is u-v-?-?-?-?-u. If (w,x) is in the cycle, then w and x
    are two of the 6 vertices. The cycle visits u,v and w,x.
    
    Let me think differently. Two edges (a,b) and (c,d) are in conflict
    if there exists a 6-cycle containing both edges. A 6-cycle is a set of
    6 vertices with 6 edges forming a cycle. If (a,b) and (c,d) are both
    in the cycle, the cycle has form a-b-?-?-?-?-a (with c,d somewhere in
    the ? positions) or similar.
    
    For efficiency: (a,b) and (c,d) conflict iff there exists a path of
    length 4 from a to c (or a to d, b to c, b to d) that avoids using
    edges (a,b) and (c,d), and the intermediate vertices don't include
    the endpoints in a way that creates a shorter cycle.
    
    Actually the cleanest: (a,b) and (c,d) conflict iff there's a simple
    path of exactly 4 edges connecting one endpoint of {a,b} to one endpoint
    of {c,d}, where the path doesn't use edge (a,b) or edge (c,d), and
    all 6 vertices {a,b,c,d} plus 2 intermediate are distinct.
    
    Path of length 4 from a to c: a - p1 - p2 - p3 - c
    Then cycle: a-b-...-c-p3-p2-p1-a? No, that's not right.
    
    If we have edges (a,b) and (c,d), and a path a-p1-p2-p3-c (length 4),
    then the cycle would be: a-b-? - ? - ? - ? - a. But (c,d) needs to be
    in the cycle. 
    
    Let me use a different approach. For C6: vertices v0-v1-v2-v3-v4-v5-v0.
    Edge (a,b) means {a,b} = {vi, vi+1 mod 6} for some i.
    Edge (c,d) means {c,d} = {vj, vj+1 mod 6} for some j.
    
    So (a,b) and (c,d) conflict if there exist 6 distinct vertices 
    v0..v5 such that {a,b} is one of the 6 edges of the cycle and 
    {c,d} is another edge of the cycle.
    
    For a given pair of edges (a,b) and (c,d), they conflict if:
    - a,b,c,d are all distinct (otherwise can't be in same C6 as distinct edges)
    - There exist 2 additional distinct vertices p, q (not in {a,b,c,d}) 
      such that {a,b,c,d,p,q} can be arranged in a 6-cycle containing 
      both edges (a,b) and (c,d).
    
    This is getting complex. Let me use a direct check:
    (a,b) and (c,d) conflict iff there's a 6-cycle containing both.
    Equivalently, in the complete graph K40, remove edges (a,b) and (c,d),
    and check if there's a path of length 4 from a to c (or appropriate 
    pairing) that with edges (a,b) and (c,d) forms a C6.
    
    The 6-cycle containing edges (a,b) and (c,d): 
    Without loss of generality, suppose the cycle is 
    a - b - x - y - z - c - d - ... no, that's 7 vertices.
    
    6-cycle: 6 vertices, 6 edges. If (a,b) is one edge and (c,d) is another:
    Case 1: a and c are the same vertex? No, edges must be distinct and 
             in a simple cycle, adjacent edges share one vertex.
             (a,b) and (c,d) are non-adjacent in the cycle (otherwise they'd
             share a vertex, but we assumed a,b,c,d distinct).
    
    So in the 6-cycle, (a,b) and (c,d) are separated by at least one edge.
    The cycle: a-b-?-?-?-c-d-? ... no, 6 vertices: 
    a-b-p-q-r-s-a where {c,d} must be one of {bp,pq,qr,rs,sa}.
    Or: a-b-c-d-p-q-a (adjacent edges, but then a,b,c,d not all distinct 
        if they share vertices... wait (a,b) and (c,d) share no vertex if 
        a,b,c,d distinct).
    
    OK let me just enumerate: for edges (a,b) and (c,d) with a<b, c<d, 
    a!=c, a!=d, b!=c, b!=d:
    They conflict if there exist distinct p, q not in {a,b,c,d} such that
    one of these 6-cycles exists:
    1. a-b-c-d-p-q-a: need edges (b,c),(d,p),(p,q),(q,a)
    2. a-b-c-q-p-d-a: need edges (b,c),(c,q),(q,p),(p,d),(d,a) -- that's 5 intermediate, wrong
    Let me be more careful.
    
    A 6-cycle is a sequence v0,v1,v2,v3,v4,v5 with edges (v0,v1),(v1,v2),...,(v5,v0).
    If (a,b) is in the cycle, WLOG a=v0, b=v1.
    If (c,d) is in the cycle, WLOG c=vk, d=v(k+1 mod 6) for some k in {2,3,4,5}
    (k=0,1 would mean same edge or adjacent sharing vertex, but we need distinct).
    Actually k can be 0..5 but (c,d) != (a,b) so {c,d} != {a,b}.
    If k=0: c=v0=a, d=v1=b -- same edge, excluded.
    If k=1: c=v1=b, d=v2 -- shares vertex b with (a,b). Then {a,b,c,d} has c=b, 
            not all distinct.
    If k=5: c=v5, d=v0=a -- shares vertex a.
    
    So for a,b,c,d all distinct, (c,d) must be at position k=2,3,4 in the cycle 
    (where positions are 0:(v0,v1), 1:(v1,v2), ..., 5:(v5,v0)):
    k=2: c=v2, d=v3. Cycle: a-b-c-d-v4-v5-a. Need: (a,b),(b,c),(c,d),(d,v4),(v4,v5),(v5,a).
         Two free vertices: v4, v5. Need edges (b,c), (d,v4), (v4,v5), (v5,a).
         So: (b,c) exists in K40 (always, it's complete), and we need v4,v5 distinct 
         from {a,b,c,d} with edges (d,v4),(v4,v5),(v5,a).
         This means: in K40 minus {a,b,c,d}, is there a path d-?-?-a of length 3?
         I.e., does there exist v4,v5 not in {a,b,c,d} with edges (d,v4),(v4,v5),(v5,a)?
         Since K40 is complete, YES as long as there are at least 2 vertices outside 
         {a,b,c,d}, which is true (36 >= 2). So (a,b) and (c,d) ALWAYS conflict?
         
    Wait, that can't be right. In a complete graph, every pair of edges is in some 
    6-cycle. But we're not working in K40; we're working in a subgraph. The conflict 
    graph should capture: given that we've already chosen some set S of edges, 
    adding edge e creates a C6 if there's a path of length 5 in G[S] connecting 
    the endpoints of e.
    
    So the "conflict" is not pairwise in general - it depends on the current set of 
    edges. This means a static conflict graph doesn't work directly.
    
    Alternative: use the incremental check as in the current program but with 
    a better search strategy.
    
    Let me pivot: use a **greedy construction with careful ordering** followed by 
    **intensive 2-opt local search** (remove 1, add 2; remove 2, add 2; remove 3, add 3).
    """
    pass


def _build_adj(edges):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _has_c6_adj(adj, n):
    """Check if graph has a 6-cycle."""
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w == u:
                    continue
                for x in adj[w]:
                    if x == u or x == v:
                        continue
                    for y in adj[x]:
                        if y == u or y == v or y == w:
                            continue
                        for z in adj[y]:
                            if z == u or z == v or z == w or z == x:
                                continue
                            if u in adj[z]:
                                return True
    return False


def _creates_c6(adj, u, v):
    """Check if adding edge (u,v) to graph with adjacency adj creates a C6.
    A C6 through (u,v): u-v-w-x-y-z-u where w,x,y,z distinct and not u,v."""
    for w in adj[v]:
        if w == u:
            continue
        for x in adj[w]:
            if x == u or x == v:
                continue
            for y in adj[x]:
                if y == u or y == v or y == w:
                    continue
                for z in adj[y]:
                    if z == u or z == v or z == w or z == x:
                        continue
                    if u in adj[z]:
                        return True
    return False


def _greedy_construct(order, deadline):
    """
    Greedily add edges in the given order if they don't create a C6.
    Returns (edge_set, adj).
    """
    start_time = time.time()
    edge_set = set()
    adj = [set() for _ in range(N)]
    
    for u, v in order:
        if time.time() - start_time > deadline:
            break
        key = (min(u, v), max(u, v))
        if key in edge_set:
            continue
        if not _creates_c6(adj, u, v):
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add(key)
    
    return edge_set, adj


def _local_search_2opt(edge_set, adj, deadline, rng):
    """
    Local search: try 2-opt moves (remove 1 edge, add 2 non-edges) 
    and 1-opt moves (remove 1, add 1).
    Accept if edge count increases and no C6.
    """
    start_time = time.time()
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    improved = True
    while improved and time.time() - start_time < deadline:
        improved = False
        edges_list = list(edge_set)
        rng.shuffle(edges_list)
        
        for e1 in edges_list:
            if time.time() - start_time > deadline:
                break
            u1, v1 = e1
            
            # Try 1-opt: remove e1, add one non-edge
            for _ in range(50):
                u2, v2 = rng.choice(all_pairs)
                key2 = (min(u2, v2), max(u2, v2))
                if key2 in edge_set:
                    continue
                
                # Remove e1
                adj[u1].discard(v1)
                adj[v1].discard(u1)
                edge_set.discard(e1)
                
                if not _creates_c6(adj, u2, v2):
                    adj[u2].add(v2)
                    adj[v2].add(u2)
                    edge_set.add(key2)
                    improved = True
                    break
                else:
                    # Revert
                    adj[u1].add(v1)
                    adj[v1].add(u1)
                    edge_set.add(e1)
            
            # Try 2-opt: remove e1, add 2 non-edges
            if time.time() - start_time > deadline:
                break
            u1, v1 = e1
            adj[u1].discard(v1)
            adj[v1].discard(u1)
            edge_set.discard(e1)
            
            added = 0
            added_edges = []
            for _ in range(200):
                u2, v2 = rng.choice(all_pairs)
                key2 = (min(u2, v2), max(u2, v2))
                if key2 in edge_set:
                    continue
                if not _creates_c6(adj, u2, v2):
                    adj[u2].add(v2)
                    adj[v2].add(u2)
                    edge_set.add(key2)
                    added_edges.append(key2)
                    added += 1
                    if added == 2:
                        break
            
            if added == 2:
                improved = True
            else:
                # Revert: remove added, re-add e1
                for ke in added_edges:
                    a, b = ke
                    adj[a].discard(b)
                    adj[b].discard(a)
                    edge_set.discard(ke)
                adj[u1].add(v1)
                adj[v1].add(u1)
                edge_set.add(e1)
    
    return edge_set, adj


def _local_search_3opt(edge_set, adj, deadline, rng):
    """
    Try 3-opt moves: remove 2 edges, add 3 non-edges.
    """
    start_time = time.time()
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    improved = True
    while improved and time.time() - start_time < deadline:
        improved = False
        edges_list = list(edge_set)
        rng.shuffle(edges_list)
        
        for i in range(0, min(len(edges_list), 30), 1):
            if time.time() - start_time > deadline:
                break
            e1 = edges_list[i]
            if i + 1 < len(edges_list):
                e2 = edges_list[i + 1]
            else:
                continue
            
            u1, v1 = e1
            u2, v2 = e2
            
            # Remove both
            adj[u1].discard(v1)
            adj[v1].discard(u1)
            adj[u2].discard(v2)
            adj[v2].discard(u2)
            edge_set.discard(e1)
            edge_set.discard(e2)
            
            added = 0
            added_edges = []
            for _ in range(300):
                ua, va = rng.choice(all_pairs)
                key = (min(ua, va), max(ua, va))
                if key in edge_set or key == e1 or key == e2:
                    continue
                if not _creates_c6(adj, ua, va):
                    adj[ua].add(va)
                    adj[va].add(ua)
                    edge_set.add(key)
                    added_edges.append(key)
                    added += 1
                    if added == 3:
                        break
            
            if added == 3:
                improved = True
            else:
                for ke in added_edges:
                    a, b = ke
                    adj[a].discard(b)
                    adj[b].discard(a)
                    edge_set.discard(ke)
                adj[u1].add(v1)
                adj[v1].add(u1)
                adj[u2].add(v2)
                adj[v2].add(u2)
                edge_set.add(e1)
                edge_set.add(e2)
    
    return edge_set, adj


def _multi_start_greedy_ls(deadline):
    """
    Multiple greedy constructions with different orderings, followed by local search.
    """
    start_time = time.time()
    best_edges = set()
    best_count = 0
    rng = random.Random(42)
    
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Different orderings
    orderings = []
    
    # 1. Random order
    order1 = list(all_pairs)
    rng.shuffle(order1)
    orderings.append(order1)
    
    # 2. Degree-based: prioritize vertices that are "central"
    # Use a heuristic: sort by sum of distances (in K40, all same), 
    # so use random with bias
    order2 = list(all_pairs)
    for _ in range(100):
        rng.shuffle(order2)
    orderings.append(order2)
    
    # 3. Cluster-based: group by vertex blocks
    order3 = []
    for i in range(N):
        for j in range(i + 1, N):
            order3.append((i, j))
    rng.shuffle(order3)
    orderings.append(order3)
    
    # 4. Reverse
    order4 = list(all_pairs)
    rng.shuffle(order4)
    order4.reverse()
    orderings.append(order4)
    
    # 5-8. More random
    for _ in range(4):
        o = list(all_pairs)
        rng.shuffle(o)
        orderings.append(o)
    
    num_starts = len(orderings)
    per_start_time = (deadline - 15) / num_starts
    
    for idx, order in enumerate(orderings):
        if time.time() - start_time > deadline - 5:
            break
        
        # Greedy construction
        edge_set, adj = _greedy_construct(order, time.time() + per_start_time * 0.3)
        count = len(edge_set)
        
        if count <= best_count:
            continue
        
        # Local search 2-opt
        ls_deadline = time.time() + per_start_time * 0.5
        edge_set, adj = _local_search_2opt(edge_set, adj, ls_deadline, rng)
        count = len(edge_set)
        
        # Local search 3-opt
        ls2_deadline = time.time() + per_start_time * 0.2
        edge_set, adj = _local_search_3opt(edge_set, adj, ls2_deadline, rng)
        count = len(edge_set)
        
        if count > best_count:
            best_count = count
            best_edges = set(edge_set)
    
    return best_edges, best_count


def _seed_graph_8K5():
    """8 disjoint K5s: 100 edges."""
    edges = []
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([base + a, base + b])
    return edges


def construct():
    n = N
    start_time = time.time()
    deadline = 130  # Safety margin before 150s kill
    
    # Multi-start greedy + local search
    edge_set, count = _multi_start_greedy_ls(deadline)
    
    if count == 0:
        return _seed_graph_8K5()
    
    edges = [[u, v] for u, v in edge_set]
    
    # Final verification
    adj = _build_adj(edges)
    if _has_c6_adj(adj, n):
        return _seed_graph_8K5()
    
    return edges
# EVOLVE-BLOCK-END
