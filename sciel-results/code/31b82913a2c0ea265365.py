"""K3,3-free graph on 20 vertices: maximize edges via greedy + local search."""
import time
import random
from itertools import combinations

N = 20
TOTAL_EDGES = N * (N - 1) // 2  # 190


def _is_k33_free(adj):
    """Check if graph with adjacency bitmasks is K3,3-free.
    A K3,3 exists iff there exist 3 vertices u,v,w with at least 3 common neighbors.
    """
    for u in range(N):
        for v in range(u + 1, N):
            common_uv = adj[u] & adj[v]
            # For each w, count common neighbors of u,v,w
            # We need: does there exist w such that popcount(adj[u]&adj[v]&adj[w]) >= 3?
            # But w can be any vertex (including u or v? No, K3,3 has 6 distinct vertices)
            # Actually in K3,3 the two parts are disjoint. So we need 3 vertices in one part
            # and 3 in the other, with all 9 edges present.
            # So: exist distinct a,b,c and distinct d,e,f such that all edges between {a,b,c} and {d,e,f} exist.
            # This means: exist a,b,c (distinct) such that |N(a) ∩ N(b) ∩ N(c)| >= 3,
            # and the 3 common neighbors are distinct from a,b,c.
            # Actually if a,b,c are in part 1, their common neighbors in part 2 must be >= 3.
            # But also those common neighbors must not include a,b,c themselves (simple graph, no self-loops).
            # Since adj[a] doesn't include a (no self-loop), common neighbors of a,b,c automatically exclude a,b,c.
            # Wait: adj[a] & adj[b] & adj[c] could include a if a is neighbor of b and c. But that's fine
            # because in K3,3 the parts are disjoint. So we need 3 common neighbors that are distinct from {a,b,c}.
            # Actually, if x is in N(a)∩N(b)∩N(c) and x is one of {a,b,c}, say x=a, then a∈N(a) which is false.
            # So common neighbors of a,b,c are automatically distinct from a,b,c.
            # Therefore: K3,3 exists iff ∃ distinct a,b,c with popcount(adj[a]&adj[b]&adj[c]) >= 3.
            
            # Efficient check: for each w, compute common = adj[u] & adj[v] & adj[w], check popcount >= 3
            # But this is O(N^3) which for N=20 is 8000, fine.
            pass
    
    # Better: iterate over all triples
    for a in range(N):
        for b in range(a + 1, N):
            ab = adj[a] & adj[b]
            if ab.bit_count() < 3:
                continue
            for c in range(b + 1, N):
                abc = ab & adj[c]
                if abc.bit_count() >= 3:
                    return False
    return True


def _is_k33_fast(adj):
    """Faster K3,3 check using the triple approach."""
    for a in range(N):
        for b in range(a + 1, N):
            ab = adj[a] & adj[b]
            if ab.bit_count() < 3:
                continue
            for c in range(b + 1, N):
                if (ab & adj[c]).bit_count() >= 3:
                    return False
    return True


def _can_add_edge(adj, u, v):
    """Check if adding edge (u,v) keeps the graph K3,3-free."""
    # Adding edge (u,v) means: for any triple containing u and v, the common neighbor count increases.
    # Specifically, for any w != u,v: common(u,v,w) gains 1 if w was already connected to both... 
    # No. Adding edge (u,v) changes adj[u] and adj[v].
    # New adj[u] = old adj[u] | (1<<v), new adj[v] = old adj[v] | (1<<u).
    # A new K3,3 can only involve u or v in a way that the new edge participates.
    # The new edge (u,v) can be part of a K3,3 only if there exist 2 other vertices in u's part and 2 in v's part
    # forming a K3,3 with the new edge.
    # More precisely: after adding (u,v), check if any triple has >= 3 common neighbors.
    # The only triples whose common-neighbor count changes are those containing u or v.
    # Trips containing u: (u, x, y) for x,y != u. Common neighbors = adj[u] & adj[x] & adj[y].
    #   adj[u] gained v. So new common = (old_common) | (v if v in adj[x] and v in adj[y] else 0).
    #   So the count can increase by at most 1, and only if v is adjacent to both x and y.
    # Trips containing v: similar.
    # Trips not containing u or v: unchanged.
    
    # So we only need to check triples that include u or v AND where the new edge causes a count increase.
    
    # Triples with u: (u, x, y) where x,y from 0..19, x!=u, y!=u, x<y.
    #   New common = (adj[u]|bit_v) & adj[x] & adj[y] = (adj[u]&adj[x]&adj[y]) | (bit_v if (adj[x]&adj[y])&bit_v else 0)
    #   The count increases iff v is adjacent to both x and y, AND the old count was exactly 2 (so new is 3).
    #   Or old count was >=3 already (but then graph wasn't K3,3-free before, contradiction).
    #   So: increase happens iff (adj[x]&adj[y]&bit_v) != 0 and old_count == 2.
    
    # Similarly for triples with v: (v, x, y) where x,y != v.
    #   Increase iff u is adjacent to both x and y, and old_count == 2.
    
    bit_u = 1 << u
    bit_v = 1 << v
    
    # Check triples containing u (but not v, since (u,v,x) is handled in both)
    # Actually let's just check all triples that include u or v.
    # Triples (u, x, y) with x < y, x != u, y != u:
    for x in range(N):
        if x == u:
            continue
        for y in range(x + 1, N):
            if y == u:
                continue
            # Check if v is adjacent to both x and y
            if (adj[x] & adj[y] & bit_v) and (adj[x] & adj[y] & bit_v):
                # v is common neighbor of x and y
                old_common = adj[u] & adj[x] & adj[y]
                if old_common.bit_count() == 2:
                    return False
    # Triples (v, x, y) with x < y, x != v, y != v:
    for x in range(N):
        if x == v:
            continue
        for y in range(x + 1, N):
            if y == v:
                continue
            if (adj[x] & adj[y] & bit_u):
                old_common = adj[v] & adj[x] & adj[y]
                if old_common.bit_count() == 2:
                    return False
    # Also triple (u, v, x) for x != u,v:
    for x in range(N):
        if x == u or x == v:
            continue
        # common = adj[u] & adj[v] & adj[x], with new adj[u] and adj[v]
        new_adj_u = adj[u] | bit_v
        new_adj_v = adj[v] | bit_u
        common = new_adj_u & new_adj_v & adj[x]
        if common.bit_count() >= 3:
            return False
    return True


def _greedy_build():
    """Build a K3,3-free graph greedily."""
    adj = [0] * N
    edges = []
    # Try all edges in a random order
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    random.shuffle(all_edges)
    for u, v in all_edges:
        if _can_add_edge(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append((u, v))
    return adj, edges


def _local_search(adj, edges, max_iters=2000, deadline=None):
    """Local search: try swapping edges to improve."""
    best_adj = adj[:]
    best_edges = edges[:]
    best_count = len(edges)
    
    edge_set = set(edges)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        # Try removing an edge and adding another
        random.shuffle(edges)
        for rem_u, rem_v in edges:
            if deadline and time.time() > deadline:
                break
            # Try to find an edge to add after removing (rem_u, rem_v)
            adj[rem_u] &= ~(1 << rem_v)
            adj[rem_v] &= ~(1 << rem_u)
            # Now try to add some edge
            all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
            random.shuffle(all_edges)
            added = False
            for add_u, add_v in all_edges:
                if (add_u, add_v) in edge_set:
                    continue
                if _can_add_edge(adj, add_u, add_v):
                    adj[add_u] |= (1 << add_v)
                    adj[add_v] |= (1 << add_u)
                    edge_set.discard((rem_u, rem_v))
                    edge_set.add((add_u, add_v))
                    edges.remove((rem_u, rem_v))
                    edges.append((add_u, add_v))
                    added = True
                    improved = True
                    break
            if not added:
                # Re-add the removed edge
                adj[rem_u] |= (1 << rem_v)
                adj[rem_v] |= (1 << rem_u)
                edge_set.add((rem_u, rem_v))
                edges.append((rem_u, rem_v))
            if improved:
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_adj = adj[:]
                    best_edges = edges[:]
                break
        if not improved:
            # No improvement, try pure removal+addition without shuffle
            pass
    
    return best_adj, best_edges, best_count


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_edges = []
    best_count = 0
    
    # Multiple random restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        adj, edges = _greedy_build()
        count = len(edges)
        if count > best_count:
            best_count = count
            best_edges = edges[:]
        
        # Local search
        if time.time() > deadline - 5:
            break
        adj_ls, edges_ls, count_ls = _local_search(adj, edges, max_iters=500, deadline=deadline)
        if count_ls > best_count:
            best_count = count_ls
            best_edges = edges_ls[:]
    
    # Convert to list of [u, v]
    return [[u, v] for u, v in best_edges]
