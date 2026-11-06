"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 20
DEADLINE_FRACTION = 0.90  # safety margin

def _has_k33(adj):
    """Check if graph with adjacency list has a K3,3 subgraph.
    A K3,3 exists iff there are 3 vertices with at least 3 common neighbours."""
    n = len(adj)
    # For each triple of vertices, check if they share >= 3 common neighbours
    # But that's O(n^3 * n) which is 20^4 = 160000, fine.
    # Actually, K3,3 has two parts of size 3. We need 3 vertices in one part
    # and 3 in the other, with all 9 edges present.
    # Equivalent: there exist 3 vertices {a,b,c} such that |N(a) ∩ N(b) ∩ N(c)| >= 3.
    # Because if a,b,c share 3 common neighbours x,y,z, then {a,b,c} x {x,y,z} is K3,3.
    
    for a in range(n):
        na = adj[a]
        for b in range(a+1, n):
            nb = adj[b]
            # common neighbours of a and b
            common_ab = na & nb
            if len(common_ab) < 3:
                continue
            for c in range(b+1, n):
                nc = adj[c]
                common_abc = common_ab & nc
                if len(common_abc) >= 3:
                    return True
    return False

def _count_k33_violations(adj):
    """Count the number of triples (a,b,c) with >= 3 common neighbours."""
    n = len(adj)
    count = 0
    for a in range(n):
        na = adj[a]
        for b in range(a+1, n):
            nb = adj[b]
            common_ab = na & nb
            if len(common_ab) < 3:
                continue
            for c in range(b+1, n):
                nc = adj[c]
                common_abc = common_ab & nc
                if len(common_abc) >= 3:
                    count += 1
    return count

def _try_add_edge(adj, u, v):
    """Try to add edge (u,v) to adj. Returns True if valid, False otherwise."""
    if u == v or v in adj[u]:
        return False
    # Adding edge (u,v) creates new common neighbours for triples involving u or v.
    # Specifically, for any pair (x,y) where x != u,v and y != u,v:
    #   - If u is adjacent to both x and y, and v is adjacent to both x and y,
    #     then after adding (u,v), u and v become common neighbours of x and y.
    #     This could create a K3,3 if x and y already had 2 common neighbours.
    # More precisely: after adding (u,v), check if any triple has >= 3 common neighbours.
    # Efficient check: the only new triples that could become violating are those
    # that include u or v.
    
    # Triples including u: (u, x, y) for x < y, x,y != v
    #   Common neighbours of u, x, y after adding (u,v) = N(u) ∩ N(x) ∩ N(y)
    #   But N(u) now includes v. So we need to check if v is in N(x) ∩ N(y)
    #   and |N(u) ∩ N(x) ∩ N(y)| >= 3.
    #   Before adding, |N(u) ∩ N(x) ∩ N(y)| was some value k.
    #   After adding, it's k + (1 if v in N(x) ∩ N(y) else 0).
    #   So we need k + (1 if v in N(x) ∩ N(y) else 0) >= 3, i.e., k >= 2 and v in N(x) ∩ N(y),
    #   or k >= 3 (already violating, but we assume current graph is K3,3-free so k <= 2).
    #   So violation iff v in N(x) ∩ N(y) AND |N(u) ∩ N(x) ∩ N(y)| >= 2.
    
    # Similarly for triples including v.
    
    # Triples not including u or v: their common neighbours don't change, so no new violation.
    
    # Check triples with u: (u, x, y) where x, y != v, x < y
    nu = adj[u]
    nv = adj[v]
    for x in range(N):
        if x == u or x == v:
            continue
        nx = adj[x]
        # common neighbours of u and x (before adding)
        common_ux = nu & nx
        for y in range(x+1, N):
            if y == u or y == v:
                continue
            ny = adj[y]
            common_uxy = common_ux & ny
            if len(common_uxy) >= 2 and v in nx and v in ny:
                # After adding (u,v), v becomes a common neighbour of u,x,y
                # so total common neighbours = len(common_uxy) + 1 >= 3
                return False
    
    # Check triples with v: (v, x, y) where x, y != u, x < y
    for x in range(N):
        if x == u or x == v:
            continue
        nx = adj[x]
        common_vx = nv & nx
        for y in range(x+1, N):
            if y == u or y == v:
                continue
            ny = adj[y]
            common_vxy = common_vx & ny
            if len(common_vxy) >= 2 and u in nx and u in ny:
                return False
    
    # Also need to check triples (u, x, v) and (v, x, u) -- but these are covered
    # Actually, triple (u, x, v): common neighbours = N(u) ∩ N(x) ∩ N(v).
    # After adding (u,v), this doesn't change because u and v are in the triple, not as neighbours.
    # Wait, no. The common neighbours of vertices u, x, v are vertices z such that
    # z is adjacent to u, x, and v. Adding edge (u,v) doesn't affect which vertices
    # are adjacent to u, x, or v (it only adds the edge between u and v, which means
    # u is now in N(v) and v is in N(u), but for the triple (u,x,v), we're looking
    # at neighbours of u, x, v, so u being in N(v) doesn't matter since u is one of the triple).
    # So triples that include both u and v: (u, v, x). Common neighbours = N(u) ∩ N(v) ∩ N(x).
    # This doesn't change when we add (u,v). So no new violation from these.
    
    # Actually wait, I need to be more careful. Let me just do a full check for safety
    # since N=20 is small.
    
    # Full check: temporarily add the edge and verify
    adj[u].add(v)
    adj[v].add(u)
    result = not _has_k33(adj)
    if not result:
        adj[u].discard(v)
        adj[v].discard(u)
    return result

def _try_remove_edge(adj, u, v):
    """Remove edge (u,v) from adj."""
    adj[u].discard(v)
    adj[v].discard(u)

def _edge_swap(adj, u1, v1, u2, v2):
    """Try to remove edge (u1,v1) and add edge (u2,v2). Returns True if valid."""
    if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
        return False
    if v1 not in adj[u1]:
        return False
    if v2 in adj[u2]:
        return False
    
    # Remove (u1,v1), add (u2,v2), check K3,3-free
    adj[u1].discard(v1)
    adj[v1].discard(u1)
    adj[u2].add(v2)
    adj[v2].add(u2)
    
    if _has_k33(adj):
        # Revert
        adj[u2].discard(v2)
        adj[v2].discard(u2)
        adj[u1].add(v1)
        adj[v1].add(u1)
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 120 * DEADLINE_FRACTION
    
    best_adj = None
    best_count = 0
    
    # Start with a random dense-ish graph and use local search
    # Or start from the seed K2,18 and improve
    
    # Strategy: Start from K2,18 (36 edges), then use edge swaps to improve
    # Also try random restarts with random initial graphs
    
    def _make_adj_from_edges(edges):
        adj = [set() for _ in range(N)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def _edge_count(adj):
        return sum(len(s) for s in adj) // 2
    
    def _local_search(adj, max_iter=50000):
        """Add-only local search: try to add any missing edge."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle order for diversity
            pairs = [(u, v) for u in range(N) for v in range(u+1, N) if v not in adj[u]]
            random.shuffle(pairs)
            for u, v in pairs:
                if _try_add_edge(adj, u, v):
                    improved = True
                    if time.time() > deadline:
                        break
        return adj
    
    def _swap_local_search(adj, max_iter=50000):
        """Edge-swap local search: try to remove one edge and add another (or two)."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try single edge swaps: remove (u1,v1), add (u2,v2)
            existing = [(u, v) for u in range(N) for v in range(u+1, N) if v in adj[u]]
            missing = [(u, v) for u in range(N) for v in range(u+1, N) if v not in adj[u]]
            
            random.shuffle(existing)
            random.shuffle(missing)
            
            for u1, v1 in existing:
                if time.time() > deadline:
                    break
                for u2, v2 in missing:
                    if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                        continue
                    if _edge_swap(adj, u1, v1, u2, v2):
                        improved = True
                        # Rebuild missing list since edge changed
                        missing = [(u, v) for u in range(N) for v in range(u+1, N) if v not in adj[u]]
                        random.shuffle(missing)
                        break
                if improved:
                    break
        return adj
    
    # Phase 1: Start from K2,18 and do add-only search
    seed_edges = [[a, b] for a in range(2) for b in range(2, 20)]
    adj = _make_adj_from_edges(seed_edges)
    adj = _local_search(adj)
    adj = _swap_local_search(adj)
    count = _edge_count(adj)
    if count > best_count:
        best_count = count
        best_adj = [set(s) for s in adj]
    
    # Phase 2: Try random restarts
    restarts = 0
    while time.time() < deadline and restarts < 50:
        restarts += 1
        # Random initial graph with moderate density
        adj = [set() for _ in range(N)]
        # Add random edges, avoiding immediate K3,3
        all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if _try_add_edge(adj, u, v):
                pass
            if time.time() > deadline:
                break
        
        adj = _local_search(adj)
        adj = _swap_local_search(adj)
        count = _edge_count(adj)
        if count > best_count:
            best_count = count
            best_adj = [set(s) for s in adj]
    
    # Convert best_adj to edge list
    if best_adj is None:
        return seed_edges
    
    edges = []
    for u in range(N):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
