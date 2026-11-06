import time
import random
import sys

def construct():
    """
    Construct a K2,3-free graph on 20 vertices maximizing edges.
    Uses greedy construction with multiple restarts and local search.
    """
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def count_common(adj, u, v):
        """Count common neighbors of u and v using bitmask."""
        return (adj[u] & adj[v]).bit_count()

    def is_valid_add(adj, u, v):
        """
        Check if adding edge (u,v) keeps graph K2,3-free.
        A K2,3 is a set of 5 vertices {u,v,a,b,c} where u,v connect to a,b,c.
        So no two vertices can share 3 or more common neighbors.
        
        When we add edge (u,v), we need to check:
        1. For every pair (x,y) where x,y are common neighbors of u and v after adding (u,v),
           we need common_neighbours(x,y) <= 2.
        
        Actually, the constraint is: for ANY pair of vertices (a,b), |N(a) ∩ N(b)| <= 2.
        
        Adding edge (u,v) changes:
        - N(u) gains v, N(v) gains u
        - For any other vertex w: if w is neighbor of u, then w gains v as a potential common neighbor with... 
          Actually let's think more carefully.
          
        The pairs that could become problematic are:
        - (u, v): they now share all their common neighbors. Need |N(u) ∩ N(v)| <= 2 after adding.
          Before adding, N(u) and N(v) are disjoint from each other in terms of the edge.
          After adding, N(u) ∩ N(v) = (old N(u) ∩ old N(v)) ∪ {v if v in N(u)... no}
          Actually after adding edge (u,v): v ∈ N(u) and u ∈ N(v).
          N(u) ∩ N(v) after = (N(u) before ∪ {v}) ∩ (N(v) before ∪ {u})
          = (N(u) before ∩ N(v) before) ∪ (N(u) before ∩ {u}) ∪ ({v} ∩ N(v) before) ∪ ({v} ∩ {u})
          = (N(u) before ∩ N(v) before) since u ∉ N(u), v ∉ N(v), and u ≠ v.
          Wait, that's not right either. Let me reconsider.
          
          Before: N(u) = A, N(v) = B, u ∉ A, v ∉ B, and (u,v) not an edge so v ∉ A and u ∉ B.
          After adding (u,v): N'(u) = A ∪ {v}, N'(v) = B ∪ {u}.
          N'(u) ∩ N'(v) = (A ∪ {v}) ∩ (B ∪ {u}) = (A ∩ B) ∪ (A ∩ {u}) ∪ ({v} ∩ B) ∪ ({v} ∩ {u})
          = (A ∩ B) ∪ ∅ ∪ (∅ if v∉B) ∪ ∅ = A ∩ B.
          
          So |N'(u) ∩ N'(v)| = |A ∩ B| = number of common neighbors before. Need this <= 2.
          
        - For any vertex w ≠ u,v: 
          If w ∈ A (w is neighbor of u before): 
            N'(w) is unchanged unless w = v, but w ≠ v.
            Actually N'(w) = N(w) if w ≠ u,v.
            But the pairs involving w: 
            Pair (w, u): N'(w) ∩ N'(u) = N(w) ∩ (A ∪ {v}) = (N(w) ∩ A) ∪ (N(w) ∩ {v}).
            Before: N(w) ∩ A. Now we add v if v ∈ N(w), i.e., if (w,v) is an edge.
            So if (w,v) is an edge, the common neighbor count for pair (w,u) increases by 1.
            We need (N(w) ∩ A) ∪ {v if (w,v) edge} to have size <= 2.
            i.e., |N(w) ∩ A| + (1 if (w,v) is edge else 0) <= 2.
            
            Similarly for pair (w, v): N'(w) ∩ N'(v) = N(w) ∩ (B ∪ {u}) = (N(w) ∩ B) ∪ {u if (w,u) edge}.
            Need |N(w) ∩ B| + (1 if (w,u) edge else 0) <= 2.
            
            For pair (w, x) where w,x ≠ u,v: N'(w) ∩ N'(x) = N(w) ∩ N(x), unchanged.
            
        So the checks are:
        1. |N(u) ∩ N(v)| <= 2 (common neighbors of u and v before adding)
        2. For each w in N(u): |N(w) ∩ N(u)| + (1 if (w,v) in edges else 0) <= 2
           But N(w) ∩ N(u) = N(w) ∩ A. Note w ∈ A means w is in N(u).
           Also, is u in N(w)? Yes, since (u,w) is an edge. So u ∈ N(w).
           N(w) ∩ N(u) includes... let's just compute it.
           Actually, let me just compute directly.
        3. For each w in N(v): |N(w) ∩ N(v)| + (1 if (w,u) in edges else 0) <= 2
        """
        # Check 1: common neighbors of u and v
        if count_common(adj, u, v) > 2:
            return False
        
        # Check 2: for each neighbor w of u (excluding v, but v not in N(u) yet)
        # Pair (w, u): new common neighbors = N(w) ∩ (N(u) ∪ {v})
        # = (N(w) ∩ N(u)) ∪ ({v} if v in N(w))
        # Size = |N(w) ∩ N(u)| + (1 if v in N(w) else 0)
        # Note: v in N(w) means edge (w,v) exists
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[u] & (1 << w):  # w is neighbor of u
                common_wu = count_common(adj, w, u)
                has_wv = bool(adj[v] & (1 << w))  # edge (w,v) exists
                if common_wu + (1 if has_wv else 0) > 2:
                    return False
        
        # Check 3: for each neighbor w of v
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v] & (1 << w):  # w is neighbor of v
                common_wv = count_common(adj, w, v)
                has_wu = bool(adj[u] & (1 << w))  # edge (w,u) exists
                if common_wv + (1 if has_wu else 0) > 2:
                    return False
        
        return True

    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    def is_k23_free(adj):
        """Verify the entire graph is K2,3-free."""
        for u in range(n):
            for v in range(u + 1, n):
                if count_common(adj, u, v) > 2:
                    return False
        return True

    def greedy_construct(order_seed=None):
        """Greedy construction: try to add edges in a given order."""
        adj = [0] * n
        if order_seed is not None:
            rng = random.Random(order_seed)
        else:
            rng = random.Random()
        
        # Generate all possible edges
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)
        
        for u, v in all_edges:
            if is_valid_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        return adj

    def local_search(adj, max_iter=1000):
        """Try to improve by swapping edges."""
        adj = adj[:]  # copy
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > deadline - 2:
                break
            
            # Try removing one edge and adding another
            edges = adj_to_edges(adj)
            random.shuffle(edges)
            
            for e1 in edges:
                u1, v1 = e1
                # Try removing e1
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                
                # Try adding a random non-edge
                all_non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) 
                                  if not (adj[u] & (1 << v))]
                if not all_non_edges:
                    # Restore
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    continue
                
                random.shuffle(all_non_edges)
                found = False
                for u2, v2 in all_non_edges[:50]:  # Limit attempts
                    if is_valid_add(adj, u2, v2):
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        found = True
                        improved = True
                        break
                
                if not found:
                    # Restore
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
        
        return adj

    best_adj = None
    best_count = 0

    # Phase 1: Multiple greedy restarts
    num_restarts = 0
    while time.time() < deadline - 30:
        seed = num_restarts
        adj = greedy_construct(order_seed=seed)
        cnt = edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
        num_restarts += 1

    # Phase 2: Local search on best and some good solutions
    if best_adj is not None:
        improved_adj = local_search(best_adj, max_iter=2000)
        cnt = edge_count(improved_adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = improved_adj

    # Phase 3: More restarts with local search if time permits
    while time.time() < deadline - 5:
        seed = num_restarts
        adj = greedy_construct(order_seed=seed)
        cnt = edge_count(adj)
        if cnt >= best_count - 2:  # Only do local search if close to best
            improved_adj = local_search(adj, max_iter=500)
            cnt2 = edge_count(improved_adj)
            if cnt2 > best_count:
                best_count = cnt2
                best_adj = improved_adj
        num_restarts += 1

    # Final verification
    if best_adj is None:
        # Fallback: five disjoint K4s
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges

    if not is_k23_free(best_adj):
        # Should not happen, but fallback
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges

    return adj_to_edges(best_adj)
