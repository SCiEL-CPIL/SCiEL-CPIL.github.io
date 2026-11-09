import time
import random
import numpy as np

def construct():
    n = 31
    m = 91
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    def count_c4(adj):
        """Count 4-cycles: sum over all pairs (u,v) of C(common_neighbors(u,v), 2)"""
        # common[u][v] = number of common neighbors of u and v
        # For each pair, C4s contributed = common*(common-1)//2
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                # Count common neighbors
                # adj[u] & adj[v] gives common neighbors
                common = np.sum(adj[u] & adj[v])
                total += common * (common - 1) // 2
        return total
    
    def count_c4_fast(adj):
        """Faster C4 count using matrix multiplication"""
        # A^2[u][v] = number of common neighbors of u and v
        A2 = adj @ adj
        # For each pair (u,v), C4s = C(A2[u][v], 2)
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                c = A2[u][v]
                total += c * (c - 1) // 2
        return total
    
    def random_graph(m):
        """Generate a random graph with m edges"""
        adj = np.zeros((n, n), dtype=np.int8)
        edges = set()
        while len(edges) < m:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                u, v = min(u, v), max(u, v)
                if (u, v) not in edges:
                    edges.add((u, v))
                    adj[u][v] = 1
                    adj[v][u] = 1
        return adj, edges
    
    def swap_delta(adj, edges, remove_edge, add_edge):
        """Compute delta in C4 count when removing remove_edge and adding add_edge"""
        a, b = remove_edge
        c, d = add_edge
        
        # If the edges are the same, no change
        if (a, b) == (c, d) or (a, b) == (d, c):
            return 0
        
        # Check validity: remove_edge must exist, add_edge must not exist
        if not adj[a][b]:
            return 10**9  # Invalid
        if adj[c][d]:
            return 10**9  # Invalid
        
        # Compute delta in C4 count
        # When we remove edge (a,b):
        #   For each neighbor x of a (x != b): common(a,x) decreases by 1 (because b was a common neighbor via edge a-b... wait no)
        #   Actually, removing edge (a,b) means:
        #     - a and b are no longer adjacent
        #     - For any vertex x that is neighbor of both a and b (x != a, x != b), 
        #       the path a-x-b no longer exists as a-b edge removed, but a-x and x-b still exist
        #       So common neighbors of a and b decrease by 1 (they lose each other as neighbors? No...)
        
        # Let me think more carefully.
        # C4 count = sum_{u<v} C(common(u,v), 2)
        # where common(u,v) = |N(u) ∩ N(v)|
        
        # When we remove edge (a,b):
        #   N(a) loses b, N(b) loses a
        #   For pair (a,b): common(a,b) was |N(a)∩N(b)|, now it's |N(a)∩N(b)| - 1? 
        #   No: N(a) and N(b) both change. New N(a) = N(a)\{b}, New N(b) = N(b)\{a}
        #   common(a,b) new = |N(a)\{b} ∩ N(b)\{a}| = |N(a)∩N(b)| - |{b}∩N(b)| - |N(a)∩{a}| + |{a}∩{b}|
        #   Since b is in N(b) and a is in N(a), but a is not in N(b) (no self-loop) and b is not in N(a)
        #   Wait, a is in N(b) because (a,b) is an edge, and b is in N(a) because (a,b) is an edge.
        #   So: common(a,b) new = |N(a)∩N(b)| - 1 - 1 + 0 = common(a,b) - 2? 
        #   Hmm, let me reconsider. N(a) includes b, N(b) includes a.
        #   N(a)∩N(b): vertices that are neighbors of both a and b. Does b count? b is in N(a) but is b in N(b)? No (no self-loop). Does a count? a is in N(b) but is a in N(a)? No.
        #   So common(a,b) = |N(a)∩N(b)| doesn't include a or b themselves.
        #   After removing (a,b): N'(a) = N(a)\{b}, N'(b) = N(b)\{a}
        #   common'(a,b) = |N'(a)∩N'(b)| = |(N(a)\{b})∩(N(b)\{a})|
        #   = |N(a)∩N(b)| - |{b}∩N(b)| - |N(a)∩{a}| + |{a}∩{b}|
        #   b ∈ N(b)? No. a ∈ N(a)? No. So = |N(a)∩N(b)| - 0 - 0 + 0 = common(a,b)
        #   Wait that's not right either. Let me just compute directly.
        
        # Actually, the cleanest way: just recompute the affected terms.
        # Removing (a,b) affects common(u,v) for pairs where u or v is a or b, 
        # AND the common neighbor count changes.
        
        # Pairs affected by removing (a,b):
        # - Pair (a, x) for x in N(b) (x != a): common(a,x) changes because b is no longer neighbor of a
        #   Actually common(a,x) = |N(a)∩N(x)|. After removal, N(a) loses b.
        #   So common(a,x) new = |N(a)∩N(x)| - 1 if b ∈ N(x), else same.
        #   b ∈ N(x) means x ∈ N(b). So for x in N(b), x != a: common(a,x) decreases by 1.
        # - Pair (b, x) for x in N(a) (x != b): common(b,x) decreases by 1 if a ∈ N(x), i.e., x ∈ N(a).
        #   So for x in N(a), x != b: common(b,x) decreases by 1.
        # - Pair (a, b): common(a,b) - as computed above, stays the same? Let me verify with small example.
        
        # This is getting complex. Let me just do a direct but efficient delta computation.
        
        # Better approach: compute the change in sum_{u<v} C(common(u,v), 2)
        # by tracking which common(u,v) values change.
        
        # For removing edge (a,b):
        #   For each x in N(b) \ {a}: common(a,x) decreases by 1 (b was in N(a)∩N(x), now not in N(a))
        #     Wait: b ∈ N(a) always (since edge a-b exists). b ∈ N(x) iff x ∈ N(b).
        #     So if x ∈ N(b) and x != a, then b ∈ N(a)∩N(x), so common(a,x) includes b.
        #     After removing (a,b), b ∉ N(a), so b ∉ N(a)∩N(x), so common(a,x) decreases by 1.
        #   Similarly for each x in N(a) \ {b}: common(b,x) decreases by 1.
        #   For pair (a,b): common(a,b) = |N(a)∩N(b)|. 
        #     b ∉ N(b) (no self loop), a ∉ N(a). So common(a,b) doesn't include a or b.
        #     After removal, N'(a) = N(a)\{b}, N'(b) = N(b)\{a}.
        #     common'(a,b) = |N'(a)∩N'(b)|. 
        #     N(a)∩N(b) contains vertices w where w∈N(a) and w∈N(b). 
        #     w=b: b∈N(a) yes, b∈N(b) no. So b not in intersection.
        #     w=a: a∈N(a) no. So a not in intersection.
        #     So N(a)∩N(b) = N'(a)∩N'(b) ∪ (possible a or b, but they're excluded).
        #     Actually N'(a)∩N'(b) = (N(a)\{b})∩(N(b)\{a}) = N(a)∩N(b) \ ({a}∪{b}) but a∉N(a)∩N(b) and b∉N(a)∩N(b), so it's the same.
        #     So common(a,b) is unchanged.
        
        # So delta for removing (a,b):
        #   For each x in N(b) \ {a}: 
        #     old_c = common(a,x), new_c = old_c - 1
        #     delta += C(new_c,2) - C(old_c,2) = (new_c*(new_c-1) - old_c*(old_c-1))//2
        #     = ((old_c-1)*(old_c-2) - old_c*(old_c-1))//2 = (old_c-1)*((old_c-2)-old_c)//2 = (old_c-1)*(-2)//2 = -(old_c-1)
        #   For each x in N(a) \ {b}:
        #     Similarly delta += -(common(b,x) - 1)
        
        # For adding edge (c,d):
        #   For each x in N(d) \ {c}: common(c,x) increases by 1
        #     delta += C(new_c,2) - C(old_c,2) where new_c = old_c+1
        #     = ((old_c+1)*old_c - old_c*(old_c-1))//2 = old_c*(old_c+1 - (old_c-1))//2 = old_c*2//2 = old_c
        #   For each x in N(c) \ {d}:
        #     delta += common(d,x)
        
        # Total delta = sum over affected pairs
        
        delta = 0
        
        # Removing (a,b)
        for x in range(n):
            if x == a or x == b:
                continue
            if adj[a][x] and adj[b][x]:
                # x is common neighbor of a and b, so x ∈ N(a) and x ∈ N(b)
                # This affects pair (a,x): common(a,x) decreases by 1
                common_ax = np.sum(adj[a] & adj[x])
                delta -= (common_ax - 1)  # C(c-1,2) - C(c,2) = -(c-1)
                # This affects pair (b,x): common(b,x) decreases by 1
                common_bx = np.sum(adj[b] & adj[x])
                delta -= (common_bx - 1)
        
        # Adding (c,d)
        for x in range(n):
            if x == c or x == d:
                continue
            if adj[c][x] and adj[d][x]:
                # x is common neighbor of c and d
                # This affects pair (c,x): common(c,x) increases by 1
                common_cx = np.sum(adj[c] & adj[x])
                delta += common_cx  # C(c+1,2) - C(c,2) = c
                # This affects pair (d,x): common(d,x) increases by 1
                common_dx = np.sum(adj[d] & adj[x])
                delta += common_dx
        
        return delta
    
    def hill_climb(adj, edges, max_iter=5000):
        """Hill climbing to minimize C4 count via edge swaps"""
        current_c4 = count_c4_fast(adj)
        best_c4 = current_c4
        best_adj = adj.copy()
        best_edges = set(edges)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            # Try random swaps
            for _ in range(100):
                if time.time() > deadline:
                    break
                
                # Pick a random edge to remove
                remove_edge = random.choice(list(edges))
                
                # Pick a random non-edge to add
                while True:
                    c = random.randint(0, n - 1)
                    d = random.randint(0, n - 1)
                    if c != d:
                        c, d = min(c, d), max(c, d)
                        if (c, d) not in edges:
                            break
                
                delta = swap_delta(adj, edges, remove_edge, (c, d))
                
                if delta < 0:
                    # Accept the swap
                    a, b = remove_edge
                    adj[a][b] = 0
                    adj[b][a] = 0
                    edges.discard(remove_edge)
                    edges.add((c, d))
                    adj[c][d] = 1
                    adj[d][c] = 1
                    
                    current_c4 += delta
                    improved = True
                    
                    if current_c4 < best_c4:
                        best_c4 = current_c4
                        best_adj = adj.copy()
                        best_edges = set(edges)
            
            if not improved:
                break
        
        return best_adj, best_edges, best_c4
    
    # Multiple random restarts
    best_overall_c4 = float('inf')
    best_overall_adj = None
    best_overall_edges = None
    
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        adj, edges = random_graph(m)
        adj_hc, edges_hc, c4_hc = hill_climb(adj, edges)
        
        if c4_hc < best_overall_c4:
            best_overall_c4 = c4_hc
            best_overall_adj = adj_hc
            best_overall_edges = edges_hc
        
        # Early stop if we found 0 or 1 C4s
        if best_overall_c4 <= 1:
            break
    
    # Convert to edge list
    result = []
    for (u, v) in best_overall_edges:
        result.append([u, v])
    
    # Verify we have exactly m edges
    assert len(result) == m, f"Expected {m} edges, got {len(result)}"
    # Verify all vertices are in range
    for edge in result:
        assert 0 <= edge[0] <= n - 1
        assert 0 <= edge[1] <= n - 1
        assert edge[0] != edge[1]
    
    return result
