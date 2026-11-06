import random
import time
from itertools import combinations

def construct():
    n = 20
    deadline = time.time() + 115  # safety margin before 120s kill

    def has_k33(adj):
        """Check if graph with bitmask adj[] contains K3,3.
        K3,3 exists iff there exist 3 vertices u,v,w with >=3 common neighbors.
        """
        for u in range(n):
            for v in range(u+1, n):
                common_uv = adj[u] & adj[v]
                # For each w in common_uv, check if adj[u]&adj[v]&adj[w] has >=3 bits
                # Actually we need: exists u,v,w such that |N(u)∩N(v)∩N(w)| >= 3
                # More efficient: for each pair (u,v), look at common_uv,
                # then for each w in common_uv, count bits in adj[u]&adj[v]&adj[w]
                # But that's O(n^3) per check. Let's do it smarter.
                
        # Direct check: for each triple (u,v,w), count common neighbors
        for u in range(n):
            for v in range(u+1, n):
                cuv = adj[u] & adj[v]
                for w in range(v+1, n):
                    if (cuv & adj[w]).bit_count() >= 3:
                        return True
        return False

    def is_add_valid(adj, a, b):
        """Check if adding edge (a,b) keeps graph K3,3-free.
        Only need to check triples that include a or b, since only those change.
        """
        # Triples containing a: (a, x, y) where x,y in V\{a,b}
        # Common neighbors of (a,x,y) = N(a)∩N(x)∩N(y)
        # Before adding (a,b), N(a) didn't include b. After, it does.
        # So new common neighbors for triples with a and x,y: add b if b∈N(x)∩N(y)
        # Similarly for triples with b.
        
        na = adj[a]
        nb = adj[b]
        
        # Check triples (a, x, y): new common neighbor set = (na & adj[x] & adj[y]) 
        # but since we're adding edge a-b, b becomes in N(a). So if b in N(x) and b in N(y),
        # then b is a new common neighbor. The old common neighbors were (na & adj[x] & adj[y])
        # which already had < 3 (since graph was valid). So we just need to check if
        # b is in adj[x] & adj[y], and if so, old count + 1 >= 3, i.e., old count >= 2.
        
        for x in range(n):
            if x == a or x == b:
                continue
            ax = adj[x]
            if not (na & ax):  # a and x not adjacent, skip (common neighbors empty or small)
                # Actually they could still have common neighbors via other vertices
                pass
            # For triple (a, x, y), y ranges over V\{a,b,x}
            for y in range(n):
                if y == a or y == b or y == x:
                    continue
                ay = adj[y]
                # Common neighbors of a, x, y before adding edge:
                old_common = (na & ax & ay).bit_count()
                # After adding a-b: b is in N(a) now, so if b in N(x) and b in N(y):
                if (nb & ax & ay) & (1 << b):  # b is neighbor of x and y
                    # Wait, nb is N(b). b in N(x) means x in N(b) means bit x in nb.
                    # Let me re-check: adj[b] is bitmask of neighbors of b.
                    # b in N(x) iff x in adj[b] iff (adj[b] >> x) & 1
                    # b in N(y) iff y in adj[b] iff (adj[b] >> y) & 1
                    # b in N(a) after adding: yes, by construction.
                    # So new common neighbor b exists iff x in adj[b] and y in adj[b]
                    if (adj[b] & (1 << x)) and (adj[b] & (1 << y)):
                        new_count = old_common + 1
                        if new_count >= 3:
                            return False
                # Also need to check if a in N(x) and a in N(y) for the symmetric case?
                # No, we're only adding edge a-b. a's neighborhood gains b, b's gains a.
                # For triple (a,x,y): common neighbors gain b only if b∈N(x)∩N(y).
                # a is already in N(a) trivially? No, a is not in N(a) (no self-loops).
                # So a is not a common neighbor of (a,x,y) since a∉N(a).
                # Similarly for triple (b,x,y): common neighbors gain a only if a∈N(x)∩N(y).
        
        # Check triples (b, x, y):
        for x in range(n):
            if x == a or x == b:
                continue
            ax = adj[x]
            for y in range(n):
                if y == a or y == b or y == x:
                    continue
                ay = adj[y]
                old_common = (nb & ax & ay).bit_count()
                # a is new neighbor of b. a in N(x) and a in N(y)?
                if (adj[a] & (1 << x)) and (adj[a] & (1 << y)):
                    new_count = old_common + 1
                    if new_count >= 3:
                        return False
        
        return True

    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def random_start(seed):
        """Generate a random K3,3-free graph."""
        rng = random.Random(seed)
        adj = [0] * n
        # Start with random sparse graph and greedily add edges
        order = list(range(n * (n - 1) // 2))
        rng.shuffle(order)
        # Map linear index to (u,v)
        def idx_to_uv(idx):
            # idx = u*n - u*(u+1)//2 + (v - u - 1) for v > u
            # Simpler: generate all pairs
            pass
        pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
        rng.shuffle(pairs)
        for (u, v) in pairs:
            if is_add_valid(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj):
        """Local search: greedy add, then 1-swap, then 2-swap."""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Greedy add: try to add any missing edge
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] & (1 << v)):
                        if is_add_valid(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            improved = True
            if not improved:
                break
            
            # 1-swap: remove edge (a,b), try to add edge (c,d)
            if time.time() >= deadline - 5:
                break
            found_swap = False
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] & (1 << v)):
                        continue
                    # Try removing (u,v)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    # Try adding some other edge
                    for c in range(n):
                        for d in range(c+1, n):
                            if (c == u and d == v) or (c == v and d == u):
                                continue
                            if not (adj[c] & (1 << d)):
                                if is_add_valid(adj, c, d):
                                    adj[c] |= (1 << d)
                                    adj[d] |= (1 << c)
                                    found_swap = True
                                    improved = True
                                    break
                        if found_swap:
                            break
                    if not found_swap:
                        # Restore
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    else:
                        break
                if found_swap:
                    break
            if not improved:
                break
        
        return adj

    best_adj = None
    best_count = 0

    # Multiple restarts
    for seed in range(50):
        if time.time() >= deadline - 10:
            break
        adj = random_start(seed)
        adj = local_search(adj)
        cnt = edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
    
    # Final greedy add pass
    if best_adj is not None:
        for u in range(n):
            for v in range(u+1, n):
                if not (best_adj[u] & (1 << v)):
                    if is_add_valid(best_adj, u, v):
                        best_adj[u] |= (1 << v)
                        best_adj[v] |= (1 << u)
    
    if best_adj is None:
        # Fallback: K2,18
        adj = [0] * n
        for a in range(2):
            for b in range(2, 20):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        best_adj = adj

    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    return edges
