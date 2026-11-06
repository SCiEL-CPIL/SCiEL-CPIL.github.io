import time
import random
import math

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 118  # Safety margin

    def build_books(adj, edges):
        books = {}
        for u, v in edges:
            books[(u, v)] = (adj[u] & adj[v]).bit_count()
        return books

    def get_max_sum(books):
        if not books:
            return 0, 0
        mx = max(books.values())
        s = sum(books.values())
        return mx, s

    def make_bipartite_plus(seed):
        rng = random.Random(seed)
        adj = [0] * n
        edges = set()
        for a in range(20):
            for b in range(20, 40):
                edges.add((a, b))
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        part = rng.choice([list(range(20)), list(range(20, 40))])
        u, v = rng.sample(part, 2)
        u, v = min(u, v), max(u, v)
        edges.add((u, v))
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return adj, edges

    def local_search(adj, edges, seed, time_limit):
        rng = random.Random(seed)
        books = build_books(adj, edges)
        cur_max, cur_sum = get_max_sum(books)
        best_max, best_sum = cur_max, cur_sum
        best_edges = edges.copy()
        
        T = 2.0
        cooling = 0.999
        min_T = 0.001
        stall = 0
        max_stall = 500
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            if T <= min_T:
                break
            
            # Pick edge to remove
            # Bias towards high book edges
            # To speed up, we can just pick a random edge, but high book edges are better candidates for removal if we want to reduce max
            # However, we need to maintain connectivity/structure? No, just min max book.
            # Let's pick from top 20%
            if rng.random() < 0.7:
                # Sort is expensive. Let's just pick random and check if it's "bad"
                # Or maintain a list of edges sorted by book? Too expensive to update.
                # Just pick random edge.
                pass
            
            e_rem = rng.choice(list(books.keys()))
            u1, v1 = e_rem
            
            adj_bak = adj[:]
            books_bak = books.copy()
            edges_bak = edges.copy()
            
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            edges.discard(e_rem)
            del books[e_rem]
            
            # Update books for neighbors of u1 and v1
            # Neighbors of u1: w where adj[u1] & (1<<w)
            # If w is also neighbor of v1, then (u1,w) and (v1,w) lose a triangle
            # Specifically, edge (u1, w) loses 1 if v1 is common neighbor.
            # Edge (v1, w) loses 1 if u1 is common neighbor.
            
            # Optimization: Iterate over intersection of neighbors?
            # Or just iterate over all w? N=40 is small.
            for w in range(n):
                if w == u1 or w == v1:
                    continue
                is_u1_w = bool(adj[u1] & (1 << w))
                is_v1_w = bool(adj[v1] & (1 << w))
                
                if is_u1_w and is_v1_w:
                    k1 = (min(u1, w), max(u1, w))
                    if k1 in books:
                        books[k1] -= 1
                    k2 = (min(v1, w), max(v1, w))
                    if k2 in books:
                        books[k2] -= 1
                elif is_u1_w:
                    # w is neighbor of u1 but not v1.
                    # Removing (u1, v1) does not affect triangles containing (u1, w) because v1 is not a common neighbor of u1 and w?
                    # Wait. Triangle (u1, w, v1) requires edges (u1,w), (w,v1), (v1,u1).
                    # We are removing (u1, v1).
                    # So any triangle containing (u1, v1) is broken.
                    # Triangles containing (u1, w) are (u1, w, x) where x is common neighbor of u1 and w.
                    # Removing (u1, v1) removes v1 from N(u1).
                    # If v1 was a common neighbor of u1 and w, then (u1, w) loses a triangle.
                    # v1 is common neighbor of u1 and w iff (u1, v1) and (v1, w) exist.
                    # We know (u1, v1) exists (before removal).
                    # So if (v1, w) exists, then v1 was a common neighbor.
                    # So if is_v1_w is True, then (u1, w) loses a triangle.
                    # My previous logic: `if is_u1_w and is_v1_w` -> update k1 and k2.
                    # This is correct.
                    # If is_u1_w is True and is_v1_w is False, then v1 is NOT a common neighbor of u1 and w.
                    # So (u1, w) does NOT lose a triangle.
                    pass

            # Find best add
            best_add = None
            best_pot = 999
            # Sample non-edges
            # We want to add an edge that creates few triangles.
            # pot = common neighbors.
            # We want min pot.
            for _ in range(30):
                u2 = rng.randint(0, n - 1)
                v2 = rng.randint(0, n - 1)
                if u2 == v2:
                    continue
                u2, v2 = min(u2, v2), max(u2, v2)
                if (u2, v2) in edges:
                    continue
                pot = (adj[u2] & adj[v2]).bit_count()
                if pot < best_pot:
                    best_pot = pot
                    best_add = (u2, v2)
            
            if best_add is None:
                adj = adj_bak
                books = books_bak
                edges = edges_bak
                T *= cooling
                continue
            
            u2, v2 = best_add
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            edges.add(best_add)
            books[best_add] = best_pot
            
            # Update books for neighbors
            for w in range(n):
                if w == u2 or w == v2:
                    continue
                is_u2_w = bool(adj[u2] & (1 << w))
                is_v2_w = bool(adj[v2] & (1 << w))
                
                if is_u2_w and is_v2_w:
                    k1 = (min(u2, w), max(u2, w))
                    if k1 in books:
                        books[k1] += 1
                    k2 = (min(v2, w), max(v2, w))
                    if k2 in books:
                        books[k2] += 1

            new_max, new_sum = get_max_sum(books)
            
            # Accept/Reject
            accept = False
            if new_max < cur_max:
                accept = True
                stall = 0
            elif new_max == cur_max and new_sum < cur_sum:
                accept = True
                stall = 0
            else:
                delta = (new_max - cur_max) * 10 + (new_sum - cur_sum)
                if delta < 0:
                    accept = True
                    stall = 0
                else:
                    if rng.random() < math.exp(-delta / T):
                        accept = True
                    else:
                        stall += 1
                        if stall > max_stall:
                            T = max(T * 0.5, min_T)
                            stall = 0
            
            if accept:
                cur_max, cur_sum = new_max, new_sum
                if cur_max < best_max or (cur_max == best_max and cur_sum < best_sum):
                    best_max, best_sum = cur_max, cur_sum
                    best_edges = edges.copy()
            else:
                adj = adj_bak
                books = books_bak
                edges = edges_bak
            
            T *= cooling
            
            if best_max <= 7:
                break
        
        return best_max, best_sum, best_edges

    global_best = None
    global_best_max = 999
    global_best_sum = 999
    
    # Phase 1: Bipartite + 1
    for seed in range(50):
        if time.time() > deadline - 5:
            break
        adj, edges = make_bipartite_plus(seed)
        tm = max(1.0, (deadline - 5 - time.time()) / 50)
        bm, bs, be = local_search(adj, edges, seed, tm)
        if bm < global_best_max or (bm == global_best_max and bs < global_best_sum):
            global_best_max, global_best_sum = bm, bs
            global_best = be
        if global_best_max <= 7:
            break
            
    if global_best is None:
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return [[u, v] for (u, v) in sorted(global_best)]
