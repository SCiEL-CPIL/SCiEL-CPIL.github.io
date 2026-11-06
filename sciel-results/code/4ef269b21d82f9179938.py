import time
import random
import math
import numpy as np

def construct():
    n = 40
    target_edges = 401
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s limit

    def make_bipartite_plus(seed):
        """Create K_{20,20} + one edge within a part."""
        rng = random.Random(seed)
        adj = [0] * n
        edges = set()
        
        # Complete bipartite K_{20,20}
        for a in range(20):
            for b in range(20, 40):
                edges.add((a, b))
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        
        # Add one edge within a part to reach 401 edges
        # Randomly choose which part and which two vertices
        if rng.random() < 0.5:
            part = list(range(20))
        else:
            part = list(range(20, 40))
        
        u, v = rng.sample(part, 2)
        u, v = min(u, v), max(u, v)
        edges.add((u, v))
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        return adj, edges

    def compute_books_fast(adj, edges):
        """Compute book size for each edge using numpy."""
        # Build adjacency matrix
        A = np.zeros((n, n), dtype=np.uint8)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
        
        # Book size for edge (u,v) = number of common neighbors = (A^2)[u,v]
        A2 = A @ A
        
        books = {}
        for u, v in edges:
            books[(u, v)] = int(A2[u, v])
        
        return books, A

    def local_search(adj, edges, seed, time_limit):
        """Simulated annealing local search to minimize max book size."""
        rng = random.Random(seed)
        books, A = compute_books_fast(adj, edges)
        
        cur_max = max(books.values()) if books else 0
        cur_sum = sum(books.values())
        
        best_max = cur_max
        best_sum = cur_sum
        best_edges = edges.copy()
        
        T = 1.0
        cooling = 0.9995
        min_T = 0.001
        stall = 0
        max_stall = 200
        
        end_time = time.time() + time_limit
        edge_list = list(edges)
        
        while time.time() < end_time:
            if T <= min_T:
                break
            
            # Pick an edge to remove
            e_rem = rng.choice(edge_list)
            u1, v1 = e_rem
            
            # Save state
            adj_bak = adj[:]
            books_bak = books.copy()
            A_bak = A.copy()
            edges_bak = edges.copy()
            edge_list_bak = edge_list[:]
            
            # Remove edge
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            edges.discard(e_rem)
            edge_list.remove(e_rem)
            del books[e_rem]
            A[u1, v1] = 0
            A[v1, u1] = 0
            
            # Update books for affected edges
            # When removing (u1, v1), any triangle containing (u1, v1) is broken.
            common = adj[u1] & adj[v1]
            for w in range(n):
                if common & (1 << w):
                    k1 = (min(u1, w), max(u1, w))
                    if k1 in books:
                        books[k1] -= 1
                    k2 = (min(v1, w), max(v1, w))
                    if k2 in books:
                        books[k2] -= 1
            
            # Find best edge to add - sample more candidates
            best_add = None
            best_pot = 999
            # Sample non-edges, prefer those with low common neighbor count
            for _ in range(80):
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
                # Restore
                adj = adj_bak
                books = books_bak
                A = A_bak
                edges = edges_bak
                edge_list = edge_list_bak
                T *= cooling
                continue
            
            u2, v2 = best_add
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            edges.add(best_add)
            edge_list.append(best_add)
            books[best_add] = best_pot
            A[u2, v2] = 1
            A[v2, u2] = 1
            
            # Update books for affected edges
            # When adding (u2, v2), new triangles are formed with common neighbors.
            common = adj[u2] & adj[v2]
            for w in range(n):
                if common & (1 << w):
                    k1 = (min(u2, w), max(u2, w))
                    if k1 in books:
                        books[k1] += 1
                    k2 = (min(v2, w), max(v2, w))
                    if k2 in books:
                        books[k2] += 1
            
            new_max = max(books.values()) if books else 0
            new_sum = sum(books.values())
            
            # Accept/Reject with smoother objective
            accept = False
            if new_max < cur_max:
                accept = True
                stall = 0
            elif new_max == cur_max and new_sum < cur_sum:
                accept = True
                stall = 0
            else:
                # Use a softer penalty that considers distribution
                delta = (new_max - cur_max) * 8 + (new_sum - cur_sum) * 0.5
                if delta < 0:
                    accept = True
                    stall = 0
                else:
                    if T > min_T and rng.random() < math.exp(-delta / T):
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
                A = A_bak
                edges = edges_bak
                edge_list = edge_list_bak
            
            T *= cooling
            
            if best_max <= 7:
                break
        
        return best_max, best_sum, best_edges

    global_best = None
    global_best_max = 999
    global_best_sum = 999
    
    # Multiple restarts with different seeds
    for seed in range(100):
        if time.time() > deadline - 3:
            break
        
        adj, edges = make_bipartite_plus(seed)
        
        # Allocate time based on remaining budget
        remaining = deadline - 3 - time.time()
        tm = max(0.5, remaining / (100 - seed))
        
        bm, bs, be = local_search(adj, edges, seed, tm)
        
        if bm < global_best_max or (bm == global_best_max and bs < global_best_sum):
            global_best_max, global_best_sum = bm, bs
            global_best = be
        
        if global_best_max <= 7:
            break
    
    if global_best is None:
        # Fallback: K_{20,20} + one edge
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    
    return [[u, v] for (u, v) in sorted(global_best)]
