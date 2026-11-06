import time
import random
import math

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 115  # Safety margin for 120s limit

    def get_books(adj, edges):
        books = {}
        max_b = 0
        sum_b = 0
        for u, v in edges:
            common = adj[u] & adj[v]
            b = common.bit_count()
            books[(u, v)] = b
            if b > max_b:
                max_b = b
            sum_b += b
        return max_b, sum_b, books

    def try_move(adj, edges, books, max_b, sum_b, T):
        # Remove a random edge, prefer high book
        # To be fast, we don't sort every time. We pick a random edge.
        # Or we pick from a small list of high book edges.
        # Let's just pick random for now, it's simpler and often works if we run many iterations.
        # But targeted removal is better.
        
        # Find a good edge to remove: one with high book size
        # We have 'books' dict.
        # Picking max book edge is O(m).
        # Let's pick a random edge that has book size > average
        # Or just random.
        
        # Actually, let's try to remove an edge that is part of the current max book.
        # If we remove the max book edge, max_b might drop or stay same.
        
        # Strategy:
        # 1. Pick a random edge to remove.
        # 2. Pick the best edge to add (lowest common neighbors).
        
        e_rem = random.choice(list(edges))
        u1, v1 = e_rem
        
        # Temporarily remove
        adj[u1] &= ~(1 << v1)
        adj[v1] &= ~(1 << u1)
        edges.discard(e_rem)
        
        # Find best edge to add
        best_add = None
        best_score = 999
        # Sample a subset of non-edges to save time? No, 1600 is small.
        # But doing it every step might be slow if we do 100k steps.
        # 100k * 1600 = 160M ops. Too slow in Python.
        
        # Optimization: Only check a random subset of non-edges, or maintain a list of candidates.
        # Let's just pick a random non-edge.
        # If we pick random non-edge, we might add a bad edge.
        
        # Let's pick 10 random non-edges and choose the best.
        candidates = []
        for _ in range(10):
            u2 = random.randint(0, n-1)
            v2 = random.randint(0, n-1)
            if u2 == v2:
                continue
            key = (min(u2,v2), max(u2,v2))
            if key not in edges:
                # Check if key is valid (not self loop)
                common = adj[u2] & adj[v2]
                score = common.bit_count()
                candidates.append((score, key, u2, v2))
        
        if not candidates:
            # Restore
            adj[u1] |= (1 << v1)
            adj[v1] |= (1 << u1)
            edges.add(e_rem)
            return False, max_b, sum_b, books
            
        candidates.sort(key=lambda x: x[0])
        score, key, u2, v2 = candidates[0]
        
        # Add best candidate
        adj[u2] |= (1 << v2)
        adj[v2] |= (1 << u2)
        edges.add(key)
        
        # Recalculate books for affected edges?
        # This is complex. Let's just recalculate all books if we accept.
        # But we need new_max_b and new_sum_b to decide acceptance.
        # Recalculating all books is O(m) = 401 ops.
        # This is fast.
        
        new_max_b, new_sum_b, new_books = get_books(adj, edges)
        
        # Decide acceptance
        accept = False
        if new_max_b < max_b:
            accept = True
        elif new_max_b == max_b and new_sum_b < sum_b:
            accept = True
        elif T > 0:
            # Simulated annealing
            delta = (new_max_b - max_b) * 10 + (new_sum_b - sum_b)
            if delta < 0:
                accept = True
            else:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    accept = True
        
        if accept:
            return True, new_max_b, new_sum_b, new_books
        else:
            # Revert
            adj[u2] &= ~(1 << v2)
            adj[v2] &= ~(1 << u2)
            edges.discard(key)
            adj[u1] |= (1 << v1)
            adj[v1] |= (1 << u1)
            edges.add(e_rem)
            return False, max_b, sum_b, books

    best_edges = None
    best_max = 999
    best_sum = 999
    
    # Multiple restarts
    for seed in range(50):
        if time.time() > deadline - 5:
            break
            
        random.seed(seed)
        adj = [0] * n
        edges = set()
        
        # Initialize with random edges
        while len(edges) < target_edges:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                key = (min(u,v), max(u,v))
                if key not in edges:
                    edges.add(key)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        max_b, sum_b, books = get_books(adj, edges)
        
        # Simulated Annealing
        T = 5.0
        min_T = 0.1
        cooling = 0.99
        
        while time.time() < deadline - 2:
            if T < min_T:
                break
                
            accepted, max_b, sum_b, books = try_move(adj, edges, books, max_b, sum_b, T)
            T *= cooling
            
            if max_b < best_max or (max_b == best_max and sum_b < best_sum):
                best_max = max_b
                best_sum = sum_b
                best_edges = edges.copy()
                
            if best_max <= 7:
                break

    if best_edges is None:
        # Fallback to seed program result
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]

    return [[u, v] for (u, v) in sorted(best_edges)]
