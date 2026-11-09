import time
import numpy as np
import random

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def make_paley(p):
        """Create Paley tournament adjacency matrix for prime p ≡ 3 mod 4."""
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        adj = np.zeros((p, p), dtype=np.int8)
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    adj[i, j] = 1
                    adj[j, i] = 0
                else:
                    adj[i, j] = 0
                    adj[j, i] = 1
        return adj
    
    def count_bad_4sets(adj, n):
        """Count bad 4-sets using bitmask in-neighborhoods."""
        in_mask = [0] * n
        for u in range(n):
            mask = 0
            for v in range(n):
                if v != u and adj[v, u] == 1:
                    mask |= (1 << v)
            in_mask[u] = mask
        
        bad = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    inter_ijk = in_mask[i] & in_mask[j] & in_mask[k]
                    for l in range(k + 1, n):
                        if (inter_ijk & in_mask[l]) == 0:
                            bad += 1
        return bad
    
    def flip_edge(adj, i, j):
        """Flip edge between i and j, return old values."""
        old_ij = adj[i, j]
        adj[i, j] = 1 - old_ij
        adj[j, i] = 1 - adj[j, i]
        return old_ij
    
    def local_search(adj, n, max_iters=5000):
        """Local search minimizing bad 4-sets via edge flips."""
        current_bad = count_bad_4sets(adj, n)
        if current_bad == 0:
            return adj, 0
        
        best_bad = current_bad
        best_adj = adj.copy()
        
        for it in range(max_iters):
            if time_left() < 5.0:
                break
            
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if j < i:
                i, j = j, i
            
            old_ij = flip_edge(adj, i, j)
            new_bad = count_bad_4sets(adj, n)
            
            if new_bad < best_bad:
                best_bad = new_bad
                best_adj = adj.copy()
                if best_bad == 0:
                    return best_adj, 0
            else:
                # Revert
                adj[i, j] = old_ij
                adj[j, i] = 1 - old_ij
        
        return best_adj, best_bad
    
    # Try n from 31 up to 66 with local search
    for n in range(31, 67):
        if time_left() < 10.0:
            break
        
        # Initialize with random tournament
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i, j] = 1
                    adj[j, i] = 0
                else:
                    adj[i, j] = 0
                    adj[j, i] = 1
        
        # Run local search
        best_adj, best_bad = local_search(adj, n, max_iters=2000)
        
        if best_bad == 0:
            # Build orientation
            orientation = []
            for i in range(n):
                for j in range(i + 1, n):
                    orientation.append(0 if best_adj[i, j] == 1 else 1)
            return {'n': n, 'orientation': orientation}
    
    # Fallback: Paley tournament on 67
    p = 67
    adj = make_paley(p)
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            orientation.append(0 if adj[i, j] == 1 else 1)
    return {'n': p, 'orientation': orientation}
