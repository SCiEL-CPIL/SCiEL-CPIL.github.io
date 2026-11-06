import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    # Seed solution (n=9) to ensure we don't return worse than seed
    seed_n = 9
    seed_colors = [min(j - i, seed_n - j + i) - 1 for i in range(seed_n) for j in range(i + 1, seed_n)]
    best = {'n': seed_n, 'colors': seed_colors}
    
    for n in range(17, 8, -1):
        if time.time() > deadline:
            break
            
        # Allocate time based on n
        time_limit = 35.0 if n == 17 else (25.0 if n == 16 else (15.0 if n == 15 else 10.0))
        target_end = time.time() + time_limit
        
        result = _local_search(n, target_end)
        if result:
            return result
            
    return best

def _count_c6s_adj(adj, n):
    """Count C6s in a graph given by adjacency bitmasks. Assumes adj[u] is bitmask of neighbors."""
    # Count by: for each edge (u,v), count paths of length 4 from u to v avoiding u,v internally.
    # Each C6 counted 12 times (6 edges x 2 directions), so divide by 12.
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] >> v) & 1:
                continue
            # Count paths u-a-b-c-v with a,b,c not in {u,v} and all distinct
            nu = adj[u] & ~(1 << v)
            nv = adj[v] & ~(1 << u)
            a_mask = nu
            while a_mask:
                lsb_a = a_mask & -a_mask
                a = lsb_a.bit_length() - 1
                a_mask ^= lsb_a
                na = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                c_mask = nv & ~(1 << a)
                while c_mask:
                    lsb_c = c_mask & -c_mask
                    c = lsb_c.bit_length() - 1
                    c_mask ^= lsb_c
                    common = na & adj[c] & ~(1 << c)
                    total += common.bit_count()
    return total // 12

def _local_search(n, deadline):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    best_colors = None
    best_c6 = float('inf')
    
    # Try multiple random restarts
    max_restarts = 3000
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize with a random coloring
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Build adjacency
        adj = [[0] * n for _ in range(4)]
        for idx, (u, v) in enumerate(edges):
            c = colors[idx]
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
        
        # Count initial C6s
        c6_counts = [_count_c6s_adj(adj[c], n) for c in range(4)]
        total_c6 = sum(c6_counts)
        
        if total_c6 < best_c6:
            best_c6 = total_c6
            best_colors = colors[:]
            if total_c6 == 0:
                return {'n': n, 'colors': colors}
        
        # Local search with simulated annealing
        temp = 10.0
        cooling = 0.9995
        no_improve = 0
        
        for step in range(200000):
            if time.time() > deadline:
                break
            
            # Pick a random edge to recolor
            idx = random.randint(0, num_edges - 1)
            u, v = edges[idx]
            old_c = colors[idx]
            
            # Try all 4 colors, pick the one with best delta
            best_new_c = old_c
            best_delta = 0
            
            for new_c in range(4):
                if new_c == old_c:
                    continue
                
                # Remove edge from old_c
                adj[old_c][u] &= ~(1 << v)
                adj[old_c][v] &= ~(1 << u)
                
                # Add edge to new_c
                adj[new_c][u] |= (1 << v)
                adj[new_c][v] |= (1 << u)
                
                # Count C6s in old_c and new_c
                new_c6_old = _count_c6s_adj(adj[old_c], n)
                new_c6_new = _count_c6s_adj(adj[new_c], n)
                
                delta = (new_c6_old - c6_counts[old_c]) + (new_c6_new - c6_counts[new_c])
                
                # Undo
                adj[old_c][u] |= (1 << v)
                adj[old_c][v] |= (1 << u)
                adj[new_c][u] &= ~(1 << v)
                adj[new_c][v] &= ~(1 << u)
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = new_c
            
            # Accept or reject based on simulated annealing
            if best_delta < 0:
                accept = True
            else:
                accept = random.random() < (2.718281828 ** (-best_delta / max(temp, 0.1)))
            
            if accept and best_new_c != old_c:
                # Apply the move
                adj[old_c][u] &= ~(1 << v)
                adj[old_c][v] &= ~(1 << u)
                adj[best_new_c][u] |= (1 << v)
                adj[best_new_c][v] |= (1 << u)
                
                c6_counts[old_c] = _count_c6s_adj(adj[old_c], n)
                c6_counts[best_new_c] = _count_c6s_adj(adj[best_new_c], n)
                
                colors[idx] = best_new_c
                total_c6 = sum(c6_counts)
                
                if total_c6 < best_c6:
                    best_c6 = total_c6
                    best_colors = colors[:]
                    if total_c6 == 0:
                        return {'n': n, 'colors': colors}
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve > 20000:
                    break
            else:
                no_improve += 1
                if no_improve > 20000:
                    break
            
            temp *= cooling
        
        if best_c6 == 0:
            return {'n': n, 'colors': best_colors}
    
    if best_c6 == 0 and best_colors:
        return {'n': n, 'colors': best_colors}
    
    return None
