import time
import random

def construct():
    start_time = time.time()
    hard_deadline = start_time + 110.0
    rng = random.Random(42)

    def edge_idx(n, i, j):
        if i > j: i, j = j, i
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

    def build_adj(n, colors):
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj

    def has_simple_path_len_5(u, v, adj_c, n):
        """Check if there's a simple path of length 5 from u to v in graph adj_c (excluding edge u-v)."""
        # adj_c is list of int bitmasks
        # u, v are indices
        # We assume edge (u,v) is NOT in adj_c
        
        mask_x1 = adj_c[u]
        while mask_x1:
            lsb = mask_x1 & -mask_x1
            x1 = lsb.bit_length() - 1
            mask_x1 -= lsb
            
            mask_x2 = adj_c[x1] & ~(1 << u)
            while mask_x2:
                lsb = mask_x2 & -mask_x2
                x2 = lsb.bit_length() - 1
                mask_x2 -= lsb
                
                if x2 == v: continue
                
                mask_x3 = adj_c[x2] & ~(1 << u) & ~(1 << x1)
                while mask_x3:
                    lsb = mask_x3 & -mask_x3
                    x3 = lsb.bit_length() - 1
                    mask_x3 -= lsb
                    
                    if x3 == v: continue
                    
                    mask_x4 = adj_c[x3] & ~(1 << u) & ~(1 << x1) & ~(1 << x2)
                    while mask_x4:
                        lsb = mask_x4 & -mask_x4
                        x4 = lsb.bit_length() - 1
                        mask_x4 -= lsb
                        
                        if x4 == v: continue
                        
                        if (adj_c[x4] >> v) & 1:
                            return True
        return False

    def count_c6s(n, colors):
        """Count total C6s (multiplied by 6) in the coloring."""
        adj = build_adj(n, colors)
        total = 0
        for c in range(4):
            adj_c = adj[c]
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj_c[i] >> j) & 1:
                        # Remove edge
                        adj_c[i] &= ~(1 << j)
                        adj_c[j] &= ~(1 << i)
                        if has_simple_path_len_5(i, j, adj_c, n):
                            total += 1
                        # Restore edge
                        adj_c[i] |= (1 << j)
                        adj_c[j] |= (1 << i)
        return total

    def local_search(n, colors, max_iters=500000):
        """Local search with incremental C6 delta."""
        num_edges = n * (n - 1) // 2
        colors = colors[:]
        adj = build_adj(n, colors)
        edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        current_total = count_c6s(n, colors)
        best_total = current_total
        best_colors = colors[:]
        
        for it in range(max_iters):
            if time.time() > hard_deadline:
                break
            
            ei = rng.randint(0, num_edges - 1)
            u, v = edge_list[ei]
            old_c = colors[ei]
            new_c = rng.randint(0, 3)
            if new_c == old_c:
                continue
            
            # Remove edge from old color
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            removed = 1 if has_simple_path_len_5(u, v, adj[old_c], n) else 0
            adj[old_c][u] |= (1 << v)
            adj[old_c][v] |= (1 << u)
            
            # Check added in new color
            added = 1 if has_simple_path_len_5(u, v, adj[new_c], n) else 0
            
            delta = added - removed
            new_total = current_total + delta
            
            if new_total <= best_total:
                # Accept
                adj[old_c][u] &= ~(1 << v)
                adj[old_c][v] &= ~(1 << u)
                adj[new_c][u] |= (1 << v)
                adj[new_c][v] |= (1 << u)
                colors[ei] = new_c
                current_total = new_total
                best_total = new_total
                best_colors = colors[:]
            else:
                # Simulated annealing / Random restart
                if delta <= 2 and rng.random() < 0.1:
                    adj[old_c][u] &= ~(1 << v)
                    adj[old_c][v] &= ~(1 << u)
                    adj[new_c][u] |= (1 << v)
                    adj[new_c][v] |= (1 << u)
                    colors[ei] = new_c
                    current_total = new_total
            
            if best_total == 0:
                return best_colors
        
        return best_colors

    def try_n(n, time_budget):
        deadline = time.time() + time_budget
        for attempt in range(100):
            if time.time() > deadline:
                break
            colors = [rng.randint(0, 3) for _ in range(n * (n - 1) // 2)]
            colors = local_search(n, colors, max_iters=100000)
            if count_c6s(n, colors) == 0:
                return colors
        return None

    for n in [18, 17, 16, 15, 14, 13, 12, 11, 10, 9]:
        if time.time() > hard_deadline - 5:
            break
        remaining = hard_deadline - time.time()
        if n == 18:
            budget = min(remaining - 5, 10.0)
        elif n == 17:
            budget = min(remaining - 5, 30.0)
        else:
            budget = min(remaining - 5, 10.0)
            
        if budget < 2:
            continue
            
        result = try_n(n, budget)
        if result is not None:
            if count_c6s(n, result) == 0:
                return {'n': n, 'colors': result}
                
    # Fallback
    for n in [9, 8, 7, 6]:
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append((i * 3 + j) % 4)
        if count_c6s(n, colors) == 0:
            return {'n': n, 'colors': colors}
            
    return {'n': 1, 'colors': []}

if __name__ == '__main__':
    construct()
