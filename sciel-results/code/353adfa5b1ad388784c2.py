"""R_4(C6) - 4-colouring of K_n without monochromatic C6, using constraint-weighted local search."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0
    
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    def edge_index(i, j, n):
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    def get_all_c6_edge_tuples(n):
        """Get all unique C6s as numpy array of edge index tuples (6 per C6)."""
        c6_edges = []
        for v0 in range(n):
            for v1 in range(v0 + 1, n):
                for v2 in range(n):
                    if v2 in (v0, v1):
                        continue
                    for v3 in range(n):
                        if v3 in (v0, v1, v2):
                            continue
                        for v4 in range(n):
                            if v4 in (v0, v1, v2, v3):
                                continue
                            for v5 in range(n):
                                if v5 in (v0, v1, v2, v3, v4):
                                    continue
                                if min(v0, v1, v2, v3, v4, v5) != v0:
                                    continue
                                if v1 > v5:
                                    continue
                                edges = [(v0, v1), (v1, v2), (v2, v3), (v3, v4), (v4, v5), (v5, v0)]
                                edge_idxs = []
                                for u, v in edges:
                                    if u > v:
                                        u, v = v, u
                                    edge_idxs.append(edge_index(u, v, n))
                                c6_edges.append(edge_idxs)
        return np.array(c6_edges, dtype=np.int32)
    
    def count_c6s_and_hotness(colors_arr, c6_edges):
        """Count monochromatic C6s and return edge hotness (count of mono-C6s each edge is in)."""
        if len(c6_edges) == 0:
            return 0, np.zeros(len(colors_arr), dtype=np.float64)
        edge_colors = colors_arr[c6_edges]  # shape: (num_c6s, 6)
        is_mono = np.all(edge_colors == edge_colors[:, 0:1], axis=1)
        count = int(np.sum(is_mono))
        
        # Compute hotness: for each edge, how many mono-C6s does it appear in?
        if count > 0:
            mono_c6_indices = np.where(is_mono)[0]
            hotness = np.zeros(len(colors_arr), dtype=np.float64)
            mono_edge_indices = c6_edges[mono_c6_indices]
            # Count occurrences
            np.add.at(hotness, mono_edge_indices, 1)
        else:
            hotness = np.zeros(len(colors_arr), dtype=np.float64)
        
        return count, hotness
    
    # Precompute C6 edge tuples
    c6_cache = {}
    for n in range(10, 19):
        if time.time() > deadline:
            break
        c6_cache[n] = get_all_c6_edge_tuples(n)
    
    def local_search_weighted(colors, n, c6_edges, max_iters=20000):
        """Constraint-weighted local search: bias toward edges in many mono-C6s."""
        num_edges = n * (n - 1) // 2
        colors_arr = np.array(colors, dtype=np.int32)
        current_count, hotness = count_c6s_and_hotness(colors_arr, c6_edges)
        
        if current_count == 0:
            return colors, 0
        
        T = 3.0
        no_improve = 0
        max_no_improve = 8000
        edge_weights = np.ones(num_edges, dtype=np.float64)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if no_improve > max_no_improve:
                break
            if current_count == 0:
                return colors_arr.tolist(), 0
            
            # Update edge weights based on hotness
            # Edges in more mono-C6s get higher weight
            edge_weights = 1.0 + hotness / max(1, current_count)
            
            # Select edge with probability proportional to weight
            total_w = np.sum(edge_weights)
            probs = edge_weights / total_w
            ei = np.random.choice(num_edges, p=probs)
            ei = int(ei)
            
            old_color = int(colors_arr[ei])
            new_color = random.randint(0, 3)
            if new_color == old_color:
                continue
            
            colors_arr[ei] = new_color
            new_count, new_hotness = count_c6s_and_hotness(colors_arr, c6_edges)
            delta = new_count - current_count
            
            if delta <= 0 or random.random() < np.exp(-delta / max(0.05, T)):
                current_count = new_count
                hotness = new_hotness
                no_improve = 0
            else:
                colors_arr[ei] = old_color
                no_improve += 1
            
            T *= 0.9998
        
        return colors_arr.tolist(), current_count
    
    # Try n from 17 down to 10, spending most time on largest n
    for n in range(17, 9, -1):
        if time.time() > deadline:
            break
        
        if n not in c6_cache:
            continue
        
        c6_edges = c6_cache[n]
        num_edges = n * (n - 1) // 2
        found = False
        
        # Allocate time based on n
        remaining = deadline - time.time()
        if n >= 16:
            max_restarts = max(10, int(remaining * 3))
            max_iters_per_restart = 15000
        else:
            max_restarts = max(5, int(remaining * 5))
            max_iters_per_restart = 10000
        
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break
            
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            colors, count = local_search_weighted(colors, n, c6_edges, max_iters=max_iters_per_restart)
            
            if count == 0:
                best_n = n
                best_colors = colors
                found = True
                break
        
        if found:
            # Try to go higher
            continue
        else:
            # If we failed at this n, try lower n
            if n == 17:
                # Spent significant time, try n=16 with remaining time
                continue
            else:
                break
    
    # Fallback: ensure we have at least n=9
    if best_n < 9:
        best_n = 9
        best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    return {'n': best_n, 'colors': best_colors}
