import random
import time

def construct():
    target_ns = [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15]
    
    start_time = time.time()
    time_limit = 115.0
    
    for n in target_ns:
        if time.time() - start_time > time_limit:
            break
            
        # Partition vertices into 5 subsets
        subsets = [[] for _ in range(5)]
        for v in range(n):
            subsets[v % 5].append(v)
            
        # Precompute edges
        edges = []
        edge_idx_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
                edge_idx_map[(i, j)] = idx
                idx += 1
        m = len(edges)
        
        best_colors = None
        best_energy = float('inf')
        
        # Multiple restarts with different partition orderings or hash seeds
        for restart in range(10):
            if time.time() - start_time > time_limit:
                break
                
            # 1. Initialize based on partition and deterministic hash
            colors = [0] * m
            
            # Intra-subset: Use a "base" pattern. 
            # For simplicity, we assign colors 0-4 randomly but constrained?
            # The directive says "based on a precomputed C4-free base graph".
            # We'll simulate this by using a fixed valid coloring for K5 (if subset size <= 5) 
            # or a random C4-free pattern for larger.
            # Since n<=26, subset sizes are ~5.
            
            # Map each vertex in subset to a position in a base K5 coloring
            # Base K5 coloring (edges of K5 colored 0-4 with no mono C4):
            # Vertices 0,1,2,3,4
            # Edges:
            # (0,1):0, (0,2):1, (0,3):2, (0,4):3
            # (1,2):4, (1,3):0, (1,4):1
            # (2,3):1, (2,4):2
            # (3,4):3
            # Check C4s:
            # Color 0: (0,1), (1,3). No C4.
            # Color 1: (0,2), (1,4), (2,3). No C4.
            # Color 2: (0,3), (2,4). No C4.
            # Color 3: (0,4), (3,4). No C4.
            # Color 4: (1,2). No C4.
            # This is a valid 5-coloring of K5.
            
            base_k5_edges = {
                (0,1):0, (0,2):1, (0,3):2, (0,4):3,
                (1,2):4, (1,3):0, (1,4):1,
                (2,3):1, (2,4):2,
                (3,4):3
            }
            
            for k in range(5):
                subset = subsets[k]
                # If subset size > 5, we can't just map to K5.
                # But n<=26, max subset size is 6 (if n=26, 26/5=5.2).
                # For size 6, we need a C4-free coloring of K6.
                # We'll just use random for intra-subset if size > 5, 
                # or extend the K5 pattern.
                
                # For now, use random for intra-subset to allow flexibility,
                # but biased to be "structured".
                for idx_a in range(len(subset)):
                    for idx_b in range(idx_a + 1, len(subset)):
                        u = subset[idx_a]
                        v = subset[idx_b]
                        if len(subset) <= 5:
                            # Map to base K5 indices
                            c = base_k5_edges[(idx_a, idx_b)]
                        else:
                            # Fallback to random for larger subsets
                            c = random.randint(0, 4)
                        
                        edge_key = (u, v) if u < v else (v, u)
                        e_idx = edge_idx_map[edge_key]
                        colors[e_idx] = c
            
            # Inter-subset: Deterministic hash
            for i in range(n):
                for j in range(i + 1, n):
                    si = i % 5
                    sj = j % 5
                    if si != sj:
                        e_idx = edge_idx_map[(i, j)]
                        # Deterministic hash
                        colors[e_idx] = (i * 3 + j * 7 + restart) % 5
            
            # Build adjacency for energy calculation
            adj = [ [set() for _ in range(n)] for _ in range(5) ]
            for k, (i, j) in enumerate(edges):
                c = colors[k]
                adj[c][i].add(j)
                adj[c][j].add(i)
                
            # Calculate initial energy
            def count_c4s_in_color(c):
                total = 0
                for u in range(n):
                    for v in adj[c][u]:
                        if u < v:
                            count = 0
                            for x in adj[c][v]:
                                if x == u: continue
                                for y in adj[c][u]:
                                    if y == v: continue
                                    if x == y: continue
                                    if y in adj[c][x]:
                                        count += 1
                            total += count
                return total // 4

            total_energy = sum(count_c4s_in_color(c) for c in range(5))
            
            if total_energy < best_energy:
                best_energy = total_energy
                best_colors = colors.copy()
                
            if total_energy == 0:
                return {'n': n, 'colors': best_colors}
                
            # Local Search / Hill Climbing to reduce energy
            # We will try to flip edges to reduce C4s
            improved = True
            while improved and time.time() - start_time < time_limit:
                improved = False
                # Shuffle edges to avoid bias
                edge_order = list(range(m))
                random.shuffle(edge_order)
                
                for k in edge_order:
                    if time.time() - start_time > time_limit:
                        break
                        
                    i, j = edges[k]
                    c_old = colors[k]
                    
                    # Try all 5 colors
                    best_c = c_old
                    best_delta = 0
                    
                    for c_new in range(5):
                        if c_new == c_old:
                            continue
                        
                        # Calculate delta
                        # Remove (i,j) from c_old, add to c_new
                        # Count C4s involving (i,j) in c_old
                        cnt_old = 0
                        for x in adj[c_old][j]:
                            if x == i: continue
                            for y in adj[c_old][i]:
                                if y == j: continue
                                if x == y: continue
                                if y in adj[c_old][x]:
                                    cnt_old += 1
                            
                        # Count C4s involving (i,j) in c_new
                        cnt_new = 0
                        for x in adj[c_new][j]:
                            if x == i: continue
                            for y in adj[c_new][i]:
                                if y == j: continue
                                if x == y: continue
                                if y in adj[c_new][x]:
                                    cnt_new += 1
                            
                        delta = cnt_new - cnt_old
                        if delta < best_delta:
                            best_delta = delta
                            best_c = c_new
                            
                    if best_delta < 0:
                        # Apply change
                        adj[c_old][i].remove(j)
                        adj[c_old][j].remove(i)
                        adj[best_c][i].add(j)
                        adj[best_c][j].add(i)
                        colors[k] = best_c
                        total_energy += best_delta
                        improved = True
                        
                        if total_energy < best_energy:
                            best_energy = total_energy
                            best_colors = colors.copy()
                            
                        if total_energy == 0:
                            return {'n': n, 'colors': best_colors}
                            
    # Fallback
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
