"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    deadline = time.time() + 55.0
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute edge indices for faster access
    edge_u = [e[0] for e in edges]
    edge_v = [e[1] for e in edges]
    m = len(edges)
    
    def fast_count_bicolored_cycles(colors):
        """Fast count of bicolored cycles using union-find per color pair."""
        # Group vertices by color
        color_to_verts = {}
        for i, c in enumerate(colors):
            if c not in color_to_verts:
                color_to_verts[c] = []
            color_to_verts[c].append(i)
        
        used_colors = sorted(color_to_verts.keys())
        total_excess = 0
        
        for idx1 in range(len(used_colors)):
            for idx2 in range(idx1, len(used_colors)):
                c1, c2 = used_colors[idx1], used_colors[idx2]
                verts_c1 = color_to_verts[c1]
                verts_c2 = color_to_verts[c2]
                verts_set = set(verts_c1) | set(verts_c2)
                
                # Union-find
                parent = {}
                rank = {}
                for v in verts_set:
                    parent[v] = v
                    rank[v] = 0
                
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                
                def union(x, y):
                    px, py = find(x), find(y)
                    if px == py:
                        return False
                    if rank[px] < rank[py]:
                        px, py = py, px
                    parent[py] = px
                    if rank[px] == rank[py]:
                        rank[px] += 1
                    return True
                
                excess = 0
                for idx in range(m):
                    u = edge_u[idx]
                    v = edge_v[idx]
                    if u in verts_set and v in verts_set:
                        if not union(u, v):
                            excess += 1
                total_excess += excess
                if total_excess > 0:
                    return total_excess  # Early exit
        return total_excess
    
    def check_coloring(colors):
        """Check if a coloring is valid."""
        for idx in range(m):
            u = edge_u[idx]
            v = edge_v[idx]
            if colors[u] == colors[v]:
                return False
        return fast_count_bicolored_cycles(colors) == 0
    
    def local_search(colors, k, max_iters=500):
        """Guided local search to minimize bicolored cycles."""
        colors = list(colors)
        current_cost = fast_count_bicolored_cycles(colors)
        best_colors = list(colors)
        best_cost = current_cost
        T = 5.0
        cooling = 0.99
        no_improve = 0
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if current_cost == 0:
                return colors
            
            # Pick vertex with highest "conflict score" or random
            if random.random() < 0.7:
                # Pick vertex involved in most color conflicts
                best_v = -1
                best_score = -1
                for v in range(n):
                    score = 0
                    for u in adj[v]:
                        if colors[u] == colors[v]:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_v = v
                if best_v >= 0 and best_score > 0:
                    v = best_v
                else:
                    v = random.randint(0, n-1)
            else:
                v = random.randint(0, n-1)
            
            old_c = colors[v]
            
            # Try all valid colors, pick best
            best_new_c = old_c
            best_new_cost = current_cost
            
            for new_c in range(k):
                if new_c == old_c:
                    continue
                # Check local validity
                valid_local = True
                for u in adj[v]:
                    if colors[u] == new_c:
                        valid_local = False
                        break
                if not valid_local:
                    continue
                
                colors[v] = new_c
                new_cost = fast_count_bicolored_cycles(colors)
                if new_cost < best_new_cost:
                    best_new_cost = new_cost
                    best_new_c = new_c
            
            if best_new_c == old_c:
                # No improvement possible with single move, try random
                new_c = random.randint(0, k-1)
                if new_c == old_c:
                    new_c = (new_c + 1) % k
                valid_local = True
                for u in adj[v]:
                    if colors[u] == new_c:
                        valid_local = False
                        break
                if not valid_local:
                    for attempt in range(5):
                        new_c = random.randint(0, k-1)
                        valid_local = True
                        for u in adj[v]:
                            if colors[u] == new_c:
                                valid_local = False
                                break
                        if valid_local:
                            break
                    if not valid_local:
                        continue
                colors[v] = new_c
                new_cost = fast_count_bicolored_cycles(colors)
            else:
                colors[v] = best_new_c
                new_cost = best_new_cost
            
            delta = new_cost - current_cost
            if delta <= 0 or random.random() < pow(2.71828, -delta / max(T, 0.01)):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = list(colors)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                colors[v] = old_c
                no_improve += 1
            
            T *= cooling
            if no_improve > 100:
                break
        
        return best_colors
    
    # Try k from 2 upward
    for k in range(2, 8):
        if time.time() > deadline:
            break
        
        found = False
        # Multiple restarts with local search
        for restart in range(500):
            if time.time() > deadline:
                break
            
            # Initialize with random assignment
            colors = [random.randint(0, k-1) for _ in range(n)]
            
            # Run local search
            colors = local_search(colors, k, max_iters=400)
            
            if check_coloring(colors):
                found = True
                break
        
        if found:
            return [int(c) for c in colors]
    
    # Fallback: try more structured colorings
    for k in range(2, 8):
        if time.time() > deadline:
            break
        patterns = []
        for seed in range(20):
            random.seed(seed)
            pattern = [random.randint(0, k-1) for _ in range(n)]
            patterns.append(pattern)
        
        # Add systematic patterns
        patterns.append([i % k for i in range(n)])
        patterns.append([(i // 7) % k for i in range(n)])
        patterns.append([(i % 7) % k for i in range(n)])
        patterns.append([(i * 7) % k for i in range(n)])
        patterns.append([(i * 13) % k for i in range(n)])
        
        for colors in patterns:
            if time.time() > deadline:
                break
            if check_coloring(colors):
                return [int(c) for c in colors]
            # Try local search from this start
            improved = local_search(colors, k, max_iters=100)
            if check_coloring(improved):
                return [int(c) for c in improved]
    
    # Final fallback: greedy coloring
    colors = [0] * n
    for v in range(n):
        used = set()
        for u in adj[v]:
            if u < v:
                used.add(colors[u])
        for c in range(k):
            if c not in used:
                colors[v] = c
                break
    
    return [int(c) for c in colors]

# EVOLVE-BLOCK-END
