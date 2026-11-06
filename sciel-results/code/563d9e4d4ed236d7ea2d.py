import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = 145  # 5s safety margin before 150s limit
    
    n = 16
    
    # Precompute all K4s and their edge indices
    # Edge index for (i,j) where i < j: sum_{k=0}^{i-1} (15-k) + (j-i-1)
    # = 15*i - i*(i-1)//2 + j - i - 1
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return 15 * i - i * (i - 1) // 2 + (j - i - 1)
    
    # Precompute K4 edge index lists
    k4s = list(itertools.combinations(range(n), 4))
    k4_edges = []
    for a, b, c, d in k4s:
        e = [
            edge_idx(a, b), edge_idx(a, c), edge_idx(a, d),
            edge_idx(b, c), edge_idx(b, d), edge_idx(c, d)
        ]
        k4_edges.append(e)
    
    def is_valid(colors):
        """Check if every K4 has at least 5 distinct colors."""
        for e in k4_edges:
            cols = {colors[e[0]], colors[e[1]], colors[e[2]],
                    colors[e[3]], colors[e[4]], colors[e[5]]}
            if len(cols) < 5:
                return False
        return True
    
    def count_violations(colors):
        """Count number of K4s with fewer than 5 distinct colors."""
        viol = 0
        for e in k4_edges:
            cols = {colors[e[0]], colors[e[1]], colors[e[2]],
                    colors[e[3]], colors[e[4]], colors[e[5]]}
            if len(cols) < 5:
                viol += 1
        return viol
    
    def get_violating_k4s(colors):
        """Return list of K4 indices that violate."""
        viol_list = []
        for idx, e in enumerate(k4_edges):
            cols = {colors[e[0]], colors[e[1]], colors[e[2]],
                    colors[e[3]], colors[e[4]], colors[e[5]]}
            if len(cols) < 5:
                viol_list.append(idx)
        return viol_list
    
    # Method 1: Circle method (15 colors)
    def circle_method():
        colors = [0] * 120
        for i in range(16):
            for j in range(i + 1, 16):
                if i == 15 or j == 15:
                    v = i if j == 15 else j
                    colors[edge_idx(i, j)] = v
                else:
                    k = (i + j) * 8 % 15
                    colors[edge_idx(i, j)] = k
        return colors
    
    # Method 2: Additive method (17 colors) - known valid
    def additive_method():
        colors = [0] * 120
        for i in range(16):
            for j in range(i + 1, 16):
                colors[edge_idx(i, j)] = (i + j) % 17
        return colors
    
    # Try circle method first
    best_sol = None
    best_colors = 999
    
    cm_colors = circle_method()
    if is_valid(cm_colors):
        return cm_colors
    
    # Local search to repair circle method
    def local_search(initial_colors, max_colors, max_iters=5000):
        colors = initial_colors[:]
        viol = count_violations(colors)
        if viol == 0:
            return colors
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            if viol == 0:
                return colors
            
            # Pick a random violating K4
            for _ in range(20):
                k4_idx = random.randint(0, len(k4_edges) - 1)
                e = k4_edges[k4_idx]
                cols = {colors[e[0]], colors[e[1]], colors[e[2]],
                        colors[e[3]], colors[e[4]], colors[e[5]]}
                if len(cols) < 5:
                    # Found violating K4, pick an edge to change
                    edge_pos = random.randint(0, 5)
                    global_idx = e[edge_pos]
                    old_color = colors[global_idx]
                    
                    # Try to find a color that helps
                    # Count how many times each color appears in this K4
                    color_counts = {}
                    for ei in e:
                        c = colors[ei]
                        color_counts[c] = color_counts.get(c, 0) + 1
                    
                    # Prefer colors not already in this K4, or colors that appear once
                    candidates = []
                    for c in range(max_colors):
                        if c == old_color:
                            continue
                        cnt = color_counts.get(c, 0)
                        if cnt == 0:
                            candidates.append((c, 0))
                        elif cnt == 1:
                            candidates.append((c, 1))
                    
                    if not candidates:
                        new_color = (old_color + 1) % max_colors
                    else:
                        candidates.sort(key=lambda x: x[1])
                        new_color = random.choice(candidates[:3])[0]
                    
                    colors[global_idx] = new_color
                    viol = count_violations(colors)
                    break
            else:
                # No violating K4 found in random picks, but count says there are some
                # Force a change on a random edge
                global_idx = random.randint(0, 119)
                old_color = colors[global_idx]
                new_color = (old_color + 1) % max_colors
                colors[global_idx] = new_color
                viol = count_violations(colors)
        
        if viol == 0:
            return colors
        return None
    
    # Try local search on circle method with 15 colors
    if time.time() < deadline - 10:
        result = local_search(cm_colors, 15, max_iters=8000)
        if result is not None and is_valid(result):
            best_sol = result
            best_colors = 15
            return best_sol
    
    # Try with more colors: 16
    if time.time() < deadline - 10:
        # Start from circle method but allow 16 colors
        result = local_search(cm_colors, 16, max_iters=6000)
        if result is not None and is_valid(result):
            best_sol = result
            best_colors = 16
            return best_sol
    
    # Fallback: additive method with 17 colors (known valid)
    add_colors = additive_method()
    if is_valid(add_colors):
        return add_colors
    
    # Ultimate fallback
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
