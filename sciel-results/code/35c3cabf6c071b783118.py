import time
import random
import math

def _get_idx(i, j, n):
    if i > j:
        i, j = j, i
    return i * (2 * n - i - 1) // 2 + (j - i - 1)

def _validate_coloring(n, colors):
    """Validate that no color class contains a C4."""
    # Precompute adjacency lists for each color for faster checking?
    # Or just use the k-array logic.
    # For validation, we can just check if any k[c][u][v] >= 2.
    # But we don't have k-array in this function. We can compute it.
    # Or we can just rely on the SA cost being 0.
    # To be safe, let's implement a fast validator.
    # For each color c, build adjacency list.
    # For each pair u, v, count common neighbors.
    # This is O(5 * n^3) which is 5 * 27^3 ~ 100k ops. Fast enough.
    
    for c in range(5):
        # Build adjacency list for color c
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = _get_idx(i, j, n)
                if colors[idx] == c:
                    adj[i].append(j)
                    adj[j].append(i)
        
        # Check for C4
        # A C4 exists if there are u, v with >= 2 common neighbors.
        # We can check for each pair (u, v) if they have 2 common neighbors.
        # Or for each pair of edges (u, w) and (v, w) ... no.
        # Just check common neighbors for all pairs.
        for u in range(n):
            for v in range(u + 1, n):
                common = 0
                # Intersection of adj[u] and adj[v]
                # Since n is small, simple loop is fine
                # But we can do better.
                # Let's just iterate over neighbors of u and check if in adj[v]
                # To speed up, we can use sets? No, lists are small.
                for w in adj[u]:
                    if w == v:
                        continue
                    if w in adj[v]:
                        common += 1
                        if common >= 2:
                            return False
    return True

def _solve_sa(n, deadline, seed=42):
    """
    Solve the C4-free 5-coloring problem for K_n using Simulated Annealing.
    Returns (colors, cost) or (None, None) if time runs out without cost 0.
    """
    rng = random.Random(seed)
    n_edges = n * (n - 1) // 2
    colors = [rng.randint(0, 4) for _ in range(n_edges)]
    
    # k[c][u][v] stores the number of common neighbors of u and v in color c.
    # We store it for u < v.
    # Initialize k
    k = [[[0] * n for _ in range(n)] for _ in range(5)]
    
    # Initialize colors and k
    # First, set colors
    # Then, compute k
    # For each color c, for each pair (u, v), count common neighbors.
    # This is O(5 * n^3).
    for c in range(5):
        # Build adjacency lists for c
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = _get_idx(i, j, n)
                if colors[idx] == c:
                    adj[i].append(j)
                    adj[j].append(i)
        
        # Compute k
        for u in range(n):
            for v in range(u + 1, n):
                # Count common neighbors
                cnt = 0
                for w in adj[u]:
                    if w == v:
                        continue
                    if w in adj[v]:
                        cnt += 1
                k[c][u][v] = cnt
                k[c][v][u] = cnt # Symmetry, though we only use u<v mostly
        
    # Compute initial cost
    cost = 0
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                cnt = k[c][u][v]
                if cnt >= 2:
                    cost += cnt * (cnt - 1) // 2
                    
    if cost == 0:
        return colors, 0

    T = 10.0
    cooling = 0.999
    min_T = 1e-6
    steps = 0
    
    while time.time() < deadline and T > min_T:
        # Pick a random edge
        idx = rng.randint(0, n_edges - 1)
        # Recover i, j from idx
        # This is a bit slow. We can precompute a list of edges.
        # But we need to map idx to (i, j).
        # Let's precompute edge list.
        
        # Actually, let's just iterate.
        # Precompute edges list outside?
        # For now, let's assume we have a way to get i, j.
        
        # We need to be careful with time.
        # Let's precompute edges.
        
        # ... (This part needs to be in the actual code)
        
        T *= cooling
        steps += 1
        
    return None, cost

# Precompute edges for speed
def _precompute_edges(n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    return edges

def _solve_sa_optimized(n, deadline, seed=42):
    rng = random.Random(seed)
    n_edges = n * (n - 1) // 2
    edges = _precompute_edges(n)
    colors = [rng.randint(0, 4) for _ in range(n_edges)]
    
    k = [[[0] * n for _ in range(n)] for _ in range(5)]
    
    # Initialize k
    for c in range(5):
        adj = [set() for _ in range(n)]
        for idx, (i, j) in enumerate(edges):
            if colors[idx] == c:
                adj[i].add(j)
                adj[j].add(i)
                
        for u in range(n):
            for v in range(u + 1, n):
                cnt = len(adj[u] & adj[v])
                k[c][u][v] = cnt
                k[c][v][u] = cnt
                
    cost = 0
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                cnt = k[c][u][v]
                if cnt >= 2:
                    cost += cnt * (cnt - 1) // 2
                    
    if cost == 0:
        return colors, 0

    T = 10.0
    cooling = 0.9995
    min_T = 1e-6
    
    while time.time() < deadline and T > min_T:
        idx = rng.randint(0, n_edges - 1)
        i, j = edges[idx]
        c_old = colors[idx]
        
        # Try all new colors
        best_delta = float('inf')
        best_c_new = -1
        
        # We can just pick a random new color different from old
        # Or try all 4 and pick the one with min delta?
        # Trying all 4 is O(4 * n). Acceptable.
        
        candidates = [c for c in range(5) if c != c_old]
        rng.shuffle(candidates)
        
        for c_new in candidates:
            delta = 0
            
            # Remove from c_old
            # Update k for pairs involving i or j
            # For each w in V \ {i, j}
            # If (j, w) in G_c_old, then k[i, w] decreases by 1
            # If (i, w) in G_c_old, then k[j, w] decreases by 1
            
            # We need to iterate over neighbors.
            # To do this efficiently, we should maintain adjacency lists/sets for each color.
            # Let's add that.
            
            # This requires maintaining adj_sets.
            # Let's refactor to include adj_sets.
            
            # Since we are in the middle of the function, let's just do the O(n) check.
            # It's fine for n=27.
            
            for w in range(n):
                if w == i or w == j:
                    continue
                idx_iw = _get_idx(i, w, n)
                idx_jw = _get_idx(j, w, n)
                
                # Check if (j, w) is in c_old
                if colors[idx_jw] == c_old:
                    # k[i, w] decreases by 1
                    u, v = (i, w) if i < w else (w, i)
                    old_k = k[c_old][u][v]
                    new_k = old_k - 1
                    delta += (new_k * (new_k - 1) // 2) - (old_k * (old_k - 1) // 2)
                    # Note: if new_k < 0, something is wrong.
                    
                # Check if (i, w) is in c_old
                if colors[idx_iw] == c_old:
                    u, v = (j, w) if j < w else (w, j)
                    old_k = k[c_old][u][v]
                    new_k = old_k - 1
                    delta += (new_k * (new_k - 1) // 2) - (old_k * (old_k - 1) // 2)
                
                # Add to c_new
                # If (j, w) is in c_new, k[i, w] increases by 1
                if colors[idx_jw] == c_new:
                    u, v = (i, w) if i < w else (w, i)
                    old_k = k[c_new][u][v]
                    new_k = old_k + 1
                    delta += (new_k * (new_k - 1) // 2) - (old_k * (old_k - 1) // 2)
                    
                # If (i, w) is in c_new, k[j, w] increases by 1
                if colors[idx_iw] == c_new:
                    u, v = (j, w) if j < w else (w, j)
                    old_k = k[c_new][u][v]
                    new_k = old_k + 1
                    delta += (new_k * (new_k - 1) // 2) - (old_k * (old_k - 1) // 2)
            
            if delta < best_delta:
                best_delta = delta
                best_c_new = c_new
        
        # Acceptance
        if best_delta <= 0 or rng.random() < math.exp(-best_delta / T):
            # Apply change
            colors[idx] = best_c_new
            cost += best_delta
            
            # Update k
            for w in range(n):
                if w == i or w == j:
                    continue
                idx_iw = _get_idx(i, w, n)
                idx_jw = _get_idx(j, w, n)
                
                if colors[idx_jw] == c_old:
                    u, v = (i, w) if i < w else (w, i)
                    k[c_old][u][v] -= 1
                    k[c_old][v][u] -= 1
                if colors[idx_iw] == c_old:
                    u, v = (j, w) if j < w else (w, j)
                    k[c_old][u][v] -= 1
                    k[c_old][v][u] -= 1
                
                if colors[idx_jw] == best_c_new:
                    u, v = (i, w) if i < w else (w, i)
                    k[best_c_new][u][v] += 1
                    k[best_c_new][v][u] += 1
                if colors[idx_iw] == best_c_new:
                    u, v = (j, w) if j < w else (w, j)
                    k[best_c_new][u][v] += 1
                    k[best_c_new][v][u] += 1
            
            if cost == 0:
                return colors, 0
                
        T *= cooling
        
    return None, cost

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Try n=27
    for seed in [42, 123, 777, 2024, 555]:
        if time.time() > deadline - 10:
            break
        colors, cost = _solve_sa_optimized(27, deadline - 5, seed)
        if cost == 0:
            return {'n': 27, 'colors': colors}
            
    # Try n=26
    for seed in [42, 123, 777, 2024, 555]:
        if time.time() > deadline - 5:
            break
        colors, cost = _solve_sa_optimized(26, deadline - 2, seed)
        if cost == 0:
            return {'n': 26, 'colors': colors}
            
    # Fallback to greedy or smaller n
    # We can just return a small valid graph
    # K_1 is always valid
    return {'n': 1, 'colors': [0]}

