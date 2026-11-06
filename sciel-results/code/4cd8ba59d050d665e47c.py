import time
import random
import sys
import math

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin for 150s limit

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _seed_32():
    """Greenwood-Gleason colouring of K32."""
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
        
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
        
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return {'n': 32, 'colors': colors}

def _compute_index(n, i, j):
    """Compute the index of pair (i,j) in lexicographic edge ordering."""
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def _build_adj(colors, n):
    """Build adjacency bitsets for a given coloring."""
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj

def _count_mono_triangles(adj, n):
    """Count total monochromatic triangles using bitsets."""
    count = 0
    for c in range(4):
        for i in range(n):
            neigh = adj[c][i] & ~((1 << (i + 1)) - 1)
            j_mask = neigh
            while j_mask:
                j_bit = j_mask & (-j_mask)
                j = j_bit.bit_length() - 1
                j_mask ^= j_bit
                common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                count += common.bit_count()
    return count

def _find_one_mono_triangle(adj, n):
    """Find one monochromatic triangle, return (i, j, k, c) or None."""
    for c in range(4):
        for i in range(n):
            neigh = adj[c][i] & ~((1 << (i + 1)) - 1)
            j_mask = neigh
            while j_mask:
                j_bit = j_mask & (-j_mask)
                j = j_bit.bit_length() - 1
                j_mask ^= j_bit
                common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                if common:
                    k_bit = common & (-common)
                    k = k_bit.bit_length() - 1
                    return (i, j, k, c)
    return None

def _min_conflicts(n, time_budget):
    """
    Min-Conflicts heuristic to find a valid 4-coloring of K_n.
    """
    m = n * (n - 1) // 2
    start_time = time.time()
    
    # Precompute index-to-edge mapping
    idx_to_edge = [None] * m
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_to_edge[idx] = (i, j)
            idx += 1

    best_result = None
    best_tri = float('inf')
    
    # Multiple restarts
    for restart in range(500):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization
        colors = [random.randint(0, 3) for _ in range(m)]
        adj = _build_adj(colors, n)
        tri_count = _count_mono_triangles(adj, n)
        
        if tri_count < best_tri:
            best_tri = tri_count
            best_result = colors[:]
        
        if tri_count == 0:
            return colors
        
        stuck_count = 0
        max_stuck = 1000
        
        while stuck_count < max_stuck:
            if time.time() - start_time > time_budget:
                break
            
            tri = _find_one_mono_triangle(adj, n)
            if tri is None:
                return colors
            
            i, j, k, c = tri
            
            # Try to fix one of the edges in the triangle
            # We evaluate all 3 edges and all 3 other colors
            best_delta = float('inf')
            best_edge_idx = -1
            best_new_c = -1
            
            edges = [(i, j), (i, k), (j, k)]
            
            for u, v in edges:
                for new_c in range(4):
                    if new_c == c:
                        continue
                    
                    # Calculate delta
                    # Removing edge (u,v) from color c
                    # Adding edge (u,v) to color new_c
                    removed = (adj[c][u] & adj[c][v]).bit_count()
                    added = (adj[new_c][u] & adj[new_c][v]).bit_count()
                    delta = added - removed
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_edge_idx = _compute_index(n, u, v)
                        best_new_c = new_c
            
            if best_delta >= 0 and stuck_count > 0:
                # If no improvement, random perturbation
                if random.random() < 0.1:
                    # Perturb a random edge
                    ri = random.randint(0, m - 1)
                    old_c = colors[ri]
                    new_c = random.randint(0, 3)
                    if new_c != old_c:
                        u, v = idx_to_edge[ri]
                        removed = (adj[old_c][u] & adj[old_c][v]).bit_count()
                        added = (adj[new_c][u] & adj[new_c][v]).bit_count()
                        tri_count += added - removed
                        colors[ri] = new_c
                        adj[old_c][u] &= ~(1 << v)
                        adj[old_c][v] &= ~(1 << u)
                        adj[new_c][u] |= (1 << v)
                        adj[new_c][v] |= (1 << u)
                else:
                    stuck_count += 1
                    continue
            elif best_delta >= 0:
                # No improvement possible, increase stuck count
                stuck_count += 1
                continue
            
            # Apply the best move
            u, v = idx_to_edge[best_edge_idx]
            old_c = c
            new_c = best_new_c
            
            removed = (adj[old_c][u] & adj[old_c][v]).bit_count()
            added = (adj[new_c][u] & adj[new_c][v]).bit_count()
            tri_count += added - removed
            
            colors[best_edge_idx] = new_c
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            adj[new_c][u] |= (1 << v)
            adj[new_c][v] |= (1 << u)
            
            if tri_count < best_tri:
                best_tri = tri_count
                best_result = colors[:]
                if tri_count == 0:
                    return colors
            
            if best_delta < 0:
                stuck_count = 0
            else:
                stuck_count += 1

    return best_result if best_tri == 0 else None

def construct():
    try:
        best = _seed_32()
        best_n = 32
        
        elapsed = time.time() - _START_TIME
        remaining = _DEADLINE - elapsed
        if remaining < 5:
            return best
        
        # Try to find valid colorings for larger n
        # Focus on 50 first, as it is the known record.
        targets = [50, 48, 46, 44, 42, 40]
        
        for target_n in targets:
            if target_n <= best_n:
                continue
            
            elapsed = time.time() - _START_TIME
            remaining = _DEADLINE - elapsed
            if remaining < 3:
                break
            
            # Allocate budget
            if target_n >= 48:
                budget = min(remaining - 2, 100)
            else:
                budget = min(remaining - 2, 20)
            
            if budget < 2:
                continue
            
            # Try Min-Conflicts
            result = _min_conflicts(target_n, budget)
            if result is not None:
                return {'n': target_n, 'colors': result}
        
        return best
    except Exception as e:
        try:
            return _seed_32()
        except:
            return {'n': 1, 'colors': []}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
