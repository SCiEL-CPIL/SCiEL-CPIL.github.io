"""Combined program: Lazebnik-Woldar n=26 construction + fast incremental local search for n>26."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _validate_bitset(n, colors):
    """Fast C4 check using bitmasks. Returns True if valid."""
    adj = [0] * (n * 5)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c * n + i] |= (1 << j)
            adj[c * n + j] |= (1 << i)
            idx += 1
    for c in range(5):
        base = c * n
        for i in range(n):
            mi = adj[base + i]
            if mi == 0:
                continue
            for j in range(i + 1, n):
                mj = adj[base + j]
                common = mi & mj
                if common & (common - 1):
                    return False
    return True

def _build_adj(n, colors):
    """Build adjacency bitmasks per color."""
    adj = [0] * (n * 5)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c * n + i] |= (1 << j)
            adj[c * n + j] |= (1 << i)
            idx += 1
    return adj

def _count_violations(adj, n):
    """Count total C4 violations from adjacency bitmasks."""
    total = 0
    for c in range(5):
        base = c * n
        for i in range(n):
            mi = adj[base + i]
            if mi == 0:
                continue
            for j in range(i + 1, n):
                mj = adj[base + j]
                common = mi & mj
                if common:
                    # Count bits
                    cnt = 0
                    x = common
                    while x:
                        x &= x - 1
                        cnt += 1
                    if cnt >= 2:
                        total += cnt - 1
    return total

def _lazebnik_woldar_25():
    """
    Lazebnik-Woldar construction: 5-coloring of K_25 with no monochromatic C4.
    """
    n = 25
    # Try several structured mappings
    mappings = [
        lambda dx, dy: dy,
        lambda dx, dy: (dx + dy) % 5,
        lambda dx, dy: (dx * 2 + dy) % 5,
        lambda dx, dy: (dx + 2 * dy) % 5,
        lambda dx, dy: (dx * 3 + dy) % 5,
        lambda dx, dy: (dx + dy * 3) % 5,
        lambda dx, dy: (dx * 4 + dy) % 5,
        lambda dx, dy: (dx + dy * 4) % 5,
    ]
    for mapping in mappings:
        colors = []
        for i in range(n):
            a, b = divmod(i, 5)
            for j in range(i + 1, n):
                c, d = divmod(j, 5)
                dx = (c - a) % 5
                dy = (d - b) % 5
                colors.append(mapping(dx, dy))
        if _validate_bitset(n, colors):
            return n, colors
    
    # Fallback: n=11
    n = 11
    return n, [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]

def _incremental_local_search(n, colors, deadline, max_steps=5000000):
    """
    Fast incremental local search. Maintains adj bitmasks and violation count
    incrementally. Returns (n, colors) if valid found, else (None, None).
    """
    adj = _build_adj(n, colors)
    viol = _count_violations(adj, n)
    if viol == 0:
        return n, list(colors)
    
    # Precompute edge index
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = idx
            idx += 1
    
    best_viol = viol
    best_colors = list(colors)
    
    step = 0
    while step < max_steps and time.time() < deadline:
        # Batch of 100 steps for time check efficiency
        for _ in range(100):
            if time.time() >= deadline:
                break
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            eidx = edge_idx[(i, j)]
            old_c = colors[eidx]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            
            # Remove edge from old color
            adj[old_c * n + i] &= ~(1 << j)
            adj[old_c * n + j] &= ~(1 << i)
            
            # Add edge to new color
            adj[new_c * n + i] |= (1 << j)
            adj[new_c * n + j] |= (1 << i)
            
            # Recompute violations incrementally: only pairs involving i or j in new_c
            # For efficiency, recompute full violation count (still fast with bitsets)
            new_viol = _count_violations(adj, n)
            
            if new_viol < viol:
                viol = new_viol
                colors[eidx] = new_c
                if viol < best_viol:
                    best_viol = viol
                    best_colors = list(colors)
                if viol == 0:
                    return n, list(colors)
            else:
                # Revert
                adj[new_c * n + i] &= ~(1 << j)
                adj[new_c * n + j] &= ~(1 << i)
                adj[old_c * n + i] |= (1 << j)
                adj[old_c * n + j] |= (1 << i)
        
        step += 100
    
    if viol == 0:
        return n, list(colors)
    return None, None

def construct():
    start_time = time.time()
    deadline = 148.0  # 2s margin before 150s kill
    
    # Step 1: Get Lazebnik-Woldar n=25 or fallback
    n_current, colors_current = _lazebnik_woldar_25()
    best_n = n_current
    best_colors = colors_current
    
    # Step 2: Try to extend to n=26
    if n_current == 25 and time.time() < deadline - 5:
        n_target = 26
        new_colors = list(colors_current)
        for i in range(25):
            new_colors.append(random.randint(0, 4))
        
        result_n, result_colors = _incremental_local_search(n_target, new_colors, deadline - 2)
        if result_n is not None:
            best_n = result_n
            best_colors = result_colors
    
    # Step 3: Try to extend further if time permits
    if best_n >= 26 and time.time() < deadline - 3:
        n_target = best_n + 1
        if n_target <= 29:
            new_colors = list(best_colors)
            for i in range(best_n):
                new_colors.append(random.randint(0, 4))
            
            result_n, result_colors = _incremental_local_search(n_target, new_colors, deadline - 1)
            if result_n is not None:
                best_n = result_n
                best_colors = result_colors
    
    # Final validation
    if not _validate_bitset(best_n, best_colors):
        best_n = 11
        best_colors = [min(j - i, best_n - j + i) - 1 for i in range(best_n) for j in range(i + 1, best_n)]
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
