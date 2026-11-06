"""Combined program: algebraic modular construction + local search refinement."""
import time
import random
import numpy as np

def _verify(colors, n=128):
    """Check if every non-constant 4-AP uses at least 3 distinct colors."""
    for a in range(n):
        for d in range(1, (n - a) // 3 + 1):
            i3 = a + 3 * d
            if i3 >= n:
                break
            c = {colors[a], colors[a + d], colors[a + 2 * d], colors[a + 3 * d]}
            if len(c) < 3:
                return False
    return True

def _count_violations(colors, aps, n=128):
    """Count number of 4-APs with < 3 distinct colors."""
    c0 = colors[aps[:, 0]]
    c1 = colors[aps[:, 1]]
    c2 = colors[aps[:, 2]]
    c3 = colors[aps[:, 3]]
    
    # All same
    all_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
    
    # Exactly 2 colors (various patterns)
    # 3 same, 1 different
    p1 = (c0 == c1) & (c1 == c2) & (c2 != c3)
    p2 = (c0 == c1) & (c1 != c2) & (c2 == c3)
    p3 = (c0 != c1) & (c1 == c2) & (c2 == c3)
    p4 = (c0 == c2) & (c2 == c3) & (c0 != c1)
    # 2 and 2
    p5 = (c0 == c1) & (c2 == c3) & (c0 != c2)
    p6 = (c0 == c2) & (c1 == c3) & (c0 != c1)
    p7 = (c0 == c3) & (c1 == c2) & (c0 != c1)
    
    two_colors = p1 | p2 | p3 | p4 | p5 | p6 | p7
    bad = all_same | two_colors
    return np.sum(bad), bad

def _local_search(colors, aps, num_colors, n=128, max_iters=5000, deadline=None):
    """Local search to fix violations by changing individual positions."""
    colors = colors.copy()
    current_bad, bad_mask = _count_violations(colors, aps, n)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if current_bad == 0:
            break
        
        bad_indices = np.where(bad_mask)[0]
        if len(bad_indices) == 0:
            break
        
        bad_ap_idx = bad_indices[random.randint(0, len(bad_indices) - 1)]
        ap = aps[bad_ap_idx]
        pos = int(ap[random.randint(0, 3)])
        old = colors[pos]
        
        best_new = old
        best_bad = current_bad
        
        for nc in range(num_colors):
            if nc == old:
                continue
            colors[pos] = nc
            new_bad, new_bad_mask = _count_violations(colors, aps, n)
            if new_bad < best_bad:
                best_bad = new_bad
                best_new = nc
                if best_bad == 0:
                    break
        colors[pos] = best_new
        current_bad = best_bad
        if best_new != old:
            _, bad_mask = _count_violations(colors, aps, n)
    
    return colors, current_bad

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin before 60s limit
    n = 128
    
    # Precompute all 4-APs
    ap_list = []
    for a in range(n):
        for d in range(1, (n - a) // 3 + 1):
            i3 = a + 3 * d
            if i3 >= n:
                break
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    aps = np.array(ap_list, dtype=np.int32)
    
    # Start with modular construction using prime 43
    m = 43
    colors = np.array([x % m for x in range(1, n + 1)], dtype=np.int32)
    
    # Verify the modular coloring
    if not _verify(colors.tolist(), n):
        # Fallback to greedy if modular fails (shouldn't happen)
        colors_list = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color} | {colors_list[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 
                        for d in range(1, (x - 1) // 3 + 1))
                    and all(any(colors_list[x - 1 - j * d] != color for j in (1, 2)) 
                            for d in range(1, (x - 1) // 2 + 1))):
                    colors_list.append(color)
                    break
        colors = np.array(colors_list, dtype=np.int32)
    
    # Try to reduce colors via local search
    # First, try to map colors to a smaller palette
    used_colors = np.unique(colors)
    num_used = len(used_colors)
    
    # Try reducing to fewer colors
    for target_colors in range(3, min(num_used, 15)):
        if time.time() > deadline:
            break
        
        # Remap colors to target palette
        color_map = {}
        for i, c in enumerate(used_colors):
            color_map[c] = i % target_colors
        
        remapped = np.array([color_map[c] for c in colors], dtype=np.int32)
        
        # Check if remapped is valid
        if _verify(remapped.tolist(), n):
            colors = remapped
            break
        
        # If not valid, try local search to fix
        fixed, bad = _local_search(remapped, aps, target_colors, n, 
                                   max_iters=2000, deadline=deadline)
        if bad == 0:
            colors = fixed
            break
    
    # Final verification
    if not _verify(colors.tolist(), n):
        # Fallback to original modular if something went wrong
        colors = np.array([x % m for x in range(1, n + 1)], dtype=np.int32)
    
    return colors.tolist()
