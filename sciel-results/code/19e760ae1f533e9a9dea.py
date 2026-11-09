"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _triple_index(n, a, b, c):
    """Compute the lexicographic index of triple (a,b,c) in the list of all triples on n vertices."""
    # Index = sum over all triples (a',b',c') that come before (a,b,c)
    # a' < a: for each a' in [0,a), sum over b' in (a',n), c' in (b',n)
    # a' == a, b' < b: for each b' in (a,b), sum over c' in (b',n)
    # a' == a, b' == b, c' < c: c' in (b,c)
    idx = 0
    for ap in range(a):
        for bp in range(ap + 1, n):
            idx += n - bp - 1
    for bp in range(a + 1, b):
        idx += n - bp - 1
    idx += c - b - 1
    return idx

def _validate_fast(n, colors):
    """
    Fast validation using numpy. colors is a numpy array of 0/1.
    Returns True if valid.
    """
    # Build a 3D lookup: color[a,b,c] for a<b<c
    # Use a 2D array where we store colors for each (a,b) pair as a bitmask or array
    # For n up to 40, C(40,3) = 9880 triples
    
    # Build color lookup as a dictionary or 3D array
    # 3D array of shape (n,n,n) is 40*40*40 = 64000, which is fine
    color3d = np.zeros((n, n, n), dtype=np.int8)
    
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                color3d[a, b, c] = colors[idx]
                idx += 1
    
    # Check red K4: no 4 vertices with all 4 triples red (color=0)
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                if color3d[a, b, c] == 0:
                    for d in range(c + 1, n):
                        if (color3d[a, b, d] == 0 and color3d[a, c, d] == 0 and color3d[b, c, d] == 0):
                            return False
                        # Also need to check d < a, a < d < b, b < d < c cases
                    # Check d between a and b
                    for d in range(a + 1, b):
                        if (color3d[a, d, c] == 0 and color3d[a, d, b] == 0 and color3d[d, b, c] == 0):
                            return False
                    # Check d between b and c
                    for d in range(b + 1, c):
                        if (color3d[a, b, d] == 0 and color3d[a, d, c] == 0 and color3d[b, d, c] == 0):
                            return False
                    # Check d < a
                    for d in range(a):
                        if (color3d[d, b, c] == 0 and color3d[d, a, c] == 0 and color3d[d, a, b] == 0):
                            return False
    
    # Check blue K5: no 5 vertices with all 10 triples blue (color=1)
    # This is more expensive. For each 5-set, check all 10 triples.
    # C(40,5) = 658008, and 10 checks each = ~6.5M checks
    # Let's use a smarter approach: for each 4-set that's all-blue, check if adding a 5th vertex keeps it all-blue
    
    # Build set of blue triples for fast lookup
    blue_set = set()
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                if colors[idx] == 1:
                    blue_set.add((a, b, c))
                idx += 1
    
    for quint in combinations(range(n), 5):
        all_blue = True
        for i in range(5):
            for j in range(i + 1, 5):
                for k in range(j + 1, 5):
                    if (quint[i], quint[j], quint[k]) not in blue_set:
                        all_blue = False
                        break
                if not all_blue:
                    break
            if not all_blue:
                break
        if all_blue:
            return False
    
    return True

def _init_violations(n, colors):
    """
    Count initial violations: number of red K4s + number of blue K5s.
    colors: numpy array of 0/1
    Returns (n_violations, red_k4_count, blue_k5_count)
    """
    # Build color3d
    color3d = np.zeros((n, n, n), dtype=np.int8)
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                color3d[a, b, c] = colors[idx]
                idx += 1
    
    red_k4 = 0
    blue_k5 = 0
    
    # Count red K4s
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                if color3d[a, b, c] == 0:
                    for d in range(c + 1, n):
                        if color3d[a, b, d] == 0 and color3d[a, c, d] == 0 and color3d[b, c, d] == 0:
                            red_k4 += 1
                    for d in range(a + 1, b):
                        if color3d[a, d, c] == 0 and color3d[a, d, b] == 0 and color3d[d, b, c] == 0:
                            red_k4 += 1
                    for d in range(b + 1, c):
                        if color3d[a, b, d] == 0 and color3d[a, d, c] == 0 and color3d[b, d, c] == 0:
                            red_k4 += 1
                    for d in range(a):
                        if color3d[d, b, c] == 0 and color3d[d, a, c] == 0 and color3d[d, a, b] == 0:
                            red_k4 += 1
    
    # Count blue K5s
    blue_set = set()
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                if colors[idx] == 1:
                    blue_set.add((a, b, c))
                idx += 1
    
    for quint in combinations(range(n), 5):
        all_blue = True
        for i in range(5):
            for j in range(i + 1, 5):
                for k in range(j + 1, 5):
                    if (quint[i], quint[j], quint[k]) not in blue_set:
                        all_blue = False
                        break
                if not all_blue:
                    break
            if not all_blue:
                break
        if all_blue:
            blue_k5 += 1
    
    return red_k4 + blue_k5, red_k4, blue_k5

def _incremental_flip_violation_delta(n, colors, triple_idx, new_color):
    """
    Compute the change in violation count when flipping triple_idx to new_color.
    Returns delta (new_violations - old_violations).
    colors: numpy array of 0/1
    """
    # Find the triple (a,b,c) corresponding to triple_idx
    # This is expensive; we need a reverse mapping
    # For now, we'll do a full re-validation which is simpler but slower
    # In a production version, we'd precompute the mapping
    
    # For efficiency, let's just do full validation
    old_viol, _, _ = _init_violations(n, colors)
    colors[triple_idx] = new_color
    new_viol, _, _ = _init_violations(n, colors)
    colors[triple_idx] = 1 - new_color  # restore
    return new_viol - old_viol

def _local_search(n, colors, max_iters=5000, timeout=30.0):
    """
    Local search to minimize violations.
    colors: numpy array of 0/1
    Returns improved colors array.
    """
    start = time.time()
    n_triples = len(colors)
    
    cur_viol, _, _ = _init_violations(n, colors)
    best_viol = cur_viol
    best_colors = colors.copy()
    
    # Precompute triple list for random access
    triples = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                triples.append((a, b, c))
    
    for it in range(max_iters):
        if time.time() > start + timeout:
            break
        if cur_viol == 0:
            break
        
        # Pick a random triple to flip
        idx = random.randint(0, n_triples - 1)
        old_color = colors[idx]
        new_color = 1 - old_color
        
        # Try the flip
        colors[idx] = new_color
        new_viol, _, _ = _init_violations(n, colors)
        
        if new_viol < cur_viol:
            cur_viol = new_viol
            if cur_viol < best_viol:
                best_viol = cur_viol
                best_colors = colors.copy()
        else:
            # Revert
            colors[idx] = old_color
    
    return best_colors, best_viol

def _simulated_annealing(n, colors, timeout=60.0):
    """
    Simulated annealing to find a valid coloring.
    colors: numpy array of 0/1
    Returns (best_colors, best_viol).
    """
    start = time.time()
    n_triples = len(colors)
    
    cur_viol, _, _ = _init_violations(n, colors)
    best_viol = cur_viol
    best_colors = colors.copy()
    
    T = 5.0
    cooling = 0.999
    
    while time.time() < start + timeout:
        if cur_viol == 0:
            break
        
        idx = random.randint(0, n_triples - 1)
        old_color = colors[idx]
        new_color = 1 - old_color
        
        colors[idx] = new_color
        new_viol, _, _ = _init_violations(n, colors)
        delta = new_viol - cur_viol
        
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            cur_viol = new_viol
            if cur_viol < best_viol:
                best_viol = cur_viol
                best_colors = colors.copy()
        else:
            colors[idx] = old_color
        
        T *= cooling
    
    return best_colors, best_viol

def construct():
    """
    Construct a valid two-coloring of triples on n vertices with no red K4^(3)
    and no blue K5^(3), maximizing n.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_n = 0
    best_colors = None
    
    # Try modular constructions for various n, starting from high n
    for n in range(34, 8, -1):
        if time.time() > deadline:
            break
        
        found = False
        for m in range(2, n + 15):
            if time.time() > deadline:
                break
            
            colors = np.zeros(n * (n - 1) * (n - 2) // 6, dtype=np.int8)
            idx = 0
            for a in range(n):
                for b in range(a + 1, n):
                    for c in range(b + 1, n):
                        if (a + b + c) % m == 0:
                            colors[idx] = 0
                        else:
                            colors[idx] = 1
                        idx += 1
            
            if _validate_fast(n, colors):
                best_n = n
                best_colors = colors.tolist()
                found = True
                break
        
        if found:
            break
    
    # If we found a valid coloring, return it
    if best_n > 0 and best_colors is not None:
        return {'n': best_n, 'colors': best_colors}
    
    # If no algebraic construction works, try local search from random starts
    for n in range(34, 8, -1):
        if time.time() > deadline:
            break
        
        # Try a few random initial colorings
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            n_triples = n * (n - 1) * (n - 2) // 6
            colors = np.random.randint(0, 2, size=n_triples, dtype=np.int8)
            
            # Run simulated annealing
            improved, viol = _simulated_annealing(n, colors, timeout=20.0)
            
            if viol == 0:
                best_n = n
                best_colors = improved.tolist()
                break
        
        if best_n == n:
            break
    
    if best_n > 0 and best_colors is not None:
        return {'n': best_n, 'colors': best_colors}
    
    # Fallback: return a small valid coloring
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 
              for a in range(n) for b in range(a+1, n) for c in range(b+1, n)]
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
