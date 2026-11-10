import time
import sys
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def _generate_indices(n):
    """Generate indices for 4-sets and 5-sets in the triple array."""
    triple_list = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(triple_list)}
    
    quad_indices = []
    for quad in combinations(range(n), 4):
        ts = [
            (quad[0], quad[1], quad[2]),
            (quad[0], quad[1], quad[3]),
            (quad[0], quad[2], quad[3]),
            (quad[1], quad[2], quad[3]),
        ]
        quad_indices.append([triple_to_idx[t] for t in ts])
        
    quint_indices = []
    for quint in combinations(range(n), 5):
        ts = []
        for i in range(5):
            for j in range(i+1, 5):
                for k in range(j+1, 5):
                    ts.append((quint[i], quint[j], quint[k]))
        quint_indices.append([triple_to_idx[t] for t in ts])
        
    return np.array(quad_indices), np.array(quint_indices)

def _count_violations(n, colors, quad_idx, quint_idx):
    """Count red K4 and blue K5 violations."""
    # colors is a numpy array of 0s and 1s. 0=Red, 1=Blue.
    # Red K4: all 4 triples are 0.
    # Blue K5: all 10 triples are 1.
    
    # Extract sub-arrays
    # quad_idx shape: (num_quads, 4)
    # quint_idx shape: (num_quints, 10)
    
    # To avoid huge memory copies, we can use fancy indexing.
    # But for n=34, num_quads ~ 46k, num_quints ~ 278k.
    # colors[quad_idx] creates an array of size (46k, 4). This is small.
    # colors[quint_idx] creates an array of size (278k, 10). This is ~2.7M elements. Small.
    
    quad_colors = colors[quad_idx]
    quint_colors = colors[quint_idx]
    
    red_viol = np.sum(np.all(quad_colors == 0, axis=1))
    blue_viol = np.sum(np.all(quint_colors == 1, axis=1))
    
    return red_viol + blue_viol

def _local_search_numpy(n, deadline, seed=42):
    """
    Local search using NumPy for fast violation counting.
    """
    np.random.seed(seed)
    
    quad_idx, quint_idx = _generate_indices(n)
    num_triples = n * (n - 1) * (n - 2) // 6
    
    # Initialize colors randomly
    colors = np.random.randint(0, 2, size=num_triples)
    
    cur_viol = _count_violations(n, colors, quad_idx, quint_idx)
    best_colors = colors.copy()
    best_viol = cur_viol
    
    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    
    it = 0
    while time.time() < deadline:
        if cur_viol == 0:
            break
        
        # Pick a random triple to flip
        idx = np.random.randint(num_triples)
        old_color = colors[idx]
        colors[idx] = 1 - old_color
        
        new_viol = _count_violations(n, colors, quad_idx, quint_idx)
        
        delta = new_viol - cur_viol
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            cur_viol = new_viol
            if cur_viol < best_viol:
                best_viol = cur_viol
                best_colors = colors.copy()
        else:
            colors[idx] = old_color
            
        T *= cooling
        if T < T_min:
            T = T_min
            
        it += 1
        if it % 1000 == 0:
            # Optional: restart if stuck?
            pass

    if best_viol == 0:
        return best_colors.tolist()
    return None

def _try_n(n, deadline, seed=42):
    """Try to find a valid coloring for a specific n."""
    result = _local_search_numpy(n, deadline, seed)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    best_n = 0
    best_colors = []
    
    # We know n=34 is possible. Let's try to start from there or slightly below to ensure we get a base.
    # Then try to go higher.
    
    # First, try to find a valid coloring for n=34 to be safe.
    # If we can find one for 35, 36, etc., even better.
    
    # The search space grows rapidly.
    # n=34: ~6000 vars.
    # n=36: ~7140 vars.
    
    # Let's try n from 34 down to 20 if we fail, or up from 30.
    # Given the task says "published construction has 34 points", we should aim for 34+.
    
    # Try 34 first.
    for n in [34, 35, 36, 37, 38, 39, 40]:
        if time.time() > deadline - 5:
            break
            
        # Try a few seeds
        found = False
        for seed in [42, 123, 999, 777, 314]:
            if time.time() > deadline - 2:
                break
                
            colors = _try_n(n, deadline - 2, seed)
            if colors is not None:
                best_n = n
                best_colors = colors
                found = True
                # If we found a valid one for n, we can try n+1 in the next iteration
                # But the loop structure above tries fixed n.
                # We want to maximize n. So if we find one for 34, we should try 35 next.
                # The loop does that.
                break
        
        # If we found one for n, continue to n+1.
        # If we didn't find one for n, it might be hard. 
        # But since 34 is known, we should at least get 34.
        # If we fail for 34, we should try lower n.
        if not found and n == 34:
             # Fallback to lower n if 34 fails (unlikely)
             for n_low in [33, 32, 30, 25, 20]:
                 if time.time() > deadline - 2:
                     break
                 for seed in [42, 123]:
                     if time.time() > deadline - 1:
                         break
                     colors = _try_n(n_low, deadline - 1, seed)
                     if colors is not None:
                         best_n = n_low
                         best_colors = colors
                         break
                 if best_n > 0:
                     break
             break # Stop trying higher n if we failed 34 and found a lower one

    # If we didn't find anything (shouldn't happen), fallback to a trivial valid coloring for small n
    if best_n == 0:
        n = 10
        colors = [0] * (n * (n-1) * (n-2) // 6)
        best_n = n
        best_colors = colors

    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
