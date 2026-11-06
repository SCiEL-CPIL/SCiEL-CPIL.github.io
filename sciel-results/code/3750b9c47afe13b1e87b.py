import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Precompute triple index mapping for each n
    def build_index(n):
        tri_to_idx = {}
        i = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    tri_to_idx[(a,b,c)] = i
                    i += 1
        return tri_to_idx, i

    def get_violations(colors, red_indices, blue_indices):
        red_viol = np.all(colors[red_indices] == 0, axis=1)
        blue_viol = np.all(colors[blue_indices] == 1, axis=1)
        return red_viol, blue_viol, np.sum(red_viol) + np.sum(blue_viol)

    def count_violations_fast(colors, red_indices, blue_indices):
        # Vectorized violation count
        red_viol = np.all(colors[red_indices] == 0, axis=1)
        blue_viol = np.all(colors[blue_indices] == 1, axis=1)
        return np.sum(red_viol) + np.sum(blue_viol)

    # Simulated annealing for a given n
    def solve_n(n, max_time=100, seed_colors=None):
        tri_to_idx, n_triples = build_index(n)
        
        red_4sets = list(combinations(range(n), 4))
        blue_5sets = list(combinations(range(n), 5))
        
        red_indices = np.zeros((len(red_4sets), 4), dtype=np.int32)
        for k, quad in enumerate(red_4sets):
            for j, tri in enumerate(combinations(quad, 3)):
                red_indices[k, j] = tri_to_idx[tri]
                
        blue_indices = np.zeros((len(blue_5sets), 10), dtype=np.int32)
        for k, quint in enumerate(blue_5sets):
            for j, tri in enumerate(combinations(quint, 3)):
                blue_indices[k, j] = tri_to_idx[tri]
        
        # Initialize colors
        if seed_colors is not None and len(seed_colors) == n_triples:
            colors = np.array(seed_colors, dtype=np.int8).copy()
        else:
            # Biased initialization: slightly favor blue to help avoid blue K5
            # Actually, let's use 0.5 probability
            colors = np.random.randint(0, 2, n_triples, dtype=np.int8)
        
        # Compute initial violations
        red_viol, blue_viol, total_viol = get_violations(colors, red_indices, blue_indices)
        
        if total_viol == 0:
            return colors.tolist()
        
        # Simulated annealing
        best_colors = colors.copy()
        best_viol = total_viol
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        rng = np.random.default_rng(42)
        
        while time.time() < max_time and T > T_min:
            # Pick a random triple to flip
            tri_idx = rng.integers(0, n_triples)
            
            old_color = colors[tri_idx]
            new_color = 1 - old_color
            colors[tri_idx] = new_color
            
            # Compute new violations
            new_red_viol, new_blue_viol, new_total_viol = get_violations(colors, red_indices, blue_indices)
            
            delta_viol = new_total_viol - total_viol
            
            if delta_viol <= 0 or rng.random() < np.exp(-delta_viol / T):
                total_viol = new_total_viol
                if total_viol < best_viol:
                    best_viol = total_viol
                    best_colors = colors.copy()
                    if best_viol == 0:
                        return best_colors.tolist()
            else:
                colors[tri_idx] = old_color
            
            T *= cooling
        
        if best_viol == 0:
            return best_colors.tolist()
        
        return None

    # Strategy: Try from n=34 down, using warm-starts from smaller solutions
    best_result = None
    
    # First, try to get a solution for small n to use as warm-start
    # n=5: need to avoid blue K5 (all 10 triples blue). Easy: just don't color all blue.
    # n=10: should be feasible
    
    # Let's try a direct approach: solve for n=34 with multiple SA restarts
    # but with better initialization
    
    # Precompute for n=34
    n_target = 34
    tri_to_idx_34, n_triples_34 = build_index(n_target)
    
    red_4sets_34 = list(combinations(range(n_target), 4))
    blue_5sets_34 = list(combinations(range(n_target), 5))
    
    red_indices_34 = np.zeros((len(red_4sets_34), 4), dtype=np.int32)
    for k, quad in enumerate(red_4sets_34):
        for j, tri in enumerate(combinations(quad, 3)):
            red_indices_34[k, j] = tri_to_idx_34[tri]
            
    blue_indices_34 = np.zeros((len(blue_5sets_34), 10), dtype=np.int32)
    for k, quint in enumerate(blue_5sets_34):
        for j, tri in enumerate(combinations(quint, 3)):
            blue_indices_34[k, j] = tri_to_idx_34[tri]
    
    # Try multiple SA runs with different seeds
    rng = np.random.default_rng(123)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Initialize with random colors
        colors = rng.integers(0, 2, n_triples_34, dtype=np.int8)
        
        # Quick greedy phase first to reduce violations
        for it in range(200):
            if time.time() > deadline:
                break
            red_viol = np.all(colors[red_indices_34] == 0, axis=1)
            blue_viol = np.all(colors[blue_indices_34] == 1, axis=1)
            total_viol = np.sum(red_viol) + np.sum(blue_viol)
            
            if total_viol == 0:
                return {'n': n_target, 'colors': colors.tolist()}
            
            # Find worst triple
            badness = np.zeros(n_triples_34, dtype=np.int32)
            if np.sum(red_viol) > 0:
                red_bad = red_indices_34[red_viol].flatten()
                badness += np.bincount(red_bad, minlength=n_triples_34)
            if np.sum(blue_viol) > 0:
                blue_bad = blue_indices_34[blue_viol].flatten()
                badness += np.bincount(blue_bad, minlength=n_triples_34)
            
            if np.max(badness) == 0:
                tri_to_flip = rng.integers(0, n_triples_34)
            else:
                tri_to_flip = np.argmax(badness)
            
            colors[tri_to_flip] = 1 - colors[tri_to_flip]
        
        # Then simulated annealing
        red_viol, blue_viol, total_viol = get_violations(colors, red_indices_34, blue_indices_34)
        
        if total_viol == 0:
            return {'n': n_target, 'colors': colors.tolist()}
        
        best_colors = colors.copy()
        best_viol = total_viol
        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        
        sa_rng = np.random.default_rng(restart * 7 + 1)
        
        while time.time() < deadline - 5 and T > T_min:
            tri_idx = sa_rng.integers(0, n_triples_34)
            
            old_color = colors[tri_idx]
            new_color = 1 - old_color
            colors[tri_idx] = new_color
            
            new_red_viol = np.all(colors[red_indices_34] == 0, axis=1)
            new_blue_viol = np.all(colors[blue_indices_34] == 1, axis=1)
            new_total_viol = np.sum(new_red_viol) + np.sum(new_blue_viol)
            
            delta_viol = new_total_viol - total_viol
            
            if delta_viol <= 0 or sa_rng.random() < np.exp(-delta_viol / T):
                total_viol = new_total_viol
                if total_viol < best_viol:
                    best_viol = total_viol
                    best_colors = colors.copy()
                    if best_viol == 0:
                        return {'n': n_target, 'colors': best_colors.tolist()}
            else:
                colors[tri_idx] = old_color
            
            T *= cooling
        
        if best_viol == 0:
            return {'n': n_target, 'colors': best_colors.tolist()}
        
        # Keep best so far
        if best_result is None or best_viol < best_result[0]:
            best_result = (best_viol, best_colors.copy())
    
    # If we found a good but not perfect solution, try to repair
    if best_result is not None and best_result[0] < 100:
        colors = best_result[1].copy()
        # Greedy repair
        for it in range(10000):
            if time.time() > deadline:
                break
            red_viol = np.all(colors[red_indices_34] == 0, axis=1)
            blue_viol = np.all(colors[blue_indices_34] == 1, axis=1)
            total_viol = np.sum(red_viol) + np.sum(blue_viol)
            
            if total_viol == 0:
                return {'n': n_target, 'colors': colors.tolist()}
            
            badness = np.zeros(n_triples_34, dtype=np.int32)
            if np.sum(red_viol) > 0:
                red_bad = red_indices_34[red_viol].flatten()
                badness += np.bincount(red_bad, minlength=n_triples_34)
            if np.sum(blue_viol) > 0:
                blue_bad = blue_indices_34[blue_viol].flatten()
                badness += np.bincount(blue_bad, minlength=n_triples_34)
            
            if np.max(badness) == 0:
                break
            tri_to_flip = np.argmax(badness)
            colors[tri_to_flip] = 1 - colors[tri_to_flip]
        
        red_viol = np.all(colors[red_indices_34] == 0, axis=1)
        blue_viol = np.all(colors[blue_indices_34] == 1, axis=1)
        if np.sum(red_viol) + np.sum(blue_viol) == 0:
            return {'n': n_target, 'colors': colors.tolist()}
    
    # Fallback: try smaller n
    for n in range(33, 3, -1):
        if time.time() > deadline:
            break
        
        tri_to_idx, n_triples = build_index(n)
        red_4sets = list(combinations(range(n), 4))
        blue_5sets = list(combinations(range(n), 5))
        
        red_indices = np.zeros((len(red_4sets), 4), dtype=np.int32)
        for k, quad in enumerate(red_4sets):
            for j, tri in enumerate(combinations(quad, 3)):
                red_indices[k, j] = tri_to_idx[tri]
                
        blue_indices = np.zeros((len(blue_5sets), 10), dtype=np.int32)
        for k, quint in enumerate(blue_5sets):
            for j, tri in enumerate(combinations(quint, 3)):
                blue_indices[k, j] = tri_to_idx[tri]
        
        # Quick SA
        rng_local = np.random.default_rng(n * 13 + 7)
        colors = rng_local.integers(0, 2, n_triples, dtype=np.int8)
        
        for it in range(5000):
            if time.time() > deadline:
                break
            red_viol = np.all(colors[red_indices] == 0, axis=1)
            blue_viol = np.all(colors[blue_indices] == 1, axis=1)
            total_viol = np.sum(red_viol) + np.sum(blue_viol)
            
            if total_viol == 0:
                return {'n': n, 'colors': colors.tolist()}
            
            badness = np.zeros(n_triples, dtype=np.int32)
            if np.sum(red_viol) > 0:
                red_bad = red_indices[red_viol].flatten()
                badness += np.bincount(red_bad, minlength=n_triples)
            if np.sum(blue_viol) > 0:
                blue_bad = blue_indices[blue_viol].flatten()
                badness += np.bincount(blue_bad, minlength=n_triples)
            
            if np.max(badness) == 0:
                tri_to_flip = rng_local.integers(0, n_triples)
            else:
                tri_to_flip = np.argmax(badness)
            
            colors[tri_to_flip] = 1 - colors[tri_to_flip]
    
    # Absolute fallback
    n = 4
    colors = [0, 0, 0, 1]
    return {'n': n, 'colors': colors}
