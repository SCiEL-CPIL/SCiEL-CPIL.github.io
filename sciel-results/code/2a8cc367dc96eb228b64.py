import time
import numpy as np
import itertools

def construct():
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer

    def precompute_indices(n):
        six_sets = list(itertools.combinations(range(n), 6))
        num_six = len(six_sets)
        
        def e_idx(u, v):
            if u > v:
                u, v = v, u
            return u * (n - 1) - (u * (u + 1)) // 2 + (v - u - 1)

        partitions = []
        for partner in range(1, 6):
            remaining = [i for i in range(6) if i != 0 and i != partner]
            a, b, c, d = remaining
            pairs1 = [(0, partner), (a, b), (c, d)]
            pairs2 = [(0, partner), (a, c), (b, d)]
            pairs3 = [(0, partner), (a, d), (b, c)]
            partitions.extend([pairs1, pairs2, pairs3])
            
        indices = np.zeros((15, num_six, 12), dtype=np.int32)
        
        for p_idx, part in enumerate(partitions):
            for s_idx, vs in enumerate(six_sets):
                p1, p2, p3 = part
                v1a, v1b = vs[p1[0]], vs[p1[1]]
                v2a, v2b = vs[p2[0]], vs[p2[1]]
                v3a, v3b = vs[p3[0]], vs[p3[1]]
                
                e1 = e_idx(v1a, v2a)
                e2 = e_idx(v1a, v2b)
                e3 = e_idx(v1b, v2a)
                e4 = e_idx(v1b, v2b)
                e5 = e_idx(v2a, v3a)
                e6 = e_idx(v2a, v3b)
                e7 = e_idx(v2b, v3a)
                e8 = e_idx(v2b, v3b)
                e9 = e_idx(v1a, v3a)
                e10 = e_idx(v1a, v3b)
                e11 = e_idx(v1b, v3a)
                e12 = e_idx(v1b, v3b)
                
                indices[p_idx, s_idx, :] = [e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12]
                
        return indices

    def count_mono_k222(colors, indices):
        c_vals = colors[indices]
        all_0 = np.all(c_vals == 0, axis=2)
        all_1 = np.all(c_vals == 1, axis=2)
        return np.sum(all_0) + np.sum(all_1)

    def get_paley_colors(n):
        squares = set()
        for x in range(1, n):
            squares.add((x * x) % n)
        
        colors = np.zeros(n * (n - 1) // 2, dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                if diff in squares:
                    colors[idx] = 0
                else:
                    colors[idx] = 1
                idx += 1
        return colors

    # Try N=29
    n_target = 29
    indices_29 = precompute_indices(n_target)
    colors = get_paley_colors(n_target)
    current_score = count_mono_k222(colors, indices_29)
    
    if current_score == 0:
        return {'n': n_target, 'colors': colors.tolist()}

    # Local Search for 29
    best_score = current_score
    best_colors = colors.copy()
    T = 10.0
    T_min = 0.01
    cooling = 0.99
    
    for iter_num in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        e = np.random.randint(0, len(colors))
        colors[e] ^= 1
        new_score = count_mono_k222(colors, indices_29)
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_colors = colors.copy()
                if best_score == 0:
                    return {'n': n_target, 'colors': best_colors.tolist()}
        else:
            colors[e] ^= 1
            
        T *= cooling
        if T < T_min:
            T = 10.0
            colors = best_colors.copy()
            current_score = best_score

    # Try N=28
    if best_score > 0:
        n_target = 28
        indices_28 = precompute_indices(n_target)
        colors = get_paley_colors(n_target)
        current_score = count_mono_k222(colors, indices_28)
        
        if current_score == 0:
            return {'n': n_target, 'colors': colors.tolist()}
            
        best_score = current_score
        best_colors = colors.copy()
        T = 10.0
        T_min = 0.01
        cooling = 0.99
        
        for iter_num in range(1000):
            if time.time() - start_time > time_limit:
                break
            e = np.random.randint(0, len(colors))
            colors[e] ^= 1
            new_score = count_mono_k222(colors, indices_28)
            delta = new_score - current_score
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors.copy()
                    if best_score == 0:
                        return {'n': n_target, 'colors': best_colors.tolist()}
            else:
                colors[e] ^= 1
            T *= cooling
            if T < T_min:
                T = 10.0
                colors = best_colors.copy()
                current_score = best_score

    # Fallback to smaller N
    for n_try in [27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]:
        if time.time() - start_time > time_limit - 5:
            break
        indices_n = precompute_indices(n_try)
        colors = get_paley_colors(n_try)
        score = count_mono_k222(colors, indices_n)
        if score == 0:
            return {'n': n_try, 'colors': colors.tolist()}
            
        best_s = score
        best_c = colors.copy()
        T = 5.0
        for _ in range(200):
            if time.time() - start_time > time_limit - 5:
                break
            e = np.random.randint(0, len(colors))
            colors[e] ^= 1
            new_s = count_mono_k222(colors, indices_n)
            delta = new_s - score
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                score = new_s
                if score < best_s:
                    best_s = score
                    best_c = colors.copy()
                    if best_s == 0:
                        return {'n': n_try, 'colors': best_c.tolist()}
            else:
                colors[e] ^= 1
            T *= 0.95
            if T < 0.1:
                T = 5.0
                colors = best_c.copy()
                score = best_s

    # Absolute fallback
    n_fallback = 17
    indices_f = precompute_indices(n_fallback)
    colors_f = get_paley_colors(n_fallback)
    if count_mono_k222(colors_f, indices_f) == 0:
         return {'n': n_fallback, 'colors': colors_f.tolist()}
         
    return {'n': 1, 'colors': []}
