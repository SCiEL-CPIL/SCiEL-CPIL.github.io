import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    best_count = float('inf')
    best_coeffs = None

    def eval_square_fast(coeffs_list):
        """
        Fast evaluation using dict for sparse representation.
        coeffs_list: list of (exp, coef) with len 12, coef != 0
        Returns: count_nonzero_in_square
        """
        cl = sorted(coeffs_list)
        square_dict = defaultdict(int)
        
        for i in range(12):
            ei, ci = cl[i]
            square_dict[2 * ei] += ci * ci
            for j in range(i + 1, 12):
                ej, cj = cl[j]
                square_dict[ei + ej] += 2 * ci * cj
        
        return sum(1 for v in square_dict.values() if v != 0)

    def to_full(coeffs_list):
        if not coeffs_list:
            return None
        max_e = max(e for e, c in coeffs_list)
        if max_e >= 41:  # Length must be <= 41, so max index 40
            return None
        
        full = [0] * (max_e + 1)
        for e, c in coeffs_list:
            full[e] = c
        
        # Verify constraints
        if len(full) > 41:
            return None
        if full[-1] == 0:
            return None
        
        nz = sum(1 for x in full if x != 0)
        if nz != 12:
            return None
        if any(abs(x) > 64 for x in full):
            return None
            
        return full

    def try_update(count, coeffs_list):
        nonlocal best_count, best_coeffs
        if count < best_count:
            full = to_full(coeffs_list)
            if full is not None:
                best_count = count
                best_coeffs = full
                return True
        return False

    def brute_force_signs(exps, time_budget=5.0):
        """
        For a fixed set of exponents, try all 2^12 sign patterns with coefficient magnitudes from a small set.
        Focus on exponents that create many collisions (arithmetic progression).
        """
        nonlocal best_count, best_coeffs
        t0 = time.time()
        
        # Use small coefficient magnitudes to encourage cancellation
        mags = [1, 2, 3, 4]
        
        # For each magnitude pattern, try all sign combinations
        # This is 4^12 * 2^12 which is too many, so we'll be smarter
        # Try: all coefficients = +/-1 first (2^12 = 4096)
        # Then: mix of 1s and 2s
        
        # Phase 1: All +/-1
        for mask in range(4096):
            if time.time() - t0 > time_budget:
                break
            cl = []
            for i in range(12):
                sign = 1 if (mask >> i) & 1 else -1
                cl.append((exps[i], sign))
            c = eval_square_fast(cl)
            if c < best_count:
                try_update(c, cl)
        
        # Phase 2: Mix of 1s and 2s (2^24 = 16M, too many)
        # Instead: try specific patterns
        if time.time() - t0 < time_budget - 1:
            # Try alternating magnitudes
            mag_patterns = [
                [1]*12,
                [2]*12,
                [1,2,1,2,1,2,1,2,1,2,1,2],
                [2,1,2,1,2,1,2,1,2,1,2,1],
                [1,1,2,2,1,1,2,2,1,1,2,2],
                [1,1,1,2,1,1,1,2,1,1,1,2],
                [1,2,2,1,1,2,2,1,1,2,2,1],
            ]
            
            for mag_pat in mag_patterns:
                if time.time() - t0 > time_budget:
                    break
                # Try a subset of sign patterns for this magnitude
                step = max(1, 4096 // 200)
                for mask in range(0, 4096, step):
                    if time.time() - t0 > time_budget:
                        break
                    cl = []
                    for i in range(12):
                        sign = 1 if (mask >> i) & 1 else -1
                        cl.append((exps[i], sign * mag_pat[i]))
                    c = eval_square_fast(cl)
                    if c < best_count:
                        try_update(c, cl)
        
        return cl

    def local_search(cl, time_budget=5.0):
        """
        Local search: change individual coefficients or exponents to reduce square sparsity.
        """
        nonlocal best_count, best_coeffs
        t0 = time.time()
        cl = [tuple(x) for x in cl]
        if len(cl) != 12:
            return cl
        
        count = eval_square_fast(cl)
        try_update(count, cl)
        
        vals = [-4, -3, -2, -1, 1, 2, 3, 4, -8, 8, -16, 16, -32, 32]
        no_improve = 0
        
        while time.time() - t0 < time_budget and no_improve < 50:
            idx = random.randrange(12)
            e, c = cl[idx]
            
            # Try changing coefficient
            new_c = random.choice(vals)
            if new_c != c and abs(new_c) <= 64:
                cl2 = list(cl)
                cl2[idx] = (e, new_c)
                c2 = eval_square_fast(cl2)
                if c2 < count:
                    cl = cl2
                    count = c2
                    no_improve = 0
                    try_update(count, cl)
                else:
                    no_improve += 1
            
            # Try shifting exponent
            shift = random.choice([-2, -1, 1, 2])
            new_e = e + shift
            if 0 <= new_e <= 40:
                exps = [cl[i][0] for i in range(12)]
                exps.remove(e)
                if new_e not in exps:
                    cl2 = list(cl)
                    cl2[idx] = (new_e, c)
                    c2 = eval_square_fast(cl2)
                    if c2 < count:
                        cl = cl2
                        count = c2
                        no_improve = 0
                        try_update(count, cl)
                    else:
                        no_improve += 1
            
            if best_count == 12:
                break
        
        return cl

    def simulated_annealing(cl, time_budget=5.0):
        nonlocal best_count, best_coeffs
        t0 = time.time()
        cl = [tuple(x) for x in cl]
        if len(cl) != 12:
            return cl, float('inf')
        
        count = eval_square_fast(cl)
        try_update(count, cl)
        
        current_temp = 2.0
        cooling = 0.995
        min_temp = 0.1
        
        vals = [-4, -3, -2, -1, 1, 2, 3, 4, -8, 8, -16, 16, -32, 32]
        
        while time.time() - t0 < time_budget and current_temp > min_temp:
            move_type = random.random()
            
            if move_type < 0.5:
                # Change coefficient
                idx = random.randrange(12)
                e, c = cl[idx]
                new_c = random.choice(vals)
                if new_c != c and abs(new_c) <= 64:
                    cl2 = list(cl)
                    cl2[idx] = (e, new_c)
                    c2 = eval_square_fast(cl2)
                    delta = c2 - count
                    if delta < 0 or random.random() < np.exp(-delta / current_temp):
                        cl = cl2
                        count = c2
                        try_update(count, cl)
            elif move_type < 0.8:
                # Shift exponent
                idx = random.randrange(12)
                e, c = cl[idx]
                shift = random.choice([-3, -2, -1, 1, 2, 3])
                new_e = e + shift
                if 0 <= new_e <= 40:
                    exps = [cl[i][0] for i in range(12)]
                    exps.remove(e)
                    if new_e not in exps:
                        cl2 = list(cl)
                        cl2[idx] = (new_e, c)
                        c2 = eval_square_fast(cl2)
                        delta = c2 - count
                        if delta < 0 or random.random() < np.exp(-delta / current_temp):
                            cl = cl2
                            count = c2
                            try_update(count, cl)
            elif move_type < 0.95:
                # Swap coefficients
                i, j = random.sample(range(12), 2)
                e_i, c_i = cl[i]
                e_j, c_j = cl[j]
                if c_i != c_j:
                    cl2 = list(cl)
                    cl2[i] = (e_i, c_j)
                    cl2[j] = (e_j, c_i)
                    c2 = eval_square_fast(cl2)
                    delta = c2 - count
                    if delta < 0 or random.random() < np.exp(-delta / current_temp):
                        cl = cl2
                        count = c2
                        try_update(count, cl)
            else:
                # Pair change
                i, j = random.sample(range(12), 2)
                e_i, c_i = cl[i]
                e_j, c_j = cl[j]
                new_c_i = random.choice(vals)
                new_c_j = random.choice(vals)
                if (new_c_i != c_i or new_c_j != c_j) and abs(new_c_i) <= 64 and abs(new_c_j) <= 64:
                    cl2 = list(cl)
                    cl2[i] = (e_i, new_c_i)
                    cl2[j] = (e_j, new_c_j)
                    c2 = eval_square_fast(cl2)
                    delta = c2 - count
                    if delta < 0 or random.random() < np.exp(-delta / current_temp):
                        cl = cl2
                        count = c2
                        try_update(count, cl)
                        
            current_temp *= cooling
            if best_count == 12:
                break
                
        return cl, count

    # =====================================================================
    # Phase 1: Brute force on consecutive exponents (0..11)
    # =====================================================================
    
    random.seed(42)
    
    exps_consec = list(range(12))
    brute_force_signs(exps_consec, time_budget=8.0)
    if best_count == 12:
        return best_coeffs
    
    # Local search from best
    if best_coeffs is not None:
        start_cl = [(i, c) for i, c in enumerate(best_coeffs) if c != 0]
        local_search(start_cl, time_budget=8.0)
        if best_count == 12:
            return best_coeffs
    
    # =====================================================================
    # Phase 2: Try other exponent structures with dense collision patterns
    # =====================================================================
    
    # Arithmetic progressions with small steps create many collisions
    ap_sets = []
    for step in [1, 2, 3]:
        if 11 * step <= 40:
            ap_sets.append([i * step for i in range(12)])
    
    # Geometric-ish structures
    ge_sets = []
    for base in [2, 3]:
        exps = [0]
        for i in range(1, 12):
            e = exps[-1] + i
            if e <= 40:
                exps.append(e)
        if len(exps) == 12:
            ge_sets.append(exps)
    
    # Random dense sets
    for _ in range(5):
        exps = sorted(random.sample(range(41), 12))
        ap_sets.append(exps)
    
    for exps in ap_sets + ge_sets:
        if time.time() > deadline - 30:
            break
        
        # Quick brute force with just +/-1 signs
        t0 = time.time()
        for mask in range(4096):
            if time.time() - t0 > 3.0:
                break
            cl = []
            for i in range(12):
                sign = 1 if (mask >> i) & 1 else -1
                cl.append((exps[i], sign))
            c = eval_square_fast(cl)
            if c < best_count:
                try_update(c, cl)
        
        if best_count == 12:
            return best_coeffs
        
        # Local search from best for this structure
        if best_coeffs is not None:
            start_cl = [(i, c) for i, c in enumerate(best_coeffs) if c != 0]
            local_search(start_cl, time_budget=3.0)
            if best_count == 12:
                return best_coeffs
    
    # =====================================================================
    # Phase 3: Simulated Annealing from multiple random starts
    # =====================================================================
    
    vals = [-4, -3, -2, -1, 1, 2, 3, 4]
    
    attempts = 0
    while time.time() < deadline - 15:
        attempts += 1
        
        r = random.random()
        if r < 0.4:
            exps = list(range(12))
        elif r < 0.6:
            step = random.randint(1, 3)
            if 11 * step > 40:
                step = 3
            exps = [i * step for i in range(12)]
        else:
            exps = sorted(random.sample(range(41), 12))
        
        cl = [(exps[i], random.choice(vals)) for i in range(12)]
        
        simulated_annealing(cl, time_budget=3.0)
        if best_count == 12:
            return best_coeffs
    
    # =====================================================================
    # Phase 4: Final polish on best found
    # =====================================================================
    
    if best_coeffs is not None:
        start_cl = [(i, c) for i, c in enumerate(best_coeffs) if c != 0]
        local_search(start_cl, time_budget=10.0)
        if best_count == 12:
            return best_coeffs
        
        simulated_annealing(start_cl, time_budget=10.0)
        if best_count == 12:
            return best_coeffs

    # Fallback
    if best_coeffs is None:
        best_coeffs = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
        
        nz = [i for i, c in enumerate(best_coeffs) if c != 0]
        if len(nz) != 12:
            best_coeffs = [0]*41
            for i in range(12):
                best_coeffs[i] = 1
        else:
            # Extend to proper length
            max_e = max(i for i, c in enumerate(best_coeffs) if c != 0)
            if max_e + 1 < 12:
                best_coeffs = best_coeffs + [0]*(12 - max_e - 1)
            # Trim trailing zeros
            while len(best_coeffs) > 1 and best_coeffs[-1] == 0:
                best_coeffs.pop()
            if len(best_coeffs) < 12:
                best_coeffs = best_coeffs + [0]*(12 - len(best_coeffs))
                # Ensure last nonzero
                while best_coeffs[-1] == 0 and len(best_coeffs) > 1:
                    best_coeffs.pop()
                if len(best_coeffs) < 12:
                    best_coeffs = [0]*12
                    for i in range(12):
                        best_coeffs[i] = 1

    return best_coeffs
