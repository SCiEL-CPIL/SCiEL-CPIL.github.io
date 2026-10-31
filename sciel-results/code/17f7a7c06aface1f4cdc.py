"""Optimized construction for sumset/difference-set exponent lower bound."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    def compute_score_fast(U):
        """Fast score computation for a set U."""
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        n = len(U)
        
        if n < 150:
            sumset = set()
            diffset = set()
            for i in range(n):
                a = U[i]
                for j in range(n):
                    b = U[j]
                    sumset.add(a + b)
                    diffset.add(a - b)
            n_sum = len(sumset)
            n_diff = len(diffset)
        else:
            max_sum = 2 * max_u
            fft_size = 1
            while fft_size < 2 * (2 * max_u + 1):
                fft_size *= 2
            
            f_sum = np.zeros(fft_size, dtype=np.float64)
            for u in U:
                f_sum[u] = 1.0
            
            F_sum = np.fft.fft(f_sum)
            conv_sum = np.fft.ifft(F_sum * F_sum).real
            n_sum = int(np.sum(conv_sum > 0.5))
            
            fft_size_d = fft_size
            f_diff = np.zeros(fft_size_d, dtype=np.float64)
            for u in U:
                f_diff[u] = 1.0
            f_rev = np.zeros(fft_size_d, dtype=np.float64)
            for u in U:
                f_rev[max_u - u] = 1.0
            
            F_diff = np.fft.fft(f_diff)
            F_rev = np.fft.fft(f_rev)
            conv_diff = np.fft.ifft(F_diff * F_rev).real
            n_diff = int(np.sum(conv_diff > 0.5))
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def compute_score(U):
        return compute_score_fast(U)

    def greedy_sidon(max_size, max_val):
        """Greedy construction of a Sidon set."""
        S = [0]
        used_diffs = set()
        for candidate in range(1, max_val + 1):
            valid = True
            for s in S:
                d = candidate - s
                if d in used_diffs:
                    valid = False
                    break
                used_diffs.add(d)
            if valid:
                S.append(candidate)
                if len(S) >= max_size:
                    break
        return S

    def spectral_candidate(L, n_harmonics, seed_val):
        """Generate a candidate set from a trigonometric polynomial level set."""
        rng = random.Random(seed_val)
        # Choose random coefficients for cos(2*pi*k*x/L)
        coeffs = [rng.gauss(0, 1) for _ in range(n_harmonics)]
        
        # Evaluate f(x) = sum c_k * cos(2*pi*k*x/L) for x in 0..L-1
        best_set = None
        best_score = -1
        
        # Try different thresholds to extract level sets
        for threshold_frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
            f_vals = np.zeros(L)
            for k, c in enumerate(coeffs, 1):
                f_vals += c * np.cos(2 * np.pi * k * np.arange(L) / L)
            
            # Threshold to get a set
            thr = np.percentile(f_vals, (1 - threshold_frac) * 100)
            S = sorted([x for x in range(L) if f_vals[x] >= thr])
            
            if not S or 0 not in S:
                # Try to include 0 by shifting
                if S:
                    S = sorted([x - min(S) for x in S])
                else:
                    continue
            
            if len(S) < 3 or len(S) > 200:
                continue
            
            s = compute_score_fast(S)
            if s > best_score:
                best_score = s
                best_set = list(S)
        
        return best_set, best_score

    def local_search_improve_score(S, max_iters=500):
        """Local search to improve the score of a set."""
        S = sorted(set(S))
        if len(S) < 3:
            return S
        
        current = list(S)
        current_score = compute_score(current)
        no_improve = 0
        max_no_improve = 100
        
        while no_improve < max_no_improve and time.time() < deadline - 10:
            improved = False
            n = len(current)
            max_cur = max(current)
            
            # Try adding an element
            if n < 200:
                candidates_to_add = set()
                for el in current:
                    for delta in range(-3, 4):
                        new_el = el + delta
                        if new_el > 0 and new_el not in current:
                            candidates_to_add.add(new_el)
                for new_el in range(max_cur + 1, max_cur + 15):
                    candidates_to_add.add(new_el)
                if max_cur > 10:
                    for _ in range(10):
                        new_el = random.randint(1, max_cur)
                        if new_el not in current:
                            candidates_to_add.add(new_el)
                
                for new_el in sorted(candidates_to_add):
                    cand = sorted(current + [new_el])
                    s = compute_score(cand)
                    if s > current_score + 1e-12:
                        current = cand
                        current_score = s
                        improved = True
                        break
            
            # Try replacing an element with smaller values
            if not improved and n >= 3:
                for i in range(n - 1, 0, -1):
                    old_el = current[i]
                    for new_el in range(max(1, old_el - 8), old_el):
                        if new_el in current:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand or len(cand) != n:
                            continue
                        s = compute_score(cand)
                        if s > current_score + 1e-12:
                            current = cand
                            current_score = s
                            improved = True
                            break
                    if improved:
                        break
            
            # Try removing an element
            if not improved and n > 5:
                for i in range(n - 1, 0, -1):
                    cand = current[:i] + current[i + 1:]
                    s = compute_score(cand)
                    if s > current_score + 1e-12:
                        current = cand
                        current_score = s
                        improved = True
                        break
            
            if not improved:
                no_improve += 1
            else:
                no_improve = 0
        
        return current

    # Phase 1: Generate candidates from multiple sources
    best_score = 0
    best_set = [0, 1, 3]
    
    # Greedy Sidon sets
    for target_size in [5, 8, 12, 15, 20, 30, 40, 50, 60, 80]:
        if time.time() > deadline - 60:
            break
        max_val = target_size * target_size + 200
        S = greedy_sidon(target_size, max_val)
        if len(S) >= 3:
            s = compute_score(S)
            if s > best_score:
                best_score = s
                best_set = list(S)
    
    # Scaled small Sidon sets
    small_sidon = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
    ]
    for S in small_sidon:
        for scale in [2, 3, 4, 5, 7, 10, 15, 20]:
            if time.time() > deadline - 60:
                break
            scaled = [x * scale for x in S]
            s = compute_score(scaled)
            if s > best_score:
                best_score = s
                best_set = list(scaled)
    
    # Phase 2: Spectral candidates
    if time.time() < deadline - 40:
        for L in [30, 50, 80, 120, 160, 200, 300]:
            if time.time() > deadline - 30:
                break
            for seed_val in range(10):
                if time.time() > deadline - 25:
                    break
                S, s = spectral_candidate(L, 5, seed_val)
                if S and s > best_score:
                    best_score = s
                    best_set = list(S)
    
    # Phase 3: Local search on best
    if time.time() < deadline - 20:
        improved_set = local_search_improve_score(best_set)
        s = compute_score(improved_set)
        if s > best_score:
            best_score = s
            best_set = improved_set
    
    # Phase 4: Random restarts with local search
    random.seed(42)
    restarts_done = 0
    max_restarts = 30
    while time.time() < deadline - 5 and restarts_done < max_restarts:
        restarts_done += 1
        
        size = random.randint(5, 60)
        max_u = random.randint(20, 500)
        S = [0]
        used_diffs = set()
        sample_size = min(size - 1, max_u)
        if sample_size > 0:
            candidates_list = sorted(random.sample(range(1, max_u + 1), sample_size))
            for candidate in candidates_list:
                valid = True
                for sv in S:
                    d = candidate - sv
                    if d in used_diffs:
                        valid = False
                        break
                    used_diffs.add(d)
                if valid:
                    S.append(candidate)
        
        if len(S) >= 3:
            S = sorted(S)
            s = compute_score(S)
            if s > best_score:
                best_score = s
                best_set = list(S)
            elif s > best_score - 0.05 and time.time() < deadline - 10:
                improved = local_search_improve_score(S)
                s2 = compute_score(improved)
                if s2 > best_score:
                    best_score = s2
                    best_set = improved
    
    return list(best_set)
# EVOLVE-BLOCK-END
