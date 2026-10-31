"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    best_score = 0
    best_set = [0, 1, 3]  # seed fallback
    
    def compute_score_fast(U):
        """Compute score using FFT for sumset/diffset sizes."""
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        n = len(U)
        
        # For small sets, use direct computation (faster overhead)
        if n <= 12:
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
            # Use FFT-based approach for larger sets
            # Sumset: convolve indicator of U with itself
            size = 2 * max_u + 1
            # Next power of 2 for FFT
            fft_size = 1
            while fft_size < size:
                fft_size *= 2
            
            indicator = np.zeros(fft_size)
            for x in U:
                indicator[x] = 1.0
            
            # Sumset via convolution
            fft_ind = np.fft.fft(indicator)
            conv = np.fft.ifft(fft_ind * fft_ind)
            n_sum = int(np.sum(np.abs(conv) > 0.5))
            
            # Diffset: convolve U with reversed U
            # U-U = {a-b : a,b in U} = {a + (-b) : a in U, b in U}
            # Shift: let V = {x + max_u : x in U}, then V-V = U-U + 2*max_u... 
            # Actually, U-U ranges from -max_u to max_u
            # Use: diffset size = |U-U| = number of distinct (a-b) for a,b in U
            # This equals the number of nonzero coefficients in the autocorrelation
            # autocorr[k] = sum_x indicator[x] * indicator[x-k] for k in [-max_u, max_u]
            # |U-U| = number of k where autocorr[k] > 0
            fft_ind_conj = np.fft.fft(np.conj(indicator))
            # Actually for real signals, autocorrelation = ifft(conj(fft) * fft)
            auto = np.fft.ifft(np.conj(fft_ind) * fft_ind)
            # auto has values at positions 0..fft_size-1, where position k corresponds to lag k
            # and position fft_size-k corresponds to lag -k
            n_diff = int(np.sum(np.abs(auto) > 0.5))
        
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
        """Fallback direct computation."""
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        n = len(U)
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
    
    # Phase 1: Try known good structured sets
    golomb_rulers = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 25, 34],
        [0, 1, 4, 10, 18, 23, 25, 34, 42],
        [0, 1, 4, 10, 18, 23, 25, 34, 42, 52],
        [0, 1, 4, 10, 18, 23, 25, 34, 42, 52, 60],
        [0, 1, 4, 10, 18, 23, 25, 34, 42, 52, 60, 72],
    ]
    
    for gr in golomb_rulers:
        s = compute_score(gr)
        if s > best_score:
            best_score = s
            best_set = list(gr)
    
    # Also try scaled variants
    for gr in golomb_rulers[:8]:
        for scale in [2, 3]:
            scaled = [x * scale for x in gr]
            s = compute_score(scaled)
            if s > best_score:
                best_score = s
                best_set = list(scaled)
    
    # Phase 2: Simulated annealing with larger sets
    if time.time() < deadline - 10:
        random.seed(42)
        
        def greedy_golomb(size, max_val):
            """Greedy construction of a Golomb-like ruler."""
            gr = [0, 1]
            used_diffs = {1}
            for _ in range(size - 2):
                best_next = -1
                for candidate in range(2, max_val + 1):
                    valid = True
                    for existing in gr:
                        d = candidate - existing
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        best_next = candidate
                        break
                if best_next == -1:
                    break
                gr.append(best_next)
                for existing in gr[:-1]:
                    used_diffs.add(gr[-1] - existing)
            return gr
        
        # Start from best known
        current = list(best_set)
        current_score = best_score
        
        T = 0.05  # initial temperature
        T_min = 0.001
        cooling = 0.995
        max_iters = 5000
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Generate a neighbor
            move_type = random.random()
            new_set = list(current)
            
            if move_type < 0.3:
                # Add an element
                if len(new_set) < 20:
                    max_cur = max(new_set)
                    # Try various positions
                    if random.random() < 0.5:
                        new_el = random.randint(1, max_cur + 20)
                    else:
                        base_el = random.choice(new_set)
                        new_el = base_el + random.randint(-5, 15)
                        new_el = max(1, new_el)
                    if new_el not in new_set:
                        new_set.append(new_el)
                    else:
                        continue
                else:
                    continue
            elif move_type < 0.6:
                # Remove an element (not 0)
                if len(new_set) > 3:
                    removable = [x for x in new_set if x != 0]
                    if removable:
                        new_set.remove(random.choice(removable))
                    else:
                        continue
                else:
                    continue
            elif move_type < 0.8:
                # Replace an element
                if len(new_set) >= 3:
                    idx = random.randint(1, len(new_set) - 1)  # not index 0
                    old_el = new_set[idx]
                    max_cur = max(new_set)
                    new_el = old_el + random.randint(-10, 20)
                    new_el = max(1, new_el)
                    if new_el not in new_set:
                        new_set[idx] = new_el
                        new_set = sorted(set(new_set))
                    else:
                        continue
                else:
                    continue
            else:
                # Swap two non-zero elements
                if len(new_set) >= 4:
                    idx1 = random.randint(1, len(new_set) - 1)
                    idx2 = random.randint(1, len(new_set) - 1)
                    if idx1 != idx2:
                        new_set[idx1], new_set[idx2] = new_set[idx2], new_set[idx1]
                        new_set = sorted(set(new_set))
                        if len(new_set) != len(current):
                            continue
                    else:
                        continue
                else:
                    continue
            
            if len(new_set) < 3 or 0 not in new_set:
                continue
            
            new_score = compute_score(new_set)
            if new_score == -1:
                continue
            
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / max(T, 1e-10)):
                current = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current)
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        # Phase 3: Local refinement with wider moves
        if time.time() < deadline - 5:
            current = list(best_set)
            current_score = best_score
            no_improve = 0
            max_no_improve = 300
            
            while no_improve < max_no_improve and time.time() < deadline - 3:
                improved = False
                
                # Try adding elements with wider range
                if len(current) < 18:
                    max_cur = max(current)
                    candidates_to_add = set()
                    # Wider range: up to max_cur + 30
                    for new_el in range(1, max_cur + 31):
                        if new_el not in current:
                            candidates_to_add.add(new_el)
                    # Also try elements near existing ones
                    for el in current:
                        for delta in range(-5, 10):
                            new_el = el + delta
                            if new_el > 0 and new_el not in current:
                                candidates_to_add.add(new_el)
                    
                    for new_el in sorted(candidates_to_add):
                        cand = sorted(current + [new_el])
                        s = compute_score(cand)
                        if s > current_score + 1e-15:
                            current = list(cand)
                            current_score = s
                            if s > best_score:
                                best_score = s
                                best_set = list(cand)
                            improved = True
                            break
                
                # Try replacing with wider range
                if len(current) >= 3:
                    max_cur = max(current)
                    for i in range(len(current)):
                        if current[i] == 0:
                            continue
                        old_el = current[i]
                        for new_el in range(max(1, old_el - 10), min(max_cur + 25, old_el + 25)):
                            if new_el == old_el or new_el in current:
                                continue
                            cand = list(current)
                            cand[i] = new_el
                            cand = sorted(set(cand))
                            if 0 not in cand or len(cand) != len(current):
                                continue
                            s = compute_score(cand)
                            if s > current_score + 1e-15:
                                current = list(cand)
                                current_score = s
                                if s > best_score:
                                    best_score = s
                                    best_set = list(cand)
                                improved = True
                                break
                        if improved:
                            break
                
                if not improved:
                    no_improve += 1
                else:
                    no_improve = 0
    
    return list(best_set)
# EVOLVE-BLOCK-END
