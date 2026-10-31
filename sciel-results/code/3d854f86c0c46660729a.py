"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    best_score = 0
    best_set = [0, 1, 3]  # seed fallback
    
    def compute_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        n = len(U)
        
        # For small n, use set-based approach (faster for n < 15)
        if n < 15:
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
            # FFT-based computation for larger n
            max_sum = 2 * max_u
            max_diff = max_u
            
            # For sumset
            len_sum = 2 * max_u + 1
            fft_size = 1
            while fft_size < 2 * len_sum:
                fft_size *= 2
            
            f_sum = np.zeros(fft_size, dtype=np.float64)
            for u in U:
                f_sum[u] = 1.0
            
            F_sum = np.fft.fft(f_sum)
            conv_sum = np.fft.ifft(F_sum * F_sum).real
            n_sum = int(np.sum(conv_sum > 0.5))
            
            # For difference set: U - U = {a - b : a, b in U}
            len_diff = 2 * max_u + 1
            fft_size_d = 1
            while fft_size_d < 2 * len_diff:
                fft_size_d *= 2
            
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
    
    def is_golomb_ruler(U):
        """Check if all pairwise differences are distinct (up to sign)."""
        U = sorted(U)
        n = len(U)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = U[j] - U[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
    
    def soft_max(values, temperature=0.1):
        """Differentiable soft-max approximation for counting distinct values."""
        if len(values) == 0:
            return 0.0
        max_val = max(values)
        # Use soft-max to approximate the number of distinct values
        # Each value contributes exp(-|v - nearest_other| / temperature)
        # This peaks when values are well-separated
        contributions = []
        for v in values:
            min_dist = min(abs(v - w) for w in values if w != v) if len(values) > 1 else 100
            contributions.append(math.exp(-min_dist / temperature))
        return sum(contributions)
    
    def continuous_score(positions, temperature=0.5):
        """
        Compute a smooth surrogate score for continuous positions.
        positions: list of real numbers (first should be 0)
        Returns: smooth approximation of 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)
        """
        if len(positions) < 2:
            return -1.0
        
        # Compute all pairwise sums and differences
        sums = []
        diffs = []
        for i in range(len(positions)):
            for j in range(len(positions)):
                sums.append(positions[i] + positions[j])
                diffs.append(positions[i] - positions[j])
        
        # Soft-count distinct values using histogram binning
        # Bin into intervals of width 1, count bins with significant mass
        def soft_count(values, bin_width=1.0):
            if not values:
                return 0.0
            # Sort and use gap-based counting
            sorted_vals = sorted(values)
            count = 1  # at least one bin
            for k in range(1, len(sorted_vals)):
                gap = sorted_vals[k] - sorted_vals[k-1]
                # Soft transition: if gap > bin_width, new bin; else partial
                # Use sigmoid to smooth
                t = gap / bin_width
                count += 1.0 / (1.0 + math.exp(-10 * (t - 1)))
            return count
        
        n_sum_soft = soft_count(sums)
        n_diff_soft = soft_count(diffs)
        
        max_u = max(positions)
        base = 2 * max_u + 1
        if base <= 1:
            return 0.0
        
        ratio = n_diff_soft / max(n_sum_soft, 1e-10)
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score
    
    def continuous_gradient(positions, epsilon=0.01):
        """
        Compute numerical gradient of continuous_score with respect to positions.
        positions[0] is fixed at 0, so gradient[0] = 0.
        """
        n = len(positions)
        grad = np.zeros(n)
        base_score = continuous_score(positions)
        
        for i in range(1, n):  # skip index 0 (fixed at 0)
            pos_plus = list(positions)
            pos_plus[i] += epsilon
            score_plus = continuous_score(pos_plus)
            
            pos_minus = list(positions)
            pos_minus[i] -= epsilon
            score_minus = continuous_score(pos_minus)
            
            grad[i] = (score_plus - score_minus) / (2 * epsilon)
        
        return grad
    
    def round_and_validate(positions):
        """Round continuous positions to integers, ensure 0 is present, valid set."""
        rounded = [round(p) for p in positions]
        # Ensure 0 is present
        if 0 not in rounded:
            rounded[0] = 0
        # Remove duplicates, sort
        rounded = sorted(set(rounded))
        # Ensure non-negative
        rounded = [x for x in rounded if x >= 0]
        if not rounded or 0 not in rounded:
            return None
        if len(rounded) < 3:
            return None
        return rounded
    
    # Pre-compute seed
    best_score = compute_score([0, 1, 3])
    
    # Phase 1: Try known good structured sets (Golomb rulers, powers, etc.)
    # Known optimal Golomb rulers
    golomb_rulers = [
        [0, 1, 3],           # size 3, max 3
        [0, 1, 4, 6],        # size 4, max 6
        [0, 1, 4, 10, 12],   # size 5, max 12
        [0, 1, 4, 10, 12, 17], # size 6, max 17
        [0, 1, 4, 10, 18, 23, 25], # size 7, max 25
        [0, 1, 4, 10, 18, 23, 25, 34], # size 8, max 34
        [0, 1, 4, 10, 18, 23, 25, 34, 42], # size 9, max 42
        [0, 1, 4, 10, 18, 23, 25, 34, 42, 52], # size 10, max 52
    ]
    
    for gr in golomb_rulers:
        s = compute_score(gr)
        if s > best_score:
            best_score = s
            best_set = list(gr)
    
    # Also try scaled/shifted variants of Golomb rulers
    for gr in golomb_rulers[:6]:
        for scale in [2, 3, 4, 5]:
            scaled = [x * scale for x in gr]
            s = compute_score(scaled)
            if s > best_score:
                best_score = s
                best_set = list(scaled)
    
    # Phase 2: Continuous optimization using gradient ascent on soft surrogate
    if time.time() < deadline - 20:
        # Try multiple initializations with different sizes
        for size in range(4, 12):
            if time.time() > deadline - 10:
                break
            
            # Multiple random initializations
            for init_iter in range(5):
                if time.time() > deadline - 10:
                    break
                
                # Initialize with geometric-like spacing
                random.seed(42 + size * 100 + init_iter)
                if init_iter == 0:
                    # Geometric
                    ratio = random.uniform(1.8, 2.5)
                    positions = [0.0]
                    val = 1.0
                    for _ in range(size - 1):
                        val *= ratio
                        positions.append(val)
                elif init_iter == 1:
                    # Golomb-ruler-like with jitter
                    base_gr = golomb_rulers[min(size - 3, len(golomb_rulers) - 1)]
                    if len(base_gr) >= size:
                        positions = [float(x) + random.uniform(-0.3, 0.3) for x in base_gr[:size]]
                        positions[0] = 0.0
                    else:
                        # Extend
                        positions = [float(x) for x in base_gr]
                        while len(positions) < size:
                            positions.append(positions[-1] * random.uniform(1.5, 2.0))
                        positions[0] = 0.0
                elif init_iter == 2:
                    # Random sparse
                    max_u = random.uniform(20, 100)
                    positions = [0.0] + sorted(random.uniform(1, max_u) for _ in range(size - 1))
                elif init_iter == 3:
                    # Powers of 2 with jitter
                    positions = [float(2 ** i) + random.uniform(-0.2, 0.2) for i in range(size)]
                    positions[0] = 0.0
                else:
                    # Fibonacci-like
                    positions = [0.0, 1.0]
                    while len(positions) < size:
                        positions.append(positions[-1] + positions[-2])
                    positions = [p + random.uniform(-0.1, 0.1) for p in positions]
                    positions[0] = 0.0
                
                # Gradient ascent
                current_pos = positions
                current_cs = continuous_score(current_pos)
                
                for step in range(50):
                    if time.time() > deadline - 10:
                        break
                    
                    grad = continuous_gradient(current_pos, epsilon=0.05)
                    
                    # Adaptive learning rate
                    lr = 0.1 / (1 + step * 0.1)
                    
                    # Gradient ascent
                    new_pos = list(current_pos)
                    new_pos[0] = 0.0  # Keep 0 fixed
                    for i in range(1, len(new_pos)):
                        new_pos[i] += lr * grad[i]
                        # Keep non-negative
                        new_pos[i] = max(0.1, new_pos[i])
                    
                    new_cs = continuous_score(new_pos)
                    
                    if new_cs > current_cs - 1e-6:
                        current_pos = new_pos
                        current_cs = new_cs
                    else:
                        # Try smaller step
                        lr *= 0.5
                        new_pos = list(current_pos)
                        new_pos[0] = 0.0
                        for i in range(1, len(new_pos)):
                            new_pos[i] += lr * grad[i]
                            new_pos[i] = max(0.1, new_pos[i])
                        new_cs = continuous_score(new_pos)
                        if new_cs > current_cs - 1e-6:
                            current_pos = new_pos
                            current_cs = new_cs
                        else:
                            break
                
                # Round and validate
                rounded = round_and_validate(current_pos)
                if rounded is not None:
                    s = compute_score(rounded)
                    if s > best_score:
                        best_score = s
                        best_set = list(rounded)
    
    # Phase 3: Local search around best set found
    if time.time() < deadline - 10:
        current = list(best_set)
        current_score = best_score
        no_improve_count = 0
        max_no_improve = 1000
        
        while no_improve_count < max_no_improve and time.time() < deadline - 5:
            improved = False
            
            # Try adding an element
            if len(current) < 40:
                max_cur = max(current)
                candidates_to_add = set()
                for el in current:
                    for delta in range(-3, 4):
                        new_el = el + delta
                        if new_el > 0 and new_el not in current:
                            candidates_to_add.add(new_el)
                for new_el in range(max_cur + 1, max_cur + 20):
                    candidates_to_add.add(new_el)
                if max_cur > 0:
                    for mult in [2, 3]:
                        new_el = max_cur * mult
                        if new_el not in current and new_el < 10000:
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
            
            # Try replacing an element
            if len(current) >= 3:
                max_cur = max(current)
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    old_el = current[i]
                    for new_el in range(max(1, old_el - 5), min(max_cur + 10, old_el + 6)):
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
            
            # Try removing an element (if set is large)
            if len(current) > 5 and not improved:
                for i in range(len(current) - 1, 0, -1):
                    cand = current[:i] + current[i+1:]
                    s = compute_score(cand)
                    if s > current_score + 1e-15:
                        current = list(cand)
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                        improved = True
                        break
            
            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0
    
    # Phase 4: Random restarts with various structured initializations
    if time.time() < deadline - 5:
        random.seed(12345)
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            
            init_type = random.randint(0, 3)
            
            if init_type == 0:
                size = random.randint(4, 12)
                max_u = random.randint(10, 100)
                gr = [0, 1]
                used_diffs = {1}
                for _ in range(size - 2):
                    best_next = -1
                    for candidate in range(2, max_u + 1):
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
                cand = gr
            elif init_type == 1:
                size = random.randint(4, 15)
                ratio = random.uniform(1.5, 3.0)
                cand = [0]
                val = 1
                for _ in range(size - 1):
                    cand.append(int(val))
                    val *= ratio
            elif init_type == 2:
                size = random.randint(5, 25)
                max_u = random.randint(20, 200)
                cand = [0] + sorted(random.sample(range(1, max_u + 1), min(size - 1, max_u)))
            else:
                size = random.randint(4, 12)
                base = random.choice([2, 3, 4, 5])
                cand = [base ** i for i in range(size)]
            
            if len(cand) >= 3 and 0 in cand:
                cand = sorted(set(cand))
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                elif s > best_score - 0.01:
                    current = list(cand)
                    current_score = s
                    for _ in range(20):
                        if time.time() > deadline - 5:
                            break
                        max_cur = max(current)
                        candidates_to_add = set()
                        for el in current:
                            for delta in range(-2, 3):
                                new_el = el + delta
                                if new_el > 0 and new_el not in current:
                                    candidates_to_add.add(new_el)
                        for new_el in range(max_cur + 1, max_cur + 5):
                            candidates_to_add.add(new_el)
                        
                        improved_inner = False
                        for new_el in sorted(candidates_to_add):
                            cand2 = sorted(current + [new_el])
                            s2 = compute_score(cand2)
                            if s2 > current_score + 1e-15:
                                current = list(cand2)
                                current_score = s2
                                if s2 > best_score:
                                    best_score = s2
                                    best_set = list(cand2)
                                improved_inner = True
                                break
                        if not improved_inner:
                            break
    
    return list(best_set)
# EVOLVE-BLOCK-END
