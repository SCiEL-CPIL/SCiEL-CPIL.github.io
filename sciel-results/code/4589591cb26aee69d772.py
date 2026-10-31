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
        def soft_count(values, bin_width=1.0):
            if not values:
                return 0.0
            sorted_vals = sorted(values)
            count = 1
            for k in range(1, len(sorted_vals)):
                gap = sorted_vals[k] - sorted_vals[k-1]
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
        
        for i in range(1, n):
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
        if 0 not in rounded:
            rounded[0] = 0
        rounded = sorted(set(rounded))
        rounded = [x for x in rounded if x >= 0]
        if not rounded or 0 not in rounded:
            return None
        if len(rounded) < 3:
            return None
        return rounded
    
    # Pre-compute seed
    best_score = compute_score([0, 1, 3])
    
    # Phase 1: Try known good structured sets (Golomb rulers, powers, etc.)
    golomb_rulers = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 25, 34],
        [0, 1, 4, 10, 18, 23, 25, 34, 42],
        [0, 1, 4, 10, 18, 23, 25, 34, 42, 52],
    ]
    
    for gr in golomb_rulers:
        s = compute_score(gr)
        if s > best_score:
            best_score = s
            best_set = list(gr)
    
    for gr in golomb_rulers[:6]:
        for scale in [2, 3, 4, 5]:
            scaled = [x * scale for x in gr]
            s = compute_score(scaled)
            if s > best_score:
                best_score = s
                best_set = list(scaled)
    
    # Phase 2: Dual-view optimization - work with difference sets (Sidon structures)
    # Strategy: For a given difference multiset D, find U such that U-U has maximal spread
    # but U+U is minimized (high additive structure within U)
    
    def find_best_U_from_diffs(target_diffs, max_size=15, max_val=500):
        """
        Given a target difference set (positive differences), find U containing 0
        such that U-U approximates target_diffs and |U+U| is minimized.
        Uses greedy construction with preference for arithmetic-progression-like structure.
        """
        if not target_diffs:
            return None
        
        target_diffs = sorted(set(target_diffs))
        if not target_diffs:
            return None
        
        best_U = None
        best_score = -1
        
        # Try multiple starting points and growth strategies
        for start_val in [1, 2, 3, 4, 5]:
            if start_val not in target_diffs and start_val > max(target_diffs):
                continue
            
            U = [0]
            used_diffs = set()
            current_max = 0
            
            # Greedily add elements to match target differences
            remaining_targets = list(target_diffs)
            
            for step in range(max_size - 1):
                if time.time() > deadline - 10:
                    break
                
                best_candidate = None
                best_candidate_score = -1
                
                # Candidate positions: near existing elements + target diffs
                candidates = set()
                for u in U:
                    for d in target_diffs[:20]:  # Only consider smaller diffs for efficiency
                        cand = u + d
                        if 0 < cand <= max_val and cand not in U:
                            candidates.add(cand)
                        cand2 = u - d
                        if 0 < cand2 <= max_val and cand2 not in U:
                            candidates.add(cand2)
                
                # Also try arithmetic progression extensions
                if len(U) >= 2:
                    last_diff = U[-1] - U[-2]
                    for mult in [1, 2, 3]:
                        cand = U[-1] + last_diff * mult
                        if 0 < cand <= max_val and cand not in U:
                            candidates.add(cand)
                
                # Try geometric extensions
                if len(U) >= 2 and U[-1] > 0:
                    for ratio in [1.5, 2.0, 2.5]:
                        cand = int(U[-1] * ratio)
                        if 0 < cand <= max_val and cand not in U:
                            candidates.add(cand)
                
                for cand in sorted(candidates):
                    new_U = sorted(U + [cand])
                    # Quick score check
                    n = len(new_U)
                    if n < 3:
                        continue
                    
                    # Compute approximate score
                    sumset = set()
                    diffset = set()
                    for i in range(n):
                        a = new_U[i]
                        for j in range(n):
                            b = new_U[j]
                            sumset.add(a + b)
                            diffset.add(a - b)
                    
                    n_sum = len(sumset)
                    n_diff = len(diffset)
                    max_u = max(new_U)
                    base = 2 * max_u + 1
                    if base <= 1:
                        continue
                    ratio = n_diff / max(n_sum, 1)
                    if ratio <= 1:
                        continue
                    s = 1 + math.log(ratio) / math.log(base)
                    
                    # Bonus for matching target differences
                    new_diffs = set()
                    for i in range(n):
                        for j in range(i + 1, n):
                            new_diffs.add(new_U[j] - new_U[i])
                    match_count = len(new_diffs & set(target_diffs))
                    s_adjusted = s + 0.01 * match_count / len(target_diffs)
                    
                    if s_adjusted > best_candidate_score:
                        best_candidate_score = s_adjusted
                        best_candidate = new_U
                        best_candidate_raw_score = s
                
                if best_candidate is None:
                    break
                
                # Check if this is a good improvement
                if best_candidate_raw_score > best_score + 0.001 or step == max_size - 2:
                    U = best_candidate
                    best_score = best_candidate_raw_score
                    best_U = list(U)
                
                # Check for local optimum
                if best_candidate_raw_score < best_score - 0.01:
                    break
            
            if best_U and best_score > -1:
                s = compute_score(best_U)
                if s > best_score:
                    best_score = s
                    best_U = list(best_U)
        
        return best_U if best_U else None
    
    def generate_sidon_like_diffs(size, max_diff=100):
        """Generate candidate difference sets with Sidon-like properties."""
        diffs = set()
        # Start with small differences
        for d in range(1, min(size + 5, max_diff + 1)):
            if random.random() < 0.7:
                diffs.add(d)
        
        # Add some larger gaps for spread
        for _ in range(size // 2):
            d = random.randint(size, max_diff)
            diffs.add(d)
        
        # Ensure Sidon property: no four distinct a,b,c,d with a+b=c+d
        # (approximate: limit multiplicity of differences)
        return sorted(diffs)
    
    # Phase 2a: Try dual-view optimization with various difference structures
    if time.time() < deadline - 30:
        for trial in range(30):
            if time.time() > deadline - 15:
                break
            
            random.seed(999 + trial)
            size = random.randint(4, 12)
            max_diff = random.randint(50, 300)
            
            # Generate a Sidon-like difference set
            target_diffs = generate_sidon_like_diffs(size, max_diff)
            
            # Try to find optimal U for this difference structure
            U_candidate = find_best_U_from_diffs(target_diffs, max_size=size + 2, max_val=max_diff + 50)
            
            if U_candidate is not None and len(U_candidate) >= 3:
                s = compute_score(U_candidate)
                if s > best_score:
                    best_score = s
                    best_set = list(U_candidate)
    
    # Phase 2b: Direct optimization with preference for AP-like structures
    # Key insight: minimize |U+U| by making U as "additively structured" as possible
    # while maximizing |U-U| through large gaps
    
    def optimize_with_structure(max_size=15, max_val=500, num_trials=20):
        """
        Optimize by starting with structured sets (APs, geometric) and perturbing
        to increase |U-U|/|U+U| ratio.
        """
        local_best = None
        local_best_score = -1
        
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                break
            
            random.seed(7777 + trial)
            
            # Choose a base structure
            struct_type = trial % 4
            
            if struct_type == 0:
                # Arithmetic progression with one outlier
                n = random.randint(4, 10)
                d = random.randint(2, 10)
                start = random.randint(1, 5)
                U = [start * i for i in range(n)]
                # Add an outlier to increase |U-U|
                outlier = random.randint(max(U) + 10, max_val)
                U.append(outlier)
                U = sorted(set(U))
                
            elif struct_type == 1:
                # Two interleaved APs
                n1 = random.randint(2, 6)
                n2 = random.randint(2, 6)
                d1 = random.randint(2, 8)
                d2 = random.randint(d1 + 1, d1 + 10)
                U1 = [d1 * i for i in range(n1)]
                U2 = [d2 * i for i in range(n2)]
                U = sorted(set(U1 + U2))
                
            elif struct_type == 2:
                # Geometric with small perturbations
                n = random.randint(4, 10)
                ratio = random.uniform(1.8, 2.5)
                U = [0]
                val = 1
                for i in range(n - 1):
                    U.append(int(val + random.uniform(-0.5, 0.5)))
                    val *= ratio
                U = sorted(set(U))
                
            else:
                # Mixed: AP base + geometric tail
                n_ap = random.randint(3, 6)
                n_geo = random.randint(2, 4)
                d = random.randint(2, 6)
                U_ap = [d * i for i in range(n_ap)]
                U_geo = [d * (n_ap - 1) * (2 ** i) for i in range(n_geo)]
                U = sorted(set(U_ap + U_geo))
            
            if len(U) < 3 or 0 not in U:
                continue
            
            # Local search: try to improve by adjusting elements
            current = sorted(set(U))
            current_score = compute_score(current)
            
            for step in range(50):
                if time.time() > deadline - 10:
                    break
                
                improved = False
                
                # Try moving one element
                if len(current) > 3:
                    for i in range(1, len(current)):
                        old = current[i]
                        for delta in range(-3, 4):
                            new_val = old + delta
                            if new_val <= 0 or new_val in current:
                                continue
                            cand = list(current)
                            cand[i] = new_val
                            cand = sorted(set(cand))
                            if len(cand) != len(current) or 0 not in cand:
                                continue
                            s = compute_score(cand)
                            if s > current_score + 1e-10:
                                current = cand
                                current_score = s
                                improved = True
                                break
                        if improved:
                            break
                
                # Try adding an element
                if len(current) < max_size and not improved:
                    max_cur = max(current)
                    for new_el in range(max_cur + 1, min(max_cur + 30, max_val + 1)):
                        cand = sorted(current + [new_el])
                        s = compute_score(cand)
                        if s > current_score + 1e-10:
                            current = cand
                            current_score = s
                            improved = True
                            break
                    # Try inserting between existing elements
                    for i in range(1, len(current)):
                        gap_start = current[i-1] + 1
                        gap_end = current[i] - 1
                        if gap_end > gap_start:
                            for new_el in range(gap_start, min(gap_end + 1, gap_start + 5)):
                                if new_el in current:
                                    continue
                                cand = sorted(current + [new_el])
                                s = compute_score(cand)
                                if s > current_score + 1e-10:
                                    current = cand
                                    current_score = s
                                    improved = True
                                    break
                        if improved:
                            break
                
                # Try removing an element
                if len(current) > 4 and not improved:
                    for i in range(len(current) - 1, 0, -1):
                        cand = current[:i] + current[i+1:]
                        s = compute_score(cand)
                        if s > current_score + 1e-10:
                            current = cand
                            current_score = s
                            improved = True
                            break
                
                if not improved:
                    break
            
            if current_score > local_best_score:
                local_best_score = current_score
                local_best = list(current)
        
        return local_best, local_best_score
    
    if time.time() < deadline - 20:
        U_struct, score_struct = optimize_with_structure(max_size=15, max_val=500, num_trials=25)
        if U_struct is not None and score_struct > best_score:
            best_score = score_struct
            best_set = list(U_struct)
    
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
            
            # Try removing an element
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
