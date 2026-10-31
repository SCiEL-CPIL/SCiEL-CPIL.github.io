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
    
    def compute_score_bitset(U):
        """Compute score using Python integers as bitsets for exact arithmetic."""
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        n = len(U)
        
        # Build bitset for U
        mask = 0
        for u in U:
            mask |= (1 << u)
        
        # Sumset: U+U = {a+b : a,b in U}
        # Use FFT for large sets, bitset for small
        if max_u <= 2048:
            # Bitset convolution via repeated shift-and-OR
            sumset_mask = 0
            for u in U:
                sumset_mask |= (mask << u)
            n_sum = sumset_mask.bit_count()
        else:
            # FFT fallback
            fft_size = 1
            while fft_size < 4 * max_u + 4:
                fft_size *= 2
            f = np.zeros(fft_size, dtype=np.float64)
            for u in U:
                f[u] = 1.0
            F = np.fft.fft(f)
            conv = np.fft.ifft(F * F).real
            n_sum = int(np.sum(conv > 0.5))
        
        # Difference set: U-U = {a-b : a,b in U}
        # This is symmetric: if d is in U-U, so is -d
        # U-U range: [-max_u, max_u], so 2*max_u+1 possible values
        # We can compute |U-U| = 2 * |{d > 0 : d in U-U}| + 1 (for d=0)
        if max_u <= 2048:
            diffset_mask = 0
            for u in U:
                # a - u for all a in U
                # This gives values in range [-u, max_u - u]
                # Shift by u to make non-negative, then combine
                shifted = (mask >> 0)  # mask itself
                # Actually: U - u = {a - u : a in U}
                # If we shift mask right by u, we get {a-u : a in U, a >= u}
                # For a < u, a-u < 0
                # Let's just compute all differences directly
                pass
            
            # Simpler: compute positive differences
            # |U-U| = 1 + 2 * |{d > 0 : exists a,b in U with a-b = d}|
            pos_diffs = set()
            for i in range(n):
                for j in range(i):
                    d = U[i] - U[j]
                    if d > 0:
                        pos_diffs.add(d)
            n_diff = 1 + 2 * len(pos_diffs)
        else:
            # FFT for large sets
            fft_size_d = 1
            while fft_size_d < 4 * max_u + 4:
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
    
    def marginal_insert(U, x, sumset, diffset):
        """
        Compute the change in |U+U| and |U-U| when inserting x into U.
        Returns (delta_n_sum, delta_n_diff, new_max_u)
        """
        n = len(U)
        new_max_u = max(max(U), x)
        
        # New sums: x + u for all u in U
        new_sums = set()
        for u in U:
            new_sums.add(x + u)
        
        # New differences: x - u and u - x for all u in U
        new_diffs = set()
        for u in U:
            new_diffs.add(x - u)
            new_diffs.add(u - x)
        
        delta_sum = len(new_sums - sumset)
        delta_diff = len(new_diffs - diffset)
        
        return delta_sum, delta_diff, new_max_u
    
    def marginal_delete(U, x, sumset, diffset):
        """
        Compute the change in |U+U| and |U-U| when deleting x from U.
        Returns (delta_n_sum, delta_n_diff)
        """
        n = len(U)
        
        # Sums that only appear with x: x+u where u != x and no other pair gives x+u
        # This is complex; for simplicity, recompute
        # But we can estimate: sums involving x are x+u for u in U\{x}
        # A sum s = x+u is lost only if no other pair (a,b) with a,b != x gives s
        
        # For efficiency in local search, we'll just recompute for small sets
        U_new = [u for u in U if u != x]
        if not U_new or 0 not in U_new:
            return None, None
        
        new_sumset = set()
        new_diffset = set()
        for i in range(len(U_new)):
            a = U_new[i]
            for j in range(len(U_new)):
                b = U_new[j]
                new_sumset.add(a + b)
                new_diffset.add(a - b)
        
        delta_sum = len(new_sumset) - len(sumset)
        delta_diff = len(new_diffset) - len(diffset)
        
        return delta_sum, delta_diff
    
    def bitset_local_search(U, max_iterations=5000):
        """
        Local search using bitset representation for fast marginal evaluations.
        """
        start_ls = time.time()
        U = sorted(set(U))
        if not U or 0 not in U:
            return U, -1
        
        # Build initial sumset and diffset
        n = len(U)
        sumset = set()
        diffset = set()
        for i in range(n):
            a = U[i]
            for j in range(n):
                b = U[j]
                sumset.add(a + b)
                diffset.add(a - b)
        
        max_u = max(U)
        n_sum = len(sumset)
        n_diff = len(diffset)
        base = 2 * max_u + 1
        if base <= 1:
            return U, 0
        ratio = n_diff / n_sum
        if ratio <= 1:
            current_score = 1.0
        else:
            current_score = 1 + math.log(ratio) / math.log(base)
        
        best_score_ls = current_score
        best_U_ls = list(U)
        
        no_improve = 0
        max_no_improve = 50
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            if no_improve > max_no_improve:
                break
            
            improved = False
            max_u = max(U)
            
            # Try inserting new elements
            candidates = set()
            for el in U:
                for delta in range(-2, 3):
                    new_el = el + delta
                    if new_el > 0 and new_el not in U:
                        candidates.add(new_el)
            for new_el in range(max_u + 1, max_u + 15):
                candidates.add(new_el)
            if max_u > 0:
                candidates.add(max_u * 2)
                candidates.add(max_u * 3)
            
            # Also try some random candidates
            for _ in range(5):
                candidates.add(random.randint(1, max_u * 2 + 20))
            
            best_delta = -1
            best_new_el = None
            best_new_score = -1
            
            for new_el in sorted(candidates):
                if new_el in U:
                    continue
                
                delta_sum, delta_diff, new_max_u = marginal_insert(U, new_el, sumset, diffset)
                new_n_sum = n_sum + delta_sum
                new_n_diff = n_diff + delta_diff
                
                if new_n_sum <= 0:
                    continue
                
                new_base = 2 * new_max_u + 1
                if new_base <= 1:
                    continue
                
                new_ratio = new_n_diff / new_n_sum
                if new_ratio <= 1:
                    new_score = 1.0
                else:
                    new_score = 1 + math.log(new_ratio) / math.log(new_base)
                
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_new_el = new_el
                    best_delta = (delta_sum, delta_diff, new_max_u)
            
            # Try the best insertion
            if best_new_el is not None and best_new_score > current_score + 1e-12:
                U.append(best_new_el)
                U.sort()
                n += 1
                
                # Update sumset and diffset
                for u in U[:-1]:
                    sumset.add(best_new_el + u)
                    diffset.add(best_new_el - u)
                    diffset.add(u - best_new_el)
                
                n_sum += best_delta[0]
                n_diff += best_delta[1]
                max_u = best_delta[2]
                current_score = best_new_score
                
                if current_score > best_score_ls:
                    best_score_ls = current_score
                    best_U_ls = list(U)
                
                improved = True
                no_improve = 0
                continue
            
            # Try deleting elements
            if len(U) > 3:
                best_del_el = None
                best_del_score = -1
                best_del_deltas = None
                
                for i in range(len(U) - 1, 0, -1):  # Don't delete 0
                    x = U[i]
                    # Quick estimate: check if x contributes unique sums/diffs
                    # Count sums/diffs that only involve x
                    unique_sums = 0
                    unique_diffs = 0
                    
                    # Sums involving x
                    for u in U:
                        if u == x:
                            continue
                        s = x + u
                        # Check if s can be formed without x
                        # s = a + b where a,b in U\{x}
                        # For efficiency, just check if s - x is in U (trivial) 
                        # and if there's another representation
                        found_alt = False
                        for a in U:
                            if a == x:
                                continue
                            b = s - a
                            if b in U and b != x:
                                found_alt = True
                                break
                        if not found_alt:
                            unique_sums += 1
                    
                    # This is O(n^2) per element, too slow for large n
                    # Skip detailed deletion check for now
                    
                    pass
                
                # Simpler: try deleting the largest element
                if len(U) > 3:
                    x = U[-1]
                    U_new = U[:-1]
                    new_sumset = set()
                    new_diffset = set()
                    for i2 in range(len(U_new)):
                        a = U_new[i2]
                        for j2 in range(len(U_new)):
                            b = U_new[j2]
                            new_sumset.add(a + b)
                            new_diffset.add(a - b)
                    
                    new_n_sum = len(new_sumset)
                    new_n_diff = len(new_diffset)
                    new_max_u = max(U_new)
                    new_base = 2 * new_max_u + 1
                    
                    if new_base > 1 and new_n_sum > 0:
                        new_ratio = new_n_diff / new_n_sum
                        if new_ratio <= 1:
                            new_score = 1.0
                        else:
                            new_score = 1 + math.log(new_ratio) / math.log(new_base)
                        
                        if new_score > current_score + 1e-12:
                            U = U_new
                            sumset = new_sumset
                            diffset = new_diffset
                            n_sum = new_n_sum
                            n_diff = new_n_diff
                            max_u = new_max_u
                            n = len(U)
                            current_score = new_score
                            
                            if current_score > best_score_ls:
                                best_score_ls = current_score
                                best_U_ls = list(U)
                            
                            improved = True
                            no_improve = 0
                            continue
            
            # Try replacing an element
            if len(U) >= 3:
                max_u = max(U)
                for i in range(len(U)):
                    if U[i] == 0:
                        continue
                    old_el = U[i]
                    for new_el in range(max(1, old_el - 5), min(max_u + 10, old_el + 6)):
                        if new_el == old_el or new_el in U:
                            continue
                        
                        # Compute new sumset and diffset
                        U_new = list(U)
                        U_new[i] = new_el
                        U_new = sorted(set(U_new))
                        if 0 not in U_new or len(U_new) != len(U):
                            continue
                        
                        new_sumset = set()
                        new_diffset = set()
                        for a in U_new:
                            for b in U_new:
                                new_sumset.add(a + b)
                                new_diffset.add(a - b)
                        
                        new_n_sum = len(new_sumset)
                        new_n_diff = len(new_diffset)
                        new_max_u = max(U_new)
                        new_base = 2 * new_max_u + 1
                        
                        if new_base <= 1 or new_n_sum <= 0:
                            continue
                        
                        new_ratio = new_n_diff / new_n_sum
                        if new_ratio <= 1:
                            new_score = 1.0
                        else:
                            new_score = 1 + math.log(new_ratio) / math.log(new_base)
                        
                        if new_score > current_score + 1e-12:
                            U = U_new
                            sumset = new_sumset
                            diffset = new_diffset
                            n_sum = new_n_sum
                            n_diff = new_n_diff
                            max_u = new_max_u
                            current_score = new_score
                            
                            if current_score > best_score_ls:
                                best_score_ls = current_score
                                best_U_ls = list(U)
                            
                            improved = True
                            break
                    if improved:
                        no_improve = 0
                        break
            
            if not improved:
                no_improve += 1
        
        return best_U_ls, best_score_ls
    
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
    
    # Try powers of 2, 3, etc.
    for base in [2, 3, 4, 5]:
        for size in range(3, 12):
            cand = [base ** i for i in range(size)]
            s = compute_score(cand)
            if s > best_score:
                best_score = s
                best_set = list(cand)
    
    # Phase 2: Bitset-based local search from best structured set
    if time.time() < deadline - 15:
        U_ls, score_ls = bitset_local_search(list(best_set), max_iterations=3000)
        if score_ls > best_score:
            best_score = score_ls
            best_set = list(U_ls)
    
    # Phase 3: More local search from different starting points
    if time.time() < deadline - 10:
        # Try local search from Golomb rulers
        for gr in golomb_rulers[:5]:
            if time.time() > deadline - 5:
                break
            U_ls, score_ls = bitset_local_search(list(gr), max_iterations=2000)
            if score_ls > best_score:
                best_score = score_ls
                best_set = list(U_ls)
    
    # Phase 4: Random restarts with local search
    if time.time() < deadline - 5:
        random.seed(12345)
        for _ in range(30):
            if time.time() > deadline - 3:
                break
            
            init_type = random.randint(0, 3)
            
            if init_type == 0:
                size = random.randint(4, 10)
                max_u = random.randint(10, 80)
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
                size = random.randint(4, 12)
                ratio = random.uniform(1.5, 2.5)
                cand = [0]
                val = 1
                for _ in range(size - 1):
                    cand.append(int(val))
                    val *= ratio
            elif init_type == 2:
                size = random.randint(5, 15)
                max_u = random.randint(20, 150)
                cand = [0] + sorted(random.sample(range(1, max_u + 1), min(size - 1, max_u)))
            else:
                size = random.randint(4, 10)
                base = random.choice([2, 3, 4])
                cand = [base ** i for i in range(size)]
            
            if len(cand) >= 3 and 0 in cand:
                cand = sorted(set(cand))
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                
                # Local search from this point
                if time.time() < deadline - 3:
                    U_ls, score_ls = bitset_local_search(list(cand), max_iterations=1000)
                    if score_ls > best_score:
                        best_score = score_ls
                        best_set = list(U_ls)
    
    # Final: one more local search from the best set found
    if time.time() < deadline - 2 and len(best_set) >= 3:
        U_final, score_final = bitset_local_search(list(best_set), max_iterations=2000)
        if score_final > best_score:
            best_score = score_final
            best_set = list(U_final)
    
    return list(best_set)
# EVOLVE-BLOCK-END
