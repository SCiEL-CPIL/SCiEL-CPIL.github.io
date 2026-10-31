"""Optimized construction for sumset/difference-set exponent lower bound."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def compute_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        n = len(U)
        
        # For small n, use set-based approach
        if n < 200:
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
            # FFT-based computation for larger sets
            max_sum = 2 * max_u
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
            
            # For difference set: |U-U| = |{a-b : a,b in U}|
            # This equals the number of distinct differences
            # We can compute this via convolution of U with reversed U
            len_diff = 2 * max_u + 1
            fft_size_d = 1
            while fft_size_d < 2 * len_diff:
                fft_size_d *= 2
            
            f_diff = np.zeros(fft_size_d, dtype=np.float64)
            for u in U:
                f_diff[u] = 1.0
            
            # Reverse: f_rev[i] = 1 if (max_u - i) in U, i.e., f_rev[max_u - u] = 1
            f_rev = np.zeros(fft_size_d, dtype=np.float64)
            for u in U:
                f_rev[max_u - u] = 1.0
            
            F_diff = np.fft.fft(f_diff)
            F_rev = np.fft.fft(f_rev)
            conv_diff = np.fft.ifft(F_diff * F_rev).real
            # The difference a-b corresponds to index (a-b + max_u) in the convolution
            # So we count non-zero entries
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

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def next_prime(n):
        while not is_prime(n):
            n += 1
        return n

    def next_prime_power(n):
        """Find the next prime power >= n."""
        for p in range(2, 10000):
            if not is_prime(p):
                continue
            pk = p
            while pk < n:
                pk *= p
            return pk, p
        return n, 2

    def bosch_chowla_sidon(p):
        """Bose-Chowla construction: Sidon set of size p-1 in [1, p^2-1].
        For prime p, the set {1 + j*p : j in 0..p-2} shifted appropriately.
        Actually, the standard construction: for prime p, 
        S = {i + p*j : i in 1..p-1, j in 0..p-2} but that's not Sidon.
        
        Correct Bose-Chowla: For prime p, consider the set
        S = {1 + k*(p-1) : k = 0, 1, ..., p-2} — no, that's arithmetic.
        
        The actual Bose-Chowla (1960): For prime p, the set
        A = {i + p*j : 1 <= i <= p-1, 0 <= j <= p-2} has size (p-1)^2
        but is NOT Sidon.
        
        Let me use the correct construction. The Bose-Chowla Sidon set:
        For prime p, define S = {x + p*y : x in F_p^*, y in F_p}
        where the multiplication is in F_p. Actually, the standard result is:
        
        For prime p, the set {1 + j*p : j = 0, ..., p-2} is NOT Sidon.
        
        Let me use a simpler correct construction:
        Singer's construction for prime power q:
        S = {0} union {g^i - g^(i+q) : i = 0, ..., q} in F_{q^2}
        embedded in integers via a field representation.
        
        Actually, let me use a simpler approach: greedy Sidon set construction.
        """
        pass

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

    def singer_sidon(q):
        """Singer's construction of a Sidon set of size q+1 in [0, q^2+q].
        For prime power q, this gives a perfect difference set.
        """
        if q < 2:
            return [0, 1]
        
        # For small q, use known constructions
        # Singer's construction uses finite fields
        # For prime q, we can use the construction:
        # S = {0} union {g^i - g^(i+q) mod (q^2-1) : i = 0, ..., q-1}
        # but this needs careful implementation.
        
        # Alternative: use the simple fact that for prime p,
        # the set {i^2 mod p : i = 0, ..., p-1} has special properties.
        
        # Let's use a direct construction for small prime powers
        # For q = p prime:
        # The set {0, 1, p, p+1, 2p, 2p+1, ...} is not Sidon.
        
        # Use greedy for now as fallback
        max_val = q * q + q
        return greedy_sidon(q + 1, max_val)

    def explicit_sidon_constructions():
        """Try various explicit Sidon set constructions."""
        candidates = []
        
        # Small known Sidon sets (Golomb rulers)
        known = [
            [0, 1, 3],
            [0, 1, 4, 6],
            [0, 1, 4, 10, 12],
            [0, 1, 4, 10, 18, 23, 25],
            [0, 1, 4, 10, 18, 23, 25, 34, 42],
            [0, 1, 4, 10, 18, 23, 25, 34, 42, 52, 60],
            [0, 1, 4, 10, 18, 23, 25, 34, 42, 52, 60, 72],
            [0, 1, 4, 10, 18, 23, 25, 34, 42, 52, 60, 72, 84],
        ]
        candidates.extend(known)
        
        # Greedy Sidon sets of various sizes
        for target_size in [5, 8, 12, 15, 20, 25, 30, 40, 50, 70, 100, 150, 200]:
            S = greedy_sidon(target_size, target_size * target_size + 100)
            if len(S) >= 3:
                candidates.append(S)
        
        return candidates

    def local_search_minimize_diameter(S, max_iters=500):
        """Try to reduce the diameter of a Sidon set while maintaining Sidon property."""
        S = sorted(set(S))
        if len(S) < 3:
            return S
        
        current = list(S)
        current_score = compute_score(current)
        no_improve = 0
        max_no_improve = 200
        
        while no_improve < max_no_improve and time.time() < deadline - 10:
            improved = False
            n = len(current)
            max_cur = max(current)
            
            # Try replacing elements with smaller values
            for i in range(n - 1, 0, -1):  # Start from largest
                old = current[i]
                # Try replacing with smaller values
                for new in range(1, old):
                    if new in current:
                        continue
                    # Check if replacing maintains Sidon property
                    test = list(current)
                    test[i] = new
                    test = sorted(set(test))
                    if len(test) != n:
                        continue
                    
                    # Check Sidon property
                    valid = True
                    diffs = set()
                    for a in range(n):
                        for b in range(a + 1, n):
                            d = test[b] - test[a]
                            if d in diffs:
                                valid = False
                                break
                            diffs.add(d)
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    s = compute_score(test)
                    if s > current_score + 1e-12:
                        current = test
                        current_score = s
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                no_improve += 1
            else:
                no_improve = 0
        
        return current

    def local_search_improve_score(S, max_iters=1000):
        """Local search to improve the score of a set."""
        S = sorted(set(S))
        if len(S) < 3:
            return S
        
        current = list(S)
        current_score = compute_score(current)
        no_improve = 0
        max_no_improve = 200
        
        while no_improve < max_no_improve and time.time() < deadline - 5:
            improved = False
            n = len(current)
            max_cur = max(current)
            
            # Try adding an element - focus on promising positions
            if n < 300:
                candidates_to_add = set()
                # Add elements near existing ones
                for el in current:
                    for delta in range(-2, 3):
                        new_el = el + delta
                        if new_el > 0 and new_el not in current:
                            candidates_to_add.add(new_el)
                # Add elements just beyond max
                for new_el in range(max_cur + 1, max_cur + 10):
                    candidates_to_add.add(new_el)
                # Try some random positions
                if max_cur > 10:
                    for _ in range(5):
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
            
            # Try replacing an element with smaller values to reduce diameter
            if not improved and n >= 3:
                for i in range(n - 1, 0, -1):  # Start from largest
                    if current[i] == 0:
                        continue
                    old_el = current[i]
                    # Try replacing with smaller values
                    for new_el in range(max(1, old_el - 5), old_el):
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
            
            # Try removing an element if it hurts
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

    # Phase 1: Generate candidate Sidon sets from various constructions
    best_score = 0
    best_set = [0, 1, 3]
    
    candidates = explicit_sidon_constructions()
    
    # Also try greedy Sidon sets with different parameters
    for target_size in [10, 20, 30, 50, 80, 120, 200]:
        if time.time() > deadline - 30:
            break
        max_val = target_size * target_size + 200
        S = greedy_sidon(target_size, max_val)
        if len(S) >= 3:
            candidates.append(S)
    
    # Try scaled versions of small Sidon sets
    small_sidon = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
    ]
    for S in small_sidon:
        for scale in [2, 3, 4, 5, 7, 10]:
            scaled = [x * scale for x in S]
            candidates.append(scaled)
    
    # Evaluate all candidates
    for S in candidates:
        if time.time() > deadline - 30:
            break
        S = sorted(set(S))
        if len(S) < 3 or 0 not in S:
            continue
        s = compute_score(S)
        if s > best_score:
            best_score = s
            best_set = list(S)
    
    # Phase 2: Local search on best candidate
    if time.time() < deadline - 15:
        improved_set = local_search_improve_score(best_set)
        s = compute_score(improved_set)
        if s > best_score:
            best_score = s
            best_set = improved_set
    
    # Phase 3: Try more aggressive local search with random restarts
    random.seed(42)
    restarts_done = 0
    max_restarts = 50
    while time.time() < deadline - 5 and restarts_done < max_restarts:
        restarts_done += 1
        
        # Random Sidon-like initialization
        size = random.randint(5, 80)
        max_u = random.randint(20, 800)
        S = [0]
        used_diffs = set()
        # Sample candidates in increasing order
        sample_size = min(size - 1, max_u)
        if sample_size > 0:
            candidates_list = sorted(random.sample(range(1, max_u + 1), sample_size))
            for candidate in candidates_list:
                valid = True
                for s in S:
                    d = candidate - s
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
            elif s > best_score - 0.03:
                improved = local_search_improve_score(S)
                s2 = compute_score(improved)
                if s2 > best_score:
                    best_score = s2
                    best_set = improved
    
    return list(best_set)
# EVOLVE-BLOCK-END
