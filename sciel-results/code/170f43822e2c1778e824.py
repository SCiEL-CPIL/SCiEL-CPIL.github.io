import time
import sys
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 12
    max_val = 10000
    max_sum = 20000

    # Helper: Check if x is P-smooth
    def is_smooth(x, primes):
        for p in primes:
            while x % p == 0:
                x //= p
        return x == 1

    # Helper: Generate P-smooth numbers up to limit
    def gen_smooth(limit, primes):
        smooth = set()
        def gen(current, idx):
            if current > limit:
                return
            smooth.add(current)
            if idx < len(primes):
                p = primes[idx]
                next_val = current
                while next_val <= limit:
                    next_val *= p
                    gen(next_val, idx)
                gen(current, idx + 1)
        gen(1, 0)
        return sorted(smooth)

    # Strategy 1: Find 12-clique in graph where edges are P-smooth sums
    # Try P = {2, 3, 5, 7} first (4 primes)
    # If fail, try P = {2, 3, 5} (3 primes)
    
    prime_sets_to_try = [
        [2, 3, 5, 7],
        [2, 3, 5],
        [2, 3]
    ]
    
    for primes in prime_sets_to_try:
        if time.time() > deadline - 10:
            break
            
        smooth_numbers = gen_smooth(max_sum, primes)
        smooth_set = set(smooth_numbers)
        
        # Build adjacency bitsets
        # adj[i] is a bitmask of j such that i+j is smooth
        adj = [0] * (max_val + 1)
        
        # For each smooth sum S, add edges (i, S-i)
        for S in smooth_numbers:
            if S > max_sum:
                continue
            # i + j = S, i != j
            # i ranges from max(1, S-max_val) to min(S-1, max_val)
            start_i = max(1, S - max_val)
            end_i = min(S - 1, max_val)
            
            for i in range(start_i, end_i + 1):
                j = S - i
                if i == j:
                    continue
                # Set bit j in adj[i] and bit i in adj[j]
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
        # Find clique of size n
        # Use backtracking with bitsets
        # Pre-compute suffix masks for ordering
        # To save time, we only need masks for vertices we might visit
        # But we can just use a helper function to get mask of > v
        
        def get_suffix_mask(v):
            # Bits set for indices > v
            # We want bits v+1 .. max_val
            # (1 << (v+1)) - 1 gives bits 0..v
            # ALL_BITS ^ ((1 << (v+1)) - 1) gives bits v+1 .. max_val
            # ALL_BITS = (1 << (max_val + 1)) - 1
            return ((1 << (max_val + 1)) - 1) ^ ((1 << (v + 1)) - 1)
        
        ALL_BITS = (1 << (max_val + 1)) - 1
        
        def find_clique(candidates, current_clique, depth):
            if time.time() > deadline - 5:
                return None
            if len(current_clique) == n:
                return current_clique[:]
            
            if candidates == 0:
                return None
            
            if candidates.bit_count() < n - len(current_clique):
                return None
            
            # Heuristic: Pick vertex with highest degree in candidates
            best_v = -1
            best_deg = -1
            temp_cands = candidates
            # Limit iteration to avoid slow scans if candidates is huge
            # But candidates shrinks rapidly in clique search
            while temp_cands:
                lsb = temp_cands & -temp_cands
                v = lsb.bit_length() - 1
                deg = (candidates & adj[v]).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp_cands ^= lsb
                
            if best_v == -1:
                return None
                
            # Try all candidates? No, standard backtracking tries one, fails, tries next.
            # But picking the "best" one first is a heuristic.
            # To ensure completeness, we should iterate all candidates.
            # However, for performance, we might just try the best one and backtrack if it fails?
            # No, that's not correct. We must try all.
            
            temp_cands = candidates
            while temp_cands:
                if time.time() > deadline - 5:
                    return None
                lsb = temp_cands & -temp_cands
                v = lsb.bit_length() - 1
                temp_cands ^= lsb
                
                current_clique.append(v)
                new_candidates = candidates & adj[v] & get_suffix_mask(v)
                
                result = find_clique(new_candidates, current_clique, depth + 1)
                if result:
                    return result
                
                current_clique.pop()
                
            return None
        
        # Initial candidates: vertices with degree >= n-1
        initial_candidates = 0
        for i in range(1, max_val + 1):
            if adj[i].bit_count() >= n - 1:
                initial_candidates |= (1 << i)
        
        clique = find_clique(initial_candidates, [], 0)
        
        if clique:
            # Verify
            valid = True
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    s = clique[i] + clique[j]
                    if not is_smooth(s, primes):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return sorted(clique)

    # Fallback: Local search
    # Start with multiples of 30
    arr = [30 * i for i in range(1, 13)]
    
    def count_primes(a):
        primes = set()
        for i in range(len(a)):
            for j in range(i+1, len(a)):
                s = a[i] + a[j]
                x = s
                d = 2
                while d * d <= x:
                    if x % d == 0:
                        primes.add(d)
                        while x % d == 0:
                            x //= d
                    d += 1
                if x > 1:
                    primes.add(x)
        return len(primes)
    
    best_count = count_primes(arr)
    best_arr = arr[:]
    
    random.seed(42)
    while time.time() < deadline - 2:
        # Local search step
        improved = False
        for pos in range(12):
            old = arr[pos]
            # Try random replacements
            for _ in range(50):
                new_val = random.randint(1, 10000)
                if new_val in arr:
                    continue
                arr[pos] = new_val
                new_count = count_primes(arr)
                if new_count < best_count:
                    best_count = new_count
                    best_arr = arr[:]
                    improved = True
                else:
                    arr[pos] = old
            if improved:
                break
        if not improved:
            # Restart with perturbation
            for pos in range(12):
                arr[pos] = random.randint(1, 10000)
            # Ensure distinct
            for i in range(12):
                while arr[i] in arr[:i]:
                    arr[i] = random.randint(1, 10000)
            best_count = count_primes(arr)
            best_arr = arr[:]
            
    return best_arr
