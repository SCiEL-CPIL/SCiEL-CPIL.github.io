import time
import random
from itertools import combinations

def construct():
    """
    Returns a list of distinct integers in the range 2..40 such that
    all 2^k subset products are distinct. Maximizes k.
    
    Uses efficient integer-encoded exponent vectors, targeted search strategies,
    and incremental collision checking to maximize set size.
    """
    start_time = time.time()
    deadline = start_time + 57  # Safety margin before 60s limit
    
    # Prime factorization for each number 2-40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Precompute exponent vectors as integers for fast comparison
    # Encode each exponent vector as a single integer using base-8 encoding
    # (max exponent for any prime in 2..40 is 5 for 2^5=32)
    def encode_exp_vec(exp):
        """Encode exponent tuple as a single integer."""
        code = 0
        for i in range(num_primes):
            code = code * 8 + exp[i]
        return code
    
    def add_encoded(a, b):
        """Add two encoded exponent vectors."""
        result = 0
        # We need to decode, add, and re-encode
        # But for speed, we'll work with tuples internally and encode for storage
        return a + b  # Placeholder - we'll use tuples
    
    # Precompute exponent vectors as tuples for all numbers 2-40
    exp_vectors = {}
    for n in range(2, 41):
        exponents = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                exponents[i] += 1
                temp //= p
        exp_vectors[n] = tuple(exponents)
    
    zero = (0,) * num_primes
    
    def _is_valid_set(nums):
        """Check if all subset products of nums are distinct using exponent vectors."""
        if not nums:
            return True
        sums = {zero}
        for n in nums:
            v = exp_vectors[n]
            new_sums = set()
            for s in sums:
                ns = tuple(s[i] + v[i] for i in range(num_primes))
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums |= new_sums
        return True
    
    def _can_add_to_sums(sums, candidate):
        """Check if candidate can be added to set with given sum set. Returns new sums or None."""
        v = exp_vectors[candidate]
        new_sums = set()
        for s in sums:
            ns = tuple(s[i] + v[i] for i in range(num_primes))
            if ns in sums:
                return None
            new_sums.add(ns)
        return new_sums
    
    def _greedy_build(order, deadline):
        """Greedy build in given order, tracking sums incrementally."""
        selected = []
        sums = {zero}
        for n in order:
            if time.time() > deadline:
                break
            new_sums = _can_add_to_sums(sums, n)
            if new_sums is not None:
                sums |= new_sums
                selected.append(n)
        return selected, sums
    
    def _try_extend(current_set, current_sums, deadline, target_size):
        """Try to extend current_set by adding elements one by one."""
        selected = list(current_set)
        sums = set(current_sums)
        remaining = [x for x in range(2, 41) if x not in selected]
        
        # Try adding elements in various orders
        for _ in range(3):  # Multiple passes
            if time.time() > deadline:
                break
            for n in remaining[:]:
                if len(selected) >= target_size:
                    break
                if time.time() > deadline:
                    break
                new_sums = _can_add_to_sums(sums, n)
                if new_sums is not None:
                    sums |= new_sums
                    selected.append(n)
                    remaining.remove(n)
        
        return selected, sums
    
    def _local_search_swap(current, sums, deadline):
        """Try swap-based local search to improve the set. More efficient version."""
        best = list(current)
        best_size = len(best)
        best_sums = set(sums)
        candidates = [x for x in range(2, 41) if x not in best]
        
        if not candidates:
            return best, best_sums
        
        improved = True
        max_iterations = 50
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            if time.time() > deadline:
                break
            
            for i in range(len(best)):
                if time.time() > deadline:
                    break
                for c in candidates:
                    if time.time() > deadline:
                        break
                    
                    # Try swapping best[i] with c
                    trial = list(best)
                    trial[i] = c
                    
                    # Check validity of trial set efficiently
                    trial_sums = {zero}
                    valid = True
                    for n in trial:
                        v = exp_vectors[n]
                        new_prods = set()
                        for s in trial_sums:
                            ns = tuple(s[j] + v[j] for j in range(num_primes))
                            if ns in trial_sums:
                                valid = False
                                break
                            new_prods.add(ns)
                        if not valid:
                            break
                        trial_sums |= new_prods
                    
                    if valid:
                        # Try to add more elements greedily
                        trial2 = list(trial)
                        tp = set(trial_sums)
                        for c2 in range(2, 41):
                            if time.time() > deadline:
                                break
                            if c2 in trial2:
                                continue
                            new_sums = _can_add_to_sums(tp, c2)
                            if new_sums is not None:
                                tp |= new_sums
                                trial2.append(c2)
                        
                        if len(trial2) > best_size:
                            best = trial2
                            best_size = len(best)
                            best_sums = tp
                            candidates = [x for x in range(2, 41) if x not in best]
                            improved = True
                            break
                if improved:
                    break
        return best, best_sums
    
    # Start with a good deterministic set as baseline
    base_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 4, 16, 9, 25]
    if _is_valid_set(base_set):
        best = list(base_set)
        # Compute sums for best
        best_sums = {zero}
        for n in best:
            new_sums = _can_add_to_sums(best_sums, n)
            if new_sums is not None:
                best_sums |= new_sums
    else:
        best = []
        best_sums = {zero}
    
    best_size = len(best)
    all_nums = list(range(2, 41))
    
    # Targeted search: try to find sets of size 17 and 18
    # Strategy 1: Start with primes (12 of them) and add composites
    prime_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    composite_candidates = [4, 8, 16, 32, 9, 27, 25, 6, 10, 14, 15, 21, 22, 26, 33, 34, 35, 38, 39, 40]
    
    # Try different orderings of composites to add to primes
    composite_orders = [
        composite_candidates,  # natural order
        sorted(composite_candidates, key=lambda x: sum(exp_vectors[x])),  # fewest prime factors
        sorted(composite_candidates, key=lambda x: -x),  # descending
        sorted(composite_candidates, key=lambda x: max(exp_vectors[x])),  # max exponent
    ]
    
    for comp_order in composite_orders:
        if time.time() > deadline:
            break
        order = prime_set + comp_order
        current_set, current_sums = _greedy_build(order, deadline)
        if len(current_set) > best_size:
            best = current_set
            best_size = len(best)
            best_sums = current_sums
    
    # Strategy 2: Randomized restarts with greedy + local search
    random.seed(42)
    restart_count = 0
    max_restarts = 200
    
    while time.time() < deadline and restart_count < max_restarts:
        restart_count += 1
        random_order = list(all_nums)
        random.shuffle(random_order)
        current_set, current_sums = _greedy_build(random_order, deadline)
        
        # If we got a good set, try to improve with local search
        if len(current_set) >= best_size - 1:
            improved_set, improved_sums = _local_search_swap(current_set, current_sums, deadline)
            if len(improved_set) > best_size:
                best = improved_set
                best_size = len(best)
                best_sums = improved_sums
    
    # Strategy 3: Direct search for size 17/18 using targeted combinations
    # Try to find which composites can be added to the 12 primes
    if best_size < 17:
        # Try all subsets of composites of size 5-6 to add to primes
        for comp_size in [6, 5, 4]:
            if time.time() > deadline:
                break
            for comp_subset in combinations(composite_candidates, comp_size):
                if time.time() > deadline:
                    break
                trial = prime_set + list(comp_subset)
                if _is_valid_set(trial):
                    if len(trial) > best_size:
                        best = trial
                        best_size = len(best)
                        best_sums = {zero}
                        for n in best:
                            new_sums = _can_add_to_sums(best_sums, n)
                            if new_sums is not None:
                                best_sums |= new_sums
    
    # Final verification
    if not _is_valid_set(best):
        # Fallback to known good solution
        best = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 4, 16, 9, 25]
    
    return [int(x) for x in best]
