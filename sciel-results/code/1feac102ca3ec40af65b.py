import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    nprimes = len(primes)
    
    # Compute exponent vectors for 2..40
    def get_vector(n):
        vec = [0] * nprimes
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)
    
    numbers = list(range(2, 41))
    vectors = [get_vector(n) for n in numbers]
    
    # Initial solution: all primes (indices where number is prime)
    # Prime numbers in 2..40: 2,3,5,7,11,13,17,19,23,29,31,37
    prime_indices = [i for i, n in enumerate(numbers) if n in primes]
    
    best = list(prime_indices)
    best_size = len(best)
    
    # Function to check if adding vector v to set of vectors S keeps all subset products distinct
    # i.e., v is not a difference of two subset sums of S
    def can_add(v, subset_sums):
        """Check if v can be added. subset_sums is a set of all subset sums of current vectors."""
        # Need: v + s != s' for all s, s' in subset_sums
        # i.e., v != s' - s for all s, s' in subset_sums
        # i.e., v + s != s' for all s in subset_sums, i.e., v + s not in subset_sums
        for s in subset_sums:
            if (s[0]+v[0], s[1]+v[1], s[2]+v[2], s[3]+v[3], s[4]+v[4], s[5]+v[5],
                s[6]+v[6], s[7]+v[7], s[8]+v[8], s[9]+v[9], s[10]+v[10], s[11]+v[11]) in subset_sums:
                return False
        return True
    
    def compute_subset_sums(vecs):
        """Compute all subset sums of a list of vectors."""
        sums = {tuple([0]*nprimes)}
        for v in vecs:
            new_sums = set()
            for s in sums:
                ns = tuple(s[i] + v[i] for i in range(nprimes))
                new_sums.add(ns)
            sums |= new_sums
        return sums
    
    # Branch and bound search
    # We'll try to build a large set
    # Use the primes as a base and try to add composites
    
    # Actually, let's do a more systematic search
    # Start with empty set and build up
    
    def search(selected, subset_sums, candidates, current_size):
        nonlocal best, best_size
        
        if time.time() > deadline:
            return
        
        if current_size > best_size:
            best = list(selected)
            best_size = current_size
        
        if not candidates:
            return
        
        # Upper bound: current_size + len(candidates)
        if current_size + len(candidates) <= best_size:
            return
        
        # Try adding each candidate
        for i, idx in enumerate(candidates):
            if time.time() > deadline:
                return
            
            v = vectors[idx]
            
            if can_add(v, subset_sums):
                # Add it
                new_subset_sums = set()
                for s in subset_sums:
                    ns = tuple(s[j] + v[j] for j in range(nprimes))
                    new_subset_sums.add(ns)
                new_subset_sums |= subset_sums
                
                new_selected = selected + [idx]
                new_candidates = candidates[i+1:]
                
                search(new_selected, new_subset_sums, new_candidates, current_size + 1)
            
            # Also try skipping (implicit in the loop structure - we continue to next candidate)
    
    # This naive search is too slow. Let me use a different strategy.
    # Start with primes and greedily try to add composites.
    
    # Actually, let me try a different approach:
    # Use the fact that we want maximum size. Start with a greedy construction
    # then try to improve.
    
    # Greedy: sort by some heuristic, add if possible
    # Heuristic: prefer numbers with "simpler" vectors (fewer non-zero entries)
    
    # Let me try: start with all primes, then try adding each composite
    selected = set(prime_indices)
    vecs = [vectors[i] for i in selected]
    subset_sums = compute_subset_sums(vecs)
    
    # Try adding each composite
    composites = [i for i in range(len(numbers)) if i not in selected]
    
    # Sort composites by "complexity" (number of prime factors)
    def complexity(idx):
        return sum(vectors[idx])
    
    composites.sort(key=complexity)
    
    for idx in composites:
        if time.time() > deadline:
            break
        v = vectors[idx]
        if can_add(v, subset_sums):
            selected.add(idx)
            # Update subset sums
            new_sums = set()
            for s in subset_sums:
                ns = tuple(s[j] + v[j] for j in range(nprimes))
                new_sums.add(ns)
            subset_sums |= new_sums
    
    if len(selected) > best_size:
        best = list(selected)
        best_size = len(best)
    
    # Now try local search: remove one element, add another
    # Or: try different orderings
    
    # Let me also try a randomized restart approach
    import random
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random ordering of all numbers
        order = list(range(len(numbers)))
        random.shuffle(order)
        
        selected = set()
        subset_sums = {tuple([0]*nprimes)}
        
        for idx in order:
            if time.time() > deadline:
                break
            v = vectors[idx]
            if can_add(v, subset_sums):
                selected.add(idx)
                new_sums = set()
                for s in subset_sums:
                    ns = tuple(s[j] + v[j] for j in range(nprimes))
                    new_sums.add(ns)
                subset_sums |= new_sums
        
        if len(selected) > best_size:
            best = list(selected)
            best_size = len(best)
    
    # Convert to actual numbers
    result = [numbers[i] for i in best]
    result.sort()
    
    # Verify
    def verify(nums):
        products = set()
        k = len(nums)
        for mask in range(1 << k):
            prod = 1
            for i in range(k):
                if mask & (1 << i):
                    prod *= nums[i]
            if prod in products:
                return False
            products.add(prod)
        return True
    
    if not verify(result):
        # Fallback to primes
        result = sorted(primes)
    
    # Ensure correct types
    result = [int(x) for x in result]
    return list(result)
