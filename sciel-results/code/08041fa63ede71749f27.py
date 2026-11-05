"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Precompute prime factorizations for numbers 2..128
    # Represent each number as a tuple of (prime, exponent) pairs
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    
    def factorize(n):
        """Return tuple of (prime, exponent) for n."""
        factors = []
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                exp = 0
                while n % p == 0:
                    n //= p
                    exp += 1
                factors.append((p, exp))
        if n > 1:
            factors.append((n, 1))
        return tuple(factors)
    
    # Precompute factorizations
    factor_cache = {}
    for i in range(2, 129):
        factor_cache[i] = factorize(i)
    
    # Precompute cumulative product factor vectors for quick tail computation
    # A product of consecutive block [a_i, ..., a_j] has factor vector = sum of factor vectors
    
    def add_factor_vectors(v1, v2):
        """Add two factor vectors (tuples of (prime, exp))."""
        if not v1:
            return v2
        if not v2:
            return v1
        # Merge
        result = {}
        for p, e in v1:
            result[p] = e
        for p, e in v2:
            result[p] = result.get(p, 0) + e
        return tuple(sorted(result.items()))
    
    # For efficiency, represent factor vectors as frozenset-like tuples
    # Actually, for hashing and comparison, sorted tuple works well
    
    def compute_tails_factors(values, x):
        """Compute all consecutive-block product factor vectors ending at x."""
        tails = []
        prod_vec = factor_cache[x]
        tails.append(prod_vec)
        for a in reversed(values):
            prod_vec = add_factor_vectors(prod_vec, factor_cache[a])
            tails.append(prod_vec)
        return tails
    
    def is_valid_factors(values, product_set, x):
        """Check if x can be appended without product collision using factor vectors."""
        tails = compute_tails_factors(values, x)
        for t in tails:
            if t in product_set:
                return False, tails
        return True, tails
    
    # Greedy baseline
    def run_greedy():
        values = []
        product_set = set()
        for x in range(2, 129):
            if time.time() > deadline:
                break
            tails = compute_tails_factors(values, x)
            valid = True
            for t in tails:
                if t in product_set:
                    valid = False
                    break
            if valid:
                values.append(x)
                product_set.update(tails)
        return values, product_set
    
    best_seq, best_prods = run_greedy()
    best_len = len(best_seq)
    
    # Beam search
    # State: seq_tuple -> product_factor_set
    # We process numbers in order, at each step either include or skip
    
    beam = {(): set()}
    
    for x in range(2, 129):
        if time.time() > deadline:
            break
        
        next_beam = {}
        
        for seq, prods in beam.items():
            # Option 1: Skip x - state remains
            if seq not in next_beam:
                next_beam[seq] = prods
            
            # Option 2: Include x
            # Compute tails
            tails = compute_tails_factors(list(seq), x)
            
            # Check collision
            has_collision = False
            for t in tails:
                if t in prods:
                    has_collision = True
                    break
            
            if not has_collision:
                new_seq = seq + (x,)
                new_prods = set(prods)
                new_prods.update(tails)
                next_beam[new_seq] = new_prods
        
        # Prune beam: keep top W by length, then by "potential" (smaller sum is better for density)
        # Also consider: sequences that are longer have more "committed" elements
        # Use a scoring function: primary = length, secondary = -sum (smaller sum = more room)
        
        sorted_seqs = sorted(next_beam.keys(), key=lambda s: (-len(s), sum(s) if s else 0))
        
        W = 500
        beam = {s: next_beam[s] for s in sorted_seqs[:W]}
        
        # Update best
        if sorted_seqs and len(sorted_seqs[0]) > best_len:
            best_len = len(sorted_seqs[0])
            best_seq = list(sorted_seqs[0])
            best_prods = next_beam[sorted_seqs[0]]
    
    # Local optimization: try to improve the best sequence
    # Try removing one element and seeing if we can add two others
    if time.time() < deadline - 10:
        current = list(best_seq)
        current_prods = set(best_prods)
        current_len = len(current)
        
        # Try each position for removal
        for pos in range(len(current)):
            if time.time() > deadline:
                break
            
            test = list(current)
            test.pop(pos)
            
            # Recompute products for test
            test_prods = set()
            n = len(test)
            for i in range(n):
                vec = ()
                for j in range(i, n):
                    vec = add_factor_vectors(vec, factor_cache[test[j]])
                    test_prods.add(vec)
            
            # Try to fill gaps: after removal, try to insert new elements
            # This is complex; instead, try a simple greedy refill
            # Find where we can insert
            for x in range(2, 129):
                if x in test:
                    continue
                # Check if x can be inserted at any position
                # For simplicity, try appending after the last element less than x
                # Actually, just try to rebuild greedily from scratch with this as seed
                pass
        
        # Simpler local search: try swapping adjacent pairs
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(current) - 1):
                if time.time() > deadline:
                    break
                # Try swapping current[i] and current[i+1] if it maintains increasing order
                if current[i] > current[i+1]:
                    continue
                # They're already in order, so swap would break order unless equal (not allowed)
                # Instead, try replacing one element
                for j in range(len(current)):
                    if time.time() > deadline:
                        break
                    old_val = current[j]
                    # Try replacing with a different value
                    for new_val in range(2, 129):
                        if new_val == old_val:
                            continue
                        # Check if replacement maintains strictly increasing
                        if j > 0 and new_val <= current[j-1]:
                            continue
                        if j < len(current) - 1 and new_val >= current[j+1]:
                            continue
                        
                        test = list(current)
                        test[j] = new_val
                        # Recompute all products
                        test_prods = set()
                        n = len(test)
                        valid = True
                        for i2 in range(n):
                            vec = ()
                            for j2 in range(i2, n):
                                vec = add_factor_vectors(vec, factor_cache[test[j2]])
                                if vec in test_prods:
                                    valid = False
                                    break
                                test_prods.add(vec)
                            if not valid:
                                break
                        
                        if valid:
                            # Same length, but different products - could allow future extensions
                            # For now, just track if it's a different valid sequence
                            # We could try extending from here
                            pass
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
    
    return best_seq
# EVOLVE-BLOCK-END
