import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55

    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    # Precompute prime factorizations for all numbers 2..128
    prime_factors = {}
    for n in range(2, 129):
        factors = {}
        temp = n
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                factors[d] = factors.get(d, 0) + 1
                temp //= d
            d += 1
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        prime_factors[n] = factors

    all_numbers = list(range(2, 129))
    primes = [x for x in all_numbers if is_prime(x)]
    composites = [x for x in all_numbers if not is_prime(x)]

    def is_valid_fast(s):
        """
        Check if all consecutive block products are distinct using prime exponent vectors.
        Two products are equal iff their prime exponent vectors are identical.
        """
        n = len(s)
        if n == 0:
            return True
        
        # For each starting position i, compute cumulative exponent vector
        # and check for collisions
        prods = set()
        for i in range(n):
            # Start with zero vector
            vec = {}
            for j in range(i, n):
                # Add prime_factors[s[j]] to vec
                pf = prime_factors[s[j]]
                for p, e in pf.items():
                    vec[p] = vec.get(p, 0) + e
                # Convert to hashable tuple for set membership
                # Use a canonical form: sort by prime
                key = tuple(sorted(vec.items()))
                if key in prods:
                    return False
                prods.add(key)
        return True

    def is_valid_incremental(s, pos, val):
        """
        Check if inserting val at position pos in sequence s keeps all block products distinct.
        Only need to check blocks that include the new element.
        """
        n = len(s)
        new_seq = s[:pos] + [val] + s[pos:]
        new_n = len(new_seq)
        
        # We need to check all blocks that include the new element at position pos
        # Blocks that don't include it were already valid (assuming s was valid)
        # But we also need to ensure new blocks don't collide with old blocks
        
        # Strategy: compute all block products in new_seq and check for duplicates
        # But we can optimize by only checking blocks that include position pos
        
        # First, collect all existing block products from s
        old_prods = set()
        for i in range(n):
            vec = {}
            for j in range(i, n):
                pf = prime_factors[s[j]]
                for p, e in pf.items():
                    vec[p] = vec.get(p, 0) + e
                key = tuple(sorted(vec.items()))
                old_prods.add(key)
        
        # Now check new blocks that include position pos
        # A block [a,b] in new_seq includes pos iff a <= pos <= b
        # In terms of new_seq indices, the new element is at index pos
        for a in range(pos + 1):  # a is start index in new_seq, 0-indexed
            for b in range(pos, new_n):  # b is end index in new_seq
                # Block is new_seq[a:b+1]
                vec = {}
                for j in range(a, b + 1):
                    val_j = new_seq[j]
                    pf = prime_factors[val_j]
                    for p, e in pf.items():
                        vec[p] = vec.get(p, 0) + e
                key = tuple(sorted(vec.items()))
                if key in old_prods:
                    return False
                old_prods.add(key)
        
        return True

    best_seq = primes[:]
    best_len = len(primes)
    
    random.seed(42)
    
    def greedy_insert(seed, check_order_strategy, deadline):
        """
        Given a valid seed, try to insert remaining numbers.
        """
        seed = sorted(seed)
        if not is_valid_fast(seed):
            return seed, len(seed)
            
        current_seq = seed[:]
        current_len = len(current_seq)
        current_set = set(current_seq)
        
        remaining = [x for x in all_numbers if x not in current_set]
        
        if check_order_strategy == 'ascending':
            check_order = remaining[:]
        elif check_order_strategy == 'descending':
            check_order = remaining[::-1]
        elif check_order_strategy == 'random':
            check_order = remaining[:]
            random.shuffle(check_order)
        elif check_order_strategy == 'middle_out':
            mid = len(remaining) // 2
            check_order = []
            for d in range(mid + 1):
                if mid - d >= 0:
                    check_order.append(remaining[mid - d])
                if d > 0 and mid + d < len(remaining):
                    check_order.append(remaining[mid + d])
        else:
            check_order = remaining[:]
        
        for c in check_order:
            if time.time() > deadline - 5:
                break
                
            # Find position to insert c to keep sequence sorted
            pos = 0
            while pos < len(current_seq) and current_seq[pos] < c:
                pos += 1
            
            if pos < len(current_seq) and current_seq[pos] == c:
                continue
                
            new_seq = current_seq[:pos] + [c] + current_seq[pos:]
            
            if is_valid_fast(new_seq):
                current_seq = new_seq
                current_set.add(c)
                current_len += 1
        
        return current_seq, current_len

    # Define multiple strategies to try
    strategies = [
        ('primes_asc', primes[:], 'ascending'),
        ('primes_desc', primes[:], 'descending'),
        ('primes_rand', primes[:], 'random'),
        ('small_asc', all_numbers[:20], 'ascending'),
        ('small_desc', all_numbers[:20], 'descending'),
        ('small_rand', all_numbers[:20], 'random'),
        ('small_mid', all_numbers[:20], 'middle_out'),
    ]
    
    # Add a random seed
    rand_seed = []
    for x in sorted(random.sample(all_numbers, 15)):
        test = rand_seed + [x]
        pos = 0
        while pos < len(test) and test[pos] < x:
            pos += 1
        test.insert(pos, x)
        if is_valid_fast(test):
            rand_seed = test
        else:
            test.pop(pos)
    if rand_seed:
        strategies.append(('rand_asc', rand_seed[:], 'ascending'))
        strategies.append(('rand_desc', rand_seed[:], 'descending'))
        strategies.append(('rand_rand', rand_seed[:], 'random'))
    
    # Also try a "primes + selected composites" seed
    sq_free_composites = [x for x in composites if all(prime_factors[x].get(p, 0) == 1 for p in prime_factors[x])]
    seed_pf = primes[:] + sq_free_composites[:10]
    seed_pf.sort()
    if is_valid_fast(seed_pf):
        strategies.append(('pf_asc', seed_pf[:], 'ascending'))
        strategies.append(('pf_desc', seed_pf[:], 'descending'))
    
    for name, seed, strategy in strategies:
        if time.time() > deadline - 10:
            break
            
        result, result_len = greedy_insert(seed, strategy, deadline)
        
        if result_len > best_len:
            best_len = result_len
            best_seq = result[:]
    
    # Second phase: local search with swap-based improvement
    if time.time() < deadline - 15:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            current_best = best_seq[:]
            current_best_set = set(current_best)
            
            # Try removing each element and see if we can add back two others
            for idx in range(len(current_best)):
                if time.time() > deadline - 10:
                    break
                removed = current_best[idx]
                without = current_best[:idx] + current_best[idx+1:]
                if not is_valid_fast(without):
                    continue
                
                # Try to add two numbers not in current_best
                missing = [x for x in all_numbers if x not in current_best_set]
                # Try pairs
                for i in range(len(missing)):
                    if time.time() > deadline - 10:
                        break
                    for j in range(i+1, len(missing)):
                        if time.time() > deadline - 10:
                            break
                        candidate = without + [missing[i], missing[j]]
                        candidate.sort()
                        if is_valid_fast(candidate) and len(candidate) > len(best_seq):
                            best_seq = candidate
                            best_len = len(candidate)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    # Third phase: try single-element swaps
    if time.time() < deadline - 5:
        for idx in range(len(best_seq)):
            if time.time() > deadline - 5:
                break
            for new_val in all_numbers:
                if time.time() > deadline - 5:
                    break
                if new_val in best_seq and new_val != best_seq[idx]:
                    continue
                if new_val == best_seq[idx]:
                    continue
                candidate = best_seq[:idx] + [new_val] + best_seq[idx+1:]
                candidate.sort()
                if len(set(candidate)) != len(candidate):
                    continue
                if is_valid_fast(candidate):
                    # Check if this improves length or is same length but different
                    if len(candidate) > len(best_seq):
                        best_seq = candidate
                        best_len = len(candidate)
    
    return best_seq
