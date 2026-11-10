"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import Counter

# Precompute prime factorization vectors for all numbers 2-128
_MAX_VAL = 128
_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
_prime_index = {p: i for i, p in enumerate(_primes)}
_num_primes = len(_primes)

def _factorize(n):
    """Return a vector (tuple) of prime exponents for n."""
    vec = [0] * _num_primes
    for i, p in enumerate(_primes):
        if p * p > n:
            break
        while n % p == 0:
            vec[i] += 1
            n //= p
    if n > 1:
        if n in _prime_index:
            vec[_prime_index[n]] += 1
    return tuple(vec)

# Precompute factorization vectors
_factor_vecs = {}
for x in range(2, _MAX_VAL + 1):
    _factor_vecs[x] = _factorize(x)

def _vec_add(a, b):
    """Add two exponent vectors element-wise."""
    return tuple(x + y for x, y in zip(a, b))

def _compute_all_product_vecs(seq):
    """Compute all consecutive block product exponent vectors for a sequence."""
    vecs = set()
    n = len(seq)
    for i in range(n):
        cur = (0,) * _num_primes
        for j in range(i, n):
            cur = _vec_add(cur, _factor_vecs[seq[j]])
            vecs.add(cur)
    return vecs

def _is_valid_sequence(seq):
    """Check if sequence is strictly increasing and all block products are distinct."""
    if not seq:
        return True
    # Check strictly increasing
    for i in range(1, len(seq)):
        if seq[i] <= seq[i-1]:
            return False
    # Check products using exponent vectors
    n = len(seq)
    expected = n * (n + 1) // 2
    vecs = _compute_all_product_vecs(seq)
    return len(vecs) == expected

def _try_insert_at(seq, x, pos, existing_vecs):
    """Try inserting x at position pos, return new vecs set if valid, else None."""
    n = len(seq)
    if pos > 0 and seq[pos-1] >= x:
        return None
    if pos < n and seq[pos] <= x:
        return None
    
    new_seq = seq[:pos] + [x] + seq[pos:]
    new_vecs = _compute_all_product_vecs(new_seq)
    expected = len(new_seq) * (len(new_seq) + 1) // 2
    if len(new_vecs) == expected:
        return new_vecs
    return None

def _local_search(seq, deadline):
    """Apply local search with insertion and swap moves to improve sequence length."""
    best = list(seq)
    best_vecs = _compute_all_product_vecs(best)
    improved = True
    max_passes = 20
    passes = 0
    
    while improved and passes < max_passes and time.time() < deadline:
        improved = False
        passes += 1
        available = [x for x in range(2, _MAX_VAL + 1) if x not in best]
        
        # 1. Try to insert any available number
        for x in available:
            if time.time() >= deadline:
                break
            # Try inserting x at all valid positions
            for pos in range(len(best) + 1):
                new_vecs = _try_insert_at(best, x, pos, best_vecs)
                if new_vecs is not None:
                    best = best[:pos] + [x] + best[pos:]
                    best_vecs = new_vecs
                    improved = True
                    break
            if improved:
                break
        
        if time.time() >= deadline:
            break
        
        # 2. If no insertion worked, try 1-out/1-in swaps to potentially unlock insertions
        found_swap = False
        for i in range(len(best)):
            if time.time() >= deadline:
                break
            for x in available:
                if time.time() >= deadline:
                    break
                # Remove best[i], try inserting x at various positions
                base = best[:i] + best[i+1:]
                for pos in range(len(base) + 1):
                    new_vecs = _try_insert_at(base, x, pos, best_vecs)
                    if new_vecs is not None:
                        # Check if this swap allows an immediate additional insertion
                        new_best = base[:pos] + [x] + base[pos:]
                        new_available = [y for y in available if y != x]
                        can_extend = False
                        for y in new_available:
                            if time.time() >= deadline:
                                break
                            for pos2 in range(len(new_best) + 1):
                                if _try_insert_at(new_best, y, pos2, new_vecs) is not None:
                                    can_extend = True
                                    break
                            if can_extend:
                                break
                        if can_extend:
                            best = new_best
                            best_vecs = new_vecs
                            available.remove(x)
                            available.append(best[i])
                            improved = True
                            found_swap = True
                            break
                if found_swap:
                    break
            if found_swap:
                break
                
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Greedy construction using exponent vectors
    values = []
    product_vecs = set()
    for x in range(2, _MAX_VAL + 1):
        # Compute all suffix product vectors when appending x
        tails = []
        cur = (0,) * _num_primes
        for a in reversed(values):
            cur = _vec_add(cur, _factor_vecs[a])
            tails.append(cur)
        tails.append(_factor_vecs[x])  # singleton
        
        # Check if any tail product vector already exists
        if not product_vecs.intersection(tails):
            values.append(x)
            product_vecs.update(tails)
    
    # Try to improve with local search if time permits
    if time.time() < deadline - 2.0:
        improved = _local_search(values, deadline)
        if len(improved) > len(values):
            values = improved
    
    return values
# EVOLVE-BLOCK-END
