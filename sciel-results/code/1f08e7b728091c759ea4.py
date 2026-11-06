"""Dense increasing sequences with unique block products."""
import time
import random
from math import log

# EVOLVE-BLOCK-START

# Precompute primes and factorizations for numbers 2..128
_PRIMES = []
_is_prime = [True] * 129
for i in range(2, 129):
    if _is_prime[i]:
        _PRIMES.append(i)
        for j in range(i * i, 129, i):
            _is_prime[j] = False

_NUM_PRIMES = len(_PRIMES)
_prime_to_idx = {p: i for i, p in enumerate(_PRIMES)}

# Factorization as tuple of exponents
def _factorize(n):
    exps = [0] * _NUM_PRIMES
    for i, p in enumerate(_PRIMES):
        while n % p == 0:
            exps[i] += 1
            n //= p
    return tuple(exps)

_FACTORS = {n: _factorize(n) for n in range(2, 129)}

def _add_factors(a, b):
    return tuple(x + y for x, y in zip(a, b))

def _mul_factors(a, b):
    return tuple(x * y for x, y in zip(a, b))

def _greedy_construct():
    """Original greedy: iterate 2..128, add if no suffix product collision."""
    values = []
    products = set()  # set of product tuples
    for x in range(2, 129):
        fx = _FACTORS[x]
        tails = [fx]
        for a in reversed(values):
            tails.append(_add_factors(tails[-1], _FACTORS[a]))
        if not any(t in products for t in tails):
            values.append(x)
            products.update(tails)
    return values

def _all_block_products(values):
    """Compute all nonempty consecutive block products as tuples."""
    prods = set()
    n = len(values)
    for i in range(n):
        cur = [0] * _NUM_PRIMES
        for j in range(i, n):
            for k in range(_NUM_PRIMES):
                cur[k] += _FACTORS[values[j]][k]
            prods.add(tuple(cur))
    return prods

def _validate(values):
    """Check strictly increasing, in range, and all block products distinct."""
    if not values:
        return True
    for i in range(len(values) - 1):
        if values[i] >= values[i + 1]:
            return False
    if values[0] < 2 or values[-1] > 128:
        return False
    prods = _all_block_products(values)
    # Count total blocks
    n = len(values)
    total = n * (n + 1) // 2
    return len(prods) == total

def _sa_optimize(initial, deadline, seed=42):
    """Simulated annealing to improve the sequence."""
    rng = random.Random(seed)
    best = list(initial)
    best_len = len(best)
    current = list(best)
    current_len = len(current)
    
    start_time = time.time()
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        T *= cooling
        if T < T_min:
            break
        
        # Random move
        move_type = rng.random()
        
        if move_type < 0.3:
            # Replace an element
            if len(current) <= 1:
                continue
            idx = rng.randrange(len(current))
            old_val = current[idx]
            # Try a random different value in [2,128]
            new_val = old_val
            for _ in range(50):
                candidate = rng.randint(2, 128)
                if candidate != old_val:
                    new_val = candidate
                    break
            current[idx] = new_val
            # Sort to maintain increasing
            current.sort()
            # Deduplicate
            deduped = []
            for v in current:
                if not deduped or v > deduped[-1]:
                    deduped.append(v)
            current = deduped
            
        elif move_type < 0.5:
            # Remove an element
            if len(current) <= 1:
                continue
            idx = rng.randrange(len(current))
            current.pop(idx)
            
        elif move_type < 0.7:
            # Insert a new element
            candidates = [v for v in range(2, 129) if v not in current]
            if not candidates:
                continue
            new_val = rng.choice(candidates)
            # Insert in sorted position
            pos = 0
            while pos < len(current) and current[pos] < new_val:
                pos += 1
            current.insert(pos, new_val)
            
        else:
            # Swap two elements
            if len(current) < 2:
                continue
            i, j = rng.sample(range(len(current)), 2)
            current[i], current[j] = current[j], current[i]
            current.sort()
        
        # Validate
        if not _validate(current):
            # Reject invalid
            if move_type < 0.3:
                current[idx] = old_val
                current.sort()
                deduped = []
                for v in current:
                    if not deduped or v > deduped[-1]:
                        deduped.append(v)
                current = deduped
            elif move_type < 0.5:
                current.insert(idx, old_val)
            elif move_type < 0.7:
                current.pop(pos)
            else:
                current[i], current[j] = current[j], current[i]
                current.sort()
            continue
        
        new_len = len(current)
        
        if new_len >= current_len:
            # Accept (improvement or equal)
            current_len = new_len
            if new_len > best_len:
                best = list(current)
                best_len = new_len
        else:
            # Worse: accept with probability
            delta = current_len - new_len
            if rng.random() < _exp_neg(delta, T):
                current_len = new_len
            else:
                # Revert
                if move_type < 0.3:
                    current[idx] = old_val
                    current.sort()
                    deduped = []
                    for v in current:
                        if not deduped or v > deduped[-1]:
                            deduped.append(v)
                    current = deduped
                elif move_type < 0.5:
                    current.insert(idx, old_val)
                elif move_type < 0.7:
                    current.pop(pos)
                else:
                    current[i], current[j] = current[j], current[i]
                    current.sort()
                current_len = len(best) if False else len(current)
    
    return best

def _exp_neg(x, T):
    """Safe exp(-x/T)"""
    if T <= 0:
        return 0.0
    val = -x / T
    if val < -50:
        return 0.0
    return 2.718281828459045 ** val

def construct():
    deadline = time.time() + 55.0  # 55s safety margin
    
    # Start with greedy
    result = _greedy_construct()
    best = list(result)
    best_len = len(best)
    
    # Try multiple SA runs with different seeds
    seed = 42
    while time.time() < deadline:
        candidate = _sa_optimize(best, deadline, seed=seed)
        if len(candidate) > best_len:
            best = candidate
            best_len = len(candidate)
        seed += 1
        # If we've been stuck for a while, try from greedy again
        if seed % 5 == 0 and time.time() < deadline:
            candidate = _sa_optimize(_greedy_construct(), deadline, seed=seed + 1000)
            if len(candidate) > best_len:
                best = candidate
                best_len = len(candidate)
    
    return best

# EVOLVE-BLOCK-END
