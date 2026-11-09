"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

PRIMES = _primes_up_to(128)
P = len(PRIMES)
PRIME_INDEX = {p: i for i, p in enumerate(PRIMES)}

def _factorize(x):
    """Return exponent vector as a tuple of length P."""
    vec = [0] * P
    for i, p in enumerate(PRIMES):
        while x % p == 0:
            vec[i] += 1
            x //= p
    return tuple(vec)

# Precompute exponent vectors for all numbers 2..128
EXP_VECTORS = {}
for x in range(2, 129):
    EXP_VECTORS[x] = _factorize(x)

def _prefix_vectors(seq):
    """Compute prefix exponent vectors for a sequence. Returns list of tuples."""
    n = len(seq)
    prefix = [tuple([0] * P)]
    cur = [0] * P
    for x in seq:
        ev = EXP_VECTORS[x]
        for i in range(P):
            cur[i] += ev[i]
        prefix.append(tuple(cur))
    return prefix

def _check_valid(seq):
    """Check if all consecutive block products are distinct. Returns True if valid."""
    n = len(seq)
    if n == 0:
        return True
    prefix = _prefix_vectors(seq)
    seen = set()
    for i in range(n):
        for j in range(i + 1, n + 1):
            # block product seq[i..j-1] = prefix[j] - prefix[i]
            diff = tuple(prefix[j][k] - prefix[i][k] for k in range(P))
            if diff in seen:
                return False
            seen.add(diff)
    return True

def _greedy_construct():
    """Original greedy approach."""
    values = []
    products = set()
    for x in range(2, 129):
        # Compute new tails
        tails = [EXP_VECTORS[x]]
        cur = list(EXP_VECTORS[x])
        for a in reversed(values):
            ev = EXP_VECTORS[a]
            for i in range(P):
                cur[i] += ev[i]
            tails.append(tuple(cur))
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values

def _try_insert(seq, x, pos):
    """Try inserting x at position pos in seq. Returns new seq if valid, else None."""
    new_seq = seq[:pos] + [x] + seq[pos:]
    if _check_valid(new_seq):
        return new_seq
    return None

def construct():
    deadline = time.time() + 80  # 80s safety margin
    best = _greedy_construct()
    best_len = len(best)
    
    # Get all numbers not in best
    used = set(best)
    available = [x for x in range(2, 129) if x not in used]
    
    # Local search: try to insert available elements
    random.seed(42)
    available_shuffled = available[:]
    random.shuffle(available_shuffled)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in available_shuffled:
            if x in set(best):
                continue
            # Try inserting at each position
            for pos in range(len(best) + 1):
                if time.time() > deadline:
                    break
                new_seq = _try_insert(best, x, pos)
                if new_seq is not None:
                    best = new_seq
                    best_len = len(best)
                    improved = True
                    break
        # Reshuffle available
        used = set(best)
        available_shuffled = [x for x in range(2, 129) if x not in used]
        random.shuffle(available_shuffled)
    
    # Try removal and reinsertion for further improvement
    while time.time() < deadline - 5:
        improved = False
        # Try removing one element and inserting a different one
        for i in range(len(best)):
            if time.time() > deadline - 5:
                break
            removed = best[i]
            seq_without = best[:i] + best[i+1:]
            # Try inserting available elements
            for x in range(2, 129):
                if x == removed or x in set(seq_without):
                    continue
                for pos in range(len(seq_without) + 1):
                    if time.time() > deadline - 5:
                        break
                    new_seq = seq_without[:pos] + [x] + seq_without[pos:]
                    if _check_valid(new_seq):
                        best = new_seq
                        best_len = len(best)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best
# EVOLVE-BLOCK-END
