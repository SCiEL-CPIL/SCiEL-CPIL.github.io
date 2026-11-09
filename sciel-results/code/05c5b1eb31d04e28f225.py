"""Binary error-correcting codes of length 18, distance 6.

Strategy:
  1. Build a strong seed via greedy construction from a diverse initial pool.
  2. Iteratively improve by:
     a. Greedy insertion of candidate words from a large pool.
     b. Local search: try to swap or add words to increase code size.
  3. Use time budget to run multiple random restarts.
"""
import time
import random
from itertools import combinations

N = 18
MIN_DIST = 6
MASK = (1 << N) - 1

def hamming_dist(a, b):
    return (a ^ b).bit_count()

def valid_code(code):
    """Check all pairs have distance >= MIN_DIST."""
    for i in range(len(code)):
        for j in range(i + 1, len(code)):
            if hamming_dist(code[i], code[j]) < MIN_DIST:
                return False
    return True

def build_seed_greedy(pool, order_seed=None):
    """Greedy: add words from pool in a fixed/random order if compatible."""
    if order_seed is not None:
        random.seed(order_seed)
    pool = list(pool)
    random.shuffle(pool)
    code = []
    for w in pool:
        ok = True
        for c in code:
            if hamming_dist(w, c) < MIN_DIST:
                ok = False
                break
        if ok:
            code.append(w)
    return code

def greedy_insert(code, pool):
    """Try to insert as many pool words as possible into code."""
    code = list(code)
    pool = list(pool)
    random.shuffle(pool)
    added = 0
    for w in pool:
        if any(hamming_dist(w, c) < MIN_DIST for c in code):
            continue
        code.append(w)
        added += 1
    return code, added

def local_improve(code, pool, deadline):
    """Attempt to grow the code by adding from pool, and try small swaps."""
    code, _ = greedy_insert(code, pool)
    # Try single-word removal + two insertions
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(code)):
            candidate = code[:i] + code[i+1:]
            # try to add two words from pool
            for p1 in range(len(pool)):
                w1 = pool[p1]
                if any(hamming_dist(w1, c) < MIN_DIST for c in candidate):
                    continue
                for p2 in range(len(pool)):
                    if p2 == p1:
                        continue
                    w2 = pool[p2]
                    if any(hamming_dist(w2, c) < MIN_DIST for c in candidate) or \
                       hamming_dist(w1, w2) < MIN_DIST:
                        continue
                    new_code = candidate + [w1, w2]
                    if len(new_code) > len(code):
                        code = new_code
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return code

def generate_pool(size):
    """Generate a diverse pool of candidate words."""
    pool = set()
    # All words of weight 0, 1, 2 (low weight helps fill gaps)
    for w in range(3):
        for comb in combinations(range(N), w):
            word = 0
            for b in comb:
                word |= (1 << b)
            pool.add(word)
    # Random words with various weights
    while len(pool) < size:
        wt = random.randint(3, N - 3)
        bits = random.sample(range(N), wt)
        word = 0
        for b in bits:
            word |= (1 << b)
        pool.add(word)
    # Include Reed-Muller words
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            pool.add(word)
    # Include extended QR-like constructions via quadratic residues
    # Use a known good set: words from the [18,6,?] code family
    # Add cyclic shifts of some base patterns
    base_patterns = [
        0b101010101010101010,
        0b111000111000111000,
        0b110100110100110100,
        0b111100001111000011,
        0b011001100110011001,
    ]
    for bp in base_patterns:
        for shift in range(N):
            pool.add((bp << shift) & MASK)
            pool.add(((~bp) & MASK) << shift & MASK)
    return list(pool)

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    best_code = []
    
    # Initial pool
    pool = generate_pool(2000)
    
    # First greedy pass
    best_code = build_seed_greedy(pool, order_seed=42)
    
    # Multiple random restarts with local improvement
    restart = 0
    while time.time() < deadline:
        restart += 1
        # New random pool mix
        pool_r = generate_pool(1500 + random.randint(0, 1000))
        
        # Greedy with different random order
        code = build_seed_greedy(pool_r, order_seed=restart * 7919)
        
        # Local improvement
        code = local_improve(code, pool_r, deadline)
        
        if len(code) > len(best_code):
            best_code = code
            # Try to further improve the best
            if time.time() < deadline:
                extra_pool = generate_pool(1000)
                best_code, _ = greedy_insert(best_code, extra_pool)
                if time.time() < deadline:
                    best_code = local_improve(best_code, extra_pool, deadline)
        
        # Early exit if we reach a very large code
        if len(best_code) >= 200:
            break
    
    return best_code
