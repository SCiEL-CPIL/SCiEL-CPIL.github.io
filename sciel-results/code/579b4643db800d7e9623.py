"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random


def construct(q, n, radius):
    """Construct a covering code for q-ary alphabet, length n, covering radius r."""
    t_start = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    if q == 3 and n == 7 and radius == 1:
        return _ternary_7_1(t_start, deadline)
    
    # Generic fallback: greedy set cover
    return _greedy_cover(q, n, radius, t_start, deadline)


def _ternary_7_1(t_start, deadline):
    """Specialized solver for K_3(7,1)."""
    q, n, r = 3, 7, 1
    total = q ** n  # 2187
    
    # Precompute: for each of the 2187 words, the set of words it covers
    # A word covers itself + all words at Hamming distance 1
    # Represent each word as an integer in [0, 2187)
    
    # Convert all words to integer indices
    # word (d0, d1, ..., d6) -> d0*3^6 + d1*3^5 + ... + d6
    
    powers = [3**(n-1-i) for i in range(n)]
    
    # Precompute coverage: cov[i] = list of indices covered by word i
    cov = np.zeros((total, total), dtype=bool)
    
    # For each word, compute the indices of itself and its 21 neighbors
    for idx in range(total):
        # Convert idx to digit tuple
        digits = []
        tmp = idx
        for p in reversed(powers):
            digits.append(tmp // p)
            tmp %= p
        digits = tuple(digits)
        
        # Self
        cov[idx, idx] = True
        
        # Neighbors: change one digit to each of the other 2 values
        for pos in range(n):
            old = digits[pos]
            for new_val in range(q):
                if new_val == old:
                    continue
                new_digits = list(digits)
                new_digits[pos] = new_val
                new_idx = sum(d * p for d, p in zip(new_digits, powers))
                cov[idx, new_idx] = True
    
    # Greedy set cover
    uncovered = np.ones(total, dtype=bool)
    codewords = []
    
    while uncovered.any():
        if time.time() > deadline:
            break
        
        # Find codeword that covers the most uncovered words
        # counts = number of uncovered words covered by each candidate
        counts = cov @ uncovered  # (total,) vector
        best = np.argmax(counts)
        
        if counts[best] == 0:
            break
        
        codewords.append(best)
        uncovered[cov[best]] = False
    
    # Local search: try to remove or replace codewords
    # Phase 1: try removing codewords (check if coverage is maintained)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        
        # Compute coverage for current codewords
        if not codewords:
            break
        
        cw_set = set(codewords)
        covered = np.zeros(total, dtype=bool)
        for cw in codewords:
            covered[cov[cw]] = True
        
        # Try removing each codeword
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Check if removing codewords[i] still covers everything
            # A word is only covered by codewords[i] if no other codeword covers it
            # Compute: words covered ONLY by codewords[i]
            
            # For efficiency, compute coverage without codewords[i]
            covered_without = np.zeros(total, dtype=bool)
            for j, cw in enumerate(codewords):
                if j == i:
                    continue
                covered_without[cov[cw]] = True
            
            if covered_without.all():
                codewords.pop(i)
                improved = True
                break
    
    # Phase 2: try replacing pairs (remove 2, add 1)
    if time.time() < deadline - 5:
        for _ in range(200):  # limited iterations
            if time.time() > deadline:
                break
            
            # Compute current coverage
            if not codewords:
                break
            covered = np.zeros(total, dtype=bool)
            for cw in codewords:
                covered[cov[cw]] = True
            
            if not covered.all():
                break
            
            # Try: remove codeword i, find best replacement
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                
                covered_without_i = np.zeros(total, dtype=bool)
                for j, cw in enumerate(codewords):
                    if j == i:
                        continue
                    covered_without_i[cov[cw]] = True
                
                uncovered_after_remove = ~covered_without_i
                if not uncovered_after_remove.any():
                    # Can remove entirely
                    codewords.pop(i)
                    break
                
                # Find best single codeword to add that covers most uncovered
                counts = cov @ uncovered_after_remove
                best_add = np.argmax(counts)
                
                if counts[best_add] >= counts[codewords[i]]:
                    # Adding best_add covers at least as many as removing codewords[i]
                    # But we need: after removing i and adding best_add, everything is covered
                    # This is not guaranteed, so check
                    new_covered = covered_without_i.copy()
                    new_covered[cov[best_add]] = True
                    if new_covered.all():
                        codewords[i] = best_add
                        break
    
    # Convert indices back to digit lists
    result = []
    for idx in codewords:
        digits = []
        tmp = idx
        for p in reversed(powers):
            digits.append(tmp // p)
            tmp %= p
        result.append(digits)
    
    # Verify coverage
    if result:
        covered = np.zeros(total, dtype=bool)
        for cw in result:
            idx = sum(d * p for d, p in zip(cw, powers))
            covered[cov[idx]] = True
        if not covered.all():
            # Fallback: use the seed construction
            return _seed_construction(q, n, radius)
    
    return result


def _seed_construction(q, n, radius):
    """The original seed construction."""
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    # Generic fallback
    return _greedy_cover(q, n, radius, time.time(), time.time() + 10)


def _greedy_cover(q, n, radius, t_start, deadline):
    """Generic greedy set cover for covering codes."""
    total = q ** n
    if total > 100000:
        # Too large for full enumeration, use random sampling
        return _random_cover(q, n, radius, t_start, deadline)
    
    powers = [q**(n-1-i) for i in range(n)]
    
    # Precompute coverage
    cov = np.zeros((total, total), dtype=bool)
    for idx in range(total):
        digits = []
        tmp = idx
        for p in reversed(powers):
            digits.append(tmp // p)
            tmp %= p
        digits = tuple(digits)
        cov[idx, idx] = True
        for pos in range(n):
            for new_val in range(q):
                if new_val == digits[pos]:
                    continue
                new_digits = list(digits)
                new_digits[pos] = new_val
                new_idx = sum(d * p for d, p in zip(new_digits, powers))
                cov[idx, new_idx] = True
    
    uncovered = np.ones(total, dtype=bool)
    codewords = []
    
    while uncovered.any():
        if time.time() > deadline:
            break
        counts = cov @ uncovered
        best = np.argmax(counts)
        if counts[best] == 0:
            break
        codewords.append(best)
        uncovered[cov[best]] = False
    
    result = []
    for idx in codewords:
        digits = []
        tmp = idx
        for p in reversed(powers):
            digits.append(tmp // p)
            tmp %= p
        result.append(digits)
    
    return result


def _random_cover(q, n, radius, t_start, deadline):
    """Random greedy for large spaces."""
    codewords = []
    seen = set()
    
    while time.time() < deadline:
        cw = [random.randint(0, q-1) for _ in range(n)]
        if tuple(cw) not in seen:
            seen.add(tuple(cw))
            codewords.append(cw)
    
    return codewords
# EVOLVE-BLOCK-END
