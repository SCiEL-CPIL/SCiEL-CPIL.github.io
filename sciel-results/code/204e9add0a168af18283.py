"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import log
from collections import Counter

def _prime_factors(n):
    """Return dict of prime -> exponent for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _num_prime_factors(n):
    """Count total prime factors with multiplicity."""
    return sum(_prime_factors(n).values())

def _all_block_products(seq):
    """Return set of all nonempty consecutive block products."""
    prods = set()
    n = len(seq)
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            prods.add(p)
    return prods

def _is_valid(seq):
    """Check strictly increasing in [2,128] and all block products distinct."""
    if not seq:
        return False
    if len(seq) != len(set(seq)):
        return False
    if seq[0] < 2 or seq[-1] > 128:
        return False
    for i in range(len(seq) - 1):
        if seq[i] >= seq[i + 1]:
            return False
    prods = _all_block_products(seq)
    return len(prods) == sum(len(seq) - i for i in range(len(seq)))

def _is_valid_fast(seq):
    """Fast validity check with early termination."""
    n = len(seq)
    if n == 0:
        return False
    if seq[0] < 2 or seq[-1] > 128:
        return False
    for i in range(n - 1):
        if seq[i] >= seq[i + 1]:
            return False
    
    prods = set()
    expected_count = n * (n + 1) // 2
    current_count = 0
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            if p in prods:
                return False
            prods.add(p)
            current_count += 1
    return current_count == expected_count

def _greedy_construct():
    """Greedy: pick numbers in order of increasing prime factor count, then value."""
    candidates = list(range(2, 129))
    # Sort by (num_prime_factors, value) to prefer "simpler" numbers
    candidates.sort(key=lambda x: (_num_prime_factors(x), x))
    
    values = []
    products = set()
    
    for x in candidates:
        # Check if x can be inserted while maintaining increasing order
        # and without creating product conflicts
        # Find insertion position
        pos = 0
        while pos < len(values) and values[pos] < x:
            pos += 1
        
        # Check if inserting x at pos creates conflicts
        # We need to check all blocks that include x
        new_products = []
        # Blocks ending at x (from i to pos-1 in values, plus x)
        for i in range(pos, -1, -1):
            p = x
            for j in range(i, pos):
                p *= values[j]
            new_products.append(p)
        # Blocks starting at x (x plus values from pos to j)
        for j in range(pos, len(values)):
            p = x
            for k in range(pos, j + 1):
                p *= values[k]
            if p not in new_products:  # avoid double-counting singleton
                new_products.append(p)
        
        if not any(p in products for p in new_products):
            values.insert(pos, x)
            products.update(new_products)
    
    return sorted(values)

def _greedy_construct_v2():
    """Alternative greedy: try to maximize length by being more selective."""
    # Use prime factorization signatures to avoid conflicts
    values = []
    products = set()
    
    # Process numbers in order of increasing "complexity"
    candidates = list(range(2, 129))
    candidates.sort(key=lambda x: (len(_prime_factors(x)), sum(_prime_factors(x).values()), x))
    
    for x in candidates:
        pos = 0
        while pos < len(values) and values[pos] < x:
            pos += 1
        
        # Check all new block products
        conflict = False
        # Left-extending blocks
        for i in range(pos, -1, -1):
            p = x
            for j in range(i, pos):
                p *= values[j]
            if p in products:
                conflict = True
                break
        if conflict:
            continue
        
        # Right-extending blocks
        for j in range(pos, len(values)):
            p = x
            for k in range(pos, j + 1):
                p *= values[k]
            if p in products:
                conflict = True
                break
        if conflict:
            continue
        
        # No conflict, add it
        new_prods = set()
        for i in range(pos, -1, -1):
            p = x
            for j in range(i, pos):
                p *= values[j]
            new_prods.add(p)
        for j in range(pos, len(values)):
            p = x
            for k in range(pos, j + 1):
                p *= values[k]
            new_prods.add(p)
        
        values.insert(pos, x)
        products.update(new_prods)
    
    return sorted(values)

def _try_improve(seq, deadline):
    """Local search with 1-out/1-in and 2-out/2-in moves."""
    best = list(seq)
    best_len = len(best)
    
    while time.time() < deadline:
        improved = False
        n = len(best)
        
        # 1-out/1-in moves
        for i in range(n):
            if time.time() >= deadline:
                break
            candidate = list(best)
            candidate.pop(i)
            for x in range(2, 129):
                if x in candidate:
                    continue
                pos = 0
                while pos < len(candidate) and candidate[pos] < x:
                    pos += 1
                new_seq = candidate[:pos] + [x] + candidate[pos:]
                
                if new_seq[0] < 2 or new_seq[-1] > 128:
                    continue
                valid_order = all(new_seq[k] < new_seq[k+1] for k in range(len(new_seq) - 1))
                if not valid_order:
                    continue
                
                if _is_valid_fast(new_seq):
                    if len(new_seq) > best_len:
                        best = new_seq
                        best_len = len(best)
                        improved = True
        
        # 2-out/2-in moves (less frequent, more expensive)
        if time.time() < deadline and not improved:
            for i in range(n):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() >= deadline:
                        break
                    candidate = list(best)
                    candidate.pop(j)
                    candidate.pop(i)
                    
                    # Try inserting two new numbers
                    for x in range(2, 129):
                        if time.time() >= deadline:
                            break
                        if x in candidate:
                            continue
                        pos_x = 0
                        while pos_x < len(candidate) and candidate[pos_x] < x:
                            pos_x += 1
                        temp = candidate[:pos_x] + [x] + candidate[pos_x:]
                        
                        for y in range(2, 129):
                            if time.time() >= deadline:
                                break
                            if y in temp:
                                continue
                            pos_y = 0
                            while pos_y < len(temp) and temp[pos_y] < y:
                                pos_y += 1
                            new_seq = temp[:pos_y] + [y] + temp[pos_y:]
                            
                            if new_seq[0] < 2 or new_seq[-1] > 128:
                                continue
                            valid_order = all(new_seq[k] < new_seq[k+1] for k in range(len(new_seq) - 1))
                            if not valid_order:
                                continue
                            
                            if _is_valid_fast(new_seq):
                                if len(new_seq) > best_len:
                                    best = new_seq
                                    best_len = len(best)
                                    improved = True
    
        if not improved:
            break
    
    return best

def construct():
    deadline = time.time() + 80  # Safety margin before 90s kill
    
    # Try multiple greedy strategies
    best = _greedy_construct()
    best_len = len(best)
    
    alt = _greedy_construct_v2()
    if len(alt) > best_len:
        best = alt
        best_len = len(best)
    
    # Try to improve with local search
    if time.time() < deadline:
        improved = _try_improve(best, deadline)
        if len(improved) > best_len:
            best = improved
            best_len = len(best)
    
    return best
# EVOLVE-BLOCK-END
