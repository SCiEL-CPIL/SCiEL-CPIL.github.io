"""Rewritten program with a stronger algorithm for integer sum-product extremal sets."""
import time
import random
import sys
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 75.0  # Safety margin before 90s kill
    best_set = None
    best_score = float('inf')
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A."""
        A = sorted(A)
        n = len(A)
        # Sumset A+A (with equal summands allowed)
        sumset = set()
        for i in range(n):
            for j in range(i, n):
                sumset.add(A[i] + A[j])
        # Product set A*A (with equal factors allowed)
        prodset = set()
        for i in range(n):
            for j in range(i, n):
                prodset.add(A[i] * A[j])
        return max(len(sumset), len(prodset))
    
    # Start with a reasonable heuristic: geometric progression
    # For n=20, a geometric progression with ratio r gives:
    # Sumset size ~ n(n+1)/2 = 210 (all partial sums distinct if r is large enough)
    # Product set size ~ n(n+1)/2 = 210 (all products distinct)
    # But we want to minimize the MAXIMUM, so we want both small.
    # 
    # Key insight: If we use a small set of bases and their products, we might get
    # overlaps in both sumset and productset.
    
    # Strategy 1: Try small integers that create structure
    # The set {1, 2, 4, 8, ..., 2^19} has:
    # - Sumset: all sums of distinct powers of 2 (with repetition allowed for equal summands)
    #   Actually with equal summands allowed, 2^i + 2^i = 2^(i+1), so we get all integers 2..2^20-1? No.
    #   Let's think: A = {2^0, 2^1, ..., 2^19}. A+A = {2^i + 2^j : 0<=i<=j<=19}.
    #   These are all distinct (binary representation), so |A+A| = 20*21/2 = 210.
    #   A*A = {2^i * 2^j : 0<=i<=j<=19} = {2^(i+j) : 0<=i<=j<=19}.
    #   i+j ranges from 0 to 38, but not all values are achievable uniquely.
    #   Actually i+j can be any value from 0 to 38, and for each k=i+j with i<=j,
    #   the number of representations doesn't matter since we're counting distinct values.
    #   So |A*A| = 39.
    #   max(210, 39) = 210.
    
    # Can we do better? The minimum possible |A+A| for 20 distinct positive integers is at least 20*21/2 = 210?
    # No! That's only true if all sums are distinct. In general, |A+A| >= 2n-1 for distinct positive integers,
    # but that's a lower bound. The actual minimum is larger.
    
    # For A = {1, 2, ..., 20}:
    # A+A = {2, 3, ..., 40}, so |A+A| = 39.
    # A*A: products of pairs from {1..20}. Many overlaps.
    # |A*A| is somewhat large. Let's compute it.
    
    # Let's try the sequential set first
    A_seq = list(range(1, 21))
    score_seq = evaluate(A_seq)
    best_set = A_seq[:]
    best_score = score_seq
    
    # Try geometric progression with ratio 2
    if time.time() > deadline:
        return best_set
    
    A_geo = [2**i for i in range(20)]
    score_geo = evaluate(A_geo)
    if score_geo < best_score:
        best_score = score_geo
        best_set = A_geo[:]
    
    # Try mixed approaches: small base set with some structure
    # Try {1, 2, 3, 4, ..., k} union {large powers}
    
    # Let's try random local search starting from sequential
    if time.time() > deadline:
        return best_set
    
    current = A_seq[:]
    current_score = score_seq
    
    # Local search: try swapping elements
    random.seed(42)
    for _ in range(50000):
        if time.time() > deadline:
            break
        i = random.randint(0, 19)
        # Try replacing current[i] with a random value
        new_val = random.randint(1, 10000)
        if new_val in current:
            continue
        old_val = current[i]
        current[i] = new_val
        current.sort()
        new_score = evaluate(current)
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_set = current[:]
        else:
            current[i] = old_val
            current.sort()
    
    # Try a structured approach: use small numbers that create many product overlaps
    # The set {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 48}
    # These are divisors of 720 or similar highly composite numbers
    
    if time.time() > deadline:
        return best_set
    
    # Try sets based on divisors of highly composite numbers
    def divisors(n):
        divs = set()
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
            i += 1
        return divs
    
    # Try 720
    divs720 = sorted([d for d in divisors(720) if d <= 10000])
    if len(divs720) >= 20:
        A_div720 = divs720[:20]
        score_div = evaluate(A_div720)
        if score_div < best_score:
            best_score = score_div
            best_set = A_div720[:]
    
    # Try 5040
    divs5040 = sorted([d for d in divisors(5040) if d <= 10000])
    if len(divs5040) >= 20:
        A_div5040 = divs5040[:20]
        score_div5 = evaluate(A_div5040)
        if score_div5 < best_score:
            best_score = score_div5
            best_set = A_div5040[:]
    
    # Try 2520
    divs2520 = sorted([d for d in divisors(2520) if d <= 10000])
    if len(divs2520) >= 20:
        A_div2520 = divs2520[:20]
        score_div25 = evaluate(A_div2520)
        if score_div25 < best_score:
            best_score = score_div25
            best_set = A_div2520[:]
    
    # Try 1260
    divs1260 = sorted([d for d in divisors(1260) if d <= 10000])
    if len(divs1260) >= 20:
        A_div1260 = divs1260[:20]
        score_div12 = evaluate(A_div1260)
        if score_div12 < best_score:
            best_score = score_div12
            best_set = A_div1260[:]
    
    # More random local search from different starting points
    if time.time() > deadline:
        return best_set
    
    for start_offset in [0, 10, 50, 100, 500]:
        if time.time() > deadline:
            break
        current = [x + start_offset for x in range(1, 21)]
        current_score = evaluate(current)
        if current_score < best_score:
            best_score = current_score
            best_set = current[:]
        
        for _ in range(10000):
            if time.time() > deadline:
                break
            i = random.randint(0, 19)
            new_val = random.randint(1, 10000)
            if new_val in current:
                continue
            old_val = current[i]
            current[i] = new_val
            current.sort()
            new_score = evaluate(current)
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_set = current[:]
            else:
                current[i] = old_val
                current.sort()
    
    # Strategy: Use sets based on divisors of lcm(1..k) or highly composite numbers
    # These create many product collisions because a*b = c*d for many quadruples
    
    if time.time() > deadline:
        return best_set
    
    # Try sets of the form {d1, d2, ..., d20} where di are divisors of N
    # with N being a highly composite number
    def get_divisors(n):
        divs = set()
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
            i += 1
        return sorted(divs)
    
    # Try various highly composite numbers
    highly_composite = [2520, 5040, 10080, 15120, 20160, 27720, 45360, 50400, 55440, 83160]
    
    for N in highly_composite:
        if time.time() > deadline:
            break
        divs = [d for d in get_divisors(N) if d <= 10000]
        if len(divs) < 20:
            continue
        
        # Try first 20 divisors
        A_test = divs[:20]
        score = evaluate(A_test)
        if score < best_score:
            best_score = score
            best_set = A_test[:]
        
        # Try last 20 divisors (larger divisors)
        if len(divs) >= 20:
            A_test = divs[-20:]
            score = evaluate(A_test)
            if score < best_score:
                best_score = score
                best_set = A_test[:]
        
        # Try a mix: smallest and largest
        if len(divs) >= 40:
            A_test = divs[:10] + divs[-10:]
            if len(set(A_test)) == 20:
                score = evaluate(A_test)
                if score < best_score:
                    best_score = score
                    best_set = A_test[:]
    
    # Strategy: Use arithmetic progressions with small difference
    # A = {a, a+d, a+2d, ..., a+19d}
    # Sumset has structure, product set may have collisions
    
    if time.time() > deadline:
        return best_set
    
    for a in range(1, 20):
        if time.time() > deadline:
            break
        for d in range(1, 10):
            if time.time() > deadline:
                break
            A_ap = [a + i * d for i in range(20)]
            if max(A_ap) > 10000:
                continue
            score = evaluate(A_ap)
            if score < best_score:
                best_score = score
                best_set = A_ap[:]
    
    # Strategy: Use sets with many small factors
    # Numbers like 2^a * 3^b * 5^c * 7^d create many product collisions
    
    if time.time() > deadline:
        return best_set
    
    # Generate numbers with small prime factors
    def small_factor_numbers(limit=10000, count=20):
        """Generate numbers <= limit with many small prime factors."""
        candidates = []
        for n in range(1, limit + 1):
            # Count number of divisors
            div_count = 0
            i = 1
            while i * i <= n:
                if n % i == 0:
                    div_count += 1
                    if i != n // i:
                        div_count += 1
                i += 1
            candidates.append((div_count, n))
        
        # Sort by number of divisors (descending)
        candidates.sort(reverse=True)
        return [n for _, n in candidates[:count]]
    
    A_small_factors = small_factor_numbers(10000, 20)
    score = evaluate(A_small_factors)
    if score < best_score:
        best_score = score
        best_set = A_small_factors[:]
    
    # Strategy: Local search with structured moves
    # Try swapping two elements, or replacing one with a multiple/divisor
    
    if time.time() > deadline:
        return best_set
    
    random.seed(123)
    current = best_set[:]
    current_score = best_score
    
    for iteration in range(30000):
        if time.time() > deadline:
            break
        
        # Random move type
        move_type = random.choice(['replace', 'swap', 'multiple', 'divisor'])
        
        if move_type == 'replace':
            i = random.randint(0, 19)
            new_val = random.randint(1, 10000)
            if new_val in current:
                continue
            old_val = current[i]
            current[i] = new_val
            current.sort()
            new_score = evaluate(current)
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_set = current[:]
            else:
                current[i] = old_val
                current.sort()
        
        elif move_type == 'swap':
            i, j = random.sample(range(20), 2)
            current[i], current[j] = current[j], current[i]
            current.sort()
            new_score = evaluate(current)
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_set = current[:]
            else:
                current[i], current[j] = current[j], current[i]
                current.sort()
        
        elif move_type == 'multiple':
            i = random.randint(0, 19)
            factor = random.choice([2, 3, 4, 5, 6])
            new_val = current[i] * factor
            if new_val > 10000 or new_val in current:
                continue
            old_val = current[i]
            current[i] = new_val
            current.sort()
            new_score = evaluate(current)
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_set = current[:]
            else:
                current[i] = old_val
                current.sort()
        
        elif move_type == 'divisor':
            i = random.randint(0, 19)
            # Try to find a proper divisor
            val = current[i]
            divisors_found = []
            for d in range(2, int(val ** 0.5) + 1):
                if val % d == 0:
                    divisors_found.append(d)
                    if d != val // d:
                        divisors_found.append(val // d)
            if not divisors_found:
                continue
            new_val = random.choice(divisors_found)
            if new_val in current:
                continue
            old_val = current[i]
            current[i] = new_val
            current.sort()
            new_score = evaluate(current)
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_set = current[:]
            else:
                current[i] = old_val
                current.sort()
    
    return best_set
# EVOLVE-BLOCK-END
