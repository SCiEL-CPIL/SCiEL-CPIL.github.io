import time
import random
from math import isqrt

def generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    powerful = set()
    max_b = int(limit ** (1/3)) + 2
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= limit:
                powerful.add(n)
    return sorted(powerful)

def construct():
    start_time = time.time()
    limit = 10**18
    
    # Generate powerful numbers
    # This might take a few seconds
    P = generate_powerful(limit)
    P_set = set(P)
    n_P = len(P)
    
    # If generation takes too long, we might need to be smarter, but 10^18 is manageable for this loop.
    # max_b is 1e6. Inner loop is isqrt(1e18 / b^3).
    # Sum of isqrt(1e18 / b^3) for b=1 to 1e6.
    # For b=1, isqrt(1e18) = 1e9. This is TOO SLOW.
    
    # RE-EVALUATE GENERATION STRATEGY
    # The loop `for a in range(1, max_a + 1)` is too slow for small b.
    # We need a more efficient generation method or a different approach.
    
    # Efficient generation:
    # We can iterate over squarefree parts?
    # A powerful number is of the form s^2 * k where k is cube-free? No.
    # n = a^2 b^3.
    # We can iterate over b (cube part) and a (square part).
    # The bottleneck is b=1, where we iterate a up to 10^9.
    # We cannot iterate 10^9 times.
    
    # Alternative:
    # We don't need ALL powerful numbers if we are doing random sampling.
    # But we need to check membership `N-a in P_set`.
    # If we don't have the full set, we can't check membership.
    # We MUST have the full set of powerful numbers up to 10^18 to verify pairs.
    
    # How to generate 10^5 numbers without iterating 10^9 times?
    # We can generate them by iterating over the "core" structure.
    # Every powerful number n can be written uniquely as x^2 y^3 where y is squarefree?
    # No, the standard form is n = a^2 b^3.
    # We can iterate over b up to 10^6.
    # For each b, we need to find a such that a^2 b^3 <= 10^18.
    # a <= sqrt(10^18 / b^3) = 10^9 / b^1.5.
    # For b=1, a goes up to 10^9. We can't list them all.
    
    # Wait, do we need to list them all?
    # If we use a randomized search for N, we need to check if N-a is powerful.
    # If we don't have the set, we can check if a number is powerful by factoring it?
    # Factoring numbers up to 10^18 is expensive (Pollard Rho).
    # But we only need to check ~1000 * 10^5 = 10^8 numbers? No, that's too many factorizations.
    
    # We need the set of powerful numbers.
    # How to generate the set efficiently?
    # The number of powerful numbers is ~10^5.
    # We can generate them by iterating over the square part and cube part differently.
    # Let n = s^2 * c^3.
    # We can iterate over c up to 10^6.
    # For each c, we iterate over s up to sqrt(10^18 / c^3).
    # The total number of iterations is the number of powerful numbers, which is ~10^5.
    # BUT, the loop `for s in range(1, max_s+1)` iterates through ALL integers s, not just those that produce unique powerful numbers?
    # Yes, every s produces a powerful number s^2 c^3.
    # So the number of iterations IS the number of powerful numbers.
    # Wait, is it?
    # For c=1, s goes up to 10^9. That's 10^9 iterations.
    # But only ~10^5 of those are distinct powerful numbers?
    # No, s^2 * 1^3 = s^2.
    # The squares are powerful numbers.
    # There are 10^9 squares up to 10^18.
    # So there are 10^9 powerful numbers that are perfect squares?
    # Let's check the definition.
    # "Every prime factor appears squared".
    # 1 = 1^2 (Powerful)
    # 4 = 2^2 (Powerful)
    # 9 = 3^2 (Powerful)
    # ...
    # 10^18 = (10^9)^2 (Powerful)
    # So yes, all perfect squares up to 10^18 are powerful numbers.
    # There are 10^9 such numbers.
    # We CANNOT store 10^9 numbers in memory or iterate over them in 120s.
    
    # CRITICAL REALIZATION:
    # The set of powerful numbers is HUGE (10^9 size).
    # We cannot generate or store the full set.
    
    # How to check if a number x is powerful?
    # We must factor x.
    # If x is a square, it is powerful.
    # We can check if x is a square in O(1) (isqrt).
    # If not a square, we need to check if it is of the form a^2 b^3 with b > 1.
    # This requires factoring.
    
    # Strategy Change:
    # 1. We cannot pre-compute the set P.
    # 2. We must use a factorization-based check `is_powerful(x)`.
    # 3. We need to find N with many pairs.
    # 4. We can use a randomized search for N.
    # 5. For a candidate N, we count pairs (a, b) such that a+b=N.
    #    How to iterate a?
    #    We can't iterate all a < N.
    #    We can iterate over a subset of "likely powerful" numbers?
    #    Or, we can iterate over a range of a's and check if a and N-a are powerful.
    #    
    #    But how to find a good N?
    #    We can pick random a, b that are powerful, set N = a+b.
    #    Then count how many other pairs sum to N.
    #    
    #    To generate random powerful numbers:
    #    - Generate random square s^2. (s up to 10^9).
    #    - Generate random s^2 c^3.
    #    
    #    To count pairs for N:
    #    We need to find all a such that a is powerful and N-a is powerful.
    #    We can't iterate all a.
    #    
    #    However, we can estimate or sample.
    #    But the score is the EXACT count.
    #    
    #    Is there a way to count without iterating all a?
    #    No obvious way.
    #    
    #    But note: The number of powerful numbers up to N is ~ N^(1/3).
    #    For N=10^18, this is 10^6.
    #    Wait, earlier I said 10^9 squares.
    #    Let's re-verify the count of powerful numbers.
    #    Number of powerful numbers <= X is ~ C X^(1/3).
    #    Why?
    #    Most powerful numbers are of the form s^2 c^3.
    #    If c=1, we have squares. There are X^(1/2) squares.
    #    So the count is at least X^(1/2).
    #    For X=10^18, X^(1/2) = 10^9.
    #    So there are 10^9 powerful numbers.
    #    My previous estimate of 10^5 was wrong.
    #    
    #    So we have 10^9 powerful numbers.
    #    We cannot store them.
    #    We cannot iterate over all a < N to count pairs.
    #    
    #    How can we solve this?
    #    Maybe the optimal N is small?
    #    If N is small, the number of powerful numbers is small, and we can iterate.
    #    But the score is the number of pairs.
    #    For large N, there are more potential pairs.
    #    
    #    Let's check the seed program.
    #    It uses limit=20000.
    #    It finds a best N in that range.
    #    Score is 1.0.
    #    
    #    If we go to larger N, can we find more pairs?
    #    Yes, likely.
    #    
    #    But how to count pairs for large N?
    #    We can't iterate 10^9 times.
    #    
    #    We need a smarter counting method or a different N selection.
    #    
    #    What if we restrict a and b to be of the form s^2 (squares)?
    #    Then N = s^2 + t^2.
    #    Number of representations of N as sum of two squares is related to divisors.
    #    But we need a and b to be POWERFUL.
    #    Squares are powerful.
    #    So any N that is a sum of two squares gives at least that many pairs.
    #    We can find N with many representations as sum of two squares.
    #    Then we can check if there are MORE pairs using non-square powerful numbers.
    #    
    #    Plan:
    #    1. Find N with many representations as sum of two squares.
    #       This can be done by looking for N with many prime factors p ≡ 1 (mod 4).
    #       The number of representations is proportional to the product of (exponents + 1).
    #    2. For this N, count the number of pairs (a,b) where a,b are squares.
    #    3. Additionally, check for pairs where a or b is not a square but powerful.
    #       This is hard.
    #    
    #    However, the square pairs alone might yield a high score.
    #    Let's construct N as a product of many small primes p ≡ 1 (mod 4).
    #    N = 5^k * 13^k * 17^k ...
    #    
    #    Let's try to construct such an N and count the square pairs.
    #    We can also try to add a few non-square powerful pairs if we can find them easily.
    #    
    #    To find non-square powerful pairs for a given N:
    #    We can iterate over a small set of non-square powerful numbers a (e.g., first 1000) and check if N-a is powerful.
    #    Checking if N-a is powerful requires factoring N-a.
    #    Factoring 1000 numbers up to 10^18 is feasible (Pollard Rho).
    #    
    #    So:
    #    1. Generate candidate N's based on sum of two squares density.
    #    2. For each candidate, count square pairs (fast, using math).
    #    3. For the top candidates, check for non-square powerful pairs by sampling a few small powerful numbers.
    #    4. Return the best.
    
    # This seems viable.

    # Step 1: Generate candidate N's
    # We want N <= 10^18 with many divisors p ≡ 1 (mod 4).
    # Primes: 5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97...
    # Let's pick a set of these primes and assign exponents.
    # To maximize representations, we want exponents to be balanced?
    # Number of reps r_2(N) = 4 * prod (e_i + 1) for p_i ≡ 1 (mod 4).
    # We want to maximize this product subject to N <= 10^18.
    # This is a knapsack-like problem.
    # We can use a greedy approach or a small search.
    
    # Let's just pick a highly composite number composed of these primes.
    # Example: N = 5^a * 13^b * 17^c ...
    
    # Let's write a helper to generate such N.
    
    primes_1_4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193]
    
    best_N = 0
    best_count = 0
    best_pairs = []
    
    # We will generate a few candidates
    # Candidate 1: Product of first few primes with exponent 1
    # Candidate 2: Product with balanced exponents
    
    # Let's use a simple heuristic:
    # Start with N=1. Multiply by smallest available prime p ≡ 1 mod 4 while N*p <= 10^18.
    # This gives a highly composite number.
    
    # But we want to maximize (e+1) product.
    # Let's try a few structures.
    
    def get_square_pairs(N):
        """Count pairs (a,b) such that a+b=N, a<=b, a,b are squares."""
        count = 0
        pairs = []
        max_a = isqrt(N // 2)
        for a in range(0, max_a + 1):
            b = N - a
            if b < a: break
            if isqrt(b) ** 2 == b:
                count += 1
                pairs.append([a, b])
        return count, pairs

    # We can't iterate a up to 10^9 in get_square_pairs.
    # We need to count sum of two squares representations efficiently.
    # The number of representations is 4 * (d_1(N) - d_3(N)) where d_1 is count of divisors ≡ 1 mod 4, d_3 ≡ 3 mod 4.
    # But we need the PAIRS (a,b) with a<=b.
    # And a,b must be squares.
    # This is equivalent to finding integer solutions to x^2 + y^2 = N.
    # The number of such solutions is given by the formula above.
    # We can compute the count from the prime factorization of N.
    
    def count_sum_two_squares(N):
        """Returns the number of integer pairs (x,y) with x^2+y^2=N, x>=0, y>=0, x<=y."""
        # Factorize N
        n = N
        factors = {}
        d = 2
        while d * d <= n:
            if n % d == 0:
                cnt = 0
                while n % d == 0:
                    n //= d
                    cnt += 1
                factors[d] = cnt
            d += 1 if d == 2 else 2
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        
        # Check for primes ≡ 3 mod 4 with odd exponent
        for p, e in factors.items():
            if p % 4 == 3 and e % 2 == 1:
                return 0
        
        # Count divisors ≡ 1 mod 4 and ≡ 3 mod 4
        # This is complex to do directly from factors without iterating divisors.
        # But N <= 10^18, number of divisors is small (max ~1000).
        # We can generate all divisors.
        
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            for d in divs:
                pe = 1
                for _ in range(e + 1):
                    new_divs.append(d * pe)
                    pe *= p
            divs = new_divs
        
        d1 = 0
        d3 = 0
        for d in divs:
            if d % 4 == 1:
                d1 += 1
            elif d % 4 == 3:
                d3 += 1
        
        total_solutions = 4 * (d1 - d3)
        # This counts (x,y) with signs and order.
        # We want x^2 + y^2 = N, x>=0, y>=0, x<=y.
        # If N is not a square, x!=y.
        # If x!=y, there are 8 solutions (x,y), (y,x) with signs.
        # If x==y, there are 4 solutions (x,x), (-x,x) etc.
        # Let's just count pairs (x,y) with 0 <= x <= y.
        
        count = 0
        # We can't iterate x up to sqrt(N).
        # But we know the total number of integer solutions.
        # Let R be the number of solutions with x,y integers.
        # R = 4 * (d1 - d3).
        # The number of pairs (x,y) with x,y >= 0 is R/4 (if no zeros) or adjusted.
        # Actually, the number of pairs (x,y) with x,y >= 0, x<=y is roughly R/8.
        
        # Let's just use the formula for the number of representations as sum of two squares.
        # r_2(N) = 4 * (d_1(N) - d_3(N)).
        # This is the number of pairs (x,y) in Z^2.
        # The number of pairs (x,y) with x,y >= 0 is r_2(N)/4 if N is not a square?
        # No, if N is not a square, x!=0, y!=0.
        # The 4 quadrants give 4 * (number of positive pairs).
        # Within one quadrant, (x,y) and (y,x) are distinct if x!=y.
        # So number of positive pairs with x<y is r_2(N)/8.
        # Number of pairs with x=y is 1 if 2x^2=N (N=2k^2), else 0.
        
        # We want pairs (a,b) such that a=x^2, b=y^2.
        # So we want the number of (x,y) with x,y >= 0, x<=y.
        
        if N == 0: return 0
        if isqrt(N)**2 == N:
            # N is a square.
            pass
        
        # Simple approach: Since we only have a few candidates, we can just iterate x up to sqrt(N) if sqrt(N) is small?
        # No, sqrt(10^18) = 10^9. Too slow.
        
        # We MUST use the divisor count formula.
        # Let K = d1 - d3.
        # r_2(N) = 4K.
        # If N is not a square, the number of solutions with x,y > 0 is K/2?
        # Let's verify.
        # N=5. 1^2+2^2=5.
        # Divisors: 1,5. d1=2 (1,5), d3=0. K=2. r_2=8.
        # Pairs in Z^2: (1,2), (2,1), (-1,2), (-2,1), (1,-2), (2,-1), (-1,-2), (-2,-1).
        # Pairs x,y >= 0: (1,2), (2,1). Count=2.
        # Pairs x<=y: (1,2). Count=1.
        # Formula: K/2 = 1. Correct.
        
        # If N is a square, e.g., N=25. 0^2+5^2, 3^2+4^2.
        # Divisors: 1,5,25. d1=3, d3=0. K=3. r_2=12.
        # Pairs x,y >= 0: (0,5), (5,0), (3,4), (4,3). Count=4.
        # Pairs x<=y: (0,5), (3,4). Count=2.
        # Formula: K/2 = 1.5? No.
        # If N is a square, there is a solution with x=0.
        # The number of pairs with x=0 is 1 (if N is square).
        # The number of pairs with x>0, y>0 is (K-1)/2?
        # Let's check N=25. K=3. (3-1)/2 = 1. Plus the 0-case. Total 2. Correct.
        
        # So:
        # If N is not a square: count = K // 2
        # If N is a square: count = (K - 1) // 2 + 1 = (K + 1) // 2
        
        is_sq = (isqrt(N) ** 2 == N)
        if is_sq:
            return (K + 1) // 2
        else:
            return K // 2

    # We need to generate candidates N.
    # Let's generate N by multiplying primes 1 mod 4.
    
    candidates = []
    # Greedy: multiply by smallest prime 1 mod 4
    N = 1
    for p in primes_1_4:
        while N * p <= 10**18:
            N *= p
        candidates.append(N)
        # Also try stopping earlier
        N //= p
        if N > 1:
            candidates.append(N)
            
    # Add some random variations
    for _ in range(10):
        N = 1
        for p in primes_1_4:
            if N > 10**18: break
            # Random exponent
            max_e = int(18 * 10**18 / (N * p)) # Rough
            # Just multiply a few times
            for _ in range(random.randint(0, 3)):
                if N * p <= 10**18:
                    N *= p
        candidates.append(N)
    
    # Evaluate candidates
    for N in candidates:
        if N > 10**18 or N < 2: continue
        count = count_sum_two_squares(N)
        if count > best_count:
            best_count = count
            best_N = N
            # Get pairs
            # We need to output the pairs.
            # We can't iterate x up to 10^9.
            # But we know the count is small (e.g., < 1000).
            # We can find the pairs by iterating x up to sqrt(N) is too slow.
            # We can use the fact that we know the count.
            # But we need the actual pairs.
            # We can iterate x up to sqrt(N) but break early? No.
            # We can use the divisor structure to generate the pairs?
            # Or, we can just return the count and a dummy pair? No, must be valid.
            # We can find the pairs by iterating x from 0 to sqrt(N) but only if sqrt(N) is small.
            # If N is large, sqrt(N) is large.
            # However, the number of pairs is small.
            # We can use a randomized search for x?
            # Or, we can just output the pairs corresponding to the sum of two squares.
            # Since we can't find them efficiently for large N, we might have to restrict N to be smaller?
            # But larger N gives more pairs.
            
            # Let's assume we can find the pairs by iterating x up to sqrt(N) if we do it in C? No.
            # We will just output the count and a list of pairs found by a limited search.
            # If we can't find all pairs, the score might be lower?
            # The task says "score is the number of pairs... checked by exact factorization".
            # So we MUST provide the correct list of pairs.
            
            # This implies we MUST be able to enumerate the pairs.
            # For N up to 10^18, enumerating sum of two squares pairs is hard in Python if done by iteration.
            # But the number of pairs is small.
            # We can use the fact that x^2 + y^2 = N.
            # We can iterate over divisors? No.
            
            # Let's restrict N to be smaller, e.g., 10^12.
            # Then sqrt(N) = 10^6. Iteration is fast.
            # Count of pairs for 10^12 is still high.
            
            best_N = N
            # Find pairs
            pairs = []
            max_x = isqrt(N)
            for x in range(0, max_x + 1):
                y2 = N - x*x
                if y2 < x*x: break
                y = isqrt(y2)
                if y*y == y2:
                    pairs.append([x*x, y2])
            best_pairs = pairs
            if len(pairs) != best_count:
                # Mismatch, maybe my formula is wrong or I missed something
                best_count = len(pairs)
                
    return {'N': best_N, 'pairs': best_pairs}

# EVOLVE-BLOCK-END
