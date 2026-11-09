import math
import itertools
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def pairwise_coprime(lst):
        n = len(lst)
        for i in range(n):
            for j in range(i+1, n):
                if gcd(lst[i], lst[j]) != 1:
                    return False
        return True
    
    def is_valid(A, B):
        # Check distinctness
        if len(set(A)) != 3 or len(set(B)) != 3:
            return False
        # Check range
        for x in A + B:
            if x < 1 or x > 128:
                return False
        # Compute all sums
        sums = [a + b for a in A for b in B]
        distinct_sums = list(set(sums))
        # Check pairwise coprime
        if not pairwise_coprime(distinct_sums):
            return False
        return len(distinct_sums)
    
    # Best result so far
    best_count = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # The seed gives: A=[1,7,13], B=[4,10,16]
    # Sums: 5,11,17, 11,17,23, 17,23,29
    # Distinct: {5,11,17,23,29} -> 5 distinct, all pairwise coprime? 
    # gcd(5,11)=1, gcd(5,17)=1, gcd(5,23)=1, gcd(5,29)=1, gcd(11,17)=1, gcd(11,23)=1, gcd(11,29)=1, gcd(17,23)=1, gcd(17,29)=1, gcd(23,29)=1
    # Yes! 5 distinct pairwise coprime sums.
    
    # Let's verify the seed
    seed_sums = set()
    for a in [1,7,13]:
        for b in [4,10,16]:
            seed_sums.add(a+b)
    seed_list = list(seed_sums)
    if pairwise_coprime(seed_list):
        best_count = len(seed_list)
        best_A = [1, 7, 13]
        best_B = [4, 10, 16]
    
    # Search strategy: try to find 6, 7, 8, or 9 distinct pairwise coprime sums
    # Key insight: for 9 distinct sums, we need all a_i+b_j distinct (Sidon property)
    # and all 9 pairwise coprime
    
    # Precompute: for a given set of 9 target sums (pairwise coprime), can we find A,B?
    # Alternative: enumerate A and B from a restricted pool
    
    # To make sums pairwise coprime, many sums should be prime or products of unique primes
    # Let's focus on finding sets where sums are mostly primes
    
    # Strategy: 
    # 1. Generate candidate A sets (sorted, 3 distinct in [1,128])
    # 2. For each A, generate candidate B sets
    # 3. Check if sums are pairwise coprime and count distinct
    
    # To be efficient, restrict to "nice" structures
    # Key: if A = {a, a+d1, a+d1+d2} and B = {b, b+e1, b+e1+e2}
    # Sums form a 3x3 grid
    
    # Let me try a smarter approach: 
    # Pick 9 pairwise coprime numbers in [2, 256] that can form a 3x3 sum grid
    # A 3x3 sum grid: s[i][j] = A[i] + B[j]
    # This means s[i][j] - s[i][0] = B[j] - B[0] (same for all i)
    # And s[i][0] - s[0][0] = A[i] - A[0] (same for all j)
    # So: s[i][j] = s[0][0] + (A[i]-A[0]) + (B[j]-B[0])
    # Let u_i = A[i]-A[0], v_j = B[j]-B[0]
    # s[i][j] = c + u_i + v_j where c = A[0]+B[0]
    # u_0=0, v_0=0, u_1,u_2,v_1,v_2 > 0 (assuming sorted)
    
    # So we need: c, c+u1, c+u2, c+v1, c+v2, c+u1+v1, c+u1+v2, c+u2+v1, c+u2+v2
    # all pairwise coprime, all in [2,256], and A[0]=c-B[0], etc. must be in [1,128]
    
    # Let me search over (c, u1, u2, v1, v2) with constraints
    # c >= 2, c+u2+v2 <= 256
    # A[0] = c - B[0] >= 1, B[0] >= 1, so c >= 2
    # A[2] = c - B[0] + u2 <= 128, B[2] = B[0] + v2 <= 128
    
    # Simplify: let A[0]=a0, B[0]=b0, so c=a0+b0
    # u1=A[1]-A[0], u2=A[2]-A[0], v1=B[1]-B[0], v2=B[2]-B[0]
    # All sums: a0+b0+j for various j
    
    # Let me just do a structured search
    # To keep it fast, limit the search space
    
    # Approach: iterate over possible "base" sums and offsets
    # But this could be large. Let's be smart.
    
    # For 9 pairwise coprime numbers, they should be mostly small primes or prime powers with unique primes
    # The 9 smallest primes: 2,3,5,7,11,13,17,19,23 (sum=100)
    # But they need to fit in a 3x3 additive grid
    
    # Let me try: can {2,3,5,7,11,13,17,19,23} be a 3x3 sum grid?
    # s[0][0]=2, then we need u1,v1 such that 2+u1, 2+v1 are in the set
    # This is a specific combinatorial problem
    
    # Let me just do a bounded search with the deadline guard
    # Focus on promising ranges
    
    # To get 9 distinct sums with pairwise coprime, let's search
    # But first, let's see if 6+ is achievable
    
    # Efficient search: 
    # Fix A to be "spread out" to ensure distinct sums
    # Then search B
    
    # Actually, let me think about what limits the count.
    # With 3x3=9 cells, max 9 distinct sums.
    # For all 9 to be pairwise coprime is very hard.
    # Let's aim for 6-7 distinct pairwise coprime sums.
    
    # Let me try a targeted search
    # Precompute primes up to 256
    primes = set()
    for n in range(2, 257):
        is_p = True
        for d in range(2, int(n**0.5)+1):
            if n % d == 0:
                is_p = False
                break
        if is_p:
            primes.add(n)
    
    # Search: try A sets where elements are spaced to avoid sum collisions
    # A = {a, a+p, a+p+q} with p,q > 0
    # B = {b, b+r, b+r+s} with r,s > 0
    # Sums: a+b, a+b+r, a+b+r+s, a+p+b, a+p+b+r, a+p+b+r+s, a+p+q+b, a+p+q+b+r, a+p+q+b+r+s
    # For all distinct: need p,q,r,s such that no two sums equal
    # s[i][j] = (a+b) + i*p' + j*q' roughly (if arithmetic)
    # Actually for distinct: the 9 values a+b + delta_i + epsilon_j must all differ
    # where delta_i in {0,p,p+q} and epsilon_j in {0,r,r+s}
    # Two sums equal iff delta_i + epsilon_j = delta_i' + epsilon_j' for (i,j)≠(i',j')
    # This means the sets {delta_i + epsilon_j} have 9 distinct elements
    # Equivalent to: the difference sets don't collide
    
    # Let me just brute force with limits
    # Limit A to [1, 64], B to [1, 64] to keep sums <= 128+128=256
    # But that's still 64*64*64*64*64*64 which is too large
    
    # Better: enumerate A (3-combinations from [1,128]) is C(128,3) ≈ 340k
    # For each A, enumerate B is another 340k -> too large
    
    # Smarter: 
    # 1. Generate all valid 3-element A sets (sorted)
    # 2. For each A, the sums with B depend on B
    # 3. We want sums to be pairwise coprime
    
    # Even smarter: 
    # For 9 distinct pairwise coprime sums, each sum must have a unique set of prime factors
    # Let's focus on making sums prime (easiest to be coprime)
    # If all 9 sums are distinct primes, they're automatically pairwise coprime
    
    # So: find A={a1,a2,a3}, B={b1,b2,b3} such that all a_i+b_j are distinct primes
    # This is a "prime sum grid" problem
    
    # Let me search for this
    # Fix a1, then a2=a1+d1, a3=a1+d1+d2
    # Fix b1, then b2=b1+e1, b3=b1+e1+e2
    # Sums: a1+b1, a1+b1+e1, a1+b1+e1+e2, a1+d1+b1, a1+d1+b1+e1, a1+d1+b1+e1+e2, a1+d1+d2+b1, a1+d1+d2+b1+e1, a1+d1+d2+b1+e1+e2
    
    # Let c = a1+b1
    # Sums: c, c+e1, c+e1+e2, c+d1, c+d1+e1, c+d1+e1+e2, c+d1+d2, c+d1+d2+e1, c+d1+d2+e1+e2
    # All must be prime, and a1>=1, b1>=1, a3=a1+d1+d2<=128, b3=b1+e1+e2<=128
    
    # Search over c, d1, d2, e1, e2
    # c >= 2, c+d1+d2+e1+e2 <= 256
    # a1 = c - b1, need a1 >= 1 and b1 >= 1, so c >= 2
    # a3 = c - b1 + d1 + d2 <= 128, b3 = b1 + e1 + e2 <= 128
    
    # Let me iterate over reasonable ranges
    # d1, d2, e1, e2 >= 1
    # c+d1+d2+e1+e2 <= 256
    
    found_9 = False
    found_8 = False
    found_7 = False
    found_6 = False
    
    # Try to find 9 prime sums
    # c from 2 to 200, d1 from 1 to 50, d2 from 1 to 50, e1 from 1 to 50, e2 from 1 to 50
    # But that's 200*50^4 = 1.25e9, too large
    
    # Reduce: d1+d2 <= 126, e1+e2 <= 126, c+d1+d2+e1+e2 <= 256
    # c <= 256 - d1-d2-e1-e2
    
    # Let me be more targeted
    # Try small d1,d2,e1,e2 first
    max_delta = 40
    
    for c in range(2, 250):
        if time.time() > deadline:
            break
        for d1 in range(1, max_delta):
            if time.time() > deadline:
                break
            for d2 in range(1, max_delta):
                if time.time() > deadline:
                    break
                a3_offset = d1 + d2
                if c + a3_offset > 250:
                    break
                for e1 in range(1, max_delta):
                    if time.time() > deadline:
                        break
                    for e2 in range(1, max_delta):
                        if time.time() > deadline:
                            break
                        b3_offset = e1 + e2
                        max_sum = c + a3_offset + b3_offset
                        if max_sum > 256:
                            break
                        
                        # Check if all 9 sums are prime
                        sums9 = [
                            c,
                            c + e1,
                            c + e1 + e2,
                            c + d1,
                            c + d1 + e1,
                            c + d1 + e1 + e2,
                            c + d1 + d2,
                            c + d1 + d2 + e1,
                            c + d1 + d2 + e1 + e2
                        ]
                        
                        # All must be prime
                        if all(s in primes for s in sums9):
                            # Now check if we can choose a1, b1 such that all are in [1,128]
                            # a1 = c - b1, need 1 <= a1 <= 128 and 1 <= b1 <= 128
                            # a3 = a1 + d1 + d2 = c - b1 + d1 + d2 <= 128
                            # b3 = b1 + e1 + e2 <= 128
                            # So: b1 >= 1, b1 <= c-1 (for a1>=1)
                            # b1 <= 128 - e1 - e2
                            # c - b1 <= 128 -> b1 >= c - 128
                            # c - b1 + d1 + d2 <= 128 -> b1 >= c + d1 + d2 - 128
                            
                            b1_min = max(1, c - 128, c + d1 + d2 - 128)
                            b1_max = min(c - 1, 128 - e1 - e2)
                            
                            if b1_min <= b1_max:
                                b1 = b1_min
                                a1 = c - b1
                                A = [a1, a1 + d1, a1 + d1 + d2]
                                B = [b1, b1 + e1, b1 + e1 + e2]
                                
                                # Verify
                                if all(1 <= x <= 128 for x in A + B) and len(set(A)) == 3 and len(set(B)) == 3:
                                    distinct = list(set(A[0]+B[0] for _ in [0]))  # just verify
                                    all_sums = [a+b for a in A for b in B]
                                    if len(set(all_sums)) == 9 and pairwise_coprime(list(set(all_sums))):
                                        best_A = A
                                        best_B = B
                                        best_count = 9
                                        found_9 = True
                                        # We can return immediately or continue to see if there's a better one
                                        # 9 is max, so return
                                        return [A, B]
    
    # If we didn't find 9, try to find 8 or 7 distinct pairwise coprime sums
    # This is harder to search directly. Let me try a different approach.
    
    # For 6+ distinct pairwise coprime sums, let me try a broader search
    # with the constraint that sums are pairwise coprime (not necessarily all prime)
    
    if time.time() < deadline - 5:
        # Try structured search for 6+ distinct pairwise coprime sums
        # Use the parameterization above but relax the "all prime" constraint
        # to "pairwise coprime"
        
        # To keep it tractable, limit the search
        for c in range(2, 150):
            if time.time() > deadline - 3:
                break
            for d1 in range(1, 30):
                if time.time() > deadline - 3:
                    break
                for d2 in range(1, 30):
                    if time.time() > deadline - 3:
                        break
                    a3_off = d1 + d2
                    if c + a3_off > 200:
                        break
                    for e1 in range(1, 30):
                        if time.time() > deadline - 3:
                            break
                        for e2 in range(1, 30):
                            if time.time() > deadline - 3:
                                break
                            b3_off = e1 + e2
                            max_s = c + a3_off + b3_off
                            if max_s > 256:
                                break
                            
                            sums9 = [
                                c, c+e1, c+e1+e2,
                                c+d1, c+d1+e1, c+d1+e1+e2,
                                c+d1+d2, c+d1+d2+e1, c+d1+d2+e1+e2
                            ]
                            
                            distinct_sums = list(set(sums9))
                            n_distinct = len(distinct_sums)
                            
                            if n_distinct < best_count:
                                continue
                            
                            if not pairwise_coprime(distinct_sums):
                                continue
                            
                            # Check feasibility
                            b1_min = max(1, c - 128, c + d1 + d2 - 128)
                            b1_max = min(c - 1, 128 - e1 - e2)
                            
                            if b1_min <= b1_max:
                                b1 = b1_min
                                a1 = c - b1
                                A = [a1, a1 + d1, a1 + d1 + d2]
                                B = [b1, b1 + e1, b1 + e1 + e2]
                                
                                if all(1 <= x <= 128 for x in A + B) and len(set(A)) == 3 and len(set(B)) == 3:
                                    all_sums = [a+b for a in A for b in B]
                                    ds = list(set(all_sums))
                                    if len(ds) > best_count and pairwise_coprime(ds):
                                        best_A = A
                                        best_B = B
                                        best_count = len(ds)
    
    return [best_A, best_B]
