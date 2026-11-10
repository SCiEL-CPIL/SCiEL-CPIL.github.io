import itertools
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s kill

    # Precompute primes up to 256
    max_val = 256
    is_prime = [True] * (max_val + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(max_val**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False

    # Precompute prime partners for each a in [1,128]
    prime_partners = {}
    for a in range(1, 129):
        partners = set()
        for b in range(1, 129):
            if is_prime[a + b]:
                partners.add(b)
        prime_partners[a] = partners

    # Search for A (odd) and B (even) such that all 9 sums are prime and distinct
    odds = [i for i in range(1, 129) if i % 2 != 0]
    
    best_solution = None
    
    for A in itertools.combinations(odds, 3):
        if time.time() > deadline:
            break
        
        # Intersection of prime partners
        valid_bs = prime_partners[A[0]]
        for a in A[1:]:
            valid_bs = valid_bs.intersection(prime_partners[a])
        
        if len(valid_bs) < 3:
            continue
        
        # Try combinations of 3 from valid_bs
        for B in itertools.combinations(sorted(valid_bs), 3):
            sums = []
            distinct = True
            seen = set()
            for a in A:
                for b in B:
                    s = a + b
                    if s in seen:
                        distinct = False
                        break
                    seen.add(s)
                    sums.append(s)
                if not distinct:
                    break
            
            if distinct and len(sums) == 9:
                # All sums are prime (by construction), so pairwise coprime
                # Verify just to be safe
                ok = True
                for i in range(9):
                    for j in range(i+1, 9):
                        if math.gcd(sums[i], sums[j]) != 1:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    best_solution = [list(A), list(B)]
                    break
        
        if best_solution:
            break

    if best_solution:
        return best_solution
    
    # Fallback: hardcoded valid solution with 9 distinct prime sums
    # A=[1, 3, 5], B=[2, 4, 6] -> sums: 3,5,7, 5,7,9, 7,9,11 -> not all distinct
    # Let me compute a valid one: A=[1,5,9], B=[2,8,14]
    # sums: 3,9,15, 7,13,19, 11,17,23 -> 3,9,15,7,13,19,11,17,23
    # 9 and 15 share gcd 3. Not good.
    # A=[1,7,13], B=[2,8,14]: sums 3,9,15,9,15,21,15,21,27 - collisions
    # A=[1,3,11], B=[2,10,18]: sums 3,11,19, 5,13,21, 13,21,29 - collisions (11,13,19,5,21,29)
    # Let me just use a verified one from the search logic above
    # A=[1, 9, 17], B=[2, 10, 18]: sums 3,11,19, 11,19,27, 19,27,35 - collisions
    # A=[1, 5, 17], B=[2, 8, 14]: sums 3,9,15, 7,13,19, 19,23,31 - 19 appears twice
    # A=[1, 7, 19], B=[2, 6, 10]: sums 3,7,11, 9,13,17, 21,25,29 - all distinct!
    # Check pairwise coprime: 3,7,11,9,13,17,21,25,29
    # gcd(9,21)=3 -> NOT coprime!
    # A=[1, 5, 19], B=[2, 6, 10]: sums 3,7,11, 7,11,15, 21,25,29 - 7,11 appear twice
    # A=[1, 7, 19], B=[4, 10, 16]: sums 5,11,17, 11,17,23, 23,29,35 - 11,17,23 appear twice
    # A=[1, 7, 23], B=[2, 10, 18]: sums 3,11,19, 9,17,25, 25,33,41 - 25 appears twice
    # A=[1, 7, 23], B=[4, 12, 20]: sums 5,13,21, 11,19,27, 27,35,43 - 27 appears twice
    # A=[1, 7, 23], B=[2, 14, 26]: sums 3,15,27, 9,21,33, 25,37,49 - 3,15,27,9,21,33,25,37,49
    # gcd(3,15)=3. Not coprime.
    # A=[1, 7, 23], B=[6, 14, 22]: sums 7,15,23, 13,21,29, 29,37,45 - 29 appears twice
    # A=[1, 7, 23], B=[4, 16, 28]: sums 5,17,29, 11,23,35, 27,39,51 - 5,17,29,11,23,35,27,39,51
    # gcd(5,35)=5. Not coprime.
    # A=[1, 7, 23], B=[2, 16, 30]: sums 3,17,31, 9,23,37, 25,39,53 - 3,17,31,9,23,37,25,39,53
    # gcd(3,9)=3. Not coprime.
    # A=[1, 7, 23], B=[4, 18, 32]: sums 5,19,33, 11,25,39, 27,41,55 - 5,19,33,11,25,39,27,41,55
    # gcd(5,25)=5. Not coprime.
    # A=[1, 7, 23], B=[2, 18, 34]: sums 3,19,35, 9,25,41, 25,41,57 - 25,41 appear twice
    # A=[1, 7, 23], B=[6, 18, 30]: sums 7,19,31, 13,25,37, 29,41,53 - 7,19,31,13,25,37,29,41,53
    # Check: 7,19,31,13,25,37,29,41,53 - all primes except 25=5^2
    # gcd(7,19)=1, gcd(7,31)=1, gcd(7,13)=1, gcd(7,25)=1, gcd(7,37)=1, gcd(7,29)=1, gcd(7,41)=1, gcd(7,53)=1
    # gcd(19,31)=1, gcd(19,13)=1, gcd(19,25)=1, gcd(19,37)=1, gcd(19,29)=1, gcd(19,41)=1, gcd(19,53)=1
    # gcd(31,13)=1, gcd(31,25)=1, gcd(31,37)=1, gcd(31,29)=1, gcd(31,41)=1, gcd(31,53)=1
    # gcd(13,25)=1, gcd(13,37)=1, gcd(13,29)=1, gcd(13,41)=1, gcd(13,53)=1
    # gcd(25,37)=1, gcd(25,29)=1, gcd(25,41)=1, gcd(25,53)=1
    # gcd(37,29)=1, gcd(37,41)=1, gcd(37,53)=1
    # gcd(29,41)=1, gcd(29,53)=1
    # gcd(41,53)=1
    # ALL pairwise coprime! And all 9 distinct!
    # A=[1,7,23], B=[6,18,30] is valid with 9 distinct pairwise-coprime sums.
    return [[1, 7, 23], [6, 18, 30]]
