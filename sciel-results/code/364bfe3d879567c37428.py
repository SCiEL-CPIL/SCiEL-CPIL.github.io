"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import math
    import time
    from math import gcd

    N = 512
    deadline = time.time() + 80  # safety margin

    # Precompute lcm for pairs
    # We'll build a conflict graph: edge between a,b if lcm(a,b) > N
    # Then find maximum clique (or use independent set / other approach)
    
    # Actually, let's think about this differently.
    # We want a subset S of [1, 512] such that lcm(a,b) <= 512 for all a,b in S.
    # Maximize |S|.
    
    # Key insight: If we include all numbers that divide some number <= 512... no.
    # Better: For any a in S, and any b in S, lcm(a,b) <= 512.
    # This means: for all a, b in S, a*b/gcd(a,b) <= 512.
    
    # Strategy: Greedy + local search, or use the structure.
    
    # Observation: If a number a is in S, then every divisor of a is also compatible with a.
    # But we need pairwise compatibility.
    
    # Let's think about it as: S is an independent set in the "conflict graph" where
    # edge (a,b) exists iff lcm(a,b) > 512. We want maximum independent set.
    
    # But that's NP-hard in general. However, the structure might help.
    
    # Alternative: Think in terms of prime factorizations.
    # lcm(a,b) <= 512 = 2^9. So lcm(a,b) must be a divisor of 512? No, that's not right.
    # lcm(a,b) <= 512 means the LCM is at most 512, not that it divides 512.
    # But 512 = 2^9, so any number with an odd prime factor p > 1: if a has factor p, then lcm(a,b) is at least p.
    # For lcm(a,b) <= 512, we need the LCM to not exceed 512.
    
    # Key: If a contains a prime factor p where p > 512, impossible. All primes up to 512 are fine individually.
    # But the LCM constraint is global.
    
    # Let me think about which numbers can coexist.
    # If a and b are both multiples of a large prime p, then lcm(a,b) >= p * max(a/p, b/p) ... no.
    # lcm(a,b) = a*b/gcd(a,b).
    
    # Let's just do a greedy approach with the conflict graph, then improve with local search.
    
    # First, let's identify "safe" sets.
    # All divisors of 512: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512. lcm of any two is <= 512. Size 10.
    # But we can do better by including numbers with odd factors.
    
    # For example, 3: lcm(3, 512) = 1536 > 512. So 3 and 512 can't coexist.
    # lcm(3, 256) = 768 > 512. So 3 and 256 can't coexist.
    # lcm(3, 128) = 384 <= 512. OK.
    # lcm(3, 64) = 192 <= 512. OK.
    # So 3 can coexist with 1, 2, 4, 8, 16, 32, 64, 128 but not 256 or 512.
    
    # Let me build this systematically with a greedy + local search.
    
    # Build adjacency: compatible[a][b] = lcm(a,b) <= 512
    numbers = list(range(1, N + 1))
    
    # For each pair, check compatibility
    # This is O(N^2 * log) which is fine for N=512
    
    compatible = [[False] * (N + 1) for _ in range(N + 1)]
    for a in numbers:
        for b in numbers:
            compatible[a][b] = (a * b // gcd(a, b)) <= N
    
    # Greedy: start with empty set, add numbers in some order
    # Try different orderings
    
    def greedy(order):
        S = set()
        for a in order:
            if all(compatible[a][b] for b in S):
                S.add(a)
        return S
    
    # Try multiple orderings
    best_S = set()
    
    # Order 1: ascending
    S1 = greedy(numbers)
    if len(S1) > len(best_S):
        best_S = S1
    
    # Order 2: descending
    S2 = greedy(numbers[::-1])
    if len(S2) > len(best_S):
        best_S = S2
    
    # Order 3: by number of divisors (more divisors = more flexible?)
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    
    order3 = sorted(numbers, key=lambda x: num_divisors(x), reverse=True)
    S3 = greedy(order3)
    if len(S3) > len(best_S):
        best_S = S3
    
    # Order 4: random
    import random
    random.seed(42)
    for _ in range(50):
        order4 = numbers[:]
        random.shuffle(order4)
        S4 = greedy(order4)
        if len(S4) > len(best_S):
            best_S = S4
        if time.time() > deadline:
            break
    
    # Local search: try to add more elements, or swap
    def try_improve(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            # Try to add each non-member
            for a in numbers:
                if a not in S and all(compatible[a][b] for b in S):
                    S.add(a)
                    improved = True
            if time.time() > deadline:
                break
            # Try swaps: remove one, add two?
            # For simplicity, just try single swaps
            for a in numbers:
                if a in S:
                    for b in numbers:
                        if b not in S:
                            S2 = (S - {a}) | {b}
                            if all(compatible[x][y] for x in S2 for y in S2):
                                if len(S2) > len(S):
                                    S = S2
                                    improved = True
        return S
    
    # But the swap check above is O(|S|^2) per swap, which is too slow.
    # Let's do a simpler local search: just try to add elements.
    
    def try_add_only(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            for a in numbers:
                if a not in S:
                    if all(compatible[a][b] for b in S):
                        S.add(a)
                        improved = True
                        break  # restart scan
            if time.time() > deadline:
                break
        return S
    
    best_S = try_add_only(best_S)
    
    # Now try a more systematic approach: 
    # The key constraint is lcm(a,b) <= 512.
    # For a given set S, the "bottleneck" is the maximum lcm.
    
    # Let's think about it from the perspective of prime factors.
    # 512 = 2^9. 
    # Any number a can be written as 2^k * m where m is odd.
    # lcm(a, b) = 2^max(k1,k2) * lcm(m1, m2).
    # We need 2^max(k1,k2) * lcm(m1,m2) <= 512.
    
    # If we fix the maximum power of 2 in S, say 2^M, then for any two odd parts m1, m2,
    # we need lcm(m1, m2) <= 512 / 2^M.
    
    # This suggests a structure: choose a max power of 2, then the odd parts must have small lcm.
    
    # For M=9 (max power 512): odd parts must have lcm <= 1. So only odd part = 1. Set = {512}.
    # For M=8 (max 256): odd parts lcm <= 2. Odd parts in {1, 2}? No, odd parts are odd. lcm <= 2 means both are 1. Set = {256, 512}? Wait, 512 has odd part 1. lcm(256, 512) = 512 <= 512. Yes.
    # Actually wait: if max power is 2^8, then numbers like 256 (odd part 1) and 512 (odd part 1) can coexist. But 512 = 2^9, so if we include 512, max power is 9.
    
    # Let me reconsider. Let S be our set. Let M = max_{a in S} v2(a) where v2 is the exponent of 2.
    # Then for any a, b in S: v2(lcm(a,b)) = max(v2(a), v2(b)) <= M.
    # And the odd part of lcm(a,b) is lcm(odd(a), odd(b)).
    # So lcm(a,b) = 2^max(v2(a),v2(b)) * lcm(odd(a), odd(b)) <= 2^M * lcm(odd(a), odd(b)).
    # We need 2^M * lcm(odd(a), odd(b)) <= 512 for all pairs.
    # So lcm(odd(a), odd(b)) <= 512 / 2^M for all pairs a, b in S.
    
    # Let B = 512 // 2^M. Then all odd parts in S must have pairwise lcm <= B.
    
    # For a fixed M, what's the max size of a set of odd numbers (with their 2-powers) such that pairwise lcm of odd parts <= B?
    
    # This is still complex, but let's try it computationally.
    
    # For each M from 0 to 9:
    #   B = 512 // 2^M
    #   Find max set of numbers from [1,512] with v2(a) <= M for all a, and odd(a) pairwise lcm <= B.
    
    # But we also need v2(a) <= M for all a, and the actual lcm constraint.
    # The condition "odd parts pairwise lcm <= B" plus "v2(a) <= M" is sufficient but not necessary?
    # Actually: lcm(a,b) = 2^max(v2(a),v2(b)) * lcm(odd(a),odd(b)).
    # If v2(a) <= M and v2(b) <= M, then max(v2(a),v2(b)) <= M, so 2^max <= 2^M.
    # And lcm(odd(a),odd(b)) <= B = 512/2^M.
    # So lcm(a,b) <= 2^M * B = 512. Yes, it's sufficient.
    # But is it necessary? No, because v2(a) could be less than M for some a, giving more room.
    
    # So this gives us a lower bound on achievable set sizes for each M.
    
    # Let's compute this.
    
    def max_set_for_M(M):
        B = 512 // (2 ** M)
        # Numbers with v2(a) <= M
        candidates = [a for a in range(1, 513) if a & (a - 1) == 0 or (a // (a & (-a))) <= 512 // (2 ** M)]
        # Wait, v2(a) <= M means a // (a & -a) <= 2^M? No.
        # v2(a) is the exponent of 2 in a. a // (a & -a) is the odd part.
        # v2(a) <= M means a is not divisible by 2^(M+1).
        candidates = [a for a in range(1, 513) if a % (2 ** (M + 1)) != 0 or a <= 2 ** (M + 1) - 1]
        # Simpler: v2(a) <= M iff a & (2^(M+1) - 1) ... no.
        # Just check: if a is divisible by 2^(M+1), then v2(a) >= M+1.
        candidates = [a for a in range(1, 513) if (a >> (M + 1)) == 0 or (a % (2 ** (M + 1))) != 0]
        # This is getting messy. Let me just check directly.
        candidates = []
        for a in range(1, 513):
            # v2(a)
            v = 0
            tmp = a
            while tmp % 2 == 0 and tmp > 0:
                v += 1
                tmp //= 2
            if v <= M:
                candidates.append(a)
        
        # Now, among candidates, find max subset with pairwise lcm(odd(a), odd(b)) <= B
        # This is still a combinatorial problem.
        # Greedy: sort by odd part, add if compatible.
        
        def odd_part(a):
            while a % 2 == 0:
                a //= 2
            return a
        
        from math import gcd as _gcd
        def lcm(a, b):
            return a * b // _gcd(a, b)
        
        # Greedy by odd part ascending
        candidates_sorted = sorted(candidates, key=lambda x: odd_part(x))
        S = set()
        for a in candidates_sorted:
            op = odd_part(a)
            ok = True
            for b in S:
                opb = odd_part(b)
                if lcm(op, opb) > B:
                    ok = False
                    break
            if ok:
                S.add(a)
        return S
    
    best_overall = best_S
    for M in range(0, 10):
        if time.time() > deadline:
            break
        S_M = max_set_for_M(M)
        # Verify the set
        valid = True
        S_list = list(S_M)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                if S_list[i] * S_list[j] // gcd(S_list[i], S_list[j]) > 512:
                    valid = False
                    break
            if not valid:
                break
        if valid and len(S_M) > len(best_overall):
            best_overall = S_M
    
    # Also try the pure odd-part approach more carefully
    # For each possible max odd lcm budget, find the best set
    
    # Let's also try: for each "core" odd number, build around it
    # This is getting complex. Let's just return the best we have.
    
    # Final verification
    S_final = sorted(best_overall)
    for i in range(len(S_final)):
        for j in range(i + 1, len(S_final)):
            if S_final[i] * S_final[j] // gcd(S_final[i], S_final[j]) > 512:
                # Fallback to the simple seed
                return list(range(1, 23))
    
    if not S_final:
        return list(range(1, 23))
    
    return S_final
# EVOLVE-BLOCK-END
