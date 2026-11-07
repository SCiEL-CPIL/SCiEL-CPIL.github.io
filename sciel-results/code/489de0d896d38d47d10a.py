import time
import itertools
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s out of 90s wall-clock budget

    N = 256
    # Sieve
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    num_primes = len(primes)
    
    # Prime factors as bitmask
    # Smallest primes: 2,3,5,7,11,13,17,19,23
    pf_mask = [0] * (N + 1)
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    for i, p in enumerate(small_primes):
        for j in range(p, N + 1, p):
            pf_mask[j] |= (1 << i)
    # Primes > 23 have unique large factors. For a composite, we only care about 
    # small prime factors. If a composite has a large prime factor > N/2, it cannot divide
    # the product of two distinct other members unless the product is a multiple of that prime.
    # Since the max product of two distinct members < (N/2)^2 if they are distinct and 
    # neither is N, but actually max product is (255)*(254) which is ~64k.
    # A prime factor p > 23 can be in a composite c <= N.
    # If p | c, then p must divide b*c' for c to divide b*c'.
    # Since p > 23 and p <= N/2, p can only divide b*c' if p divides b or p divides c'.
    # But p is not a factor of any b or c' that is prime (since p is not prime).
    # If p divides a composite b, then p is in pf_small(b) ? No, p is not in small_primes.
    # So if c has a large prime factor p > 23, then c cannot divide b*c' unless 
    # b or c' is a multiple of p.
    # But if b is a multiple of p, then p divides b. Since p > 23, b is a composite 
    # (or p itself, but p > 23 and we're looking at c | b*c').
    # This is complex. To be safe, we'll only include composites that are "smooth" 
    # w.r.t small_primes, or primes.
    # A composite is "smooth" if all its prime factors are in small_primes.
    
    def is_smooth_composite(c):
        if is_prime[c]: return False
        x = c
        for p in small_primes:
            while x % p == 0:
                x //= p
        return x == 1

    smooth_composites = [c for c in range(4, N + 1) if not is_prime[c] and is_smooth_composite(c)]
    
    # We'll do a direct local search on the set S.
    # Initial S: all primes.
    # Moves: add a smooth composite c if it doesn't conflict with existing S.
    # Conflict: a in S, a divides b*c for some b,c in S distinct from a.
    # Actually, if c is a smooth composite, and S contains all primes, then 
    # c divides b*d for b,d in S iff b and d contain all prime factors of c.
    # If S contains all primes, then c divides p1*p2 if c is a product of p1,p2 (with multiplicity).
    # But b,d must be distinct from c and from each other.
    # So if c = p1*p2, then b=p1, d=p2, and c divides p1*p2. Conflict!
    # So we cannot add any composite c that is a product of two primes in S.
    # More generally, c divides b*d if the prime factors of c are covered by b and d.
    
    # This means we cannot simply add all smooth composites to the set of primes.
    # We must remove some primes or carefully choose composites.
    
    # Strategy: 
    # 1. Start with all primes.
    # 2. For each smooth composite c (in descending order of size?), check if it can be added.
    #    If not, consider removing the primes that conflict with it to see if it's worth it.
    #    But removing a prime might allow adding multiple composites.
    
    # A better strategy: Local search on S.
    # State: S (frozenset for fast operations, but we need to iterate).
    # Use a set for S.
    
    best_S = set(primes)
    best_size = len(primes)
    
    # Precompute for each composite c, which primes in S are "required" to cover its factors?
    # Actually, c divides b*d iff for every prime power p^k in c, p^k divides b*d.
    # This is equivalent to: for each prime p dividing c, the exponent of p in b*d >= exponent in c.
    
    # To check if c can be added to S:
    # For all pairs (b, d) in S with b != d, b != c, d != c, check if c divides b*d.
    # This is O(|S|^2) per composite, which is ~53^2 = 2800.
    # We have ~100 smooth composites. 100 * 2800 = 280k checks. Fast.
    
    # But we also need to consider removing elements from S to add more composites.
    # Local search moves:
    # 1. Add c if valid.
    # 2. Remove x from S if it allows adding more composites (not directly checkable).
    # 3. Swap x for c: remove x, add c. Check if valid.
    
    # Let's implement a simple hill-climbing:
    # - Try to add any smooth composite c not in S.
    # - If can't add any, try to swap: for each x in S, remove x, then try to add composites.
    
    # Actually, let's first try to add as many composites as possible to the initial set of primes.
    # We can't add any composite c that is a product of two primes in S.
    # So we can only add composites that are "prime powers" like p^k where p^k > p.
    # But p^k divides p * p? No, b and d must be distinct.
    # If c = p^2, then c divides b*d iff p^2 | b*d.
    # If S contains p and another multiple of p, say q*p, then p*(q*p) = q*p^2, which is divisible by p^2.
    # So c=p^2 conflicts if S contains p and any other multiple of p.
    # Since S contains all primes, the only multiple of p in S (other than p) is a composite.
    # But S initially only has primes. So the only multiple of p in S is p itself.
    # So for c=p^2, we need b,d in S such that p^2 | b*d.
    # If S only has primes, b and d must be p and some other prime q.
    # p*q is not divisible by p^2 (since q != p).
    # So c=p^2 does NOT conflict with the set of primes!
    # Similarly, c=p^k for k>=2 does not conflict with the set of primes.
    # What about c = p*q? 
    # c divides b*d iff p*q | b*d.
    # If b=p, d=q, then p*q | p*q. Conflict!
    # So c=p*q conflicts if p,q in S.
    # What about c = p^a * q^b?
    # It conflicts if we can find b,d in S such that p^a * q^b | b*d.
    # If S contains all primes, we can pick b=p, d=q. Then b*d = p*q.
    # p^a * q^b | p*q iff a<=1 and b<=1.
    # So if c has a prime power > 1, it might not conflict with the set of primes.
    
    # Let's test this hypothesis.
    # S = primes. Add c. Check if valid.
    
    def can_add(c, S):
        # Check if c divides b*d for any distinct b,d in S.
        # If b=d is not allowed, we only check distinct pairs.
        # But if c divides b^2, that's not a conflict because b and d must be distinct.
        # So we only need to check distinct pairs (b, d).
        # Also, c itself is not in S yet.
        # We can optimize: if c is a product of two primes p,q, then b=p, d=q is a conflict.
        # More generally, if c has prime factors p1^a1 * p2^a2 * ..., 
        # then a conflict exists if we can split the factors between two distinct elements in S.
        # Since S contains all primes, we can pick b=p1, d=p2, etc.
        # If a1>1, then we need p1^2 | b*d. If we pick b=p1, then we need p1 | d.
        # Since d is a prime, d must be p1. But b and d must be distinct.
        # So we cannot pick b=p1, d=p1.
        # We can pick b=p1, d=some other multiple of p1. But S only has primes.
        # So d must be a prime. If d is a prime, d is not a multiple of p1 unless d=p1.
        # So if a1>1, we cannot satisfy p1^2 | b*d with distinct primes b,d.
        # Thus, if c has a prime power > 1, it does NOT conflict with the set of primes.
        # If c is square-free (product of distinct primes), then it conflicts with the set of primes.
        
        # So we can add all non-square-free smooth composites to the set of primes.
        # And we cannot add any square-free smooth composite.
        
        # Let's verify with a few examples.
        # c=4=2^2. Primes in S: 2,3,5,...
        # Does 4 divide b*d? b,d distinct primes.
        # If b=2, d=3, 2*3=6, 4 not divide 6.
        # If b=2, d=5, 2*5=10, 4 not divide 10.
        # So 4 can be added.
        # c=6=2*3. b=2, d=3, 2*3=6, 6 divide 6. Conflict.
        # c=12=2^2*3. b=2, d=3, 2*3=6, 12 not divide 6.
        # b=2, d=6? 6 not in S.
        # So 12 can be added.
        
        # This is a key insight! We can add all non-square-free composites to the set of primes.
        # Let's count how many non-square-free smooth composites there are.
        
        return True  # We'll do a more rigorous check below.
    
    # Let's do a rigorous check for each smooth composite.
    # S = set(primes)
    # For each c in smooth_composites:
    #   if can_add(c, S): S.add(c)
    
    # But we must be careful: adding c might conflict with another composite d already added.
    # So we should add them in an order that minimizes conflicts, or check all pairs.
    
    # Actually, if c and d are both non-square-free, can c divide d*e or e*d?
    # If c and d are in S, and we add e, we must check e against c and d.
    # Since we are adding composites one by one, we check the new composite against all existing S.
    
    # Let's implement this.
    
    S = set(primes)
    
    # Sort smooth composites by size (descending?) or just add in any order.
    # If we add a composite that conflicts with another, we might not be able to add both.
    # But if both are non-square-free, they might not conflict.
    # Let's just try to add them all and see.
    
    for c in smooth_composites:
        # Check if c can be added to S
        valid = True
        # We need to check if c divides b*d for any distinct b,d in S.
        # This is O(|S|^2). |S| is ~53 + (number of composites added so far).
        # To speed up, we can precompute for each c the set of primes that must be in b and d.
        # But for now, let's just do the O(|S|^2) check.
        # Since |S| is small (~60), and we have ~100 composites, it's fast.
        
        # Optimization: if c is square-free, it will conflict with the set of primes.
        # So we can skip square-free composites immediately.
        
        # Check if c is square-free
        x = c
        for p in small_primes:
            if (x // p) % p == 0:
                break
            x //= p
        else:
            # No p^2 divides c
            continue  # Skip square-free composites
        
        # Now check if c divides b*d for any distinct b,d in S.
        # Since c is not square-free, it might still conflict.
        # Example: c=12=2^2*3. S contains 2,3,5,...
        # Does 12 divide b*d?
        # b=2, d=6? 6 not in S.
        # b=4, d=3? 4 not in S.
        # So 12 does not conflict with S.
        
        # But what if S contains 4 and 3? Then 4*3=12, conflict.
        # But S initially only has primes. So 4 is not in S.
        # So we are safe.
        
        # However, if we add 4 to S, and then try to add 12, we must check 12 against 4.
        # 12 divides 4*3? 4*3=12, yes. So 12 conflicts with {4,3}.
        # So the order matters.
        
        # To be safe, we'll do the O(|S|^2) check.
        
        # We can optimize by noting that c divides b*d iff for each prime power p^k in c,
        # the exponent of p in b*d is >= k.
        # Since b and d are in S, and S contains primes and some composites,
        # this is a bit complex. Let's just do the direct check.
        
        # For performance, we'll only check if c is not square-free.
        # And we'll break early if a conflict is found.
        
        # Actually, we can just add all non-square-free composites and then remove conflicts.
        # But that's more complex. Let's just do the O(|S|^2) check.
        
        # To speed up, we'll precompute the prime factorization of each c.
        # But for now, let's just do the direct check.
        
        # Since |S| is ~60, and we have ~100 composites, it's 100 * 60^2 = 360k checks. Fast.
        
        # We'll do the check.
        valid = True
        S_list = list(S)
        for i in range(len(S_list)):
            b = S_list[i]
            for j in range(i + 1, len(S_list)):
                d = S_list[j]
                if (b * d) % c == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S.add(c)
    
    best_size = len(S)
    
    # Now, can we improve by removing some primes to add more composites?
    # For example, if we remove prime p, we might be able to add composites that were conflicting with p.
    # But this is a local search problem.
    
    # Let's do a few rounds of local search:
    # 1. Try to add any smooth composite not in S.
    # 2. Try to swap a prime p with a composite c.
    
    # We'll skip this for now and return the current S.
    
    # But we must ensure S is valid.
    # Let's do a final check.
    
    S_list = list(S)
    valid = True
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            for k in range(j + 1, len(S_list)):
                c = S_list[k]
                if (b * c) % a == 0:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to primes
        S = set(primes)
    
    return list(S)

# EVOLVE-BLOCK-END
