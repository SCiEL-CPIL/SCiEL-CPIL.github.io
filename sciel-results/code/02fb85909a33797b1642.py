import time
import sys
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit

    N = 256
    candidates = list(range(2, N + 1))
    num_candidates = len(candidates)
    
    # Map value to index for quick lookup
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Precompute prime factorization or just check divisibility?
    # Divisibility a | b*c is equivalent to:
    # For every prime p, v_p(a) <= v_p(b) + v_p(c).
    # This is fast if we precompute prime factors.
    
    # Let's precompute the prime factorization of each number in candidates.
    # factors[i] is a dict {prime: exponent}
    factors = [None] * num_candidates
    for i, v in enumerate(candidates):
        x = v
        f = {}
        d = 2
        while d * d <= x:
            while x % d == 0:
                f[d] = f.get(d, 0) + 1
                x //= d
            d += 1
        if x > 1:
            f[x] = f.get(x, 0) + 1
        factors[i] = f

    # Helper: check if a | b*c using factorizations
    # idx_a, idx_b, idx_c are indices into candidates
    def divides_product(idx_a, idx_b, idx_c):
        fa = factors[idx_a]
        fb = factors[idx_b]
        fc = factors[idx_c]
        
        # Check for each prime in fa
        for p, exp_a in fa.items():
            exp_bc = fb.get(p, 0) + fc.get(p, 0)
            if exp_a > exp_bc:
                return False
        return True

    # can_add(idx_new, S_indices): 
    # S_indices is a list of indices of elements currently in S.
    # We need to check if adding idx_new creates any violation.
    # A violation exists if there exist distinct i, j in S_indices such that:
    # 1. divides_product(idx_new, i, j) -> new | c_i * c_j
    # 2. divides_product(i, idx_new, j) -> c_i | new * c_j
    # 3. divides_product(j, i, idx_new) -> c_j | c_i * new
    # Note: Since i, j, new are distinct, these cover all permutations where new is involved.
    
    def can_add(idx_new, S_indices):
        n = len(S_indices)
        for i_idx in range(n):
            i = S_indices[i_idx]
            for j_idx in range(i_idx + 1, n):
                j = S_indices[j_idx]
                if divides_product(idx_new, i, j):
                    return False
                if divides_product(i, idx_new, j):
                    return False
                if divides_product(j, i, idx_new):
                    return False
        return True

    def is_valid_set(S_indices):
        n = len(S_indices)
        for i_idx in range(n):
            i = S_indices[i_idx]
            for j_idx in range(i_idx + 1, n):
                j = S_indices[j_idx]
                for k_idx in range(j_idx + 1, n):
                    k = S_indices[k_idx]
                    if divides_product(i, j, k):
                        return False
        return True

    best_S = []
    best_size = 0

    # Strategy: 
    # 1. Greedy construction with multiple random orderings.
    # 2. Local search with swaps to escape local optima.
    # 3. "Pruning" candidates that are "too dangerous" early on?
    #    Actually, small numbers are more likely to divide products.
    #    So maybe we should try to include as many large numbers as possible first.
    
    # Let's try a specific heuristic ordering:
    # Sort by number of divisors in the set? No, dynamic.
    # Sort by value descending? Large numbers are less likely to divide b*c.
    # But small numbers are less likely to be divided by? No, the condition is a | b*c.
    # If a is small, it's easier for a | b*c. So small 'a' is "bad" (restrictive).
    # If a is large, it's harder for a | b*c. So large 'a' is "good" (less restrictive).
    # Therefore, we should prioritize adding large numbers.
    
    # However, if we add large numbers, do they restrict others?
    # If we have large L1, L2, can we add small S?
    # We need S | L1*L2 (likely true if S is small and L1,L2 are multiples or contain factors)
    # and L1 | S*L2 (unlikely if L1 is large and S, L2 are smaller? Not necessarily)
    # and L2 | S*L1 (unlikely)
    
    # So large numbers are "safe" as the divisor 'a'.
    # But they might be "dangerous" if they are the product side? No, 'a' is the divisor.
    # The constraint is symmetric in the sense that any element can be 'a'.
    # So if we have a set of large numbers, adding a small number S requires:
    # S | L_i * L_j for all pairs (likely true)
    # L_i | S * L_j for all pairs (This is the tight constraint. If L_i has a prime factor p with high exponent, and S, L_j don't have enough p, it fails.)
    
    # So the strategy of "Large first" is good because it establishes a set where the "divisor" role is played by large numbers, which is safe.
    # Then adding small numbers is risky because the small number becomes 'a' (safe) but the large numbers become 'b' or 'c' and might fail the divisibility check if the small number 'a' is actually the one being checked?
    # Wait.
    # If S is small, and L_i is large.
    # Check: L_i | S * L_j.
    # If L_i = 254 = 2 * 127. S = 2. L_j = 127.
    # 254 | 2 * 127 = 254. Yes! Violation.
    # So if we have 254 and 127, we cannot have 2.
    # So large composite numbers with small prime factors are dangerous to include if we also want small numbers.
    
    # Better strategy:
    # 1. Start with all primes > 128? 
    #    If a is prime p > 128, p | b*c implies p|b or p|c.
    #    Since b,c < 256, and p > 128, the only multiples of p in 2..256 are p itself and 2p (if 2p <= 256).
    #    If b,c are distinct from p, and b,c < 2p, then b,c cannot be multiples of p.
    #    So p ∤ b*c.
    #    Thus, any set of distinct primes > 128 is valid.
    #    How many? Primes in (128, 256]:
    #    131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251.
    #    Count: ~23.
    
    # 2. Can we add more?
    #    What about primes in (64, 128]?
    #    If p is in (64, 128], multiples are p, 2p, 3p (if 3p<=256).
    #    If we have p, 2p, 3p in the set, then p | 2p * 3p? No, p is the divisor.
    #    a=p, b=2p, c=3p. a | b*c => p | 6p^2. Yes.
    #    So we can't have p, 2p, 3p together.
    #    But we can have p and 2p?
    #    a=p, b=2p, c=x. p | 2p*x. Yes, always.
    #    So if p and 2p are in the set, ANY third element x will cause a violation?
    #    Wait. The condition is for EVERY three distinct members a,b,c.
    #    If we have p and 2p, and we add x.
    #    Triple (p, 2p, x): p | 2p * x. True. Violation.
    #    So we CANNOT have both p and 2p in the set if the set has size >= 3.
    #    Actually, if the set has size >= 3, any pair (p, 2p) in the set will form a violating triple with any other element x.
    #    So, for a set of size >= 3, we cannot contain both p and 2p for any prime p.
    #    Similarly, we cannot contain p and 3p if 3p <= 256?
    #    a=p, b=3p, c=x. p | 3p*x. True.
    #    So we cannot contain p and kp for any k >= 2 if the set size is >= 3.
    
    # This is a strong constraint!
    # For any prime p, we can include at most ONE multiple of p in the set (if size >= 3).
    # Wait, is that true?
    # If we include m1*p and m2*p.
    # Let a = m1*p, b = m2*p, c = x.
    # Does a | b*c?
    # m1*p | m2*p*x => m1 | m2*x.
    # This is not always true.
    # Example: p=2. m1=3 (6), m2=5 (10).
    # a=6, b=10, c=x.
    # 6 | 10*x => 3 | 5*x => 3 | x.
    # So if x is not a multiple of 3, 6 ∤ 10*x.
    # But we must check ALL triples.
    # If the set contains 6, 10, and 7.
    # Triple (6, 10, 7): 6 | 70? No. 10 | 42? No. 7 | 60? No.
    # So {6, 10, 7} is valid!
    # So my deduction "cannot contain p and 2p" was based on a=p, b=2p.
    # a=p, b=2p. p | 2p*x is ALWAYS true.
    # So if the set contains p and 2p, then for ANY x, the triple (p, 2p, x) is a violation.
    # Therefore, if |S| >= 3, S cannot contain both p and 2p for any prime p.
    # Similarly, S cannot contain p and kp for any k>=2?
    # a=p, b=kp. p | kp*x is always true.
    # Yes.
    # So, for each prime p, the set S can contain at most one number that is a multiple of p?
    # NO. It can contain 2p and 3p.
    # a=2p, b=3p. 2p | 3p*x => 2 | 3x => 2 | x.
    # So if x is even, violation. If x is odd, no violation from this pair.
    # But we must check all triples.
    
    # So the constraint "At most one multiple of p" is FALSE.
    # The constraint "At most one number of the form p * (integer)" is FALSE.
    # The constraint "Cannot contain p and kp" is TRUE (for k>=2).
    
    # This suggests that including small primes is very restrictive.
    # If we include 2, we cannot include 4, 6, 8, ... 254.
    # We can only include numbers NOT divisible by 2 (odd numbers) along with 2.
    # And for those odd numbers, we have similar constraints with other primes.
    
    # So, a good strategy might be:
    # 1. Choose a set of "base" primes.
    # 2. For each base prime p, we can include p itself.
    # 3. We can also include multiples of p, but we must be careful not to include p itself if we include multiples?
    #    Actually, if we include p, we can't include 2p, 3p, etc.
    #    So if we want to include many numbers, maybe we should AVOID the primes themselves and use their multiples?
    #    Or include the primes and use numbers coprime to them?
    
    # Let's stick to the Greedy + Local Search approach but with better heuristics.
    # Heuristic 1: Sort candidates by "danger".
    # Danger of x: How many pairs (b,c) does x divide?
    # If x divides many pairs, it's dangerous to add.
    # Calculate danger(x) = count of pairs (b,c) in candidates such that x | b*c.
    # Add elements with lowest danger first?
    # Or highest?
    # If x is dangerous, adding it early might block many others.
    # If x is safe, adding it is good.
    # So add SAFEST first.
    
    # Let's precompute danger for each candidate.
    danger = [0] * num_candidates
    for i in range(num_candidates):
        x = candidates[i]
        # Count pairs (j,k) such that x | candidates[j] * candidates[k]
        # This is O(N^2) per x, so O(N^3) total. 255^3 ~ 16 million. Acceptable in Python? Maybe slow.
        # Optimization: Only check j < k.
        # We can speed this up by checking if x | b*c.
        # Let's just do it.
        cnt = 0
        for j in range(num_candidates):
            b = candidates[j]
            for k in range(j + 1, num_candidates):
                c = candidates[k]
                if (b * c) % x == 0:
                    cnt += 1
        danger[i] = cnt
    
    # Sort indices by danger ascending (safest first)
    sorted_indices = sorted(range(num_candidates), key=lambda i: danger[i])
    
    # Greedy construction
    def greedy_construct(order):
        S = []
        for idx in order:
            if can_add(idx, S):
                S.append(idx)
        return S

    # Local Search: Swap
    def local_search(S_indices):
        S_set = set(S_indices)
        not_in_S = [i for i in range(num_candidates) if i not in S_set]
        
        # Try to add
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try additions
            for idx in sorted(not_in_S, key=lambda i: danger[i]):
                if can_add(idx, S_indices):
                    S_indices.append(idx)
                    S_set.add(idx)
                    not_in_S.remove(idx)
                    improved = True
                    break
            
            if improved:
                continue
                
            # Try swaps
            # Randomly pick i in S, j not in S
            if S_indices and not_in_S:
                for _ in range(10): # Limited swap attempts
                    if time.time() > deadline - 5: break
                    i = random.choice(S_indices)
                    j = random.choice(not_in_S)
                    
                    # Check if swap is valid
                    S_minus_i = [x for x in S_indices if x != i]
                    if can_add(j, S_minus_i):
                        # Perform swap
                        S_indices = S_minus_i + [j]
                        S_set.remove(i)
                        S_set.add(j)
                        not_in_S.remove(j)
                        not_in_S.append(i)
                        improved = True
                        break
        
        return S_indices

    # Run multiple restarts with different random orders
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Create a random order, but biased towards safe elements
        # Mix of random and sorted
        if r % 2 == 0:
            order = sorted_indices[:]
        else:
            order = sorted_indices[:]
            random.shuffle(order)
            # Or just random
            # order = list(range(num_candidates))
            # random.shuffle(order)
            
        S = greedy_construct(order)
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S[:]
            
    # Convert indices back to values
    result = [candidates[i] for i in best_S]
    
    # Final validation check
    if not is_valid_set(best_S):
        # Fallback to a known valid set (e.g., large primes)
        fallback = []
        for i in range(num_candidates):
            if candidates[i] > 128:
                if can_add(i, fallback):
                    fallback.append(i)
        return [candidates[i] for i in fallback]
        
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
