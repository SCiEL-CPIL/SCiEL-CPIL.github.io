import time
from math import gcd
from itertools import combinations
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    N = 512

    # 1. Precomputation
    # Sieve for primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return is_prime

    is_prime = sieve(511)
    primes = [p for p in range(2, 512) if is_prime[p]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Prime factor masks for each number 1..511
    masks = [0] * 512
    for a in range(2, 512):
        m = a
        mask_val = 0
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                mask_val |= (1 << prime_to_idx[p])
                while m % p == 0:
                    m //= p
        if m > 1:
            # m is a prime
            mask_val |= (1 << prime_to_idx[m])
        masks[a] = mask_val
    masks[1] = 0 # 1 is coprime to everything, no prime factors

    # Weights: use Fraction for exact arithmetic in B&B to avoid floating point issues
    # But for speed, we can use float for comparisons if we are careful, or stick to Fraction.
    # Given the small size, Fraction is likely fast enough and safer.
    # However, the task says "reported as a float", so we just need the final value.
    # Let's use float for speed in the heuristic part, and maybe exact for B&B?
    # Actually, let's stick to float for speed but with high precision?
    # The differences are small. Let's use Fraction for the B&B part to ensure correctness.
    # Or just use float and be careful. The current program uses float and got 1.0400.
    # The improvement to 1.0410 came from "Exact optimization via Branch-and-Bound".
    # Let's use float but with a very tight upper bound.

    weights = [0.0] * 512
    for a in range(1, 512):
        weights[a] = 1.0 / (512 - a)
        
    # Candidates sorted by weight descending
    candidates = list(range(511, 0, -1))
    candidates.sort(key=lambda x: weights[x], reverse=True)
    
    # Precompute for each prime, the list of candidates divisible by it, sorted by weight desc
    prime_cands = [[] for _ in range(num_primes)]
    for a in candidates:
        m = masks[a]
        temp = m
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            prime_cands[idx].append(a)
            temp ^= lsb

    # 2. Greedy Initial Solution
    def greedy_solve():
        used_primes = 0
        score = 0.0
        sol = []
        for a in candidates:
            if (masks[a] & used_primes) == 0:
                sol.append(a)
                used_primes |= masks[a]
                score += weights[a]
        return sol, score

    initial_sol, initial_score = greedy_solve()
    best_score = initial_score
    best_sol = initial_sol[:]
    
    # 3. Improved Branch and Bound
    # State: used_primes (int), current_score (float), avail_mask (int, 512 bits)
    # We will use a more aggressive upper bound:
    # The upper bound is the sum of the weights of the best available candidates
    # that are pairwise coprime with each other and with the current used_primes.
    # This is the "Greedy Coloring" or "Greedy Independent Set" upper bound.
    
    def calculate_ub(used_primes, avail_mask):
        """
        Greedy upper bound:
        1. Find all candidates in avail_mask that are compatible with used_primes.
        2. Sort them by weight descending.
        3. Greedily select a maximal independent set from these candidates.
        4. Return the sum of their weights.
        """
        # Collect compatible candidates
        compatible = []
        temp = avail_mask
        while temp:
            lsb = temp & -temp
            a = lsb.bit_length() - 1
            if (masks[a] & used_primes) == 0:
                compatible.append(a)
            temp ^= lsb
        
        if not compatible:
            return 0.0
            
        # Sort by weight descending (they are already mostly sorted if avail_mask is built carefully,
        # but let's be safe. Actually, we can just iterate 'candidates' which is sorted)
        # Let's just filter 'candidates' instead of rebuilding from mask for speed?
        # No, iterating the mask is O(N). Filtering 'candidates' is O(N).
        # Let's just use the 'compatible' list and sort it.
        compatible.sort(key=lambda x: weights[x], reverse=True)
        
        ub_score = 0.0
        used_in_ub = 0
        for a in compatible:
            if (masks[a] & used_in_ub) == 0:
                ub_score += weights[a]
                used_in_ub |= masks[a]
                
        return ub_score

    # BNB Search
    # To make it faster, we can use a priority queue (Best-First Search) or just DFS with good pruning.
    # DFS is simpler and often sufficient for this size if the UB is tight.
    # Let's use DFS.
    
    # Optimization: Precompute the "next" compatible candidate for a given used_primes and avail_mask?
    # That's hard. Let's stick to the simple loop but optimize the inner loop.
    
    # One key optimization: If we pick a pivot, we should pick the one with the highest weight.
    # The branching factor is reduced if we pick high weight items first.
    
    def bnb_search(used_primes, current_score, avail_mask, sol):
        nonlocal best_score, best_sol
        
        if time.time() > deadline - 1:
            return
            
        if current_score > best_score:
            best_score = current_score
            best_sol = sol[:]
            
        if avail_mask == 0:
            return
            
        # Calculate Upper Bound
        # This is the most expensive part.
        ub = calculate_ub(used_primes, avail_mask)
        if current_score + ub <= best_score + 1e-9:
            return
            
        # Select a pivot candidate
        # Choose the highest weight candidate from avail_mask that is compatible with used_primes
        pivot = -1
        # Iterate through candidates sorted by weight
        for a in candidates:
            if not (avail_mask & (1 << a)):
                continue
            if masks[a] & used_primes:
                continue
            pivot = a
            break
            
        if pivot == -1:
            return
            
        # Branch 1: Include pivot
        new_used = used_primes | masks[pivot]
        new_avail = avail_mask & ~(1 << pivot)
        
        # Optimization: We can prune 'new_avail' by removing all candidates that conflict with pivot.
        # This makes the next UB calculation tighter and faster.
        # A candidate b conflicts with pivot if (masks[b] & masks[pivot]) != 0
        # Let's do this pruning.
        temp = new_avail
        while temp:
            lsb = temp & -temp
            b = lsb.bit_length() - 1
            if (masks[b] & masks[pivot]) != 0:
                new_avail ^= lsb
            temp ^= lsb
            
        new_sol = sol + [pivot]
        bnb_search(new_used, current_score + weights[pivot], new_avail, new_sol)
        
        if time.time() > deadline - 1:
            return

        # Branch 2: Exclude pivot
        new_avail_excl = avail_mask & ~(1 << pivot)
        bnb_search(used_primes, current_score, new_avail_excl, sol)

    # Run BNB
    # Start with empty set
    full_avail_mask = 0
    for a in range(1, 512):
        full_avail_mask |= (1 << a)
        
    bnb_search(0, 0.0, full_avail_mask, [])
    
    return best_sol
