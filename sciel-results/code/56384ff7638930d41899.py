"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    N = 512
    # Candidates: a in 1..511
    # Score for a: 1/(512 - a)
    
    # Get all primes < 512
    def get_primes(n):
        sieve = [True] * n
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(N)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Precompute prime factor masks for each candidate
    # mask[a] is an integer bitmask where bit i is 1 if primes[i] divides a
    masks = [0] * N
    scores = [0.0] * N
    for a in range(1, N):
        n = a
        mask = 0
        for i, p in enumerate(primes):
            if p * p > n:
                break
            if n % p == 0:
                mask |= (1 << i)
                while n % p == 0:
                    n //= p
        if n > 1:
            # n is a prime factor
            if n in prime_to_idx:
                mask |= (1 << prime_to_idx[n])
        masks[a] = mask
        scores[a] = 1.0 / (N - a)
        
    # Greedy construction
    # Sort candidates by score descending (a descending)
    candidates = list(range(N-1, 0, -1))
    
    chosen = []
    chosen_set = set()
    used_mask = 0
    
    for a in candidates:
        if time.time() > deadline:
            break
        # Check if a is coprime to all chosen (i.e., mask[a] & used_mask == 0)
        if masks[a] & used_mask == 0:
            chosen.append(a)
            chosen_set.add(a)
            used_mask |= masks[a]
            
    current_score = sum(scores[a] for a in chosen)
    
    # Local Search: 1-swap and 2-swap
    # We will iterate until no improvement or time limit
    
    # Helper to check if a list of new elements is valid against a base mask
    def is_valid_addition(new_elements, base_mask):
        # new_elements must be pairwise coprime and coprime to base_mask
        local_mask = 0
        for e in new_elements:
            m = masks[e]
            if m & local_mask:
                return False
            if m & base_mask:
                return False
            local_mask |= m
        return True

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1-Swap
        # For each a in chosen, try to replace with b not in chosen
        # Optimization: Precompute list of non-chosen
        non_chosen = [a for a in range(N-1, 0, -1) if a not in chosen_set]
        
        for i, a in enumerate(chosen):
            if time.time() > deadline:
                break
            mask_without_a = used_mask ^ masks[a] # This works if a's mask is subset of used_mask and disjoint from others? 
            # Wait, used_mask is OR of all masks. 
            # If two elements share a prime, they wouldn't be in the set. So masks are disjoint.
            # So used_mask ^ masks[a] is indeed the mask of all other elements.
            
            best_b = -1
            best_gain = 0.0
            
            for b in non_chosen:
                if masks[b] & mask_without_a == 0:
                    gain = scores[b] - scores[a]
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b
            
            if best_b != -1 and best_gain > 1e-12:
                # Perform swap
                chosen[i] = best_b
                chosen_set.remove(a)
                chosen_set.add(best_b)
                used_mask = mask_without_a | masks[best_b]
                current_score += best_gain
                improved = True
                non_chosen.remove(best_b)
                non_chosen.append(a)
                break # Restart loop with new state
        
        if improved:
            continue # Restart while loop to re-evaluate 1-swaps with new state
            
        # 2-Swap
        # Try to replace pair (a, c) with pair (b, d)
        # This is O(N^4) in worst case, but we can prune.
        # We only care if scores[b]+scores[d] > scores[a]+scores[c]
        
        # To make this faster, we can just do a few random 2-swaps or limit the search.
        # Given the small size, let's try a structured search but with early termination.
        
        # Actually, let's just do 1-swap repeatedly until convergence, then try 2-swaps.
        # 2-swap logic:
        found_2swap = False
        n_chosen = len(chosen)
        
        # Iterate over pairs in chosen
        for i in range(n_chosen):
            if time.time() > deadline:
                break
            a = chosen[i]
            for j in range(i+1, n_chosen):
                c = chosen[j]
                mask_without_ac = used_mask ^ masks[a] ^ masks[c]
                target_score = scores[a] + scores[c]
                
                # Look for b, d in non_chosen
                # Optimization: Sort non_chosen by score descending, break early if sum of top 2 <= target_score
                # But non_chosen changes. 
                # Let's just iterate.
                best_pair = None
                best_gain = 0.0
                
                # To avoid O(N^2) for every pair, we can limit the search for b,d to top K non-chosen
                # Or just check all. 511^2 is ~260k. 20^2 pairs in chosen is 190. 190 * 260k is 50M. Might be too slow in Python.
                # Let's limit the search for b,d to the top 50 non-chosen elements by score.
                top_non_chosen = sorted(non_chosen, key=lambda x: scores[x], reverse=True)[:50]
                
                for b_idx, b in enumerate(top_non_chosen):
                    if masks[b] & mask_without_ac != 0:
                        continue
                    for d in top_non_chosen[b_idx+1:]:
                        if masks[d] & mask_without_ac != 0:
                            continue
                        if masks[b] & masks[d] != 0:
                            continue
                        gain = scores[b] + scores[d] - target_score
                        if gain > best_gain:
                            best_gain = gain
                            best_pair = (b, d)
                            
                if best_pair is not None and best_gain > 1e-12:
                    b, d = best_pair
                    # Perform swap
                    chosen[i] = b
                    chosen[j] = d
                    chosen_set.remove(a)
                    chosen_set.remove(c)
                    chosen_set.add(b)
                    chosen_set.add(d)
                    used_mask = mask_without_ac | masks[b] | masks[d]
                    current_score += best_gain
                    improved = True
                    found_2swap = True
                    # Update non_chosen
                    non_chosen.remove(b)
                    non_chosen.remove(d)
                    non_chosen.append(a)
                    non_chosen.append(c)
                    break # Restart while loop
            if found_2swap:
                break
                
    return chosen
# EVOLVE-BLOCK-END
