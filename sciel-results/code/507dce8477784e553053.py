"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    N = 512
    # We want to select a subset of {1, ..., 511} that is pairwise coprime
    # maximizing sum_{a in S} 1/(512 - a)
    
    # The weight 1/(512-a) is largest for a=511 (weight=1), a=510 (weight=1/2), etc.
    # So we prioritize larger a values.
    
    # Greedy approach with backtracking on top candidates
    # Start with a greedy solution, then try to improve
    
    # Greedy: pick candidates in order of decreasing weight (a from 511 down to 1)
    # if they are coprime to all already picked
    
    def greedy_solution():
        chosen = []
        for a in range(511, 0, -1):
            if all(gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
        return chosen
    
    best_set = greedy_solution()
    best_score = sum(1.0 / (N - a) for a in best_set)
    
    # Now try local search: try to swap out low-weight elements for high-weight ones
    # or add elements that fit
    
    # Try adding elements not in best_set
    current_set = set(best_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add any element not in current_set
        for a in range(511, 0, -1):
            if a not in current_set:
                if all(gcd(a, b) == 1 for b in current_set):
                    new_score = best_score + 1.0 / (N - a)
                    if new_score > best_score:
                        current_set.add(a)
                        best_score = new_score
                        improved = True
    
    # Try swapping: remove a low-weight element, see if we can add two high-weight ones
    # This is more complex, let's try a different approach
    
    # Better: use a priority-based search
    # The optimal solution likely contains many of the largest numbers
    # Let's try a more systematic approach with limited backtracking
    
    # Process candidates in order of decreasing weight
    # For each, decide include or exclude, but prune aggressively
    
    # Precompute which primes divide each candidate
    def get_primes(n):
        primes = set()
        d = 2
        while d * d <= n:
            if n % d == 0:
                primes.add(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            primes.add(n)
        return primes
    
    candidates = list(range(511, 0, -1))
    cand_primes = [get_primes(a) for a in candidates]
    cand_weights = [1.0 / (N - a) for a in candidates]
    
    # Suffix sum of weights for upper bound
    suffix = [0.0] * (len(candidates) + 1)
    for i in range(len(candidates) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + cand_weights[i]
    
    best_score_bb = best_score
    best_set_bb = list(current_set)
    
    def bb(idx, chosen, score, used_primes):
        nonlocal best_score_bb, best_set_bb
        
        if time.time() > deadline:
            return
        
        # Upper bound
        if score + suffix[idx] <= best_score_bb:
            return
        
        if idx == len(candidates):
            if score > best_score_bb:
                best_score_bb = score
                best_set_bb = list(chosen)
            return
        
        a = candidates[idx]
        
        # Try including a
        if not (cand_primes[idx] & used_primes):
            chosen.append(a)
            bb(idx + 1, chosen, score + cand_weights[idx], used_primes | cand_primes[idx])
            chosen.pop()
        
        # Try excluding a
        bb(idx + 1, chosen, score, used_primes)
    
    # Start branch and bound from the greedy solution's perspective
    # Actually, let's just run a fresh branch and bound with the greedy as initial bound
    
    bb(0, [], 0.0, set())
    
    if best_score_bb > best_score:
        best_set = best_set_bb
        best_score = best_score_bb
    
    return best_set
# EVOLVE-BLOCK-END
