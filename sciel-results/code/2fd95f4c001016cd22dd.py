"""Improved program: greedy pairwise coprime selection with merge-based local search."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 140  # 140s safety margin (150s kill)
    
    N = 512
    # All primes below 512
    def primes_below(n):
        sieve = [True] * n
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n, i):
                    sieve[j] = False
        return [i for i in range(n) if sieve[i]]
    
    P = primes_below(N)
    # Split: large primes (close to N) and small primes
    # Value of a number a is 1/(N - a), so larger a = higher value
    # But we need pairwise coprime
    
    # Strategy:
    # 1. Start with all primes as individual candidates
    # 2. Greedily select highest-value candidates that are coprime to all selected
    # 3. Then try to merge small primes into composites to free up coprime slots
    #    and potentially get higher total value
    
    # Phase 1: Greedy selection from primes (highest value first)
    # Value of prime p is 1/(N - p), so sort by p descending
    P_sorted = sorted(P, key=lambda p: N - p)  # closest to N first
    
    def is_coprime_to_all(a, chosen):
        for b in chosen:
            if gcd(a, b) != 1:
                return False
        return True
    
    chosen = []
    for p in P_sorted:
        if is_coprime_to_all(p, chosen):
            chosen.append(p)
    
    def score(set_a):
        total = 0.0
        for a in set_a:
            total += 1.0 / (N - a)
        return total
    
    best_set = chosen[:]
    best_score = score(best_set)
    
    # Phase 2: Local search - try swapping in composites
    # For pairs of small primes p, q that are in chosen, check if p*q < N
    # and if replacing {p, q} with {p*q} gives better score and is coprime to rest
    # Also try replacing one prime p with composite p*q where q is not in chosen
    # but p*q < N and p*q is coprime to all others
    
    # Build set of chosen primes for fast lookup
    chosen_set = set(chosen)
    
    # Find small primes in chosen (those where merging could help)
    # A merge is beneficial if 1/(N - p*q) > 1/(N-p) + 1/(N-q)
    # i.e., N-p < (N-p)(N-q)/(N-p-q) ... let's just compute directly
    
    # Try all pairs in chosen
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(chosen)
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                p, q = chosen[i], chosen[j]
                prod = p * q
                if prod >= N:
                    continue
                # Check if prod is coprime to all other chosen
                rest = [x for k, x in enumerate(chosen) if k != i and k != j]
                if not is_coprime_to_all(prod, rest):
                    continue
                # Compare scores
                old_score = 1.0/(N-p) + 1.0/(N-q)
                new_score = 1.0/(N-prod)
                if new_score > old_score:
                    # Perform the merge
                    chosen = rest + [prod]
                    chosen.sort()
                    improved = True
                    break
            if improved:
                break
    
    best_set = chosen[:]
    best_score = score(best_set)
    
    # Phase 3: Try adding 1 (it's coprime to all)
    if 1 not in chosen:
        candidate = chosen + [1]
        s = score(candidate)
        if s > best_score:
            best_set = candidate
            best_score = s
    
    # Phase 4: Try swapping individual elements
    # For each element in chosen, try replacing it with another number not in chosen
    # that is coprime to all other elements
    # This is expensive, so limit to promising swaps
    if time.time() < deadline - 30:
        chosen = best_set[:]
        chosen_set = set(chosen)
        # Candidate numbers: composites of primes, or primes not yet chosen
        # Focus on numbers close to N (high value)
        
        # Generate candidate composites: products of 2 or 3 small primes
        small_primes = [p for p in P if p < 100]
        candidates = set()
        
        # Single primes not in chosen
        for p in P:
            if p not in chosen_set:
                candidates.add(p)
        
        # Products of 2 primes
        for i in range(len(small_primes)):
            for j in range(i + 1, len(small_primes)):
                prod = small_primes[i] * small_primes[j]
                if prod < N and prod > 1:
                    candidates.add(prod)
        
        # Products of 3 small primes
        for i in range(len(small_primes)):
            for j in range(i + 1, len(small_primes)):
                for k in range(j + 1, len(small_primes)):
                    prod = small_primes[i] * small_primes[j] * small_primes[k]
                    if prod < N and prod > 1:
                        candidates.add(prod)
        
        # Sort candidates by value (closest to N first)
        candidates = [c for c in candidates if c < N and c >= 1]
        candidates.sort(key=lambda c: N - c)
        
        for c in candidates:
            if time.time() > deadline:
                break
            if c in chosen_set:
                continue
            # Try replacing each element in chosen with c
            for idx, old in enumerate(chosen):
                rest = [x for k, x in enumerate(chosen) if k != idx]
                if is_coprime_to_all(c, rest):
                    new_set = rest + [c]
                    s = score(new_set)
                    if s > best_score:
                        best_score = s
                        best_set = new_set
                        chosen = new_set[:]
                        chosen_set = set(chosen)
                        break
    
    return best_set
# EVOLVE-BLOCK-END
