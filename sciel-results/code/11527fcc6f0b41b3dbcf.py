"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START

def _sieve(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    primes = _sieve(1000)
    N = 100000
    
    # Precompute coverage arrays for each prime-residue pair
    # cover[p][r] is a numpy bool array of length N+1 where index i is True if i % p == r and i >= 1
    # Actually, we can just generate the indices for each (p, r)
    # To save memory and time, let's just store the indices.
    # indices[p][r] = np.array of indices in 1..N such that i % p == r
    
    indices = {}
    for p in primes:
        indices[p] = {}
        for r in range(p):
            # Start from the smallest positive integer congruent to r mod p
            # If r == 0, smallest is p. If r > 0, smallest is r.
            start = r if r > 0 else p
            if start <= N:
                indices[p][r] = np.arange(start, N + 1, p)
            else:
                indices[p][r] = np.array([], dtype=int)
    
    # Helper to calculate L given a set of (p, r) pairs
    def get_L(pairs):
        # Use a byte array for speed
        covered = np.zeros(N + 1, dtype=np.uint8)
        for p, r in pairs:
            idx = indices[p][r]
            if len(idx) > 0:
                covered[idx] = 1
        # Find first 0 in 1..N
        # We can use np.flatnonzero
        # But checking 1..N is fast enough if we break early?
        # np.argmax on ~covered[1:]
        # Let's just loop in Python for the first gap, it's very fast (microseconds)
        # Or use numpy:
        # first_gap = np.argmax(covered[1:] == 0)
        # if first_gap == 0: return N
        # else: return first_gap
        
        # Actually, np.argmax returns 0 if all elements are False (i.e., covered[1:] is all 1s? No, ~covered is all 0s).
        # Let's be careful.
        # ~covered[1:] is True where covered is 0.
        # If all covered, ~covered[1:] is all False. argmax returns 0.
        # If there is a gap at index k (0-based in covered[1:]), argmax returns k.
        # The actual integer is k + 1.
        
        # Let's just use a simple loop for robustness and speed (10^5 iterations is trivial)
        L = 0
        for i in range(1, N + 1):
            if covered[i]:
                L = i
            else:
                break
        return L

    # Strategy:
    # 1. Select a set of primes.
    #    We want sum(1/p) <= 1.
    #    To maximize L, we generally want small primes, but they are expensive.
    #    Let's try a few different prime sets.
    #    Set A: Greedy smallest primes until budget exhausted.
    #    Set B: Primes >= 5, greedy.
    #    Set C: Primes >= 13, greedy.
    
    def select_primes(min_p=2):
        sel = []
        s = Fraction(0, 1)
        for p in primes:
            if p < min_p:
                continue
            cost = Fraction(1, p)
            if s + cost <= Fraction(1, 1):
                sel.append(p)
                s += cost
        return sel, s

    # Try multiple prime sets
    prime_sets = [
        select_primes(2)[0],
        select_primes(3)[0],
        select_primes(5)[0],
        select_primes(13)[0],
        # Maybe a random subset of primes >= 5 that fits budget?
    ]
    
    # Remove duplicates and empty sets
    unique_sets = []
    seen = set()
    for ps in prime_sets:
        key = tuple(ps)
        if key not in seen:
            seen.add(key)
            unique_sets.append(ps)
    
    best_L = 0
    best_pairs = []
    
    for ps in unique_sets:
        if not ps:
            continue
        
        # Initialize residues
        # Heuristic: r = 1 for all? Or random?
        # Let's try a few initializations.
        inits = [
            {p: 1 for p in ps},
            {p: random.randint(0, p-1) for p in ps},
            {p: 0 for p in ps}, # r=0 means multiples of p
        ]
        
        for init in inits:
            current_pairs = [(p, init[p]) for p in ps]
            current_L = get_L(current_pairs)
            
            if current_L > best_L:
                best_L = current_L
                best_pairs = current_pairs.copy()
            
            # Local Search / Simulated Annealing
            T = 1.0
            cooling = 0.9995
            no_improve = 0
            max_no_improve = 2000 # Restart if stuck
            
            while time.time() < deadline - 5:
                # Pick a random prime to change
                idx = random.randint(0, len(ps) - 1)
                p = ps[idx]
                old_r = init[p]
                new_r = random.randint(0, p - 1)
                
                if new_r == old_r:
                    continue
                
                init[p] = new_r
                # Update pair
                current_pairs[idx] = (p, new_r)
                
                new_L = get_L(current_pairs)
                
                if new_L > current_L:
                    current_L = new_L
                    no_improve = 0
                    if current_L > best_L:
                        best_L = current_L
                        best_pairs = current_pairs.copy()
                else:
                    # Accept with probability exp((new_L - current_L) / T)
                    # Since L is integer, delta is integer.
                    delta = new_L - current_L
                    if delta < 0:
                        # Accept bad moves to escape local minima
                        if random.random() < np.exp(delta / T):
                            current_L = new_L
                        else:
                            # Revert
                            init[p] = old_r
                            current_pairs[idx] = (p, old_r)
                    # If delta == 0, accept?
                    # Usually yes, to wander.
                    
                    no_improve += 1
                    if no_improve > max_no_improve:
                        break # Restart this prime set with new random init? Or just break SA.
                
                T *= cooling
                if T < 0.01:
                    T = 0.01
            
            # If we broke due to no_improve, we might try another init for the same prime set
            # But for now, we just move to the next init.
    
    # Convert best_pairs to required format [[p, r], ...]
    return best_pairs
# EVOLVE-BLOCK-END
