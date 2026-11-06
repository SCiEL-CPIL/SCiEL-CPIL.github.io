"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    primes = _sieve(1000)
    all_primes = primes
    N = 100000
    
    best_L = 0
    best_classes = []
    
    def compute_L(covered):
        L = 0
        for i in range(1, N + 1):
            if covered[i]:
                L = i
            else:
                break
        return L
    
    def run_greedy(window_size, prefer_small_on_tie, use_efficiency, deadline, seed=0):
        nonlocal best_L, best_classes
        
        rng = random.Random(seed)
        covered = bytearray(N + 1)
        used_primes = set()
        budget = Fraction(1)
        classes = []
        
        # Pre-compute costs
        costs = {p: Fraction(1, p) for p in all_primes}
        
        for n in range(1, N + 1):
            if time.time() > deadline:
                return
            
            if covered[n]:
                continue
            
            best_p = None
            best_r = 0
            best_score = -1
            
            # Candidates: unused primes that fit in budget
            candidates = [p for p in all_primes if p not in used_primes and budget >= costs[p]]
            
            if not candidates:
                break
            
            # Shuffle candidates slightly to break ties randomly if needed, 
            # but primarily rely on score.
            # To ensure diversity in runs, we can sort by score then break ties randomly.
            
            scored_candidates = []
            for p in candidates:
                r = n % p
                
                # Count new coverage in [n, n+window_size]
                cnt = 0
                if p > window_size:
                    if not covered[n]:
                        cnt = 1
                else:
                    max_k = n + window_size
                    if max_k > N:
                        max_k = N
                    for k in range(n, max_k + 1, p):
                        if not covered[k]:
                            cnt += 1
                
                # Calculate Score
                if use_efficiency:
                    # Higher cnt and higher p (cheaper) is better?
                    # cnt * p means "coverage per unit cost" roughly?
                    # Cost is 1/p. So cnt / (1/p) = cnt * p.
                    score = cnt * p
                else:
                    score = cnt
                    # Tie breaker
                    if prefer_small_on_tie:
                        score += (1000 - p) * 0.01 # Small bias
                    else:
                        score += p * 0.01
                
                # Add small random noise to break ties and explore
                noise = rng.uniform(0, 1) * 0.1
                scored_candidates.append((score + noise, p, r, cnt))
            
            if not scored_candidates:
                break
                
            # Sort by score descending
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            
            best_score, best_p, best_r, _ = scored_candidates[0]
            
            # Commit to best_p
            used_primes.add(best_p)
            budget -= costs[best_p]
            classes.append([best_p, best_r])
            
            # Mark covered
            start_k = best_r
            if start_k == 0:
                start_k = best_p
            for k in range(start_k, N + 1, best_p):
                covered[k] = 1
        
        # Calculate L
        L = compute_L(covered)
        
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]
            
        return L

    # Define a set of strategies
    # (window_size, prefer_small, use_efficiency)
    strategies = [
        (1000, True, False),
        (1000, False, False),
        (5000, True, False),
        (5000, False, False),
        (10000, True, False),
        (10000, False, False),
        (1000, True, True),
        (1000, False, True),
        (5000, True, True),
        (5000, False, True),
        (10000, True, True),
        (10000, False, True),
    ]
    
    # Run strategies
    # Each run is fast, so we can run them all, and maybe repeat some if time allows
    # Or just run them once.
    
    for w, ps, ue in strategies:
        if time.time() > deadline - 5:
            break
        run_greedy(w, ps, ue, deadline, seed=0)
        
    # If time remains, we can try to do a "local search" or more random restarts
    # But the greedy is deterministic. To get randomness, we could perturb scores.
    # Let's try a few random perturbations if time is left.
    
    if time.time() < deadline - 10:
        # Try a few random weightings
        for i in range(50):
            if time.time() > deadline - 5:
                break
            # Random window
            w = random.choice([500, 1000, 2000, 5000])
            ps = random.choice([True, False])
            ue = random.choice([True, False])
            run_greedy(w, ps, ue, deadline, seed=i)

    # Fallback
    if not best_classes:
        # Seed-like solution
        primes_13_plus = [p for p in all_primes if p >= 13]
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_13_plus)]
        
    return best_classes
# EVOLVE-BLOCK-END
