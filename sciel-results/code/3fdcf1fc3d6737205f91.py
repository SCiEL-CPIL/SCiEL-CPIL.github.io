import time
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
    
    best_L = 0
    best_classes = []
    
    def run_greedy(window_size, prefer_small_on_tie, use_efficiency, deadline):
        """
        Run a greedy covering algorithm.
        window_size: How far ahead to look for new coverage.
        prefer_small_on_tie: If new_cov is equal, prefer smaller p (True) or larger p (False).
        use_efficiency: Score by cnt * p instead of just cnt.
        """
        nonlocal best_L, best_classes
        
        covered = bytearray(100001)
        used_primes = set()
        budget = Fraction(1)
        classes = []
        
        # Pre-compute costs as Fractions for speed? 
        # Fraction(1, p) is fast enough for 168 primes.
        
        for n in range(1, 100001):
            if time.time() > deadline:
                return
            
            if covered[n]:
                continue
            
            # n is uncovered, need to pick a prime
            best_p = None
            best_r = 0
            best_score = -1
            
            # Candidates: unused primes that fit in budget
            # We iterate all primes, filter by used and budget
            for p in all_primes:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if budget < cost:
                    continue
                
                r = n % p
                # Count new coverage in [n, n+window_size]
                cnt = 0
                # Optimization: if p > window_size, only n is in the window
                # Actually, step is p.
                # k = n, n+p, ...
                # We only care if covered[k] is 0.
                
                # Fast path for large p
                if p > window_size:
                    if not covered[n]:
                        cnt = 1
                else:
                    # Iterate
                    # max_k = min(n + window_size, 100000)
                    # for k in range(n, max_k + 1, p):
                    #     if not covered[k]:
                    #         cnt += 1
                    # 
                    # This loop is fast.
                    max_k = n + window_size
                    if max_k > 100000:
                        max_k = 100000
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
                        score += (1000 - p)
                    else:
                        score += p
                
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r
            
            if best_p is None:
                # Can't cover n
                break
            
            # Commit to best_p
            used_primes.add(best_p)
            budget -= Fraction(1, best_p)
            classes.append([best_p, best_r])
            
            # Mark covered
            # Start at best_r if best_r > 0 else best_p
            # Note: if best_r == 0, the first positive number is best_p
            start_k = best_r
            if start_k == 0:
                start_k = best_p
            for k in range(start_k, 100001, best_p):
                covered[k] = 1
        
        # Calculate L
        L = 0
        for i in range(1, 100001):
            if covered[i]:
                L = i
            else:
                break
        
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
        run_greedy(w, ps, ue, deadline)
        
    # If time remains, we can try to do a "local search" or more random restarts
    # But the greedy is deterministic. To get randomness, we could perturb scores.
    # Let's try a few random perturbations if time is left.
    
    if time.time() < deadline - 10:
        import random
        # Fixed seed for reproducibility? No, let's just use time-based seed or fixed.
        # The task doesn't require deterministic output, just max score.
        
        # Try a few random weightings
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            # Random window
            w = random.choice([500, 1000, 2000, 5000])
            ps = random.choice([True, False])
            ue = random.choice([True, False])
            # Add small random noise to score? 
            # The current run_greedy doesn't support noise.
            # We can just rely on the different parameters.
            # Actually, the parameters are discrete. 
            # To add diversity, we could modify the score function slightly.
            # But for now, just cycling through the list above is a good start.
            # Let's just re-run the best performing ones or random ones.
            run_greedy(w, ps, ue, deadline)

    # Fallback
    if not best_classes:
        # Seed-like solution
        primes_13_plus = [p for p in all_primes if p >= 13]
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_13_plus)]
        
    return best_classes
