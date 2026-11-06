import time
import random
from fractions import Fraction
import numpy as np

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
    n_primes = len(all_primes)
    
    best_L = 0
    best_classes = []
    
    # Pre-compute costs as floats for fast comparison, and exact for final check
    cost_float = {p: 1.0 / p for p in all_primes}
    cost_frac = {p: Fraction(1, p) for p in all_primes}
    
    # Precompute numpy arrays for fast coverage marking
    # For each prime, precompute the indices it covers
    prime_indices = {}
    for p in all_primes:
        prime_indices[p] = np.arange(p - 1, N, p, dtype=np.int32)  # 0-indexed: positions p-1, 2p-1, ...
        # Actually, we need 1-indexed: positions p, 2p, 3p, ...
        # Let me redo: for residue r (0 <= r < p), covered positions are r, r+p, r+2p, ...
        # 1-indexed: if r==0, positions are p, 2p, 3p, ...; if r>0, positions are r, r+p, ...
    
    def compute_L(covered):
        # Find the first gap using numpy
        if not covered[1]:
            return 0
        arr = np.frombuffer(covered[1:], dtype=np.uint8)
        zeros = np.nonzero(arr == 0)[0]
        if len(zeros) == 0:
            return N
        return int(zeros[0])
    
    def run_greedy_np(window_size, prefer_small_on_tie, use_efficiency, deadline, seed=0):
        """Fast greedy using NumPy for coverage counting and marking."""
        nonlocal best_L, best_classes
        
        rng = random.Random(seed)
        covered = np.zeros(N + 1, dtype=np.uint8)
        used_primes = set()
        budget_float = 1.0
        classes = []
        
        # Precompute which primes are affordable
        affordable_primes = [p for p in all_primes if cost_float[p] <= 1.0]
        
        for n in range(1, N + 1):
            if time.time() > deadline:
                return
            
            if covered[n]:
                continue
            
            # n is uncovered - find best prime
            best_p = None
            best_r = 0
            best_score = -1e18
            
            for p in affordable_primes:
                if p in used_primes:
                    continue
                cost_f = cost_float[p]
                if budget_float < cost_f - 1e-15:
                    continue
                
                r = n % p
                
                # Count new coverage in [n, min(n+window_size, N)] using numpy
                if p > window_size:
                    cnt = 1  # Only n itself (and possibly more, but at least 1)
                    # Actually if p > window_size, only n is in [n, n+window_size]
                else:
                    max_k = min(n + window_size, N)
                    # Positions in [n, max_k] congruent to r mod p
                    # First such position >= n: if n % p == r, then n; else n + (r - n % p + p) % p
                    # But we want positions k where k % p == r and n <= k <= max_k
                    # k = r + t*p for t >= 0, and k >= n, k <= max_k
                    if r == 0:
                        first_k = ((n + p - 1) // p) * p  # smallest multiple of p >= n
                    else:
                        first_k = n + (r - n % p) % p
                        if first_k < n:
                            first_k += p
                    if first_k > max_k:
                        cnt = 0
                    else:
                        last_k = first_k + ((max_k - first_k) // p) * p
                        cnt = ((last_k - first_k) // p) + 1
                        # But we need to count only UNCOVERED positions
                        # Use numpy for this
                        idxs = np.arange(first_k, max_k + 1, p, dtype=np.int32)
                        cnt = int(np.sum(covered[idxs] == 0))
                
                # Calculate Score
                if use_efficiency:
                    score = cnt * p
                else:
                    score = cnt
                    if prefer_small_on_tie:
                        score += (1000 - p)
                    else:
                        score += p
                
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r
            
            if best_p is None:
                break
            
            # Commit to best_p
            used_primes.add(best_p)
            budget_float -= cost_float[best_p]
            classes.append([best_p, best_r])
            
            # Mark covered using numpy
            if best_r == 0:
                start_k = best_p
            else:
                start_k = best_r
            covered[start_k::best_p] = 1
        
        # Verify budget with exact rationals
        total_cost = sum(cost_frac[p] for p, r in classes)
        if total_cost > Fraction(1):
            return
        
        L = compute_L(covered)
        
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]
        
        return L
    
    def local_improve_np(classes, deadline):
        """
        Try to improve by swapping classes: remove one class, add another to extend L.
        Uses numpy for speed.
        """
        nonlocal best_L, best_classes
        
        if not classes:
            return
        
        # Build coverage
        covered = np.zeros(N + 1, dtype=np.uint8)
        for p, r in classes:
            start_k = r if r != 0 else p
            covered[start_k::p] = 1
        
        L = compute_L(covered)
        used_primes = set(p for p, r in classes)
        total_cost_frac = sum(cost_frac[p] for p, r in classes)
        
        # Try removing each class and adding a better one
        for idx in range(len(classes)):
            if time.time() > deadline:
                return
            
            p_remove, r_remove = classes[idx]
            
            # Temporarily remove
            covered2 = covered.copy()
            start_k = r_remove if r_remove != 0 else p_remove
            covered2[start_k::p_remove] = 0
            
            # Find first gap in covered2
            L2 = compute_L(covered2)
            
            # Try to fill the gap by adding a new prime
            if L2 >= L:
                continue
            
            # Find the first uncovered number in covered2
            n_gap = L2 + 1 if L2 < N else N
            
            # Try adding each unused prime to cover n_gap
            best_new_p = None
            best_new_r = 0
            best_new_L = L2
            
            # Budget check: total_cost without p_remove + 1/p <= 1
            remaining_budget_frac = Fraction(1) - (total_cost_frac - cost_frac[p_remove])
            
            for p in all_primes:
                if p in used_primes and p != p_remove:
                    continue
                if cost_frac[p] > remaining_budget_frac:
                    continue
                
                r = n_gap % p
                # Apply this class
                covered3 = covered2.copy()
                start_k = r if r != 0 else p
                covered3[start_k::p] = 1
                
                L3 = compute_L(covered3)
                if L3 > best_new_L:
                    best_new_L = L3
                    best_new_p = p
                    best_new_r = r
            
            if best_new_p is not None and best_new_L > L:
                # Apply the improvement
                new_classes = [c for i, c in enumerate(classes) if i != idx]
                new_classes.append([best_new_p, best_new_r])
                
                # Verify
                total_cost = sum(cost_frac[pp] for pp, rr in new_classes)
                if total_cost <= Fraction(1):
                    covered_new = np.zeros(N + 1, dtype=np.uint8)
                    for pp, rr in new_classes:
                        start_k = rr if rr != 0 else pp
                        covered_new[start_k::pp] = 1
                    L_new = compute_L(covered_new)
                    if L_new > best_L:
                        best_L = L_new
                        best_classes = [list(c) for c in new_classes]
                        classes = new_classes
                        covered = covered_new
                        used_primes = set(pp for pp, rr in new_classes)
                        total_cost_frac = total_cost
    
    # Run a mix of strategies
    strategies = [
        # (window, prefer_small, use_eff, seed)
        (1000, True, True, 1),
        (1000, False, True, 2),
        (5000, True, True, 3),
        (5000, False, True, 4),
        (2000, True, True, 5),
        (2000, False, True, 6),
        (500, True, True, 7),
        (500, False, True, 8),
        (10000, True, True, 9),
        (10000, False, True, 10),
    ]
    
    # First pass: fixed strategies
    for w, ps, ue, s in strategies:
        if time.time() > deadline - 15:
            break
        run_greedy_np(w, ps, ue, deadline, s)
    
    # Second pass: Randomized perturbations
    if time.time() < deadline - 20:
        for i in range(40):
            if time.time() > deadline - 10:
                break
            w = random.choice([500, 1000, 2000, 5000, 10000])
            ps = random.choice([True, False])
            ue = random.choice([True, False])
            s = 100 + i
            run_greedy_np(w, ps, ue, deadline, s)
    
    # Third pass: Local improvement on best solution
    if best_classes and time.time() < deadline - 5:
        local_improve_np(best_classes, deadline)
    
    if not best_classes:
        # Fallback
        primes_13_plus = [p for p in all_primes if p >= 13]
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_13_plus)]
    
    return best_classes
