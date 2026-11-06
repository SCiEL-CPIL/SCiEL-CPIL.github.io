import time
from fractions import Fraction
import random

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
    # Evaluator kills at 150s. Keep a safety margin.
    deadline = start_time + 140.0
    
    primes = _sieve(1000)
    P = len(primes)
    
    best_L = 0
    best_classes = []
    
    def run_greedy(L_max, use_efficiency, prefer_small, deadline):
        nonlocal best_L, best_classes
        
        covered = bytearray(L_max + 1)
        used_primes = set()
        # Use float for budget check in the loop for speed, verify at the end.
        # Or use Fraction if we want to be safe. Fraction is exact but slower.
        # Given P <= 168, Fraction is likely fine. Let's stick to Fraction for correctness.
        budget = Fraction(1)
        classes = []
        
        n = 1
        
        while n <= L_max:
            if time.time() > deadline:
                return
            
            if covered[n]:
                n += 1
                continue
            
            best_p = None
            best_r = 0
            best_score = -1
            
            for p in primes:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if budget < cost:
                    continue
                
                r = n % p
                cnt = 0
                k = n
                while k <= L_max:
                    if not covered[k]:
                        cnt += 1
                    k += p
                
                if use_efficiency:
                    score = cnt * p
                else:
                    score = cnt
                
                # Tie breaking
                if score == best_score:
                    if best_p is not None:
                        if prefer_small:
                            if p < best_p:
                                best_score = score
                                best_p = p
                                best_r = r
                        else:
                            if p > best_p:
                                best_score = score
                                best_p = p
                                best_r = r
                    else:
                        best_score = score
                        best_p = p
                        best_r = r
                elif score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r
            
            if best_p is None:
                break
            
            used_primes.add(best_p)
            budget -= Fraction(1, best_p)
            classes.append([best_p, best_r])
            
            start_k = best_r
            if start_k == 0:
                start_k = best_p
            for k in range(start_k, L_max + 1, best_p):
                covered[k] = 1
                
            n += 1
            
        L = 0
        for i in range(1, L_max + 1):
            if covered[i]:
                L = i
            else:
                break
        
        if L > best_L:
            # Verify exact budget
            total_cost = sum(Fraction(1, p) for p, r in classes)
            if total_cost <= 1:
                best_L = L
                best_classes = [list(c) for c in classes]
        return L

    def run_local_search(L_max, deadline):
        # Start with a good greedy solution, then try to swap primes to improve L.
        # This is a different structural approach: optimization via swaps.
        # We need an initial solution. Let's generate one quickly with greedy.
        # For speed, use a smaller L_max for initialization if needed, but L_max is the target.
        
        # To save time, we might just use the best_classes found so far if valid?
        # Or generate a fresh one.
        
        # Let's generate a fresh one with a specific heuristic:
        # Prefer small primes?
        init_classes = []
        covered = bytearray(L_max + 1)
        used_primes = set()
        budget = Fraction(1)
        
        n = 1
        while n <= L_max:
            if time.time() > deadline:
                return
            if covered[n]:
                n += 1
                continue
            
            best_p = None
            best_r = 0
            best_score = -1
            
            for p in primes:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if budget < cost:
                    continue
                r = n % p
                cnt = 0
                k = n
                while k <= L_max:
                    if not covered[k]:
                        cnt += 1
                    k += p
                score = cnt * p
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r
            
            if best_p is None:
                break
            
            used_primes.add(best_p)
            budget -= Fraction(1, best_p)
            init_classes.append([best_p, best_r])
            
            start_k = best_r
            if start_k == 0:
                start_k = best_p
            for k in range(start_k, L_max + 1, best_p):
                covered[k] = 1
            n += 1
            
        # Calculate initial L
        L = 0
        for i in range(1, L_max + 1):
            if covered[i]:
                L = i
            else:
                break
        
        current_classes = [list(c) for c in init_classes]
        current_L = L
        improved = True
        swaps_done = 0
        max_swaps = 50 # Limit swaps to keep time bounded
        
        while improved and swaps_done < max_swaps:
            if time.time() > deadline:
                break
            improved = False
            # Try swapping one prime in the set with another unused prime
            # Or just try replacing a prime with a different residue for the same prime?
            # The problem allows distinct primes. So we can't change p.
            # But we can change r.
            # Wait, the greedy picks r = n % p.
            # Maybe a different r for the same p covers more?
            # But the prime p is already used. We can't add it again.
            # So we can only change the residue r for an existing prime p.
            
            # Strategy: For each prime p in current_classes, try all residues r' != r.
            # Check if the new set covers a larger prefix.
            
            for idx, (p, r) in enumerate(current_classes):
                if time.time() > deadline:
                    break
                # Try other residues
                for r_new in range(p):
                    if r_new == r:
                        continue
                    
                    # Temporarily change
                    old_r = current_classes[idx][1]
                    current_classes[idx][1] = r_new
                    
                    # Recalculate coverage?
                    # This is expensive if done naively.
                    # We need to know the impact.
                    # Impact:
                    # 1. Positions covered by old r: k % p == r
                    # 2. Positions covered by new r: k % p == r_new
                    #
                    # We can maintain a count of how many classes cover each number?
                    # That would allow O(1) update per position.
                    # But we are in a loop.
                    
                    # Let's just recalculate L for this candidate.
                    # It's O(L_max). With ~20 primes, 1000 residues, 20*1000*1000 = 20M ops.
                    # Too slow for Python if done many times.
                    
                    # Optimization: Only try residues that cover the first gap?
                    # Find the first gap in current coverage.
                    # If changing r for p doesn't cover the gap, it won't increase L.
                    
                    # Let's find the first gap in current coverage.
                    gap = -1
                    for i in range(1, L_max + 1):
                        # Check if i is covered by current_classes
                        cov = False
                        for p2, r2 in current_classes:
                            if i % p2 == r2:
                                cov = True
                                break
                        if not cov:
                            gap = i
                            break
                    
                    if gap == -1:
                        break # Full coverage
                    
                    # Does the new residue cover the gap?
                    if gap % p != r_new:
                        continue
                    
                    # It covers the gap. Does it uncover anything before the gap?
                    # If it uncovers anything before the gap, L might decrease.
                    # We need to check if any i < gap is only covered by this class with old r.
                    # This is complex.
                    # Let's just compute the new L.
                    
                    new_L = 0
                    for i in range(1, L_max + 1):
                        cov = False
                        for p2, r2 in current_classes:
                            if i % p2 == r2:
                                cov = True
                                break
                        if cov:
                            new_L = i
                        else:
                            break
                    
                    if new_L > current_L:
                        current_L = new_L
                        improved = True
                        swaps_done += 1
                        break # Restart the outer loop? Or continue?
                        # Breaking inner, continuing outer might miss other improvements.
                        # But we broke out of residue loop.
                        # We should break out of prime loop too to restart from top?
                        # Or just continue to next prime.
                    else:
                        # Revert
                        current_classes[idx][1] = old_r
            
        if current_L > best_L:
            # Verify budget
            total_cost = sum(Fraction(1, p) for p, r in current_classes)
            if total_cost <= 1:
                best_L = current_L
                best_classes = [list(c) for c in current_classes]

    # Run Greedy variants
    configs = [
        (1000, False, True),
        (1000, True, True),
        (2000, False, True),
        (2000, True, True),
        (5000, False, True),
        (5000, True, True),
        (10000, False, True),
        (10000, True, True),
        (10000, False, False),
        (10000, True, False),
        (50000, False, True),
        (50000, True, True),
    ]
    
    for l_max, eff, small in configs:
        if time.time() > deadline - 5:
            break
        run_greedy(l_max, eff, small, deadline)
        
    # Run Local Search
    if time.time() < deadline - 10:
        # Try local search on a few promising L_max values
        for l_max in [1000, 2000, 5000]:
            if time.time() > deadline - 5:
                break
            run_local_search(l_max, deadline)

    if not best_classes:
        # Fallback
        budget = Fraction(1)
        for p in primes:
            if budget >= Fraction(1, p):
                best_classes.append([p, 1 % p])
                budget -= Fraction(1, p)
            else:
                break
                
    return best_classes
