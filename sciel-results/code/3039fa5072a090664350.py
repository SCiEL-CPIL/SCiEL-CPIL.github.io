import time
from fractions import Fraction
import random

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    return [i for i in range(2, n+1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    primes = _sieve(1000)
    L_MAX = 100000  # Cap at 10^5 as per contract
    random.seed(42)
    
    best_L = 0
    best_classes = []
    
    def compute_L(covered):
        L = 0
        for i in range(1, L_MAX + 1):
            if covered[i]:
                L = i
            else:
                break
        return L
    
    def get_coverage(classes):
        covered = bytearray(L_MAX + 1)
        for p, r in classes:
            start_k = r
            if start_k == 0:
                start_k = p
            for k in range(start_k, L_MAX + 1, p):
                covered[k] = 1
        return covered
    
    def greedy_build(deadline, window, eff):
        covered = bytearray(L_MAX + 1)
        used_primes = set()
        budget = Fraction(1, 1)
        classes = []
        
        n = 1
        while n <= L_MAX:
            if time.time() > deadline:
                break
            if covered[n]:
                n += 1
                continue
            
            best_p = None
            best_r = 0
            best_score = -1
            
            hi = min(n + window, L_MAX)
            
            for p in primes:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if budget < cost:
                    continue
                
                r = n % p
                # Estimate gain within window
                gain = 0
                k = n
                while k <= hi:
                    if not covered[k]:
                        gain += 1
                    k += p
                
                # Efficiency score
                if eff:
                    score = gain * p
                else:
                    score = gain
                
                if score > best_score:
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
            for k in range(start_k, L_MAX + 1, best_p):
                covered[k] = 1
                
            n += 1
            
        return classes, budget
    
    def local_search(classes, deadline):
        nonlocal best_L, best_classes
        used_primes = set(p for p, r in classes)
        remaining_budget = Fraction(1, 1) - sum(Fraction(1, p) for p, r in classes)
        
        covered = get_coverage(classes)
        L = compute_L(covered)
        
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]
        
        no_improve = 0
        max_no_improve = 30
        
        while time.time() < deadline - 3 and no_improve < max_no_improve:
            U = L + 1
            if U > L_MAX:
                break
            
            improved = False
            
            # Move 1: Try reassigning residue of an existing prime to cover U
            candidates = []
            for p, r in classes:
                new_r = U % p
                if new_r != r:
                    candidates.append((p, r, new_r))
            
            random.shuffle(candidates)
            for p, old_r, new_r in candidates[:8]:
                if time.time() > deadline - 3:
                    break
                
                # Find index of this class
                idx = None
                for i, (pp, rr) in enumerate(classes):
                    if pp == p and rr == old_r:
                        idx = i
                        break
                if idx is None:
                    continue
                
                trial_classes = [list(c) for c in classes]
                trial_classes[idx] = [p, new_r]
                trial_cov = get_coverage(trial_classes)
                L2 = compute_L(trial_cov)
                
                if L2 > L:
                    classes = trial_classes
                    covered = trial_cov
                    L = L2
                    no_improve = 0
                    improved = True
                    if L > best_L:
                        best_L = L
                        best_classes = [list(c) for c in classes]
                    if L >= L_MAX:
                        return
                    break
            
            # Move 2: Add an unused prime with residue covering U
            unused = [p for p in primes if p not in used_primes and remaining_budget >= Fraction(1, p)]
            random.shuffle(unused)
            unused = unused[:10]
            
            for p in unused:
                if time.time() > deadline - 3:
                    break
                r = U % p
                new_classes = classes + [[p, r]]
                new_cov = get_coverage(new_classes)
                L2 = compute_L(new_cov)
                if L2 > L:
                    classes = new_classes
                    used_primes.add(p)
                    remaining_budget -= Fraction(1, p)
                    covered = new_cov
                    L = L2
                    no_improve = 0
                    improved = True
                    if L > best_L:
                        best_L = L
                        best_classes = [list(c) for c in classes]
                    if L >= L_MAX:
                        return
                    break
            
            # Move 3: Try swapping - remove a large prime (high cost) and add a small one
            if not improved:
                sorted_used = sorted(used_primes, key=lambda p: -1.0/p)
                for p_remove in sorted_used[:4]:
                    if time.time() > deadline - 3:
                        break
                    
                    trial_classes = [[p, r] for p, r in classes if p != p_remove]
                    trial_used = set(p for p, r in trial_classes)
                    trial_remaining = remaining_budget + Fraction(1, p_remove)
                    
                    # Try adding a small unused prime
                    small_unused = [p for p in primes[:20] if p not in trial_used and trial_remaining >= Fraction(1, p)]
                    for p_add in small_unused[:5]:
                        r_add = U % p_add
                        new_classes = trial_classes + [[p_add, r_add]]
                        new_cov = get_coverage(new_classes)
                        L2 = compute_L(new_cov)
                        if L2 > L:
                            classes = new_classes
                            used_primes = set(p for p, r in classes)
                            remaining_budget = Fraction(1, 1) - sum(Fraction(1, p) for p in used_primes)
                            covered = new_cov
                            L = L2
                            no_improve = 0
                            improved = True
                            if L > best_L:
                                best_L = L
                                best_classes = [list(c) for c in classes]
                            break
                    if improved:
                        break
            
            if not improved:
                no_improve += 1
        
        return classes
    
    # Phase 1: Multiple greedy strategies with different windows and efficiency metrics
    strategies = [(500, True), (1000, True), (2000, True), (3000, True), (5000, True), (10000, True)]
    strategies += [(500, False), (1000, False), (2000, False), (3000, False), (5000, False)]
    
    for window, eff in strategies:
        if time.time() > deadline - 20:
            break
        classes, _ = greedy_build(deadline, window, eff)
        if classes:
            classes = local_search(classes, deadline)
    
    # Final check
    if not best_classes:
        # Fallback: simple greedy
        classes, _ = greedy_build(deadline, 3000, True)
        best_classes = classes
    
    # Ensure budget is exact and <= 1
    total_cost = sum(Fraction(1, p) for p, r in best_classes)
    if total_cost > 1:
        # This shouldn't happen, but just in case
        pass
    
    return best_classes
