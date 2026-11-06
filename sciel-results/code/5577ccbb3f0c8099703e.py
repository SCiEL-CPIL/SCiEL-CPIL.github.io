import time
from fractions import Fraction
import numpy as np

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
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    primes = _sieve(1000)
    L_MAX = 100000
    
    # Precompute masks for each prime (as numpy arrays)
    # mask[p][r] is a boolean array of length L_MAX+1 where index k is True if k%p==r
    # But storing all p*r masks would be 168*1000*100001 booleans ~ 16.8 GB. Too much.
    # Instead, we compute masks on the fly but cache them.
    # Actually, for a given prime p, the mask for residue r is just a periodic pattern.
    # We can generate it quickly.
    
    mask_cache = {}
    
    def get_mask(p, r):
        key = (p, r)
        if key in mask_cache:
            return mask_cache[key]
        # Create boolean array where positions k (1..L_MAX) with k%p==r are True
        # Position 0 is unused (we cover 1..L_MAX)
        mask = np.zeros(L_MAX + 1, dtype=bool)
        if r == 0:
            start = p
        else:
            start = r
            if start == 0:
                start = p
        if start <= L_MAX:
            mask[start::p] = True
        mask_cache[key] = mask
        return mask
    
    def compute_L(coverage):
        # coverage is a numpy boolean array of length L_MAX+1
        # Find the first index i>=1 where coverage[i] is False
        # If all 1..L_MAX are True, return L_MAX
        if coverage[L_MAX]:
            # Check if all 1..L_MAX are covered
            # Actually, we need the first gap
            pass
        # Find first False in coverage[1:]
        # np.argmax returns index of first True. We want first False.
        # ~coverage[1:] gives True where coverage is False.
        neg = ~coverage[1:]
        if not neg.any():
            return L_MAX
        first_false_idx = np.argmax(neg)
        return first_false_idx  # This is the index in coverage[1:], so the number is first_false_idx + 1? No.
        # coverage[1] corresponds to number 1. If coverage[1] is False, first_false_idx=0, L=0.
        # If coverage[1..k] is True and coverage[k+1] is False, first_false_idx=k, L=k.
        # So L = first_false_idx. Wait.
        # coverage[1] = number 1. 
        # If number 1 is not covered, L=0. neg[0]=True, argmax=0. Return 0. Correct.
        # If number 1,2 covered, 3 not. coverage[1]=T, coverage[2]=T, coverage[3]=F.
        # neg[0]=F, neg[1]=F, neg[2]=T. argmax=2. L should be 2. Return 2. Correct.
        return first_false_idx
    
    def greedy_build(deadline, strategy):
        coverage = np.zeros(L_MAX + 1, dtype=bool)
        used_primes = set()
        budget = Fraction(1, 1)
        classes = []
        
        while True:
            if time.time() > deadline:
                break
            
            L = compute_L(coverage)
            if L >= L_MAX:
                break
                
            U = L + 1
            
            best_p = None
            best_r = 0
            best_score = -1
            
            for p in primes:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if budget < cost:
                    continue
                
                r = U % p
                mask_pr = get_mask(p, r)
                
                # Gain = number of newly covered positions
                # Total positions covered by this mask: np.count_nonzero(mask_pr)
                # Already covered: np.count_nonzero(coverage & mask_pr)
                # Gain = total - already
                # But we only care about positions 1..L_MAX. mask_pr is already 0..L_MAX.
                
                # Optimization: if the mask doesn't cover U, skip. But r=U%p ensures it covers U.
                # So gain is at least 1.
                
                # For speed, we can estimate gain without full count if possible, but count is O(L_MAX) in numpy, which is fast.
                # Actually, np.count_nonzero is very fast.
                
                # However, doing this for all 168 primes every iteration might be slow if L_MAX is large.
                # L_MAX=100000. 168 * 100000 = 16.8M operations per iteration.
                # If we do 100 iterations, that's 1.68B operations. Might be too slow.
                
                # Better: only compute exact gain for top candidates?
                # Or use a heuristic first?
                # Heuristic: prefer small p (covers more numbers per unit length? No, small p has shorter period, covers 1/p of numbers).
                # Actually, all primes cover ~1/p of the numbers.
                # The key is how many of those are NOT already covered.
                
                # Let's just compute it. Numpy is fast.
                already = np.count_nonzero(coverage & mask_pr)
                total = np.count_nonzero(mask_pr)
                gain = total - already
                
                if gain == 0:
                    continue
                
                if strategy == 0:
                    score = gain * p  # Efficiency
                elif strategy == 1:
                    score = gain
                else:
                    score = gain / p  # Prefer small p? No, gain/p is roughly 1, so this doesn't help much.
                    # Maybe prefer large p (cheap)? score = gain * p already does that.
                    # Let's try score = gain / cost = gain * p. Same as strategy 0.
                    # Strategy 2: prefer large p (cheap). score = gain * p^2?
                    score = gain * p * p
                    
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r
            
            if best_p is None:
                break
                
            used_primes.add(best_p)
            budget -= Fraction(1, best_p)
            classes.append([best_p, best_r])
            coverage |= get_mask(best_p, best_r)
            
        return classes, budget
    
    best_L = 0
    best_classes = []
    
    strategies = [0, 1, 2]
    for s in strategies:
        if time.time() > deadline - 15:
            break
        classes, budget = greedy_build(deadline - 10, s)
        if not classes:
            continue
        coverage = np.zeros(L_MAX + 1, dtype=bool)
        for p, r in classes:
            coverage |= get_mask(p, r)
        L = compute_L(coverage)
        
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]
            
        # Local Search
        if time.time() > deadline - 5:
            continue
            
        used_primes = set(p for p, r in classes)
        remaining_budget = budget
        
        no_improve = 0
        max_no_improve = 30
        
        while time.time() < deadline - 2 and no_improve < max_no_improve:
            U = L + 1
            if U > L_MAX:
                break
                
            improved = False
            
            # Move 1: Add new prime
            candidates_add = []
            for p in primes:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if remaining_budget < cost:
                    continue
                r = U % p
                mask_pr = get_mask(p, r)
                already = np.count_nonzero(coverage & mask_pr)
                total = np.count_nonzero(mask_pr)
                gain = total - already
                if gain > 0:
                    candidates_add.append((gain * p, p, r, cost))
            
            candidates_add.sort(key=lambda x: x[0], reverse=True)
            
            for _, p, r, cost in candidates_add[:5]:
                if time.time() > deadline - 2:
                    break
                new_coverage = coverage | get_mask(p, r)
                new_L = compute_L(new_coverage)
                if new_L > L:
                    classes.append([p, r])
                    used_primes.add(p)
                    remaining_budget -= cost
                    coverage = new_coverage
                    L = new_L
                    if L > best_L:
                        best_L = L
                        best_classes = [list(c) for c in classes]
                    no_improve = 0
                    improved = True
                    break
                    
            # Move 2: Change residue
            if not improved:
                for idx, (p, old_r) in enumerate(classes):
                    new_r = U % p
                    if new_r == old_r:
                        continue
                    # Recompute coverage without old mask, with new mask
                    # This is expensive. Limit to a few.
                    old_mask = get_mask(p, old_r)
                    new_mask = get_mask(p, new_r)
                    
                    # Trial coverage
                    trial_cov = coverage.copy()
                    # Remove old mask? No, we can't simply subtract because other masks might overlap.
                    # We must recompute from scratch.
                    trial_cov = np.zeros(L_MAX + 1, dtype=bool)
                    for j, (pp, rr) in enumerate(classes):
                        if j == idx:
                            trial_cov |= get_mask(pp, new_r)
                        else:
                            trial_cov |= get_mask(pp, rr)
                    
                    new_L = compute_L(trial_cov)
                    if new_L > L:
                        classes[idx] = [p, new_r]
                        coverage = trial_cov
                        L = new_L
                        if L > best_L:
                            best_L = L
                            best_classes = [list(c) for c in classes]
                        no_improve = 0
                        improved = True
                        break
                        
            # Move 3: Swap (remove one, add one)
            if not improved:
                # Try removing each prime and adding a new one
                # Limit to a few removals
                for idx_remove in range(min(len(classes), 5)):
                    if time.time() > deadline - 2:
                        break
                    
                    p_remove, r_remove = classes[idx_remove]
                    trial_classes = [c for i, c in enumerate(classes) if i != idx_remove]
                    trial_used = set(p for p, r in trial_classes)
                    trial_budget = remaining_budget + Fraction(1, p_remove)
                    
                    trial_cov = np.zeros(L_MAX + 1, dtype=bool)
                    for pp, rr in trial_classes:
                        trial_cov |= get_mask(pp, rr)
                    
                    L_prime = compute_L(trial_cov)
                    U_prime = L_prime + 1
                    
                    best_add_score = -1
                    best_add_p = None
                    best_add_r = 0
                    best_add_cost = None
                    
                    for p_add in primes:
                        if p_add in trial_used:
                            continue
                        cost_add = Fraction(1, p_add)
                        if trial_budget < cost_add:
                            continue
                        r_add = U_prime % p_add
                        mask_add = get_mask(p_add, r_add)
                        already = np.count_nonzero(trial_cov & mask_add)
                        total = np.count_nonzero(mask_add)
                        gain = total - already
                        
                        if gain > 0:
                            score = gain * p_add
                            if score > best_add_score:
                                best_add_score = score
                                best_add_p = p_add
                                best_add_r = r_add
                                best_add_cost = cost_add
                                
                    if best_add_p:
                        final_classes = trial_classes + [[best_add_p, best_add_r]]
                        final_cov = trial_cov | get_mask(best_add_p, best_add_r)
                        final_L = compute_L(final_cov)
                        
                        if final_L > L:
                            classes = final_classes
                            coverage = final_cov
                            L = final_L
                            remaining_budget = trial_budget - best_add_cost
                            used_primes = set(p for p, r in classes)
                            if L > best_L:
                                best_L = L
                                best_classes = [list(c) for c in classes]
                            no_improve = 0
                            improved = True
                            break
                            
            if not improved:
                no_improve += 1
    
    # Final validation
    if best_classes:
        total_cost = sum(Fraction(1, p) for p, r in best_classes)
        if total_cost > 1:
            pass
        ps = [p for p, r in best_classes]
        if len(ps) != len(set(ps)):
            pass
            
    return best_classes

# EVOLVE-BLOCK-END
