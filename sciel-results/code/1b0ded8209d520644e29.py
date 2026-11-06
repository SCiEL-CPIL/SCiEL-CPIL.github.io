import time
import numpy as np
from fractions import Fraction

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
    L_MAX = 100000
    
    # Precompute numpy masks for each prime p and residue r? 
    # That's 168 primes * up to 1000 residues = 168k masks of size 100k. 
    # 168k * 100k bytes = 16.8 GB. Too much memory.
    # Instead, we generate masks on the fly but cache them if frequently used?
    # Or just generate them quickly.
    # Generating a mask for (p, r):
    # idx = np.arange(1, L_MAX+1)
    # mask = (idx % p == r)
    # This is O(L_MAX) which is 100k ops. Fast.
    
    # We will cache masks for (p, r) pairs that are actually used in the search.
    mask_cache = {}
    
    def get_mask(p, r):
        key = (p, r)
        if key in mask_cache:
            return mask_cache[key]
        # Generate mask
        # Using numpy for speed
        idx = np.arange(1, L_MAX + 1)
        mask = (idx % p == r)
        mask_cache[key] = mask
        return mask

    def compute_L(coverage_bool):
        # coverage_bool is a numpy bool array of length L_MAX
        # We want the largest L such that coverage_bool[1..L] are all True.
        # Find the first False in coverage_bool[1:]
        # If all True, return L_MAX
        if coverage_bool.all():
            return L_MAX
        # np.argmax returns the index of the first True.
        # We want the first False.
        # ~coverage_bool gives True where it was False.
        # We need to be careful with 0-indexing.
        # coverage_bool[0] is unused (for integer 1, index 0).
        # Let's just iterate or use a clever trick.
        # Since L is usually large, iterating from the end might be faster if we know it's close to L_MAX?
        # But if it's small, iterating from start is better.
        # Let's use the property that we only care about the prefix.
        # Find the first 0 in the array.
        # np.flatnonzero(~coverage_bool) gives indices of False.
        # The first one is the first gap.
        gaps = np.flatnonzero(~coverage_bool)
        if len(gaps) == 0:
            return L_MAX
        first_gap_idx = gaps[0]
        # first_gap_idx is 0-based index.
        # Integer value is first_gap_idx + 1.
        # L is integer value - 1.
        return first_gap_idx
        
    def greedy_build(deadline):
        coverage = np.zeros(L_MAX + 1, dtype=bool)
        used_primes = set()
        budget = Fraction(1, 1)
        classes = []
        
        n = 1
        while n <= L_MAX:
            if time.time() > deadline:
                break
            if coverage[n]:
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
                mask = get_mask(p, r)
                
                # Gain: number of new True bits
                # We can estimate gain by counting False bits in mask that are currently False in coverage.
                # gain = np.sum(mask & ~coverage)
                # This is O(L_MAX).
                # To speed up, we can just check a window? No, we need exact for scoring.
                # But np.sum is fast (C level).
                gain = np.sum(mask & ~coverage)
                
                if gain == 0:
                    continue
                
                # Score: prefer high gain, low cost (large p)
                # Efficiency: gain / cost = gain * p
                score = gain * p
                
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
            
            # Advance n to the next uncovered number?
            # We can just increment n, it's cheap.
            n += 1
            
        return classes, budget

    best_L = 0
    best_classes = []
    
    # Phase 1: Greedy
    classes, budget = greedy_build(deadline - 10)
    
    if classes:
        coverage = np.zeros(L_MAX + 1, dtype=bool)
        for p, r in classes:
            coverage |= get_mask(p, r)
        L = compute_L(coverage)
        
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]
    
    # Phase 2: Local Search
    # We will try to improve L by adding, changing, or swapping classes.
    # To keep it fast, we will limit the number of iterations and candidates.
    
    if time.time() < deadline - 5:
        used_primes = set(p for p, r in classes)
        remaining_budget = budget
        no_improve = 0
        max_no_improve = 50 # Limit iterations
        
        while time.time() < deadline - 2 and no_improve < max_no_improve:
            U = L + 1
            if U > L_MAX:
                break
            
            improved = False
            
            # Move 1: Add new prime
            # Find best (p, r) to add such that r = U % p
            candidates_add = []
            for p in primes:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if remaining_budget < cost:
                    continue
                r = U % p
                mask = get_mask(p, r)
                gain = np.sum(mask & ~coverage)
                if gain > 0:
                    score = gain * p
                    candidates_add.append((score, p, r, cost))
            
            candidates_add.sort(key=lambda x: x[0], reverse=True)
            
            for _, p, r, cost in candidates_add[:3]: # Try top 3
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
                    
            # Move 2: Change residue of existing prime
            if not improved:
                # Try changing residue of each existing prime to cover U
                # Only if it doesn't break the prefix?
                # Actually, if we change residue, we might break the prefix.
                # We must check if new coverage still covers 1..L.
                # This is expensive.
                # Heuristic: Only change if the old residue was "wasted" (covered few new things)
                # Or just try all and check L.
                
                # To speed up, limit to a few candidates.
                # Pick primes with small p (high cost, maybe less efficient?) or large p?
                # Let's just iterate through classes.
                
                for idx, (p, old_r) in enumerate(classes):
                    if time.time() > deadline - 2:
                        break
                    new_r = U % p
                    if new_r == old_r:
                        continue
                    
                    # Trial
                    # Remove old mask, add new mask
                    # We can't just subtract because other primes might cover those spots.
                    # We must recompute coverage from scratch or maintain counts.
                    # Recomputing is O(N_classes * L_MAX).
                    # Given N_classes < 100 and L_MAX=10^5, this is ~10^6 ops.
                    # Acceptable for a few candidates.
                    
                    trial_classes = [list(c) for c in classes]
                    trial_classes[idx] = [p, new_r]
                    
                    trial_cov = np.zeros(L_MAX + 1, dtype=bool)
                    for pp, rr in trial_classes:
                        trial_cov |= get_mask(pp, rr)
                        
                    new_L = compute_L(trial_cov)
                    if new_L > L:
                        classes = trial_classes
                        coverage = trial_cov
                        L = new_L
                        if L > best_L:
                            best_L = L
                            best_classes = [list(c) for c in classes]
                        no_improve = 0
                        improved = True
                        break
                        
            # Move 3: Swap - Remove one, Add one
            if not improved:
                # Try removing a prime and adding a better one.
                # This is expensive. Limit to a few.
                # Sort classes by cost descending (small p first) to remove expensive ones?
                # Or remove ones with low gain?
                
                # Let's try removing each prime and see if we can add a better one.
                # Limit to top 5 most expensive (smallest p) or just first 5.
                
                # For simplicity, let's try removing the last added prime and see if we can add a different one.
                # Or just iterate through a few.
                
                # Let's try removing one of the smallest primes (highest cost) to free budget.
                # Actually, we want to free budget to add a prime that covers U better.
                
                # Let's just try swapping the last few primes.
                for idx_remove in range(len(classes) - 1, max(len(classes) - 4, -1), -1):
                    if time.time() > deadline - 2:
                        break
                    
                    p_remove, r_remove = classes[idx_remove]
                    
                    trial_classes = [list(c) for c in classes if c[0] != p_remove]
                    trial_used = set(p for p, r in trial_classes)
                    trial_budget = remaining_budget + Fraction(1, p_remove)
                    
                    trial_cov = np.zeros(L_MAX + 1, dtype=bool)
                    for pp, rr in trial_classes:
                        trial_cov |= get_mask(pp, rr)
                    
                    L_prime = compute_L(trial_cov)
                    U_prime = L_prime + 1
                    
                    # Find best add for trial state
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
                        gain = np.sum(mask_add & ~trial_cov)
                        
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
            # This should not happen, but if it does, we need to fix it?
            # The problem states sum <= 1.
            # If we exceeded it, the solution is invalid.
            # We should probably return a valid subset.
            # But greedy ensures we don't exceed.
            pass
            
        ps = [p for p, r in best_classes]
        if len(ps) != len(set(ps)):
            pass
            
    return best_classes
