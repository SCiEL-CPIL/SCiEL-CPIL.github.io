import time
from fractions import Fraction

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

PRIMES = _sieve(1000)
L_MAX = 5000

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin

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

    def is_prefix_covered(covered, U):
        # Check if 1..U is covered
        for i in range(1, U + 1):
            if not covered[i]:
                return False
        return True

    def greedy_build(deadline):
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
            
            for p in PRIMES:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if budget < cost:
                    continue
                
                r = n % p
                # Estimate gain: how many new numbers in [n, L_MAX] does this cover?
                # Optimization: We only need to know if it covers n.
                # But the heuristic score = gain * p.
                # To keep it fast, we can approximate gain or just use a simpler metric.
                # Inspiration 1 used a while loop to count gain up to L_MAX.
                # With L_MAX=5000, this is ~5000/p iterations.
                # For p=2, 2500 iterations. For p=997, 5 iterations.
                # This is acceptable if we don't do it for every prime every step?
                # We do it for every prime at every uncovered n.
                # Number of uncovered n is roughly L_MAX / (avg coverage).
                # This might be slow.
                
                # Let's use a faster heuristic:
                # score = 1.0 / p * (some factor) ??
                # Inspiration 1's score = gain * p.
                # Let's stick to Inspiration 1's logic but optimize the gain calculation.
                # We can stop counting gain if the score can't beat the current best.
                
                # Actually, let's just use the simple heuristic from Inspiration 1.
                # It worked well.
                gain = 0
                k = n
                # We only need to count up to a certain point?
                # If we cover n, that's 1.
                # The "gain" is the number of *currently uncovered* spots this prime covers.
                # We have `covered` array.
                while k <= L_MAX:
                    if not covered[k]:
                        gain += 1
                    k += p
                    
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
            
            start_k = best_r
            if start_k == 0:
                start_k = best_p
            for k in range(start_k, L_MAX + 1, best_p):
                covered[k] = 1
                
            n += 1
            
        return classes, budget

    # Run a few greedy builds to find a good starting point
    best_classes = []
    best_L = 0
    best_budget = Fraction(0)

    for _ in range(3): # 3 restarts
        if time.time() > deadline - 30:
            break
        classes, remaining_budget = greedy_build(deadline)
        if not classes:
            continue
        covered = get_coverage(classes)
        L = compute_L(covered)
        if L > best_L:
            best_L = L
            best_classes = classes
            best_budget = remaining_budget

    if not best_classes:
        # Fallback
        return [[2, 1]]

    # Local Search
    classes = best_classes
    remaining_budget = best_budget
    used_primes = set(p for p, r in classes)
    
    # Helper to quickly check if a move is valid and improves L
    # We maintain `covered` and `L`
    covered = get_coverage(classes)
    L = best_L
    
    improved = True
    no_improve_count = 0
    max_no_improve = 50
    
    while improved and time.time() < deadline - 5:
        improved = False
        U = L + 1
        if U > L_MAX:
            break
        
        # Candidates
        # 1. Add new prime
        # 2. Change existing prime
        # 3. Swap (Remove one, Add one) - Try limited swaps
        
        best_move = None
        best_new_L = L
        best_new_classes = None
        best_new_budget = remaining_budget
        
        # 1. Add new prime
        for p in PRIMES:
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if remaining_budget < cost:
                continue
            r = U % p
            # Check if adding this covers U and doesn't break prefix
            # We can simulate the change on a copy or check logic
            # Since we only add, it cannot break the prefix. It only adds coverage.
            # So we just need to check if it covers U.
            # If it covers U, the new L is at least U.
            # We need to calculate the new L.
            
            # Optimization: If we add a prime, L will extend.
            # Let's just calculate new coverage for the best candidate?
            # Or check if it covers U, then calculate L.
            
            # Does it cover U?
            # U % p == r, so yes, by definition.
            
            # Calculate new L
            # This is O(L_MAX).
            # To save time, we can just accept the move if it covers U, 
            # and update L in the next loop iteration?
            # No, we need to compare L.
            
            # Let's do a quick check:
            # If we add p, r, the new covered array is covered + new_marks.
            # We can check the new L by scanning from U.
            
            # For now, let's just mark it as a candidate and evaluate fully if it's the best?
            # No, let's evaluate fully. It's O(L_MAX) ~ 5000 ops.
            # Number of candidates: ~168 primes.
            # 168 * 5000 = 840,000 ops. Fast enough.
            
            new_covered = bytearray(covered)
            start_k = r
            if start_k == 0:
                start_k = p
            for k in range(start_k, L_MAX + 1, p):
                new_covered[k] = 1
            
            # Compute new L
            new_L = L
            for i in range(U, L_MAX + 1):
                if new_covered[i]:
                    new_L = i
                else:
                    break
            
            if new_L > best_new_L:
                best_new_L = new_L
                best_move = ('add', p, r)
                best_new_classes = classes + [[p, r]]
                best_new_budget = remaining_budget - cost
        
        # 2. Change existing prime
        for idx, (p, old_r) in enumerate(classes):
            new_r = U % p
            if new_r == old_r:
                continue
            
            # Changing a prime might uncover numbers in 1..L.
            # So we must check the whole prefix 1..U.
            
            new_covered = bytearray(covered)
            # Remove old coverage? No, other primes might cover those spots.
            # We must recalculate coverage from scratch or carefully update.
            # Recalculating is O(N * K) where K is number of classes.
            # K is small (~10-20). N is 5000.
            # 100 * 5000 = 500,000.
            
            # Let's just recalculate coverage for the trial classes.
            trial_classes = [list(c) for c in classes]
            trial_classes[idx] = [p, new_r]
            
            # Check prefix 1..U
            # We can do a quick check:
            # Is 1..U covered by trial_classes?
            
            # Optimization: Only check 1..U.
            trial_covered = bytearray(U + 1) # Only need up to U
            for pp, rr in trial_classes:
                start_k = rr
                if start_k == 0:
                    start_k = pp
                for k in range(start_k, U + 1, pp):
                    if k <= U:
                        trial_covered[k] = 1
            
            # Check if 1..U is all 1
            valid = True
            for i in range(1, U + 1):
                if not trial_covered[i]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # If valid, calculate full L
            new_covered_full = bytearray(L_MAX + 1)
            for pp, rr in trial_classes:
                start_k = rr
                if start_k == 0:
                    start_k = pp
                for k in range(start_k, L_MAX + 1, pp):
                    new_covered_full[k] = 1
            
            new_L = L
            for i in range(U, L_MAX + 1):
                if new_covered_full[i]:
                    new_L = i
                else:
                    break
            
            if new_L > best_new_L:
                best_new_L = new_L
                best_move = ('change', idx, p, new_r)
                best_new_classes = trial_classes
                best_new_budget = remaining_budget # Cost unchanged
        
        # 3. Swap (Limited)
        # Try removing a large prime (low cost, low gain?) and adding a small one?
        # Or removing a prime that is "redundant"?
        # This is complex. Let's skip for now to keep it simple and fast,
        # as Inspiration 1 did well without it.
        
        if best_move:
            classes = best_new_classes
            remaining_budget = best_new_budget
            used_primes = set(p for p, r in classes)
            covered = get_coverage(classes)
            L = best_new_L
            improved = True
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break

    # Final verification
    total_cost = sum(Fraction(1, p) for p, r in classes)
    if total_cost > 1:
        # Should not happen, but if it does, drop the largest prime?
        # For now, just return as is, hoping it's valid.
        pass

    return classes

if __name__ == "__main__":
    classes = construct()
    s = sum(Fraction(1, p) for p, r in classes)
    print(f"Sum={float(s)}, Classes={len(classes)}")
    covered = bytearray(L_MAX + 1)
    for p, r in classes:
        start = r
        if start == 0:
            start = p
        for k in range(start, L_MAX + 1, p):
            covered[k] = 1
    L = 0
    for i in range(1, L_MAX + 1):
        if covered[i]:
            L = i
        else:
            break
    print(f"L={L}")
