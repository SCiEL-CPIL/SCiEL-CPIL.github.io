import time
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
    deadline = start_time + 140.0
    
    primes = _sieve(1000)
    L_MAX = 5000
    
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
            best_gain = -1
            
            for p in primes:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if budget < cost:
                    continue
                
                r = n % p
                # Estimate gain
                gain = 0
                k = n
                while k <= L_MAX:
                    if not covered[k]:
                        gain += 1
                    k += p
                
                # Efficiency score
                score = gain * p
                if score > best_gain:
                    best_gain = score
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

    # 1. Initial Greedy
    classes, remaining_budget = greedy_build(deadline)
    
    # 2. Local Search / Repair
    # We maintain the list of classes and the used primes set
    used_primes = set(p for p, r in classes)
    
    # Helper to check if 1..U-1 is covered
    def is_prefix_covered(covered, U):
        for i in range(1, U):
            if not covered[i]:
                return False
        return True

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        covered = get_coverage(classes)
        L = compute_L(covered)
        
        # Find first uncovered
        U = L + 1
        if U > L_MAX:
            break
        
        # Candidates to cover U
        candidates = []
        
        # Option A: Add new prime
        for p in primes:
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if remaining_budget < cost:
                continue
            r = U % p
            candidates.append(('add', p, r, cost))
            
        # Option B: Change existing prime
        for idx, (p, r) in enumerate(classes):
            if r == U % p:
                continue # Already covers U
            r_new = U % p
            # We need to check if changing r to r_new breaks 1..U-1
            # This is expensive to check fully, so we do it in the evaluation step
            candidates.append(('change', idx, p, r_new, Fraction(0))) # Cost 0
            
        # Option C: Remove redundant prime (if it doesn't reduce L)
        # Only try if we have budget to spare? No, removing frees budget.
        # But we are stuck at U, so we need to cover U. Removing won't cover U.
        # So skip removal unless we are trying to free budget for a better 'add' later?
        # Let's focus on covering U first.

        best_move = None
        best_L = L
        best_new_classes = None
        best_new_budget = remaining_budget
        
        for move in candidates:
            if time.time() > deadline - 5:
                break
                
            if move[0] == 'add':
                _, p, r, cost = move
                # Apply
                new_classes = classes + [[p, r]]
                new_budget = remaining_budget - cost
                new_covered = bytearray(L_MAX + 1)
                for pp, rr in new_classes:
                    start_k = rr
                    if start_k == 0:
                        start_k = pp
                    for k in range(start_k, L_MAX + 1, pp):
                        new_covered[k] = 1
                
                # Check prefix
                if not is_prefix_covered(new_covered, U):
                    continue
                    
                new_L = compute_L(new_covered)
                if new_L > best_L:
                    best_L = new_L
                    best_move = move
                    best_new_classes = new_classes
                    best_new_budget = new_budget
                    
            elif move[0] == 'change':
                _, idx, p, r_new, cost = move
                # Apply
                new_classes = [list(c) for c in classes]
                new_classes[idx] = [p, r_new]
                new_budget = remaining_budget # Cost same
                
                new_covered = bytearray(L_MAX + 1)
                for pp, rr in new_classes:
                    start_k = rr
                    if start_k == 0:
                        start_k = pp
                    for k in range(start_k, L_MAX + 1, pp):
                        new_covered[k] = 1
                        
                # Check prefix
                if not is_prefix_covered(new_covered, U):
                    continue
                
                new_L = compute_L(new_covered)
                if new_L > best_L:
                    best_L = new_L
                    best_move = move
                    best_new_classes = new_classes
                    best_new_budget = new_budget

        if best_move:
            classes = best_new_classes
            remaining_budget = best_new_budget
            used_primes = set(p for p, r in classes)
            improved = True
            # Small pause to allow time check? No, tight loop.
            
    # Final Check and Cleanup
    # Ensure budget is exact and <= 1
    total_cost = sum(Fraction(1, p) for p, r in classes)
    if total_cost > 1:
        # This shouldn't happen if we track budget, but just in case
        # We might need to drop some classes
        # But greedy ensures this.
        pass
        
    return classes

# EVOLVE-BLOCK-END
