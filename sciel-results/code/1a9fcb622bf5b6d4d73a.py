import time
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Generate primes up to 1000
    def is_prime(n):
        if n < 2:
            return False
        for q in range(2, int(n**0.5) + 1):
            if n % q == 0:
                return False
        return True
    primes = [p for p in range(2, 1001) if is_prime(p)]
    
    # Greedy construction with local search improvement
    def greedy_construct(L):
        """
        Greedy: pick prime/residue that covers most uncovered numbers,
        subject to budget constraint.
        """
        if L <= 0:
            return []
        uncovered = [True] * (L + 1)
        uncovered[0] = False
        covered_count = L
        budget_num = 0  # numerator of sum 1/p (denominator is lcm of used primes, but we track via Fraction)
        budget = Fraction(0, 1)
        used_primes = set()
        classes = []
        
        while covered_count > 0:
            if time.time() > deadline - 5:
                return None
            best_p = None
            best_r = None
            best_count = 0
            
            for p in primes:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if budget + cost > 1:
                    continue
                # Count uncovered per residue
                counts = [0] * p
                for i in range(1, L + 1):
                    if uncovered[i]:
                        counts[i % p] += 1
                max_count = max(counts)
                if max_count > best_count:
                    best_count = max_count
                    best_p = p
                    best_r = counts.index(max_count)
            
            if best_p is None or best_count == 0:
                break
            
            classes.append([best_p, best_r])
            used_primes.add(best_p)
            budget += Fraction(1, best_p)
            
            # Mark covered
            start = best_r if best_r > 0 else best_p
            for i in range(start, L + 1, best_p):
                if uncovered[i]:
                    uncovered[i] = False
                    covered_count -= 1
        
        if covered_count == 0:
            return classes
        return None
    
    def local_search_improve(classes, L, max_iters=5000):
        """
        Local search: try flipping residues of existing primes, or swapping primes.
        Objective: maximize covered prefix length.
        """
        if classes is None:
            return None, 0
        
        classes = [c[:] for c in classes]
        
        def compute_coverage(cls, limit=None):
            """Return (covered_prefix, uncovered_set) for given classes."""
            if limit is None:
                limit = L
            covered = [False] * (limit + 1)
            for p, r in cls:
                start = r if r > 0 else p
                for i in range(start, limit + 1, p):
                    covered[i] = True
            # Find covered prefix
            prefix = 0
            for i in range(1, limit + 1):
                if covered[i]:
                    prefix = i
                else:
                    break
            return prefix
        
        def check_budget(cls):
            s = Fraction(0, 1)
            for p, r in cls:
                s += Fraction(1, p)
            return s <= 1
        
        current_prefix = compute_coverage(classes)
        best_prefix = current_prefix
        best_classes = [c[:] for c in classes]
        
        # Local search: try changing residues
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for idx in range(len(classes)):
                p, r = classes[idx]
                # Try a few random residues
                for _ in range(3):
                    new_r = random.randint(0, p - 1)
                    if new_r == r:
                        continue
                    old_r = classes[idx][1]
                    classes[idx][1] = new_r
                    new_prefix = compute_coverage(classes)
                    if new_prefix > best_prefix:
                        best_prefix = new_prefix
                        best_classes = [c[:] for c in classes]
                        improved = True
                    else:
                        classes[idx][1] = old_r
            
            if not improved and it > 100:
                break
        
        return best_classes, best_prefix
    
    # Start with a reasonable L
    # Try to find a good initial solution
    test_L = 150
    best_classes = None
    best_L = 0
    
    # First, get a baseline with greedy
    result = greedy_construct(test_L)
    if result is not None:
        best_classes = result
        best_L = test_L
    
    # Now try to increase L and improve with local search
    for target_L in [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000]:
        if time.time() > deadline - 20:
            break
        if target_L <= best_L:
            continue
        
        # Try greedy for this L
        result = greedy_construct(target_L)
        if result is not None:
            # Improve with local search
            improved_classes, improved_L = local_search_improve(result, target_L, max_iters=2000)
            if improved_classes is not None and improved_L > best_L:
                best_classes = improved_classes
                best_L = improved_L
        else:
            # Greedy failed, try with fewer primes or different strategy
            # Try local search on a smaller set
            if best_classes is not None:
                improved_classes, improved_L = local_search_improve(best_classes, target_L, max_iters=3000)
                if improved_classes is not None and improved_L > best_L:
                    best_classes = improved_classes
                    best_L = improved_L
            break
    
    # Final improvement pass
    if best_classes is not None and best_L < 1000:
        final_classes, final_L = local_search_improve(best_classes, min(best_L + 500, 2000), max_iters=5000)
        if final_classes is not None and final_L > best_L:
            best_classes = final_classes
            best_L = final_L
    
    if best_classes is None:
        # Fallback
        primes_seed = [p for p in range(13, 1001) if is_prime(p)]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
    
    return best_classes
