"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # wall-clock limit with margin

    base = 51975  # 3^3 * 5^2 * 7 * 11
    full_mask = (1 << base) - 1

    # Precompute all divisors > 1 of base
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    divisors.sort()

    # Precompute bitsets for all (m, r) pairs
    # masks[m][r] is a bitset where bit i is 1 if i % m == r
    masks = {}
    for m in divisors:
        masks[m] = [0] * m
        for r in range(m):
            mask_val = 0
            # Optimization: generate bits efficiently
            # Instead of looping 51975 times for each r, we can use a pattern
            # But 51975 * sum(m) is too big for naive precomputation if we do it badly.
            # Sum of divisors of 51975 is approx 100k. 100k * 51975 is 5e9 operations. Too slow.
            
            # Better precomputation:
            # For a given m, the residues are periodic.
            # We can build the mask for r=0, then rotate? No, bit rotation is not trivial for arbitrary length.
            
            # Let's try a smarter way to build masks.
            # Actually, 51975 is small enough that we can just loop, but we need to be careful.
            # Total operations: sum_{m|N} m * N.
            # Divisors: 3,5,7,9,11,15,21,33,35,45,55,63,77,99,105,135,165,231,315,385,495,693,945,1155,1386,1845,2205,2772,3465,4356,5544,6930,11155,13860,17325,22210,25950,34635,51975... wait, I need the actual divisors.
            
            # Let's just compute the sum of divisors.
            # 51975 = 3^3 * 5^2 * 7 * 11
            # Sigma(3^3) = 1+3+9+27 = 40
            # Sigma(5^2) = 1+5+25 = 31
            # Sigma(7) = 8
            # Sigma(11) = 12
            # Sigma(51975) = 40 * 31 * 8 * 12 = 12160 * 96 = 1167360.
            # 1167360 * 51975 is approx 6e10. This is too slow in Python.
            
            # We need a faster way to generate the masks or a different strategy.
            # Strategy: Don't precompute ALL masks.
            # Compute masks on the fly?
            # Or, compute masks only for small m?
            # Or, use a different representation?
            
            # Let's stick to the bitset idea but optimize the generation.
            # We can generate the mask for r=0, then for r=1, we can shift?
            # No, the pattern is not a simple shift of the whole integer because the length is fixed to N.
            
            # Alternative:
            # For a specific m, the set of indices is {r, r+m, r+2m, ...}.
            # We can construct the integer by setting bits.
            # Number of bits set is N/m.
            # Total bit sets is sum(N/m) = N * sum(1/m).
            # Sum(1/m) for divisors of 51975.
            # This is much smaller.
            # Let's try this.
            
            pass # Placeholder for logic below

    # Revised Precomputation
    # We will store masks in a dictionary: masks[m][r]
    # To speed up, we will compute masks for all m, but we will do it efficiently.
    # Actually, let's just compute them. 6e10 bit operations is too many.
    # But wait, `mask_val |= (1 << i)` is expensive.
    
    # Let's try a different approach for the bitset generation.
    # For each m, we can create an array of bytes or use numpy to generate the mask, then convert to int?
    # Numpy is available.
    import numpy as np
    
    masks = {}
    for m in divisors:
        masks[m] = []
        # Create a boolean array for r=0
        # Actually, let's just iterate r and build the int.
        # If it's too slow, we fall back to on-the-fly for large m.
        # But 1.1M total masks is not huge if we can generate them fast.
        
        # Let's try generating them.
        for r in range(m):
            # Generate bits for indices r, r+m, ...
            # Using numpy to create the boolean array is fast.
            # But converting numpy bool array to int is tricky.
            
            # Let's use a Python loop but optimized.
            # Or, since N is fixed, we can precompute a "step" mask?
            
            # Let's try just doing it. If it takes too long, the time guard will stop it.
            # But precomputation happens before the search.
            
            # Let's estimate time. 1.1M masks. Each mask has ~N/m bits set.
            # Total bit sets ~ 1.1M * (avg N/m).
            # This might take a few seconds. Acceptable.
            
            mask_val = 0
            # Optimization: build from bytes?
            # No, let's just use the loop.
            for i in range(r, base, m):
                mask_val |= (1 << i)
            masks[m].append(mask_val)

    def compute_score(coverage):
        # Count 0 bits in coverage within the range [0, base)
        # ~coverage & full_mask gives the uncovered bits
        uncovered = (~coverage) & full_mask
        return uncovered.bit_count()

    def greedy_construct(order):
        """Greedy: process moduli in given order, each picks best residue."""
        coverage = 0
        classes = []
        for m in order:
            best_r = 0
            best_newly_covered = -1
            for r in range(m):
                mask = masks[m][r]
                # Newly covered bits = mask & ~coverage
                newly = mask & (~coverage)
                count = newly.bit_count()
                if count > best_newly_covered:
                    best_newly_covered = count
                    best_r = r
            classes.append([m, best_r])
            coverage |= masks[m][best_r]
        return classes, compute_score(coverage)

    def local_search(classes, max_iter=50000):
        """Local search: try changing one class at a time."""
        # Reconstruct coverage
        coverage = 0
        for m, r in classes:
            coverage |= masks[m][r]
        current_score = compute_score(coverage)
        
        n_classes = len(classes)
        
        for _ in range(max_iter):
            if time.time() - start_time > deadline - 5:
                break
            
            # Pick a random class to modify
            idx = random.randint(0, n_classes - 1)
            m, r_old = classes[idx]
            
            # Find the best alternative residue for this modulus
            # We want to minimize the score after swapping.
            # Score = popcount(~coverage)
            # New coverage = (coverage & ~old_mask) | new_mask
            # We want to maximize popcount(new_coverage)
            
            # old_mask is known.
            old_mask = masks[m][r_old]
            
            # We can evaluate all r in O(m) time.
            # If m is large, this might be slow, but m <= 51975.
            # Max m is 51975. 51975 iterations per step.
            # 50000 steps * 51975 ops = 2.5e9 ops. Too slow.
            
            # We need to be smarter.
            # Only evaluate if m is small?
            # Or sample?
            
            if m > 1000:
                # Sample residues
                candidates = random.sample(range(m), min(100, m))
            else:
                candidates = range(m)
            
            best_new_r = r_old
            best_new_score = current_score
            
            # Optimization:
            # coverage_without_old = coverage & ~old_mask
            coverage_without_old = coverage & ~old_mask
            
            for new_r in candidates:
                if new_r == r_old:
                    continue
                new_mask = masks[m][new_r]
                new_coverage = coverage_without_old | new_mask
                new_score = compute_score(new_coverage)
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_r = new_r
            
            if best_new_r != r_old:
                classes[idx][1] = best_new_r
                # Update coverage
                coverage = coverage_without_old | masks[m][best_new_r]
                current_score = best_new_score
                
        return classes, current_score

    def simulated_annealing(classes, max_iter=50000):
        """Simulated annealing to escape local optima."""
        coverage = 0
        for m, r in classes:
            coverage |= masks[m][r]
        current_score = compute_score(coverage)
        best_classes = [c[:] for c in classes]
        best_score = current_score
        
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.9995
        
        n_classes = len(classes)
        
        for _ in range(max_iter):
            if time.time() - start_time > deadline - 5:
                break
            if T < T_min:
                break
                
            idx = random.randint(0, n_classes - 1)
            m, r_old = classes[idx]
            
            if m > 1000:
                new_r = random.randint(0, m - 1)
            else:
                new_r = random.randint(0, m - 1)
                
            if new_r == r_old:
                continue
                
            old_mask = masks[m][r_old]
            new_mask = masks[m][new_r]
            
            coverage_without_old = coverage & ~old_mask
            new_coverage = coverage_without_old | new_mask
            new_score = compute_score(new_coverage)
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < (2.718281828 ** (-delta / T)):
                classes[idx][1] = new_r
                coverage = new_coverage
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_classes = [c[:] for c in classes]
            
            T *= cooling_rate
        
        return best_classes, best_score

    # Try multiple initial orderings
    orderings = []
    orderings.append(divisors.copy())
    orderings.append(divisors[::-1].copy())
    
    # Random orderings
    for _ in range(5):
        perm = divisors.copy()
        random.shuffle(perm)
        orderings.append(perm)
    
    best_classes = None
    best_score = base  # worst case: nothing covered
    
    # Phase 1: Greedy construction
    for i, order in enumerate(orderings):
        if time.time() - start_time > deadline - 60:
            break
        try:
            classes, sc = greedy_construct(order)
            if sc < best_score:
                best_score = sc
                best_classes = classes
        except Exception:
            pass
    
    # Phase 2: Local search
    if best_classes is not None and time.time() - start_time < deadline - 20:
        try:
            best_classes, best_score = local_search(best_classes, max_iter=20000)
        except Exception:
            pass
    
    # Phase 3: Simulated annealing
    if best_classes is not None and time.time() - start_time < deadline - 10:
        try:
            best_classes, best_score = simulated_annealing(best_classes, max_iter=50000)
        except Exception:
            pass
            
    # Phase 4: More local search if time permits
    if best_classes is not None and time.time() - start_time < deadline - 5:
        try:
            best_classes, best_score = local_search(best_classes, max_iter=10000)
        except Exception:
            pass

    return best_classes if best_classes is not None else [[3, 0]]
# EVOLVE-BLOCK-END
