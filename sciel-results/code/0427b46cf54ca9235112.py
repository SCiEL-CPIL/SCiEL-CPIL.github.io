import numpy as np
import time
import random

def construct():
    N = 51975
    start_time = time.time()
    deadline = start_time + 115.0

    # Get all divisors > 1 of 51975 = 3^3 * 5^2 * 7 * 11
    divisors = []
    for i in range(2, int(N**0.5) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    if N not in divisors:
        divisors.append(N)
    divisors.sort()
    
    # Build divisor hierarchy: for each m, find its divisors that are also in our set
    div_set = set(divisors)
    m_divisors = {}  # m -> list of divisors d of m where d is in div_set and d < m
    for m in divisors:
        m_divisors[m] = [d for d in divisors if d < m and m % d == 0]
    
    # Sort by size descending (larger moduli first - they cover fewer residues but are "finer")
    divs_desc = sorted(divisors, reverse=True)
    divs_asc = sorted(divisors)
    
    # Precompute bit masks for each (m, r) pair
    # mask[m][r] = integer with bits set at positions r, r+m, r+2m, ... < N
    masks = {}
    for m in divisors:
        masks[m] = []
        for r in range(m):
            # Build bit mask for positions r, r+m, r+2m, ...
            mask = 0
            pos = r
            while pos < N:
                mask |= (1 << pos)
                pos += m
            masks[m].append(mask)
    
    def popcount(x):
        """Count set bits in integer x."""
        return bin(x).count('1')
    
    def coverage_count(covered_bits):
        """Count uncovered residues."""
        total_bits = (1 << N) - 1
        uncovered = total_bits & ~covered_bits
        return popcount(uncovered)
    
    def find_best_residue(covered_bits, m):
        """Find residue r for modulus m that covers most uncovered residues."""
        total_bits = (1 << N) - 1
        uncovered = total_bits & ~covered_bits
        best_r = 0
        best_gain = -1
        for r in range(m):
            gain = popcount(uncovered & masks[m][r])
            if gain > best_gain:
                best_gain = gain
                best_r = r
        return best_r, best_gain
    
    # Phase 1: Hierarchy-aware greedy
    # Process larger moduli first. For each, pick the residue that maximizes new coverage.
    # Key insight: larger moduli are subsets of smaller ones, so we should place them first
    # to "reserve" coverage in specific residue classes.
    
    def greedy_hierarchy(order, time_limit):
        covered = 0
        used = set()
        classes = []
        
        for m in order:
            if time.time() > time_limit:
                break
            if m in used:
                continue
            r, gain = find_best_residue(covered, m)
            if gain > 0:
                classes.append([m, r])
                used.add(m)
                covered |= masks[m][r]
        
        return classes, covered
    
    def greedy_any_order(time_limit):
        """Greedy that at each step picks the (m, r) with maximum gain from all unused moduli."""
        covered = 0
        used = set()
        classes = []
        
        while time.time() < time_limit:
            best_gain = 0
            best_m = None
            best_r = None
            
            for m in divisors:
                if m in used:
                    continue
                r, gain = find_best_residue(covered, m)
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
            
            if best_m is None or best_gain <= 0:
                break
            
            classes.append([best_m, best_r])
            used.add(best_m)
            covered |= masks[m][best_r]
        
        return classes, covered
    
    # Phase 2: Local search - try swapping residues
    def local_search(classes, covered, time_limit, rng):
        best_classes = classes[:]
        best_covered = covered
        best_score = coverage_count(covered)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            n = len(best_classes)
            
            # Try changing residue of each class
            for i in range(n):
                if time.time() > time_limit:
                    break
                m, r_old = best_classes[i]
                
                # Remove this class
                covered_without = best_covered & ~masks[m][r_old]
                
                # Try all other residues for this m
                for r_new in range(m):
                    if r_new == r_old:
                        continue
                    new_covered = covered_without | masks[m][r_new]
                    new_score = coverage_count(new_covered)
                    if new_score < best_score:
                        best_score = new_score
                        best_classes[i] = [m, r_new]
                        best_covered = new_covered
                        improved = True
            
            # Try adding an unused class
            used_mods = set(m for m, r in best_classes)
            for m in divisors:
                if time.time() > time_limit:
                    break
                if m in used_mods:
                    continue
                r, gain = find_best_residue(best_covered, m)
                if gain > 0:
                    new_covered = best_covered | masks[m][r]
                    new_score = coverage_count(new_covered)
                    if new_score < best_score:
                        best_score = new_score
                        best_classes.append([m, r])
                        used_mods.add(m)
                        best_covered = new_covered
                        improved = True
            
            # Try removing a class that doesn't help
            for i in range(len(best_classes)):
                if time.time() > time_limit:
                    break
                m, r = best_classes[i]
                without = best_covered & ~masks[m][r]
                score_without = coverage_count(without)
                if score_without < best_score:
                    best_score = score_without
                    best_classes.pop(i)
                    best_covered = without
                    improved = True
        
        return best_classes, best_covered, best_score
    
    # Phase 3: Simulated annealing / random restarts
    def random_restart(time_limit, rng):
        covered = 0
        used = set()
        classes = []
        
        # Random order
        order = divisors[:]
        rng.shuffle(order)
        
        for m in order:
            if time.time() > time_limit:
                break
            if m in used:
                continue
            r, gain = find_best_residue(covered, m)
            if gain > 0:
                classes.append([m, r])
                used.add(m)
                covered |= masks[m][r]
        
        return classes, covered
    
    # Main search
    best_classes = []
    best_score = N  # worst case: nothing covered
    
    # Try multiple strategies
    rng = random.Random(42)
    
    # Strategy 1: Descending order greedy
    if time.time() < deadline - 30:
        classes, covered = greedy_hierarchy(divs_desc, deadline - 25)
        score = coverage_count(covered)
        if score < best_score:
            best_score = score
            best_classes = classes[:]
    
    # Strategy 2: Ascending order greedy
    if time.time() < deadline - 30:
        classes, covered = greedy_hierarchy(divs_asc, deadline - 25)
        score = coverage_count(covered)
        if score < best_score:
            best_score = score
            best_classes = classes[:]
    
    # Strategy 3: Any-order greedy
    if time.time() < deadline - 30:
        classes, covered = greedy_any_order(deadline - 25)
        score = coverage_count(covered)
        if score < best_score:
            best_score = score
            best_classes = classes[:]
    
    # Local search on best
    if time.time() < deadline - 15 and best_classes:
        # Rebuild covered from best_classes
        covered = 0
        for m, r in best_classes:
            covered |= masks[m][r]
        best_classes, covered, best_score = local_search(best_classes, covered, deadline - 10, rng)
    
    # Random restarts with local search
    num_restarts = 0
    while time.time() < deadline - 10:
        num_restarts += 1
        classes, covered = random_restart(deadline - 15, rng)
        if not classes:
            continue
        score = coverage_count(covered)
        
        # Local search on this restart
        if time.time() < deadline - 5:
            classes, covered, score = local_search(classes, covered, deadline - 5, rng)
        
        if score < best_score:
            best_score = score
            best_classes = classes[:]
    
    # Final local search on the best
    if time.time() < deadline - 3 and best_classes:
        covered = 0
        for m, r in best_classes:
            covered |= masks[m][r]
        best_classes, covered, best_score = local_search(best_classes, covered, deadline - 1, rng)
    
    # Validate: ensure distinct moduli, valid residues
    final_classes = []
    used = set()
    for m, r in best_classes:
        if m > 1 and m in div_set and 0 <= r < m and m not in used:
            final_classes.append([m, r])
            used.add(m)
    
    return final_classes
