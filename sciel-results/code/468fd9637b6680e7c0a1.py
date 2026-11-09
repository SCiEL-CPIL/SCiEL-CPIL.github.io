"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _get_primes(limit=1000):
    """Return list of primes up to limit."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

def _compute_coverage(classes, cap=100000):
    """Compute the largest L such that every integer 1..L is covered."""
    if not classes:
        return 0
    covered = [False] * (cap + 2)
    for p, r in classes:
        start = r if r >= 1 else p
        for n in range(start, cap + 1, p):
            covered[n] = True
    L = 0
    for n in range(1, cap + 1):
        if covered[n]:
            L = n
        else:
            break
    return L

def _compute_coverage_fast(classes, cap=100000):
    """Faster coverage computation using numpy-like approach with bitsets."""
    if not classes:
        return 0
    # Use a bytearray for speed
    covered = bytearray(cap + 2)
    for p, r in classes:
        start = r if r >= 1 else p
        for n in range(start, cap + 1, p):
            covered[n] = 1
    L = 0
    for n in range(1, cap + 1):
        if covered[n]:
            L = n
        else:
            break
    return L

def _sum_reciprocals(classes):
    """Compute sum of 1/p as Fraction."""
    s = Fraction(0)
    for p, r in classes:
        s += Fraction(1, p)
    return s

def _greedy_construct(primes, deadline):
    """Greedy construction: pick primes to maximize consecutive coverage."""
    classes = []
    current_sum = Fraction(0)
    covered = bytearray(100002)
    
    # Start with small primes that give good coverage
    # Sort primes by size - smaller primes give more frequent hits
    for p in sorted(primes):
        if time.time() > deadline:
            break
        new_sum = current_sum + Fraction(1, p)
        if new_sum > Fraction(1):
            continue
        
        # Try each residue and pick the one that extends coverage most
        best_r = None
        best_L = 0
        # Only try a subset of residues for speed
        for r in range(p):
            # Check if this residue helps extend coverage
            start = r if r >= 1 else p
            # Quick check: does this cover any new numbers in 1..500?
            temp_covered = bytearray(100002)
            for pp, rr in classes:
                s = rr if rr >= 1 else pp
                for n in range(s, 501, pp):
                    temp_covered[n] = 1
            for n in range(start, 501, p):
                temp_covered[n] = 1
            
            L = 0
            for n in range(1, 501):
                if temp_covered[n]:
                    L = n
                else:
                    break
            
            if L > best_L:
                best_L = L
                best_r = r
        
        if best_r is not None and best_L > 0:
            classes.append([p, best_r])
            current_sum = new_sum
            
            # Update covered array
            start = best_r if best_r >= 1 else p
            for n in range(start, 100001, p):
                covered[n] = 1
    
    return classes, current_sum

def _local_search(classes, primes, best_L, best_sum, deadline):
    """Local search to improve coverage."""
    random.seed(42)
    current_classes = [c[:] for c in classes]
    current_sum = best_sum
    current_L = best_L
    no_improve_count = 0
    
    while time.time() < deadline:
        if no_improve_count > 500:
            break
            
        improvement = False
        strategy = random.random()
        
        if strategy < 0.25:
            # Strategy 1: Change residue of a random class
            if current_classes:
                idx = random.randint(0, len(current_classes) - 1)
                old_p, old_r = current_classes[idx]
                new_r = random.randint(0, old_p - 1)
                if new_r != old_r:
                    current_classes[idx] = [old_p, new_r]
                    new_L = _compute_coverage_fast(current_classes)
                    if new_L > current_L:
                        current_L = new_L
                        improvement = True
                    else:
                        current_classes[idx] = [old_p, old_r]
        
        elif strategy < 0.45:
            # Strategy 2: Targeted residue change - focus on extending coverage
            if current_classes:
                # Find which classes are most relevant to the current boundary
                gap = current_L + 1
                # Try changing residues of classes that could cover the gap
                candidates = []
                for idx, (p, r) in enumerate(current_classes):
                    # Check if changing this residue could help
                    if gap % p != r:
                        candidates.append(idx)
                
                if candidates:
                    idx = random.choice(candidates)
                    old_p, old_r = current_classes[idx]
                    new_r = gap % old_p
                    if new_r != old_r:
                        current_classes[idx] = [old_p, new_r]
                        new_L = _compute_coverage_fast(current_classes)
                        if new_L > current_L:
                            current_L = new_L
                            improvement = True
                        else:
                            # Try a few more residues
                            for offset in range(1, min(5, old_p)):
                                new_r2 = (gap + offset) % old_p
                                if new_r2 != old_r and new_r2 != new_r:
                                    current_classes[idx] = [old_p, new_r2]
                                    new_L2 = _compute_coverage_fast(current_classes)
                                    if new_L2 > current_L:
                                        current_L = new_L2
                                        improvement = True
                                        break
                            if not improvement:
                                current_classes[idx] = [old_p, old_r]
        
        elif strategy < 0.65:
            # Strategy 3: Swap residues between two classes
            if len(current_classes) >= 2:
                idx1, idx2 = random.sample(range(len(current_classes)), 2)
                p1, r1 = current_classes[idx1]
                p2, r2 = current_classes[idx2]
                current_classes[idx1] = [p1, r2 % p1]
                current_classes[idx2] = [p2, r1 % p2]
                new_L = _compute_coverage_fast(current_classes)
                if new_L > current_L:
                    current_L = new_L
                    improvement = True
                else:
                    current_classes[idx1] = [p1, r1]
                    current_classes[idx2] = [p2, r2]
        
        elif strategy < 0.80:
            # Strategy 4: Replace a prime with another
            if current_classes:
                idx = random.randint(0, len(current_classes) - 1)
                old_p, old_r = current_classes[idx]
                used_primes = set(p for p, r in current_classes)
                candidate_primes = [p for p in primes if p not in used_primes]
                if candidate_primes:
                    # Prefer primes that are close in size to maintain budget
                    candidate_primes.sort(key=lambda p: abs(1/p - 1/old_p))
                    new_p = candidate_primes[0]
                    new_sum = current_sum - Fraction(1, old_p) + Fraction(1, new_p)
                    if new_sum <= Fraction(1):
                        gap = current_L + 1
                        new_r = gap % new_p
                        current_classes[idx] = [new_p, new_r]
                        new_L = _compute_coverage_fast(current_classes)
                        if new_L > current_L:
                            current_L = new_L
                            current_sum = new_sum
                            improvement = True
                        else:
                            current_classes[idx] = [old_p, old_r]
        
        elif strategy < 0.90:
            # Strategy 5: Add a new prime if budget allows
            if current_sum < Fraction(1):
                used_primes = set(p for p, r in current_classes)
                candidate_primes = [p for p in primes if p not in used_primes]
                if candidate_primes:
                    # Prefer larger primes to save budget
                    candidate_primes.sort(reverse=True)
                    for new_p in candidate_primes[:5]:
                        new_sum = current_sum + Fraction(1, new_p)
                        if new_sum <= Fraction(1):
                            gap = current_L + 1
                            new_r = gap % new_p
                            current_classes.append([new_p, new_r])
                            new_L = _compute_coverage_fast(current_classes)
                            if new_L > current_L:
                                current_L = new_L
                                current_sum = new_sum
                                improvement = True
                                break
                            else:
                                current_classes.pop()
        
        else:
            # Strategy 6: Remove a poorly contributing class
            if len(current_classes) > 5:
                # Find class with least impact
                best_idx = -1
                best_L_after_removal = -1
                for idx in range(len(current_classes)):
                    temp = current_classes[:idx] + current_classes[idx+1:]
                    L_temp = _compute_coverage_fast(temp)
                    if L_temp > best_L_after_removal:
                        best_L_after_removal = L_temp
                        best_idx = idx
                
                if best_idx >= 0:
                    removed = current_classes.pop(best_idx)
                    new_L = _compute_coverage_fast(current_classes)
                    if new_L >= current_L:
                        current_L = new_L
                        current_sum = current_sum - Fraction(1, removed[0])
                        improvement = True
                    else:
                        current_classes.insert(best_idx, removed)
        
        if improvement:
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    return current_classes, current_L, current_sum

def _refine_residues(classes, primes, deadline):
    """Final refinement pass to optimize residues."""
    current_classes = [c[:] for c in classes]
    current_L = _compute_coverage_fast(current_classes)
    
    # Try to improve by adjusting residues of existing classes
    for _ in range(20):
        if time.time() > deadline:
            break
        
        improved = False
        for idx, (p, r) in enumerate(current_classes):
            if time.time() > deadline:
                break
            
            # Try a few nearby residues
            for delta in [-2, -1, 1, 2]:
                new_r = (r + delta) % p
                if new_r == r:
                    continue
                current_classes[idx] = [p, new_r]
                new_L = _compute_coverage_fast(current_classes)
                if new_L > current_L:
                    current_L = new_L
                    improved = True
                    break
                else:
                    current_classes[idx] = [p, r]
            
            if improved:
                break
    
    return current_classes, current_L

def construct():
    deadline = time.time() + 115  # 115s safety margin for 120s limit
    primes = _get_primes(1000)
    
    # Phase 1: Greedy construction
    classes, total_sum = _greedy_construct(primes, deadline)
    best_L = _compute_coverage_fast(classes)
    best_classes = [c[:] for c in classes]
    best_sum = total_sum
    
    # Phase 2: Local search
    classes, L, s = _local_search(best_classes, primes, best_L, best_sum, deadline)
    if L > best_L:
        best_L = L
        best_classes = classes
        best_sum = s
    
    # Phase 3: Final refinement
    classes, L = _refine_residues(best_classes, primes, deadline)
    if L > best_L:
        best_L = L
        best_classes = classes
    
    return best_classes
# EVOLVE-BLOCK-END
