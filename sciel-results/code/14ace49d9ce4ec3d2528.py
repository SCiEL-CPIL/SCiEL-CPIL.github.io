import time
import random

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Generate all divisors of 51975 greater than 1
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    # Sort by size for efficiency
    divisors.sort()
    
    # Helper: count uncovered residues covered by class (m, r)
    def count_uncovered(covered, m, r):
        # covered[i] = 0 means uncovered
        count = 0
        for i in range(r, base, m):
            if covered[i] == 0:
                count += 1
        return count
    
    # Helper: apply class to covered
    def apply_class(covered, m, r):
        for i in range(r, base, m):
            covered[i] = 1
    
    best_score = base  # worst case: nothing covered
    best_classes = None
    
    # Multiple randomized greedy attempts
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        covered = bytearray(base)  # all zeros = uncovered
        classes = []
        used_moduli = set()
        
        # Randomized greedy: at each step, pick best (m,r) from available
        # with some randomization
        for step in range(len(divisors)):
            if time.time() > deadline - 2.0:
                break
            
            best_gain = -1
            best_m = None
            best_r = None
            
            # Try all available moduli
            for m in divisors:
                if m in used_moduli:
                    continue
                
                # For each residue, compute gain
                # Optimization: only need to check residues 0..m-1
                # but we can sample if m is large
                if m > 1000:
                    # Sample residues
                    residues_to_try = random.sample(range(m), min(m, 50))
                else:
                    residues_to_try = range(m)
                
                for r in residues_to_try:
                    gain = count_uncovered(covered, m, r)
                    if gain > best_gain:
                        best_gain = gain
                        best_m = m
                        best_r = r
            
            if best_m is None or best_gain <= 0:
                break
            
            classes.append([best_m, best_r])
            used_moduli.add(best_m)
            apply_class(covered, best_m, best_r)
        
        # Compute score
        score = sum(1 for i in range(base) if covered[i] == 0)
        
        if score < best_score:
            best_score = score
            best_classes = [c[:] for c in classes]
        
        # If perfect, stop
        if best_score == 0:
            break
    
    # Local search on best solution if time permits
    if best_classes is not None and time.time() < deadline - 10.0:
        covered = bytearray(base)
        for m, r in best_classes:
            apply_class(covered, m, r)
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for idx in range(len(best_classes)):
                m, r = best_classes[idx]
                # Try changing residue for this modulus
                for r2 in range(m):
                    if r2 == r:
                        continue
                    # Compute new score
                    # Remove old class, add new class
                    old_covered = bytearray(covered)
                    # Reset positions covered only by this class... this is complex
                    # Simpler: recompute
                    new_covered = bytearray(base)
                    for i, (m2, r2_) in enumerate(best_classes):
                        if i == idx:
                            r2_ = r2
                        apply_class(new_covered, m2, r2_)
                    new_score = sum(1 for i in range(base) if new_covered[i] == 0)
                    if new_score < best_score:
                        best_score = new_score
                        best_classes[idx] = [m, r2]
                        covered = new_covered
                        improved = True
                        break
                if improved:
                    break
    
    return best_classes if best_classes is not None else []
