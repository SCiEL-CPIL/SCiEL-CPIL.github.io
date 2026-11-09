"""Combined approach: NumPy-accelerated greedy with multiple restarts and local search."""
import time
import numpy as np

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    # Precompute all odd divisors of 51975 > 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()  # ascending order

    # Precompute index arrays for each divisor: indices[m] = array of positions covered by residue r
    # For modulus m, residue r covers positions r, r+m, r+2m, ... < base
    # We'll compute these on the fly but cache the "range" structure
    def get_indices(m, r):
        """Return numpy array of indices covered by class (m, r)."""
        return np.arange(r, base, m)

    def score_solution(classes):
        """Count uncovered residues for a list of [m, r] classes."""
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            covered[r::m] = 1
        return int(np.sum(covered == 0))

    def greedy_once(divisors, base, deadline, randomize_order=False, seed=None):
        """One greedy pass: pick best (m,r) at each step."""
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        available = set(divisors)
        
        if randomize_order and seed is not None:
            import random
            rng = random.Random(seed)
            order = list(available)
            rng.shuffle(order)
        else:
            order = sorted(available)
        
        while available:
            if time.time() > deadline - 3.0:
                break
            
            best_gain = -1
            best_m = None
            best_r = None
            
            for m in order:
                if m not in available:
                    continue
                # Max possible coverage for this modulus
                max_possible = (base + m - 1) // m
                if max_possible <= best_gain:
                    continue
                
                # Find best residue for this modulus
                # Count uncovered in each residue class
                # For speed, compute all at once using reshape trick or loop
                # Use numpy: for each r in range(m), count zeros in covered[r::m]
                # This is O(m * base/m) = O(base) per modulus, which is fine
                
                best_r_for_m = 0
                best_count_for_m = -1
                for r in range(m):
                    cnt = int(np.sum(covered[r::m] == 0))
                    if cnt > best_count_for_m:
                        best_count_for_m = cnt
                        best_r_for_m = r
                
                if best_count_for_m > best_gain:
                    best_gain = best_count_for_m
                    best_m = m
                    best_r = best_r_for_m
                
                # Early exit if we've covered everything
                if best_gain == int(np.sum(covered == 0)):
                    break
            
            if best_m is None or best_gain <= 0:
                break
            
            classes.append([best_m, best_r])
            covered[best_r::best_m] = 1
            available.discard(best_m)
            
            if int(np.sum(covered == 0)) == 0:
                break
        
        return classes, int(np.sum(covered == 0))

    def local_search(classes, base, deadline):
        """Improve solution by changing residues of existing moduli."""
        if not classes:
            return classes, base
        
        best_classes = [c[:] for c in classes]
        best_score = score_solution(best_classes)
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for idx in range(len(best_classes)):
                m, r = best_classes[idx]
                # Try all residues for this modulus
                for r2 in range(m):
                    if r2 == r:
                        continue
                    # Compute new score efficiently
                    old_class = (m, r)
                    new_class = (m, r2)
                    
                    # Count how many residues are covered ONLY by the old class
                    # and how many are newly covered by the new class
                    # Simple approach: recompute
                    covered = np.zeros(base, dtype=np.uint8)
                    for i, (m2, r2_) in enumerate(best_classes):
                        if i == idx:
                            covered[r2::m2] = 1
                        else:
                            covered[r2_::m2] = 1
                    new_score = int(np.sum(covered == 0))
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_classes[idx] = [m, r2]
                        improved = True
                        break
                if improved:
                    break
        
        return best_classes, best_score

    # Phase 1: Multiple greedy restarts
    best_classes = None
    best_score = base
    
    # First, do a deterministic greedy
    classes, score = greedy_once(divisors, base, deadline)
    if score < best_score:
        best_score = score
        best_classes = classes
    
    if best_score == 0:
        return best_classes

    # More restarts with different orderings
    import random
    attempt = 0
    while time.time() < deadline - 15.0:
        attempt += 1
        seed = attempt * 7919
        classes, score = greedy_once(divisors, base, deadline, randomize_order=True, seed=seed)
        if score < best_score:
            best_score = score
            best_classes = classes
            if best_score == 0:
                break

    # Phase 2: Local search on best solution
    if best_classes is not None and time.time() < deadline - 5.0:
        best_classes, best_score = local_search(best_classes, base, deadline)

    return best_classes if best_classes is not None else []
