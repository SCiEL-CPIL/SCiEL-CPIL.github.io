import time
import numpy as np
import random

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    # Precompute all odd divisors of 51975 > 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()  # ascending order
    
    def score_solution(classes):
        """Count uncovered residues for a list of [m, r] classes."""
        if not classes:
            return base
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            covered[r::m] = 1
        return int(np.sum(covered == 0))
    
    def greedy_pass(div_order, base, deadline, covered_init=None):
        """One greedy pass: pick best (m,r) at each step from div_order."""
        covered = np.zeros(base, dtype=np.uint8) if covered_init is None else covered_init.copy()
        classes = []
        available = set(div_order)
        
        while available:
            if time.time() > deadline - 5.0:
                break
            
            best_gain = -1
            best_m = None
            best_r = None
            
            for m in div_order:
                if m not in available:
                    continue
                
                # Max possible coverage for this modulus
                max_possible = (base + m - 1) // m
                if max_possible <= best_gain:
                    continue
                
                # Find best residue for this modulus
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
            
            if best_m is None or best_gain <= 0:
                break
            
            classes.append([best_m, best_r])
            covered[best_r::best_m] = 1
            available.remove(best_m)
        
        return classes, covered
    
    def local_search(classes, base, deadline):
        """Local search: try changing residues and swapping classes."""
        if not classes:
            return classes, base
        
        best_classes = [c[:] for c in classes]
        best_score = score_solution(best_classes)
        
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            
            # Try changing residue for each class
            for idx in range(len(best_classes)):
                if time.time() > deadline - 10.0:
                    break
                m, r = best_classes[idx]
                # Try a few alternative residues
                for r2 in [0, m//2, m-1, (r+1)%m, (r-1)%m]:
                    if r2 == r:
                        continue
                    new_classes = [c[:] for c in best_classes]
                    new_classes[idx] = [m, r2]
                    new_score = score_solution(new_classes)
                    if new_score < best_score:
                        best_score = new_score
                        best_classes = new_classes
                        improved = True
                        break
                if improved:
                    break
            
            # Try swapping two classes
            if not improved and len(best_classes) >= 2:
                for i in range(len(best_classes)):
                    if time.time() > deadline - 10.0:
                        break
                    for j in range(i+1, len(best_classes)):
                        new_classes = [c[:] for c in best_classes]
                        new_classes[i], new_classes[j] = new_classes[j], new_classes[i]
                        new_score = score_solution(new_classes)
                        if new_score < best_score:
                            best_score = new_score
                            best_classes = new_classes
                            improved = True
                            break
                    if improved:
                        break
        
        return best_classes, best_score
    
    best_score = base
    best_classes = []
    
    # Strategy 1: Greedy with ascending divisors
    if time.time() < deadline - 30.0:
        classes, covered = greedy_pass(divisors, base, deadline)
        score = int(np.sum(covered == 0))
        if score < best_score:
            best_score = score
            best_classes = classes
    
    # Strategy 2: Greedy with descending divisors
    if time.time() < deadline - 30.0:
        div_desc = list(reversed(divisors))
        classes, covered = greedy_pass(div_desc, base, deadline)
        score = int(np.sum(covered == 0))
        if score < best_score:
            best_score = score
            best_classes = classes
    
    # Strategy 3: Randomized greedy attempts
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline - 40.0 and attempts < 5:
        attempts += 1
        div_rand = list(divisors)
        rng.shuffle(div_rand)
        classes, covered = greedy_pass(div_rand, base, deadline)
        score = int(np.sum(covered == 0))
        if score < best_score:
            best_score = score
            best_classes = classes
    
    # Local search refinement
    if best_classes and time.time() < deadline - 20.0:
        best_classes, best_score = local_search(best_classes, base, deadline)
    
    return best_classes if best_classes else []
