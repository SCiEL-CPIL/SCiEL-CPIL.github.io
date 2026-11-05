"""Improved program for odd covering systems with moduli dividing 51975."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    base = 51975
    deadline = time.time() + 135  # safety margin before 150s limit
    
    # All odd divisors of 51975 = 3^3 * 5^2 * 7 * 11, greater than 1
    # 51975 = 27 * 25 * 7 * 11
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    
    # Precompute residue indices for each divisor for fast coverage
    residue_indices = {}
    for m in divisors:
        for r in range(m):
            idx = np.arange(r, base, m)
            residue_indices[(m, r)] = idx
    
    def count_uncovered(classes_list):
        """Count uncovered residues for a list of [m, r] classes."""
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes_list:
            idx = residue_indices[(m, r)]
            covered[idx] = 1
        return base - int(covered.sum())
    
    def greedy_construct(order):
        """Greedy construction with given divisor order."""
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        for m in order:
            if time.time() > deadline:
                break
            best_r = 0
            best_count = -1
            for r in range(m):
                idx = residue_indices[(m, r)]
                uncovered = int((1 - covered[idx]).sum())
                if uncovered > best_count:
                    best_count = uncovered
                    best_r = r
            classes.append([m, best_r])
            idx = residue_indices[(m, best_r)]
            covered[idx] = 1
        return classes
    
    def local_search(classes, max_passes=5):
        """Iterative local search to improve residue assignments."""
        current_score = count_uncovered(classes)
        improved = True
        passes = 0
        while improved and time.time() < deadline - 10 and passes < max_passes:
            improved = False
            passes += 1
            for i in range(len(classes)):
                if time.time() > deadline - 10:
                    break
                m, r = classes[i]
                best_r = r
                best_score = current_score
                for new_r in range(m):
                    if new_r == r:
                        continue
                    classes[i][1] = new_r
                    score = count_uncovered(classes)
                    if score < best_score:
                        best_score = score
                        best_r = new_r
                    else:
                        classes[i][1] = r
                if best_r != r:
                    classes[i][1] = best_r
                    current_score = best_score
                    improved = True
        return classes, current_score
    
    def try_remove_classes(classes):
        """Try removing classes that don't help."""
        current_score = count_uncovered(classes)
        i = 0
        while i < len(classes) and time.time() < deadline - 15:
            saved = classes.pop(i)
            score_without = count_uncovered(classes)
            if score_without <= current_score:
                current_score = score_without
            else:
                classes.insert(i, saved)
                i += 1
        return classes, current_score
    
    # Strategy 1: Deterministic greedy with largest moduli first
    divisors_desc = sorted(divisors, reverse=True)
    classes1 = greedy_construct(divisors_desc)
    classes1, score1 = local_search(classes1, max_passes=8)
    classes1, score1 = try_remove_classes(classes1)
    
    # Strategy 2: Randomized greedy with adaptive ordering
    best_classes = classes1
    best_score = score1
    
    # Multiple randomized attempts
    n_random = 5
    for attempt in range(n_random):
        if time.time() > deadline - 60:
            break
        
        # Create a randomized order: mix of large and small moduli
        order = list(divisors)
        random.shuffle(order)
        # Bias towards larger moduli being considered earlier
        # Sort by a random weight + size bias
        weights = [random.random() * 0.5 + (m / max(divisors)) * 0.5 for m in order]
        order.sort(key=lambda m: -weights[divisors.index(m)])
        
        classes_r = greedy_construct(order)
        if not classes_r:
            continue
        classes_r, score_r = local_search(classes_r, max_passes=4)
        classes_r, score_r = try_remove_classes(classes_r)
        
        if score_r < best_score:
            best_score = score_r
            best_classes = classes_r
    
    # Strategy 3: Start from best, do more intensive local search
    if time.time() < deadline - 20:
        best_classes, best_score = local_search(best_classes, max_passes=15)
        best_classes, best_score = try_remove_classes(best_classes)
        # Another local search pass
        if time.time() < deadline - 10:
            best_classes, best_score = local_search(best_classes, max_passes=10)
    
    # Final verification and return
    return best_classes
# EVOLVE-BLOCK-END
