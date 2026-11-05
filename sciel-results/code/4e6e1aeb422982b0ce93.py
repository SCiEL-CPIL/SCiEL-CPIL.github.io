"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    base = 51975
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # All odd divisors of 51975 = 3^3 * 5^2 * 7 * 11, greater than 1
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    
    # Greedy: larger moduli first (they cover fewer residues per class, so harder to cover)
    divisors.sort(reverse=True)
    
    covered = np.zeros(base, dtype=np.uint8)
    classes = []  # list of [m, r]
    
    for m in divisors:
        if time.time() > deadline:
            break
        # Find residue r that covers most uncovered residues
        best_r = 0
        best_count = -1
        for r in range(m):
            count = covered[r::m].sum()
            uncovered = len(range(r, base, m)) - count
            if uncovered > best_count:
                best_count = uncovered
                best_r = r
        classes.append([m, best_r])
        covered[best_r::m] = 1
    
    # Local search: try reassigning residues for each class
    def get_uncovered_count(classes_list):
        cov = np.zeros(base, dtype=np.uint8)
        for m, r in classes_list:
            cov[r::m] = 1
        return base - cov.sum()
    
    current_score = get_uncovered_count(classes)
    
    # Iterative improvement passes
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(classes)):
            if time.time() > deadline:
                break
            m, r = classes[i]
            best_r = r
            best_score = current_score
            for new_r in range(m):
                if new_r == r:
                    continue
                classes[i][1] = new_r
                score = get_uncovered_count(classes)
                if score < best_score:
                    best_score = score
                    best_r = new_r
                else:
                    classes[i][1] = r
            if best_r != r:
                classes[i][1] = best_r
                current_score = best_score
                improved = True
    
    # Try removing classes that don't help
    if time.time() < deadline:
        i = 0
        while i < len(classes) and time.time() < deadline:
            m, r = classes[i]
            # Temporarily remove
            saved = classes.pop(i)
            score_without = get_uncovered_count(classes)
            if score_without <= current_score:
                current_score = score_without
                improved = True
            else:
                classes.insert(i, saved)
                i += 1
    
    # Final polish: one more local search pass
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(classes)):
                if time.time() > deadline - 5:
                    break
                m, r = classes[i]
                best_r = r
                best_score = current_score
                for new_r in range(m):
                    if new_r == r:
                        continue
                    classes[i][1] = new_r
                    score = get_uncovered_count(classes)
                    if score < best_score:
                        best_score = score
                        best_r = new_r
                    else:
                        classes[i][1] = r
                if best_r != r:
                    classes[i][1] = best_r
                    current_score = best_score
                    improved = True
    
    return classes
# EVOLVE-BLOCK-END
