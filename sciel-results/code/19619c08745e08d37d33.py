import numpy as np

def construct():
    base = 51975
    # Get all divisors > 1 of 51975
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    # Sort in decreasing order (largest modulus first) for greedy
    divisors.sort(reverse=True)
    
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    
    # Greedy phase: process moduli in decreasing order
    for m in divisors:
        best_r = 0
        best_count = -1
        for r in range(m):
            count = int(np.sum(covered[r::m] == 0))
            if count > best_count:
                best_count = count
                best_r = r
        classes.append([m, best_r])
        covered[best_r::m] = 1
    
    # Local search phase: iteratively improve residue assignments
    improved = True
    max_iters = 50
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(len(classes)):
            m, r = classes[i]
            # Uncover this class's current residue
            covered[r::m] = 0
            best_r = r
            best_uncovered = int(base - np.sum(covered))
            for r2 in range(m):
                if r2 == r:
                    continue
                covered[r2::m] = 1
                uncovered = int(base - np.sum(covered))
                if uncovered < best_uncovered:
                    best_uncovered = uncovered
                    best_r = r2
                covered[r2::m] = 0
            # Re-cover with best residue
            classes[i] = [m, best_r]
            covered[best_r::m] = 1
            if best_r != r:
                improved = True
    
    return classes
