import numpy as np
import time
import random

def construct():
    N = 51975
    
    # Generate divisors > 1
    divisors = []
    for i in range(1, int(N**0.5) + 1):
        if N % i == 0:
            if i > 1:
                divisors.append(i)
            j = N // i
            if j != i and j > 1:
                divisors.append(j)
    
    moduli = sorted(divisors)
    
    def compute_covered(classes):
        """Compute the covered boolean array from a list of (m, r) classes."""
        cov = np.zeros(N, dtype=np.uint8)
        for m, r in classes:
            cov[r::m] = 1
        return cov
    
    def greedy_init(order):
        """Greedy initialization: process moduli in given order, picking best residue."""
        cov = np.zeros(N, dtype=np.uint8)
        classes = []
        for m in order:
            best_r = 0
            best_gain = -1
            # Vectorized gain computation: for each r, count zeros in cov[r::m]
            # We can compute this more efficiently using a reshape trick or just loop
            # Since m * (N/m) = N total elements, looping r is N operations total per m
            # But we want to be fast. Let's use numpy.
            # For each r, gain = sum(cov[r::m] == 0)
            # We can compute all gains at once by reshaping? Not directly.
            # Let's just loop; it's fast enough for N=51975.
            for r in range(m):
                gain = np.count_nonzero(cov[r::m] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            if best_gain > 0:
                cov[best_r::m] = 1
                classes.append((m, best_r))
        return classes, cov
    
    def local_search(classes, cov, max_passes=20):
        """Local search: try reassigning residues of individual classes."""
        best_uncovered = int(np.count_nonzero(cov == 0))
        
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(classes)):
                m, r_old = classes[i]
                # Remove old
                cov[r_old::m] = 0
                # Find best new residue
                best_r = r_old
                best_gain = -1
                for r in range(m):
                    gain = np.count_nonzero(cov[r::m] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                # Apply new
                cov[best_r::m] = 1
                classes[i] = (m, best_r)
                new_uncovered = int(np.count_nonzero(cov == 0))
                if new_uncovered < best_uncovered:
                    best_uncovered = new_uncovered
                    improved = True
                else:
                    # Revert
                    cov[best_r::m] = 0
                    cov[r_old::m] = 1
                    classes[i] = (m, r_old)
            if not improved:
                break
        
        return classes, cov, best_uncovered
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_classes = []
    best_cov = np.zeros(N, dtype=np.uint8)
    best_uncovered = N
    
    # Strategy 1: Greedy ascending
    classes, cov = greedy_init(moduli)
    unc = int(np.count_nonzero(cov == 0))
    if unc < best_uncovered:
        best_uncovered = unc
        best_classes = classes[:]
        best_cov = cov.copy()
    
    # Strategy 2: Greedy descending
    classes, cov = greedy_init(sorted(moduli, reverse=True))
    unc = int(np.count_nonzero(cov == 0))
    if unc < best_uncovered:
        best_uncovered = unc
        best_classes = classes[:]
        best_cov = cov.copy()
    
    # Strategy 3: Multiple random restarts
    random.seed(42)
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        order = moduli[:]
        random.shuffle(order)
        classes, cov = greedy_init(order)
        unc = int(np.count_nonzero(cov == 0))
        if unc < best_uncovered:
            best_uncovered = unc
            best_classes = classes[:]
            best_cov = cov.copy()
    
    # Local search on the best solution found
    if time.time() < deadline:
        best_classes, best_cov, best_uncovered = local_search(best_classes, best_cov, max_passes=30)
    
    # Additional local search passes if time permits
    while time.time() < deadline - 5:
        prev_uncovered = best_uncovered
        best_classes, best_cov, best_uncovered = local_search(best_classes, best_cov, max_passes=5)
        if best_uncovered == prev_uncovered:
            break
    
    return [[m, r] for m, r in best_classes]
