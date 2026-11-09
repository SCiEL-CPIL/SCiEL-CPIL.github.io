"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    base = 51975
    
    # Get all odd divisors > 1 of base
    divs = sorted([d for d in range(3, base + 1, 2) if base % d == 0])
    
    # Use numpy for efficient coverage tracking
    covered = np.zeros(base, dtype=np.uint8)
    
    def greedy(order):
        """Greedy assignment: for each modulus in order, pick residue covering most uncovered."""
        cov = np.zeros(base, dtype=np.uint8)
        classes = {}
        for m in order:
            # Find residue r that covers most uncovered residues
            best_r = 0
            best_count = -1
            # Use vectorized approach: for each residue, count uncovered in that class
            # covered[r::m] gives the residues in class (m, r)
            # We want to maximize (uncovered in this class)
            # uncovered = 1 - cov
            # count uncovered in class (m, r) = sum(1 - cov[r::m])
            # = len(range) - sum(cov[r::m])
            # To speed up, we can compute all at once using reshape
            # But let's do it efficiently
            for r in range(m):
                idx = np.arange(r, base, m)
                cnt = np.count_nonzero(cov[idx] == 0)
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
            classes[m] = best_r
            cov[best_r::m] = 1
        return classes, cov
    
    def local_search(classes, max_iter=50):
        """Try to improve by reassigning residues for individual moduli."""
        cov = np.zeros(base, dtype=np.uint8)
        for m, r in classes.items():
            cov[r::m] = 1
        
        best_uncovered = np.count_nonzero(cov == 0)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            for m in list(classes.keys()):
                old_r = classes[m]
                # Remove old assignment
                cov[old_r::m] = 0
                # Find best new residue
                best_r = old_r
                best_count = -1
                for r in range(m):
                    idx = np.arange(r, base, m)
                    cnt = np.count_nonzero(cov[idx] == 0)
                    if cnt > best_count:
                        best_count = cnt
                        best_r = r
                # Apply new assignment
                cov[best_r::m] = 1
                classes[m] = best_r
                new_uncovered = np.count_nonzero(cov == 0)
                if new_uncovered < best_uncovered:
                    best_uncovered = new_uncovered
                    improved = True
                else:
                    # Revert
                    cov[best_r::m] = 0
                    cov[old_r::m] = 1
                    classes[m] = old_r
            if not improved:
                break
        
        return classes, best_uncovered
    
    best_classes = None
    best_uncovered = base
    
    # Strategy 1: Greedy ascending
    classes, cov = greedy(divs)
    unc = np.count_nonzero(cov == 0)
    if unc < best_uncovered:
        best_uncovered = unc
        best_classes = dict(classes)
    
    # Strategy 2: Greedy descending
    classes, cov = greedy(sorted(divs, reverse=True))
    unc = np.count_nonzero(cov == 0)
    if unc < best_uncovered:
        best_uncovered = unc
        best_classes = dict(classes)
    
    # Strategy 3: Random restarts
    random.seed(42)
    num_restarts = 30
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        order = divs[:]
        random.shuffle(order)
        classes, cov = greedy(order)
        unc = np.count_nonzero(cov == 0)
        if unc < best_uncovered:
            best_uncovered = unc
            best_classes = dict(classes)
    
    # Local search on best
    if best_classes is not None and time.time() < deadline:
        best_classes, best_uncovered = local_search(best_classes, max_iter=30)
    
    result = [[m, r] for m, r in best_classes.items()]
    return result
# EVOLVE-BLOCK-END
