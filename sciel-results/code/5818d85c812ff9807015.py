"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _score_classes(classes, base=51975):
    """Count uncovered residues."""
    covered = np.zeros(base, dtype=np.int8)
    for m, r in classes:
        covered[r::m] = 1
    return base - int(covered.sum())

def _greedy_construct(base=51975):
    """Greedy: moduli in increasing order, each takes the class covering most uncovered residues."""
    covered = bytearray(base)
    classes = []
    for m in [d for d in range(3, base + 1, 2) if base % d == 0]:
        best = max(range(m), key=lambda r: covered[r::m].count(0))
        classes.append([m, best])
        covered[best::m] = b'\x01' * len(range(best, base, m))
    return classes

def _local_search(classes, base=51975, time_limit=10):
    """Local search: try changing residues for each modulus, using efficient coverage tracking."""
    start_time = time.time()
    best_score = _score_classes(classes, base)
    best_classes = [c[:] for c in classes]
    
    # Build coverage array
    covered = np.zeros(base, dtype=np.int8)
    for m, r in best_classes:
        covered[r::m] = 1
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i, (m, r) in enumerate(best_classes):
            if time.time() - start_time > time_limit:
                break
            # Remove current class
            covered[r::m] = 0
            current_uncovered = int((covered == 0).sum())
            
            # Try each residue for this modulus
            best_new_r = r
            best_new_score = best_score
            
            for new_r in range(m):
                if new_r == r:
                    continue
                # Temporarily add new_r
                new_covered = covered.copy()
                new_covered[new_r::m] = 1
                new_uncovered = int((new_covered == 0).sum())
                
                if new_uncovered < best_new_score:
                    best_new_score = new_uncovered
                    best_new_r = new_r
            
            # Restore and apply best
            if best_new_r != r:
                covered[best_new_r::m] = 1
                best_classes[i] = [m, best_new_r]
                best_score = best_new_score
                improved = True
            else:
                covered[r::m] = 1
    
    return best_classes

def _two_opt_swap(classes, base=51975, time_limit=5):
    """Try swapping residues between pairs of moduli."""
    start_time = time.time()
    best_score = _score_classes(classes, base)
    best_classes = [c[:] for c in classes]
    
    covered = np.zeros(base, dtype=np.int8)
    for m, r in best_classes:
        covered[r::m] = 1
    
    n = len(best_classes)
    for i in range(n):
        if time.time() - start_time > time_limit:
            break
        for j in range(i + 1, n):
            if time.time() - start_time > time_limit:
                break
            mi, ri = best_classes[i]
            mj, rj = best_classes[j]
            
            # Remove both
            covered[ri::mi] = 0
            covered[rj::mj] = 0
            
            # Try swapping residues
            for new_ri in range(mi):
                if new_ri == ri:
                    continue
                for new_rj in range(mj):
                    if new_rj == rj:
                        continue
                    covered[new_ri::mi] = 1
                    covered[new_rj::mj] = 1
                    new_score = int((covered == 0).sum())
                    covered[new_ri::mi] = 0
                    covered[new_rj::mj] = 0
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_classes[i] = [mi, new_ri]
                        best_classes[j] = [mj, new_rj]
                        covered[new_ri::mi] = 1
                        covered[new_rj::mj] = 1
                        ri, rj = new_ri, new_rj
            
            # Restore
            covered[ri::mi] = 1
            covered[rj::mj] = 1
    
    return best_classes

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Start with greedy
    classes = _greedy_construct(base)
    best_score = _score_classes(classes, base)
    
    # Local search pass 1
    remaining = deadline - time.time()
    if remaining > 10:
        classes = _local_search(classes, base, time_limit=min(remaining - 10, 30))
        new_score = _score_classes(classes, base)
        if new_score < best_score:
            best_score = new_score
    
    # Two-opt swap
    remaining = deadline - time.time()
    if remaining > 10:
        classes = _two_opt_swap(classes, base, time_limit=min(remaining - 10, 20))
        new_score = _score_classes(classes, base)
        if new_score < best_score:
            best_score = new_score
    
    # Local search pass 2
    remaining = deadline - time.time()
    if remaining > 10:
        classes = _local_search(classes, base, time_limit=min(remaining - 10, 20))
        new_score = _score_classes(classes, base)
        if new_score < best_score:
            best_score = new_score
    
    # Final local search if time permits
    remaining = deadline - time.time()
    if remaining > 5:
        classes = _local_search(classes, base, time_limit=min(remaining - 5, 10))
    
    return classes
# EVOLVE-BLOCK-END
