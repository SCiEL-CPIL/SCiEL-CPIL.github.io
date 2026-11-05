"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    base = 51975
    
    # All odd divisors of 51975 greater than 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()  # increasing order: small moduli first
    
    def greedy_build(seed_offset=0):
        """Build a covering system using greedy with optional randomization."""
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        rng = np.random.RandomState(seed_offset)
        
        for m in divisors:
            if time.time() > deadline - 25:
                break
            # Find residue covering most uncovered positions
            if m <= 300:
                # Exact search for small moduli
                best_r = 0
                best_count = -1
                for r in range(m):
                    seg = covered[r::m]
                    cnt = len(seg) - int(seg.sum())
                    if cnt > best_count:
                        best_count = cnt
                        best_r = r
            else:
                # Sample-based for large moduli
                best_r = 0
                best_count = -1
                # Deterministic candidates
                cands = list(range(min(m, 32)))
                cands += [int(rng.randint(m)) for _ in range(32)]
                cands = list(set(cands))
                for r in cands:
                    seg = covered[r::m]
                    cnt = len(seg) - int(seg.sum())
                    if cnt > best_count:
                        best_count = cnt
                        best_r = r
            
            classes.append([m, best_r])
            covered[best_r::m] = 1
        
        return classes, covered
    
    def compute_uncovered(classes):
        cov = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            cov[r::m] = 1
        return int((cov == 0).sum())
    
    def local_search_incremental(classes, max_time=60):
        """Fast local search using incremental coverage updates."""
        cov = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            cov[r::m] = 1
        cur_uncov = int((cov == 0).sum())
        
        end_time = min(start_time + max_time, deadline - 10)
        improved = True
        passes = 0
        while improved and time.time() < end_time:
            improved = False
            passes += 1
            for i in range(len(classes)):
                if time.time() > end_time:
                    break
                m, r = classes[i]
                # Remove this class from coverage
                # Find positions covered ONLY by this class
                cov_without = np.zeros(base, dtype=np.uint8)
                for j, (m2, r2) in enumerate(classes):
                    if j != i:
                        cov_without[r2::m2] = 1
                base_uncov = int((cov_without == 0).sum())
                
                # Find best new residue
                best_r = r
                best_uncov = base_uncov
                
                if m <= 200:
                    for r2 in range(m):
                        if r2 == r:
                            continue
                        newly = int((cov_without[r2::m] == 0).sum())
                        unc = base_uncov - newly
                        if unc < best_uncov:
                            best_uncov = unc
                            best_r = r2
                else:
                    # Sample candidates for large moduli
                    rng_ls = np.random.RandomState(i * 7919 + passes * 31 + 42)
                    cands = list(range(min(m, 24)))
                    cands += [int(rng_ls.randint(m)) for _ in range(24)]
                    cands = list(set(cands))
                    for r2 in cands:
                        if r2 == r:
                            continue
                        newly = int((cov_without[r2::m] == 0).sum())
                        unc = base_uncov - newly
                        if unc < best_uncov:
                            best_uncov = unc
                            best_r = r2
                
                if best_r != r:
                    classes[i][1] = best_r
                    # Update coverage incrementally
                    cov[r::m] = 0
                    cov[best_r::m] = 1
                    cur_uncov = best_uncov
                    improved = True
        
        return classes, cur_uncov
    
    def remove_redundant(classes):
        """Remove classes that don't improve coverage."""
        removed = True
        while removed and time.time() < deadline - 8:
            removed = False
            cur_uncov = compute_uncovered(classes)
            for i in range(len(classes) - 1, -1, -1):
                if time.time() > deadline - 8:
                    break
                saved = classes.pop(i)
                unc_without = compute_uncovered(classes)
                if unc_without <= cur_uncov:
                    cur_uncov = unc_without
                    removed = True
                else:
                    classes.insert(i, saved)
        return classes, cur_uncov
    
    def final_polish(classes):
        """Exact polish for small moduli."""
        if time.time() > deadline - 5:
            return classes
        for i in range(len(classes)):
            if time.time() > deadline - 4:
                break
            m, r = classes[i]
            if m > 150:
                continue
            cov_without = np.zeros(base, dtype=np.uint8)
            for j, (m2, r2) in enumerate(classes):
                if j != i:
                    cov_without[r2::m2] = 1
            base_uncov = int((cov_without == 0).sum())
            best_r = r
            best_uncov = base_uncov
            for r2 in range(m):
                newly = int((cov_without[r2::m] == 0).sum())
                unc = base_uncov - newly
                if unc < best_uncov:
                    best_uncov = unc
                    best_r = r2
            if best_r != r:
                classes[i][1] = best_r
        return classes
    
    # --- Main: Try multiple seeds, keep best ---
    best_classes = None
    best_score = base + 1
    
    # Try several seeds
    max_seeds = 5
    for seed in range(max_seeds):
        if time.time() > deadline - 40:
            break
        
        classes, covered = greedy_build(seed * 104729)
        score = compute_uncovered(classes)
        
        if score < best_score:
            best_score = score
            best_classes = classes[:]
        
        # Local search on this candidate
        if time.time() < deadline - 15:
            classes_ls, score_ls = local_search_incremental(classes, max_time=30)
            if score_ls < best_score:
                best_score = score_ls
                best_classes = classes_ls[:]
    
    # Final cleanup on best
    if best_classes is not None:
        best_classes, _ = remove_redundant(best_classes)
        best_classes = final_polish(best_classes)
    
    if best_classes is None:
        # Fallback: simple greedy
        best_classes, _ = greedy_build(0)
    
    return best_classes
# EVOLVE-BLOCK-END
