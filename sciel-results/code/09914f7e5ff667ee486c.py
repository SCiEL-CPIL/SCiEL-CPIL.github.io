# EVOLVE-BLOCK-START
import time
import numpy as np

_BASE = 51975
_DEADLINE = 115.0  # safety margin before 150s kill

def _divisors(n):
    """Return all divisors > 1 of n, sorted."""
    divs = []
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    divs.append(n)
    return sorted(divs)

def _greedy_build(moduli_order):
    """Greedy: for each modulus in order, pick residue covering most uncovered residues."""
    covered = np.zeros(_BASE, dtype=np.uint8)
    classes = []
    for m in moduli_order:
        best_r = 0
        best_count = -1
        for r in range(m):
            count = int(np.sum(covered[r::m] == 0))
            if count > best_count:
                best_count = count
                best_r = r
        classes.append([m, best_r])
        covered[best_r::m] = 1
    return classes

def _local_search(classes, max_time):
    """
    Local search using the Z-mask optimization.
    For each class, calculate the set of positions that would be uncovered if this class were removed (Z).
    Then, for each possible residue, calculate how many positions in Z are covered.
    Maximize this coverage to minimize new uncovered count.
    """
    classes = [list(c) for c in classes]
    
    # Initialize counts
    counts = np.zeros(_BASE, dtype=np.int32)
    for m, r in classes:
        counts[r::m] += 1
    
    current_unc = int(np.sum(counts == 0))
    
    improved = True
    start_time = time.time()
    
    # Precompute masks? No, they change.
    
    while improved:
        improved = False
        if time.time() - start_time > max_time:
            break
        
        # Shuffle order to avoid bias
        indices = list(range(len(classes)))
        np.random.shuffle(indices)
        
        for i in indices:
            if time.time() - start_time > max_time:
                break
            
            m, old_r = classes[i]
            
            # Compute mask for current residue
            mask_old = np.zeros(_BASE, dtype=np.uint8)
            mask_old[old_r::m] = 1
            
            # Compute Z: positions that would be uncovered if class i were removed
            # Z is True where counts is 0, OR (counts is 1 AND mask_old is True)
            # Note: counts includes class i.
            # If counts[x] == 0, then x is not covered by anyone, so if we remove i (which doesn't cover x), it's still uncovered.
            # If counts[x] == 1, then x is covered by exactly one class. If that class is i (mask_old[x]==1), then removing i leaves it uncovered.
            # If counts[x] > 1, removing i still leaves it covered.
            Z = (counts == 0) | ((counts == 1) & mask_old)
            
            Z_count = int(np.sum(Z))
            
            best_r = old_r
            best_new_unc = current_unc
            
            # Try all residues
            for r in range(m):
                if r == old_r:
                    continue
                
                # Count how many positions in Z are covered by new residue r
                # Z[r::m] sums the True values in Z at positions r, r+m, ...
                covered_in_Z = int(np.sum(Z[r::m]))
                
                new_unc = Z_count - covered_in_Z
                
                if new_unc < best_new_unc:
                    best_new_unc = new_unc
                    best_r = r
            
            if best_r != old_r:
                # Apply change
                # Update counts
                counts[old_r::m] -= 1
                counts[best_r::m] += 1
                classes[i][1] = best_r
                current_unc = best_new_unc
                improved = True
                
    return classes, current_unc

def construct():
    start_time = time.time()
    moduli = _divisors(_BASE)
    
    best_classes = None
    best_unc = _BASE + 1
    
    # Try several initializations
    initializations = []
    
    # 1. Greedy Ascending
    if time.time() - start_time < _DEADLINE - 20:
        classes_asc = _greedy_build(moduli)
        initializations.append(classes_asc)
        
    # 2. Greedy Descending
    if time.time() - start_time < _DEADLINE - 20:
        classes_desc = _greedy_build(sorted(moduli, reverse=True))
        initializations.append(classes_desc)
        
    # 3. Random Initialization
    if time.time() - start_time < _DEADLINE - 20:
        classes_rand = [[m, np.random.randint(m)] for m in moduli]
        initializations.append(classes_rand)
        
    # 4. Another Random
    if time.time() - start_time < _DEADLINE - 20:
        classes_rand2 = [[m, np.random.randint(m)] for m in moduli]
        initializations.append(classes_rand2)

    # Run local search on each
    time_per_ls = 15.0 # 15s per local search run
    for init_classes in initializations:
        if time.time() - start_time > _DEADLINE - 5:
            break
        
        ls_classes, ls_unc = _local_search(init_classes, time_per_ls)
        
        if ls_unc < best_unc:
            best_unc = ls_unc
            best_classes = ls_classes
            
    # If we have time, do a final polish on the best result
    if best_classes is not None and time.time() - start_time < _DEADLINE - 5:
        final_classes, final_unc = _local_search(best_classes, _DEADLINE - 5 - (time.time() - start_time))
        if final_unc < best_unc:
            best_unc = final_unc
            best_classes = final_classes

    if best_classes is None:
        # Fallback
        best_classes = _greedy_build(moduli)
        
    return best_classes
# EVOLVE-BLOCK-END
