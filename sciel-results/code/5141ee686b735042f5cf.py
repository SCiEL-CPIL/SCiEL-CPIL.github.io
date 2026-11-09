import time
import random
import numpy as np

def _get_divisors(N):
    """Get all divisors of N greater than 1."""
    divs = []
    for d in range(2, N + 1):
        if N % d == 0:
            divs.append(d)
    return divs

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    N = 51975
    divisors = _get_divisors(N)
    n_divs = len(divisors)
    
    # Precompute index arrays for each divisor
    # idx[m_idx] = array of positions in [0, N) that are congruent to r mod m
    # We'll store as list of arrays, one per divisor
    # For each divisor m, precompute the base indices: range(0, N, m)
    base_indices = []
    for m in divisors:
        base_indices.append(np.arange(0, N, m))
    
    # counts[x] = number of classes covering x
    counts = np.zeros(N, dtype=np.int32)
    uncovered = np.int32(N)
    
    # current_r[m_idx] = chosen residue
    current_r = np.zeros(n_divs, dtype=np.int32)
    
    # 1. Greedy Initialization (sorted by divisor size, smallest first)
    sorted_order = sorted(range(n_divs), key=lambda i: divisors[i])
    
    for idx in sorted_order:
        m = divisors[idx]
        base = base_indices[idx]
        best_r = 0
        best_gain = -1
        
        # For each residue r, gain = number of currently uncovered positions in (r + base) % N
        # But since base = [0, m, 2m, ...], positions for residue r are (r + base) but capped at N
        # Actually positions are: r, r+m, r+2m, ... < N
        # Which is exactly: min(r + base, N) but only those < N
        # Simpler: positions = np.where(base + r < N, base + r, -1) then filter
        # Even simpler: the positions are just range(r, N, m)
        
        for r in range(m):
            # Compute gain: count of positions in range(r, N, m) where counts == 0
            positions = np.arange(r, N, m)
            gain = np.sum(counts[positions] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        current_r[idx] = best_r
        # Update counts
        positions = np.arange(best_r, N, m)
        counts[positions] += 1
        uncovered -= np.sum(counts[positions] == 1)
    
    best_uncovered = int(uncovered)
    best_config = current_r.copy()
    
    # 2. Local Search with simulated annealing
    T = 2.0
    cooling = 0.9995
    
    # Precompute for faster delta calculation
    # For each divisor index, we need to quickly compute:
    #   remove_gain: count of positions in old range where counts == 1
    #   add_gain: count of positions in new range where counts == 0
    
    while time.time() < deadline:
        if T < 0.05:
            # Reset temperature for continued search
            T = 1.0
            # Try to escape local minima with a random restart of a few variables
            for _ in range(3):
                idx = random.randint(0, n_divs - 1)
                m = divisors[idx]
                old_r = current_r[idx]
                new_r = random.randint(0, m - 1)
                if new_r == old_r:
                    continue
                old_pos = np.arange(old_r, N, m)
                new_pos = np.arange(new_r, N, m)
                counts[old_pos] -= 1
                counts[new_pos] += 1
                current_r[idx] = new_r
                uncovered = int(np.sum(counts == 0))
                if uncovered < best_uncovered:
                    best_uncovered = uncovered
                    best_config = current_r.copy()
            continue
        
        idx = random.randint(0, n_divs - 1)
        m = divisors[idx]
        r_old = int(current_r[idx])
        r_new = random.randint(0, m - 1)
        if r_new == r_old:
            T *= cooling
            continue
        
        old_pos = np.arange(r_old, N, m)
        new_pos = np.arange(r_new, N, m)
        
        # Delta in uncovered count
        # Removing old: positions where counts==1 become uncovered (+1 each)
        remove_delta = int(np.sum(counts[old_pos] == 1))
        # Adding new: positions where counts==0 become covered (-1 each)
        add_delta = int(np.sum(counts[new_pos] == 0))
        delta = remove_delta - add_delta
        
        # Acceptance
        if delta <= 0:
            accept = True
        else:
            accept = random.random() < np.exp(-delta / T)
        
        if accept:
            counts[old_pos] -= 1
            counts[new_pos] += 1
            current_r[idx] = r_new
            uncovered = int(np.sum(counts == 0))
            
            if uncovered < best_uncovered:
                best_uncovered = uncovered
                best_config = current_r.copy()
        
        T *= cooling
    
    # 3. Final polish: coordinate descent
    # Try each variable and pick the best residue
    for pass_num in range(5):
        if time.time() > deadline - 2.0:
            break
        improved = False
        for idx in range(n_divs):
            if time.time() > deadline - 2.0:
                break
            m = divisors[idx]
            r_old = int(current_r[idx])
            best_r = r_old
            best_uncovered_local = int(np.sum(counts == 0))
            
            old_pos = np.arange(r_old, N, m)
            counts[old_pos] -= 1  # temporarily remove
            
            for r in range(m):
                pos = np.arange(r, N, m)
                counts[pos] += 1
                unc = int(np.sum(counts == 0))
                counts[pos] -= 1
                if unc < best_uncovered_local:
                    best_uncovered_local = unc
                    best_r = r
            
            # Re-add old
            counts[old_pos] += 1
            
            if best_r != r_old:
                # Apply best
                counts[old_pos] -= 1
                new_pos = np.arange(best_r, N, m)
                counts[new_pos] += 1
                current_r[idx] = best_r
                improved = True
                
                unc = int(np.sum(counts == 0))
                if unc < best_uncovered:
                    best_uncovered = unc
                    best_config = current_r.copy()
        
        if not improved:
            break
    
    # Convert best_config to list of [m, r]
    result = []
    for i, m in enumerate(divisors):
        result.append([m, int(best_config[i])])
    
    return result
