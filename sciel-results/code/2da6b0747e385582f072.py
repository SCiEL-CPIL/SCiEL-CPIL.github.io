import time
import numpy as np
import random

def _greedy_init(indices_cache, divisors, N, order='asc'):
    """Greedy initialization with specified order."""
    counts = np.zeros(N, dtype=np.int32)
    residues = {}
    
    if order == 'asc':
        m_list = divisors
    else:  # desc
        m_list = list(reversed(divisors))
    
    for m in m_list:
        best_r = 0
        max_new = -1
        for r in range(m):
            idx = indices_cache[m][r]
            new_covered = np.sum(counts[idx] == 0)
            if new_covered > max_new:
                max_new = new_covered
                best_r = r
        residues[m] = best_r
        counts[indices_cache[m][best_r]] += 1
    
    uncovered = int(np.sum(counts == 0))
    return counts, residues, uncovered

def _local_search(indices_cache, divisors, N, counts, residues, uncovered, best_uncovered, best_residues, start_time, time_limit, random):
    """Local search with restarts and random escapes."""
    no_improve_steps = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        # Pick random modulus
        m = random.choice(divisors)
        r_old = residues[m]
        
        best_r = r_old
        best_delta = 0
        
        # Evaluate all residues for m
        for r_new in range(m):
            if r_new == r_old:
                continue
                
            idx_old = indices_cache[m][r_old]
            idx_new = indices_cache[m][r_new]
            
            L = np.sum(counts[idx_old] == 1)
            G = np.sum(counts[idx_new] == 0)
            
            delta = L - G
            
            if delta < best_delta:
                best_delta = delta
                best_r = r_new
                
        if best_r != r_old:
            # Apply move
            idx_old = indices_cache[m][r_old]
            idx_new = indices_cache[m][best_r]
            
            counts[idx_old] -= 1
            counts[idx_new] += 1
            
            uncovered += best_delta
            
            residues[m] = best_r
            
            if uncovered < best_uncovered:
                best_uncovered = uncovered
                best_residues = residues.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1
            
        # If stuck, try a random move to escape local minimum
        if no_improve_steps > max_no_improve:
            # Pick a random m and random r
            m_rand = random.choice(divisors)
            r_rand = random.randint(0, m_rand - 1)
            
            r_old_rand = residues[m_rand]
            if r_rand != r_old_rand:
                idx_old = indices_cache[m_rand][r_old_rand]
                idx_new = indices_cache[m_rand][r_rand]
                
                L = np.sum(counts[idx_old] == 1)
                G = np.sum(counts[idx_new] == 0)
                delta = L - G
                
                counts[idx_old] -= 1
                counts[idx_new] += 1
                uncovered += delta
                residues[m_rand] = r_rand
                
            no_improve_steps = 0
    
    return best_uncovered, best_residues

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    N = 51975
    
    # 1. Generate divisors
    divisors = []
    for i in range(2, int(np.sqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    divisors.append(N)
    divisors = list(set(divisors))
    divisors.sort()
    
    # 2. Precompute indices for each modulus and residue
    indices_cache = {}
    for m in divisors:
        indices_cache[m] = []
        for r in range(m):
            indices_cache[m].append(np.arange(r, N, m))
    
    # 3. Try both ascending and descending greedy, pick the better one
    counts_asc, residues_asc, unc_asc = _greedy_init(indices_cache, divisors, N, 'asc')
    counts_desc, residues_desc, unc_desc = _greedy_init(indices_cache, divisors, N, 'desc')
    
    # Pick the better initialization
    if unc_asc <= unc_desc:
        counts = counts_asc.copy()
        residues = residues_asc.copy()
        uncovered = unc_asc
    else:
        counts = counts_desc.copy()
        residues = residues_desc.copy()
        uncovered = unc_desc
    
    best_uncovered = uncovered
    best_residues = residues.copy()
    
    # 4. Local search
    random.seed(42)
    best_uncovered, best_residues = _local_search(
        indices_cache, divisors, N, counts, residues, uncovered,
        best_uncovered, best_residues, start_time, time_limit, random
    )
    
    # 5. Final polish: do a few more passes with tighter tolerance
    if time.time() - start_time < time_limit - 5:
        # Do a final pass focusing on uncovered residues
        for _ in range(10):
            if time.time() - start_time >= time_limit - 2:
                break
            improved = False
            for m in divisors:
                r_old = residues[m]
                best_r = r_old
                best_delta = 0
                for r_new in range(m):
                    if r_new == r_old:
                        continue
                    idx_old = indices_cache[m][r_old]
                    idx_new = indices_cache[m][r_new]
                    L = np.sum(counts[idx_old] == 1)
                    G = np.sum(counts[idx_new] == 0)
                    delta = L - G
                    if delta < best_delta:
                        best_delta = delta
                        best_r = r_new
                if best_r != r_old:
                    idx_old = indices_cache[m][r_old]
                    idx_new = indices_cache[m][best_r]
                    counts[idx_old] -= 1
                    counts[idx_new] += 1
                    uncovered += best_delta
                    residues[m] = best_r
                    improved = True
                    if uncovered < best_uncovered:
                        best_uncovered = uncovered
                        best_residues = residues.copy()
            if not improved:
                break
    
    # Return best found
    result = [[m, best_residues[m]] for m in divisors]
    return result
