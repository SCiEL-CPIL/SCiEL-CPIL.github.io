"""Flow-shop permutation via ILS with VND (Swap, Or-Opt, 2-Opt)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan_fast(pt_list, perm):
    """
    Fast makespan calculation using pure Python lists.
    pt_list: list of lists, shape [n_jobs][n_machines].
    perm: list of job indices.
    """
    n_machines = len(pt_list[0]) if pt_list else 0
    n = len(perm)
    if n == 0 or n_machines == 0:
        return 0.0

    prev = [0.0] * n_machines
    for job in perm:
        row = pt_list[job]
        curr = [0.0] * n_machines
        curr[0] = prev[0] + row[0]
        for m in range(1, n_machines):
            val = prev[m] if prev[m] > curr[m-1] else curr[m-1]
            curr[m] = val + row[m]
        prev = curr
    return prev[-1]

def _neh_initial(pt):
    """NEH heuristic."""
    n_jobs, n_machines = pt.shape
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    perm = [order[0]]
    pt_list = pt.tolist()
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            test = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(pt_list, test)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    return perm

def _vnd_adjacent_swap(pt_list, perm, deadline):
    """
    VND Neighborhood 1: Adjacent swaps.
    Returns (new_perm, new_ms, improved)
    """
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_list, perm), False
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    
    for i in range(n - 1):
        if time.time() > deadline:
            return current, current_ms, False
        # Swap i and i+1
        test = current[:]
        test[i], test[i+1] = test[i+1], test[i]
        ms = _makespan_fast(pt_list, test)
        if ms < current_ms - 1e-9:
            current = test
            current_ms = ms
            # Restart scan from beginning? Or continue?
            # First improvement strategy usually restarts or continues.
            # Restarting is safer for VND to ensure local opt in this neighborhood.
            # But restarting is expensive. Let's just restart the loop.
            return _vnd_adjacent_swap(pt_list, current, deadline)
    return current, current_ms, False

def _vnd_or_opt(pt_list, perm, deadline):
    """
    VND Neighborhood 2: Or-Opt (relocate block of size 1, 2, 3).
    Returns (new_perm, new_ms, improved)
    """
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_list, perm), False
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    
    for block_size in [1, 2, 3]:
        if n < block_size:
            continue
        for i in range(n - block_size + 1):
            if time.time() > deadline:
                return current, current_ms, False
            block = current[i:i+block_size]
            rest = current[:i] + current[i+block_size:]
            # Insert block at position j in rest
            # j can be 0 to len(rest)
            for j in range(len(rest) + 1):
                # Avoid inserting back at original position
                # Original position in rest is i (if i < i) or i (if i > i)?
                # If we remove [i, i+block_size), the rest has length n-block_size.
                # The position i in original corresponds to:
                # If j < i, the block is inserted before i.
                # If j >= i, the block is inserted after i.
                # We want to skip the case where we put it back.
                # The "back" position is:
                # If we insert at index i in rest, it goes back to original spot?
                # Let's trace:
                # Original: A B C D
                # i=1, size=2, block=[B,C], rest=[A,D]
                # j=0: [B,C,A,D] (moved left)
                # j=1: [A,B,C,D] (back to original) -> Skip
                # j=2: [A,D,B,C] (moved right)
                # So skip if j == i.
                if j == i:
                    continue
                
                test = rest[:j] + block + rest[j:]
                ms = _makespan_fast(pt_list, test)
                if ms < current_ms - 1e-9:
                    current = test
                    current_ms = ms
                    # Restart Or-Opt search
                    return _vnd_or_opt(pt_list, current, deadline)
    return current, current_ms, False

def _vnd_2opt(pt_list, perm, deadline):
    """
    VND Neighborhood 3: 2-Opt (reverse segment).
    Returns (new_perm, new_ms, improved)
    """
    n = len(perm)
    if n <= 2:
        return perm, _makespan_fast(pt_list, perm), False
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    
    for i in range(n - 1):
        if time.time() > deadline:
            return current, current_ms, False
        for j in range(i + 1, n):
            if time.time() > deadline:
                return current, current_ms, False
            test = current[:i] + current[i:j+1][::-1] + current[j+1:]
            ms = _makespan_fast(pt_list, test)
            if ms < current_ms - 1e-9:
                current = test
                current_ms = ms
                # Restart 2-Opt search
                return _vnd_2opt(pt_list, current, deadline)
    return current, current_ms, False

def _vnd(pt_list, perm, deadline):
    """
    Variable Neighborhood Descent.
    Cycles through Swap, Or-Opt, 2-Opt.
    Stops when no neighborhood yields an improvement.
    """
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    
    while True:
        if time.time() > deadline:
            break
        
        # 1. Adjacent Swap
        new_perm, new_ms, _ = _vnd_adjacent_swap(pt_list, current, deadline)
        if new_ms < current_ms - 1e-9:
            current = new_perm
            current_ms = new_ms
            continue # Restart cycle
        
        # 2. Or-Opt
        new_perm, new_ms, _ = _vnd_or_opt(pt_list, current, deadline)
        if new_ms < current_ms - 1e-9:
            current = new_perm
            current_ms = new_ms
            continue # Restart cycle
        
        # 3. 2-Opt
        new_perm, new_ms, _ = _vnd_2opt(pt_list, current, deadline)
        if new_ms < current_ms - 1e-9:
            current = new_perm
            current_ms = new_ms
            continue # Restart cycle
        
        # No improvement in any neighborhood
        break
        
    return current, current_ms

def _perturb(pt_list, perm, rng, strength=3):
    """
    Perturb the permutation with random Or-Opt moves.
    strength: number of random moves.
    """
    n = len(perm)
    current = list(perm)
    for _ in range(strength):
        if n < 2:
            break
        # Random block size 1, 2, or 3
        max_size = min(3, n)
        size = rng.randint(1, max_size)
        i = rng.randint(0, n - size)
        block = current[i:i+size]
        rest = current[:i] + current[i+size:]
        j = rng.randint(0, len(rest))
        if j == i:
            j = (j + 1) % (len(rest) + 1)
        current = rest[:j] + block + rest[j:]
    return current

def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = pt.shape
    start = time.time()
    deadline = start + 145.0 # Safety margin

    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))

    pt_list = pt.tolist()
    rng = random.Random(42)

    # 1. NEH Initialization
    best_perm = _neh_initial(pt)
    best_ms = _makespan_fast(pt_list, best_perm)
    
    # 2. Initial VND
    best_perm, best_ms = _vnd(pt_list, best_perm, deadline)

    # 3. ILS Loop
    T = best_ms * 0.05
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline - 5.0:
        # Perturb
        perturbed = _perturb(pt_list, best_perm, rng, strength=3)
        
        # Local Search (VND)
        local_perm, local_ms = _vnd(pt_list, perturbed, deadline - 5.0)
        
        # Acceptance Criterion (Simulated Annealing)
        delta = local_ms - best_ms
        if delta < 0:
            best_perm = local_perm
            best_ms = local_ms
        else:
            if rng.random() < np.exp(-delta / T):
                best_perm = local_perm
                best_ms = local_ms
        
        T *= cooling
        if T < T_min:
            T = T_min

    # Final polish
    if time.time() < deadline - 1.0:
        best_perm, best_ms = _vnd(pt_list, best_perm, deadline)

    return best_perm
# EVOLVE-BLOCK-END
