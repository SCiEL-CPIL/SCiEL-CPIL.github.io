"""Optimized odd covering system solver for moduli dividing 51975."""
import time
import random
import math
import numpy as np

BASE = 51975

def _get_divisors():
    divs = []
    for i in range(1, int(math.isqrt(BASE)) + 1):
        if BASE % i == 0:
            if i > 1:
                divs.append(i)
            other = BASE // i
            if other != i and other > 1:
                divs.append(other)
    divs.sort()
    return divs

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    moduli = _get_divisors()
    n = len(moduli)
    
    # Precompute index arrays for each modulus/residue using NumPy
    # For modulus m, residue r: indices are r, r+m, r+2m, ... < BASE
    idx_cache = [None] * n
    for i, m in enumerate(moduli):
        idx_cache[i] = [np.arange(r, BASE, m, dtype=np.int32) for r in range(m)]
    
    def build_coverage(residues):
        """Return a boolean numpy array of covered positions."""
        covered = np.zeros(BASE, dtype=np.bool_)
        for i, m in enumerate(moduli):
            r = residues[i]
            covered[idx_cache[i][r]] = True
        return covered
    
    def count_uncovered(covered):
        return int(BASE - covered.sum())
    
    # Initial greedy solution with multiple orderings
    def greedy(order):
        covered = np.zeros(BASE, dtype=np.bool_)
        residues = [0] * n
        for i in order:
            m = moduli[i]
            best_r = 0
            best_gain = -1
            for r in range(m):
                idx = idx_cache[i][r]
                gain = int((~covered[idx]).sum())
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            residues[i] = best_r
            covered[idx_cache[i][best_r]] = True
        return residues
    
    best_residues = None
    best_score = BASE + 1
    
    # Try greedy with different orders
    rng = random.Random(42)
    orders = [
        list(range(n)),           # increasing
        list(range(n-1, -1, -1)), # decreasing
    ]
    for _ in range(10):
        orders.append(rng.sample(list(range(n)), n))
    
    for order in orders:
        if time.time() > deadline - 30:
            break
        r = greedy(order)
        cov = build_coverage(r)
        score = count_uncovered(cov)
        if score < best_score:
            best_score = score
            best_residues = r[:]
    
    # Random starts
    for _ in range(20):
        if time.time() > deadline - 30:
            break
        r = [random.randint(0, moduli[i]-1) for i in range(n)]
        cov = build_coverage(r)
        score = count_uncovered(cov)
        if score < best_score:
            best_score = score
            best_residues = r[:]
    
    # Simulated annealing with efficient delta calculation using NumPy
    # Maintain coverage count array for fast delta computation
    def init_cov_count(residues):
        cov_count = np.zeros(BASE, dtype=np.int32)
        for i, m in enumerate(moduli):
            r = residues[i]
            cov_count[idx_cache[i][r]] += 1
        return cov_count
    
    current_residues = best_residues[:]
    current_cov_count = init_cov_count(current_residues)
    current_uncovered = int((current_cov_count == 0).sum())
    
    temp = 5.0
    cooling = 0.9995
    min_temp = 0.001
    
    while time.time() < deadline - 20 and temp > min_temp:
        i = random.randint(0, n - 1)
        m = moduli[i]
        old_r = current_residues[i]
        new_r = random.randint(0, m - 1)
        
        if new_r == old_r:
            continue
        
        # Compute delta using NumPy
        old_idx = idx_cache[i][old_r]
        new_idx = idx_cache[i][new_r]
        
        # Remove old residue class
        old_counts = current_cov_count[old_idx]
        becomes_uncovered = int((old_counts == 1).sum())
        current_cov_count[old_idx] -= 1
        
        # Add new residue class
        new_counts = current_cov_count[new_idx]
        becomes_covered = int((new_counts == 0).sum())
        current_cov_count[new_idx] += 1
        
        delta = becomes_uncovered - becomes_covered
        new_uncovered = current_uncovered + delta
        
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_residues[i] = new_r
            current_uncovered = new_uncovered
            if current_uncovered < best_score:
                best_score = current_uncovered
                best_residues = current_residues[:]
        else:
            # Revert
            current_cov_count[new_idx] -= 1
            current_cov_count[old_idx] += 1
        
        temp *= cooling
    
    # Final local search: for each modulus, find best residue given others fixed
    if time.time() < deadline - 5:
        best_cov_count = init_cov_count(best_residues)
        best_uncovered = int((best_cov_count == 0).sum())
        
        for i in range(n):
            if time.time() > deadline - 5:
                break
            m = moduli[i]
            old_r = best_residues[i]
            
            # Compute coverage from all other classes
            other_covered = np.zeros(BASE, dtype=np.bool_)
            for j in range(n):
                if j != i:
                    other_covered[idx_cache[j][best_residues[j]]] = True
            
            # For each residue, compute uncovered count
            best_r = old_r
            best_unc = best_uncovered
            for new_r in range(m):
                if new_r == old_r:
                    continue
                # Uncovered = positions not covered by others and not covered by this new class
                idx = idx_cache[i][new_r]
                # Positions covered by this new class
                covered_by_new = np.zeros(BASE, dtype=np.bool_)
                covered_by_new[idx] = True
                # Total covered = other_covered OR covered_by_new
                total_covered = other_covered | covered_by_new
                unc = int(BASE - total_covered.sum())
                if unc < best_unc:
                    best_unc = unc
                    best_r = new_r
            
            if best_r != old_r:
                best_residues[i] = best_r
                best_uncovered = best_unc
                if best_uncovered < best_score:
                    best_score = best_uncovered
    
    # One more polish pass
    if time.time() < deadline - 2:
        best_cov_count = init_cov_count(best_residues)
        best_uncovered = int((best_cov_count == 0).sum())
        
        for i in range(n):
            if time.time() > deadline - 2:
                break
            m = moduli[i]
            old_r = best_residues[i]
            
            other_covered = np.zeros(BASE, dtype=np.bool_)
            for j in range(n):
                if j != i:
                    other_covered[idx_cache[j][best_residues[j]]] = True
            
            best_r = old_r
            best_unc = best_uncovered
            for new_r in range(m):
                if new_r == old_r:
                    continue
                idx = idx_cache[i][new_r]
                covered_by_new = np.zeros(BASE, dtype=np.bool_)
                covered_by_new[idx] = True
                total_covered = other_covered | covered_by_new
                unc = int(BASE - total_covered.sum())
                if unc < best_unc:
                    best_unc = unc
                    best_r = new_r
            
            if best_r != old_r:
                best_residues[i] = best_r
                best_uncovered = best_unc
                if best_uncovered < best_score:
                    best_score = best_uncovered
    
    return [[moduli[i], best_residues[i]] for i in range(n)]
