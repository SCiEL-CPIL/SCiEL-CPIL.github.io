"""Improved program for odd covering systems with moduli dividing 51975."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    # Precompute all odd divisors of 51975 > 1, sorted in increasing order
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    n_div = len(divisors)
    
    # Precompute positions for each (modulus, residue) pair as numpy arrays for fast slicing
    positions_cache = {}
    for m in divisors:
        positions_cache[m] = [np.arange(r, base, m) for r in range(m)]
        
    div_index = {d: i for i, d in enumerate(divisors)}

    best_uncovered = base
    best_residues = [0] * n_div
    best_cover_count = None

    def evaluate_residues(residues):
        cover_count = np.zeros(base, dtype=np.uint8)
        for i, m in enumerate(divisors):
            r = residues[i]
            cover_count[positions_cache[m][r]] += 1
        return int(np.sum(cover_count == 0)), cover_count

    # Greedy initialization: try multiple orders
    # Key insight: process larger moduli first (decreasing order) for better coverage
    orders = [
        list(range(n_div - 1, -1, -1)),  # Decreasing (largest first)
        list(range(n_div)),               # Increasing
    ]
    # Add a few random orders
    for _ in range(2):
        o = list(range(n_div))
        random.shuffle(o)
        orders.append(o)
    
    for order in orders:
        if time.time() > deadline - 40:
            break
        
        residues = [0] * n_div
        cover_count = np.zeros(base, dtype=np.uint8)
        
        for idx in order:
            m = divisors[idx]
            best_r = 0
            best_gain = -1
            for r in range(m):
                pos = positions_cache[m][r]
                gain = np.sum(cover_count[pos] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            residues[idx] = best_r
            cover_count[positions_cache[m][best_r]] += 1
            
        current_uncovered, current_cover = evaluate_residues(residues)
        
        if current_uncovered < best_uncovered:
            best_uncovered = current_uncovered
            best_residues = residues[:]
            best_cover_count = current_cover

    # Local Search with Incremental Update
    residues = best_residues[:]
    cover_count = best_cover_count.copy() if best_cover_count is not None else np.zeros(base, dtype=np.uint8)
    current_uncovered = best_uncovered
    
    # Simulated Annealing
    temp = 2.0
    cooling = 0.997
    no_improve_count = 0
    max_no_improve = 10000
    
    while time.time() < deadline - 10:
        if temp < 0.05:
            break
        if no_improve_count > max_no_improve:
            break
            
        i = random.randint(0, n_div - 1)
        mi = divisors[i]
        old_ri = residues[i]
        new_ri = random.randint(0, mi - 1)
        if new_ri == old_ri:
            continue
            
        old_pos = positions_cache[mi][old_ri]
        new_pos = positions_cache[mi][new_ri]
        
        old_counts = cover_count[old_pos]
        new_counts = cover_count[new_pos]
        
        delta = int(np.sum(old_counts == 1)) - int(np.sum(new_counts == 0))
        
        if delta < 0 or random.random() < np.exp(-delta / temp):
            cover_count[old_pos] -= 1
            cover_count[new_pos] += 1
            residues[i] = new_ri
            current_uncovered += delta
            no_improve_count = 0
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_residues = residues[:]
                best_cover_count = cover_count.copy()
        else:
            no_improve_count += 1
            
        temp *= cooling

    # Final Polish: Hill Climbing with multiple passes
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for i in range(n_div):
            m = divisors[i]
            old_r = residues[i]
            best_r = old_r
            best_local = current_uncovered
            
            old_pos = positions_cache[m][old_r]
            old_counts = cover_count[old_pos]
            
            for r in range(m):
                if r == old_r:
                    continue
                new_pos = positions_cache[m][r]
                new_counts = cover_count[new_pos]
                delta = int(np.sum(old_counts == 1)) - int(np.sum(new_counts == 0))
                local_uncovered = current_uncovered + delta
                if local_uncovered < best_local:
                    best_local = local_uncovered
                    best_r = r
            
            if best_r != old_r:
                new_pos = positions_cache[m][best_r]
                cover_count[old_pos] -= 1
                cover_count[new_pos] += 1
                current_uncovered = best_local
                residues[i] = best_r
                improved = True
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_residues = residues[:]
                    best_cover_count = cover_count.copy()

    classes = [[divisors[i], best_residues[i]] for i in range(n_div)]
    return classes
# EVOLVE-BLOCK-END
