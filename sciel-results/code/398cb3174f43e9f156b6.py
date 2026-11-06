import time
import random
import numpy as np

BASE = 51975

def _get_divisors(n):
    divs = set()
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(divs)

DIVISORS = _get_divisors(BASE)

# Precompute index arrays for each (m, r)
# This is the key speedup from Inspiration 2
IDX_CACHE = {}
for m in DIVISORS:
    IDX_CACHE[m] = [np.arange(r, BASE, m) for r in range(m)]

def _greedy_init(order, deadline):
    """
    Greedy initialization: Add moduli in the specified order.
    For each modulus, pick the residue that covers the most currently uncovered residues.
    """
    counts = np.zeros(BASE, dtype=np.int8)
    uncovered = BASE
    assignments = {}
    
    for m in order:
        if time.time() > deadline:
            break
            
        best_r = 0
        best_gain = -1
        
        # Find best residue for this modulus
        for r in range(m):
            idx = IDX_CACHE[m][r]
            # Count how many of these indices are currently uncovered (count == 0)
            gain = int(np.count_nonzero(counts[idx] == 0))
            if gain > best_gain:
                best_gain = gain
                best_r = r
                
        assignments[m] = best_r
        idx = IDX_CACHE[m][best_r]
        counts[idx] += 1
        uncovered -= best_gain
        
    return assignments, counts, uncovered

def _local_search(assignments, counts, uncovered, deadline):
    """
    Local search: Hill climbing.
    Tries to change residue for each modulus to improve coverage.
    """
    assignments = assignments.copy()
    counts = counts.copy()
    best_uncovered = uncovered
    best_assignments = assignments.copy()
    
    divs = DIVISORS.copy()
    
    while time.time() < deadline:
        random.shuffle(divs)
        improved = False
        
        for m in divs:
            if time.time() > deadline:
                break
                
            r_old = assignments[m]
            idx_old = IDX_CACHE[m][r_old]
            
            # Try all other residues for this modulus
            # Optimization: If m is large, this loop is long, but each step is cheap (small array).
            # If m is small, loop is short, but each step is expensive (large array).
            # Total cost per m is roughly O(BASE).
            
            for r_new in range(m):
                if r_new == r_old:
                    continue
                    
                idx_new = IDX_CACHE[m][r_new]
                
                # Calculate delta
                # Loss: residues in idx_old that are covered ONLY by this class (count becomes 0)
                loss = int(np.count_nonzero(counts[idx_old] == 1))
                # Gain: residues in idx_new that are currently UNCOVERED (count becomes 1)
                gain = int(np.count_nonzero(counts[idx_new] == 0))
                
                delta = gain - loss
                
                if delta < 0:
                    # Accept the move
                    counts[idx_old] -= 1
                    counts[idx_new] += 1
                    uncovered += delta
                    assignments[m] = r_new
                    improved = True
                    
                    if uncovered < best_uncovered:
                        best_uncovered = uncovered
                        best_assignments = assignments.copy()
                        
                    # Move to next modulus (first improvement strategy)
                    break
                    
        if not improved:
            # Local minimum found. 
            # We could try a perturbation here, but multi-restart is simpler and effective.
            break
            
    return best_assignments, best_uncovered

def construct():
    start_time = time.time()
    # Safety margin: 145s limit, stop at 140s
    deadline = start_time + 140.0
    
    best_uncovered = BASE
    best_assignments = {}
    
    # Generate multiple orders for greedy initialization
    orders = [
        DIVISORS,
        sorted(DIVISORS, reverse=True)
    ]
    
    # Add random orders
    for _ in range(20):
        shuffled = DIVISORS.copy()
        random.shuffle(shuffled)
        orders.append(shuffled)
        
    # Time management
    # We have 140 seconds.
    # Greedy is fast (<1s per restart).
    # Local search takes the bulk of the time.
    # We will allocate time slices to each restart.
    
    num_restarts = len(orders)
    time_per_restart = (deadline - start_time) / num_restarts
    
    for i, order in enumerate(orders):
        # Calculate deadline for this restart
        # Ensure we don't exceed the global deadline
        current_deadline = min(deadline - 2.0, start_time + (i + 1) * time_per_restart)
        
        if time.time() >= current_deadline:
            continue
            
        # Phase 1: Greedy Init
        assignments, counts, uncovered = _greedy_init(order, current_deadline)
        
        if uncovered < best_uncovered:
            best_uncovered = uncovered
            best_assignments = assignments.copy()
            
        # Phase 2: Local Search
        # Use remaining time for this restart
        if time.time() < current_deadline:
            a, u = _local_search(assignments, counts, uncovered, current_deadline)
            if u < best_uncovered:
                best_uncovered = u
                best_assignments = a

    # Construct result
    result = [[m, r] for m, r in best_assignments.items()]
    
    return result

if __name__ == "__main__":
    classes = construct()
    covered = np.zeros(BASE, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    score = int(BASE - np.sum(covered))
    print(f"Score: {score}")
