"""HP lattice protein folding using greedy growth with local search refinement."""
# EVOLVE-BLOCK-START
import random
import time
import math


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute HP contact energy: -1 for each H-H pair on neighbouring lattice sites not adjacent in chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_search(sequence, chain, max_iters=2000, rng=None, time_limit=None):
    """
    Monte Carlo search using pivot moves.
    A pivot move rotates the suffix of the chain (from pivot to end) around the pivot point.
    This guarantees that if the new suffix does not collide with the prefix, the chain remains
    a valid self-avoiding walk with unit steps.
    """
    if rng is None:
        rng = random.Random()
    
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Simulated Annealing parameters
    T_start = 1.0
    T_end = 0.05
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
    T = T_start
    
    start_time = time.time()
    
    for iteration in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
            
        # Pick a pivot point (index of the residue before the one being rotated)
        # Pivot index k means we rotate the chain from k+1 to n-1 around chain[k]
        # k ranges from 0 to n-2
        k = rng.randint(0, n - 2)
        
        # The axis of rotation is the bond between k and k+1.
        # In 2D, we can rotate the suffix by 90, 180, or 270 degrees.
        
        # Get the vector from k to k+1
        v_x = current_chain[k+1][0] - current_chain[k][0]
        v_y = current_chain[k+1][1] - current_chain[k][1]
        
        # We choose one of the 3 non-identity rotations
        rot_choice = rng.choice([1, 2, 3]) # 1: 90deg, 2: 180deg, 3: 270deg
        
        if rot_choice == 1:
            new_dx, new_dy = -v_y, v_x
        elif rot_choice == 2:
            new_dx, new_dy = -v_x, -v_y
        else:
            new_dx, new_dy = v_y, -v_x
        
        # Apply the rotation to the entire suffix [k+1, ..., n-1]
        # Let P0 = current_chain[k]
        # For j from k+1 to n-1:
        #   new_P[j] = P0 + R * (old_P[j] - P0)
        
        # Optimization: 
        # 1. Compute new positions for suffix.
        # 2. Check if any new position is in the set of occupied positions of the prefix [0, ..., k-1].
        
        prefix_occupied = set(current_chain[:k]) # Positions 0 to k-1
        
        valid = True
        new_suffix = []
        
        # Pre-compute the rotation function based on rot_choice
        if rot_choice == 1:
            def rot(x, y): return (-y, x)
        elif rot_choice == 2:
            def rot(x, y): return (-x, -y)
        else:
            def rot(x, y): return (y, -x)
            
        px, py = current_chain[k]
        
        new_chain = current_chain[:k+1] # Copy prefix including pivot
        new_chain.extend([None] * (n - k - 1))
        
        # Calculate new positions for k+1 to n-1
        for j in range(k + 1, n):
            ox = current_chain[j][0] - px
            oy = current_chain[j][1] - py
            nx, ny = rot(ox, oy)
            new_pos = (px + nx, py + ny)
            
            # Check collision with prefix (0 to k-1)
            if new_pos in prefix_occupied:
                valid = False
                break
            
            new_chain[j] = new_pos
        
        if not valid:
            # Move rejected due to collision
            continue
        
        # Calculate new energy
        new_energy = _energy(sequence, new_chain)
        
        # Acceptance criterion: Simulated Annealing
        delta_E = new_energy - current_energy
        if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        T *= cooling
        
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using greedy growth followed by local search refinement."""
    start_time = time.time()
    # 120s CPU budget for ALL sequences.
    # Assuming 10 sequences, we have ~12s per sequence on average.
    # However, the prompt says "120 s CPU budget for all sequences together" and "Each evaluation must finish within 180 s".
    # To be safe and maximize score, we should use as much of the 120s as possible, but leave a safety margin.
    # Let's assume the worst case is that we are the only sequence running or the budget is shared loosely.
    # But "120s CPU budget for all sequences together" implies a total cap.
    # If we are called 10 times, and the total time is 120s, each call should ideally take ~12s.
    # However, if we run for 120s in one call, we might timeout the whole batch if others are slow.
    # Let's aim for ~10-15s per sequence to be safe within the 120s total, or simply track global time if possible?
    # No, we can't easily track global time across separate function calls unless we use a global variable.
    # Let's use a global variable to track total elapsed time across all fold calls.
    
    global _total_elapsed
    if not hasattr(fold, '_total_elapsed'):
        fold._total_elapsed = 0.0
        fold._start_time = time.time()
    
    # Check if we have exceeded the total budget
    elapsed_total = time.time() - fold._start_time
    remaining_total = 120.0 - elapsed_total
    
    # If we've used up most of the budget, return quickly
    if remaining_total < 1.0:
         # Return a simple valid chain
         n = len(sequence)
         return [list(p) for p in [(i, 0) for i in range(n)]]
    
    # Allocate a portion of the remaining budget to this sequence
    # Heuristic: allocate 1/10 of remaining, or a fixed 10s, whichever is smaller but at least 1s
    time_limit = max(1.0, min(10.0, remaining_total * 0.15))
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    n = len(sequence)
    
    # Phase 1: Greedy growth with multiple restarts
    # Use fewer restarts as pivot search is more powerful per iteration
    num_grow = 20 if n <= 30 else (10 if n <= 60 else 5)
    
    for trial in range(num_grow):
        if time.time() - start_time > time_limit:
            break
        
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Phase 2: Pivot Search refinement
        # Allocate more iterations to pivot search as it is the main optimizer
        # We don't know exactly how many iterations we can do, so let pivot_search handle time limiting
        max_pivot_iters = 5000 if n <= 50 else 3000
        
        # Pass the remaining time for this specific trial
        trial_start = time.time()
        trial_limit = time_limit - (trial_start - start_time)
        
        refined_chain, refined_energy = _pivot_search(
            sequence, chain, 
            max_iters=max_pivot_iters,
            rng=rng,
            time_limit=trial_limit
        )
        
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain
            
            # If we have time, run a second pass of pivot search on the best chain found so far
            time_elapsed_so_far = time.time() - start_time
            if time_limit - time_elapsed_so_far > 2.0:
                further_chain, further_energy = _pivot_search(
                    sequence, best_chain,
                    max_iters=max_pivot_iters,
                    rng=rng,
                    time_limit=time_limit - time_elapsed_so_far
                )
                if further_energy < best_energy:
                    best_energy = further_energy
                    best_chain = further_chain
    
    if best_chain is None:
        # Fallback: a straight line is always self-avoiding
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
