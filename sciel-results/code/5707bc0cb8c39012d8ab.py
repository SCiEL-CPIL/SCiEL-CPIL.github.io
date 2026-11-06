"""Rewritten program for HP lattice protein folding with local search optimization."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global time management
_GLOBAL_START_TIME = time.time()
_GLOBAL_LIMIT = 180.0  # Safety margin for 210s kill

def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START_TIME)

def _neighbours(point):
    """Yield 4-neighbours of a 2D point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _calculate_energy(sequence, chain):
    """Calculate the HP energy of a chain. Energy = - (number of non-adjacent H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    # Optimization: only check H residues
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _initial_growth(sequence, rng):
    """
    Generate a self-avoiding walk using a greedy randomized approach.
    Prioritizes moves that create H-H contacts, then free space.
    """
    if not sequence:
        return []
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    last_h_index = -1
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            # Heuristic score
            score = rng.random()
            
            # Check for potential H-H contacts if current residue is H
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            score += 10.0  # Strong preference for contacts
                
                # Also check if this site is near other Hs in the chain (not strictly contact, but proximity)
                # This helps compactness
            
            # Slight preference for free space to avoid trapping
            free_space = sum(1 for q in _neighbours(site) if q not in occupied)
            score += free_space * 0.1
            
            options.append((score, site))
        
        if not options:
            return None  # Trapped
            
        # Pick the best option
        best_score, best_site = max(options)
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain

def _pivot_move(chain, pivot_index, direction, occupied_set):
    """
    Attempt a pivot move at pivot_index.
    direction: 0 (no move), 1 (right), 2 (up), 3 (left), 4 (down) relative to the segment.
    Returns (new_chain, new_occupied_set, success)
    """
    if pivot_index <= 0 or pivot_index >= len(chain) - 1:
        return None, None, False
        
    # The segment to rotate is chain[pivot_index+1:]
    # The pivot is chain[pivot_index]
    # We rotate the segment around the pivot.
    
    # To simplify, we'll construct the new segment positions.
    # Original segment: S = chain[pivot_index+1:]
    # We need to apply a rotation matrix to the coordinates relative to the pivot.
    
    pivot = chain[pivot_index]
    segment = chain[pivot_index+1:]
    
    # Rotation directions in 2D:
    # 0: identity
    # 1: (x,y) -> (y, -x)  (90 deg CW)
    # 2: (x,y) -> (-x, -y) (180 deg)
    # 3: (x,y) -> (-y, x)  (90 deg CCW)
    # Note: We must check if the new positions are valid (not occupied by the rest of the chain).
    
    # The "rest" of the chain is chain[0:pivot_index+1]
    # We need to temporarily remove the segment from the occupied set to check for collisions.
    
    rest_chain = chain[:pivot_index+1]
    rest_occupied = set(rest_chain)
    
    new_segment = []
    valid = True
    
    for p in segment:
        # Relative coordinate
        dx = p[0] - pivot[0]
        dy = p[1] - pivot[1]
        
        if direction == 1:
            nx, ny = dy, -dx
        elif direction == 2:
            nx, ny = -dx, -dy
        elif direction == 3:
            nx, ny = -dy, dx
        else:
            nx, ny = dx, dy
            
        new_p = (pivot[0] + nx, pivot[1] + ny)
        
        if new_p in rest_occupied or new_p in new_segment:
            valid = False
            break
        new_segment.append(new_p)
        
    if not valid:
        return None, None, False
        
    new_chain = rest_chain + new_segment
    new_occupied = set(new_chain)
    
    return new_chain, new_occupied, True

def _optimize(sequence, chain, time_budget):
    """
    Optimize a single chain using Metropolis Monte Carlo with pivot moves.
    """
    if not chain:
        return chain
    
    n = len(chain)
    current_chain = chain
    current_energy = _calculate_energy(sequence, current_chain)
    best_chain = current_chain
    best_energy = current_energy
    current_occupied = set(current_chain)
    
    rng = random.Random()
    
    # Annealing parameters
    start_temp = 2.0
    end_temp = 0.01
    # We don't know the number of steps in advance, so we use time-based stopping.
    # We'll estimate a total number of steps based on time and adjust temp dynamically.
    
    start_time = time.time()
    steps = 0
    
    while (time.time() - start_time) < time_budget and _time_left() > 1.0:
        steps += 1
        
        # Dynamic temperature: cool down linearly over the allocated time
        # We need to know the total expected time. Let's assume we run for the full budget.
        # progress = (time.time() - start_time) / time_budget
        # temp = start_temp - (start_temp - end_temp) * progress
        
        # Simpler: use a fixed schedule of 1000 "virtual" steps, but map them to time.
        # Or just use a random temperature that decreases.
        # Let's use a simple cooling: T = T0 * exp(-alpha * steps)
        # But steps rate varies.
        # Let's stick to a simple linear cooling over the duration.
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / time_budget)
        temp = start_temp - (start_temp - end_temp) * progress
        
        # Choose a move
        # Pivot moves are effective.
        # Pick a random pivot index (1 to n-2)
        pivot_idx = rng.randint(1, n - 2)
        direction = rng.randint(1, 4)
        
        new_chain, new_occupied, success = _pivot_move(current_chain, pivot_idx, direction, current_occupied)
        
        if not success:
            continue
            
        new_energy = _calculate_energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Metropolis criterion
        if delta_e < 0 or rng.random() < math.exp(-delta_e / temp):
            current_chain = new_chain
            current_energy = new_energy
            current_occupied = new_occupied
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = new_chain
        else:
            # Reject move, do nothing
            pass
            
    return best_chain

def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    """
    if dimension != 2:
        raise ValueError("Only 2D dimension is supported")
        
    if not sequence:
        return []
        
    n = len(sequence)
    
    # Allocate time for this sequence
    # We have 10 sequences total.
    # Let's reserve some time for overhead.
    # Total limit 180s.
    # If this is the first call, we assume we have ~18s per sequence.
    # We don't know the index, so we just use a fixed budget per call?
    # No, we should track how much time we've spent in total.
    # But the function is called independently? "fold is called once per benchmark sequence in a sandboxed subprocess"
    # Wait, "subprocess" implies separate processes? No, "sandboxed subprocess" might mean the whole program is a subprocess.
    # "fold is called once per benchmark sequence" -> This implies the same process handles all 10 calls?
    # Or does it mean the evaluator calls fold() 10 times in the same process?
    # "The evaluator checks... fold is called once per benchmark sequence in a sandboxed subprocess with a 120 s CPU budget for all sequences together."
    # This suggests the entire execution of the program (which handles all 10 sequences) is in one subprocess with 120s (or 210s wall) budget.
    # So `fold` might be called 10 times in a loop by the evaluator, or the evaluator imports the module and calls fold 10 times.
    # We must manage time globally.
    
    # Let's assume the evaluator calls fold() 10 times sequentially.
    # We will track the global start time.
    # We will allocate a fraction of the remaining time to this call.
    
    # To be safe, let's assume we have 1/10th of the total time for each call on average.
    # But if previous calls were fast, we can use more.
    # If previous calls were slow, we must be careful.
    
    # Simple strategy:
    # 1. Calculate time left.
    # 2. Assume there are at most 10 calls.
    # 3. Allocate (time_left / 10) * 1.2 (slight oversubscription if we are fast) to this call?
    # No, better to be conservative.
    
    # Let's just use a fixed budget per call if we don't know the call index.
    # But we can track how many times fold has been called? No, state is not guaranteed to persist if it's a fresh import?
    # Usually in these tasks, the module is imported once.
    
    # Let's use a global counter to estimate remaining calls.
    global _FOLD_CALL_COUNT
    if not hasattr(fold, '_count'):
        fold._count = 0
    fold._count += 1
    
    total_calls_expected = 10
    calls_done = fold._count
    calls_remaining = max(1, total_calls_expected - calls_done + 1)
    
    time_left_global = _time_left()
    # Reserve 1s for safety
    available_time = max(1.0, time_left_global - 1.0)
    # Allocate proportional share
    time_budget = available_time / calls_remaining
    
    # Cap the budget for very long sequences to ensure we finish?
    # No, the budget is already proportional.
    
    # Generate initial candidates
    # For short sequences, we can do more initializations.
    # For long sequences, initialization is expensive.
    
    num_initials = 5
    if n < 30:
        num_initials = 10
    elif n > 80:
        num_initials = 3
        
    best_chain = None
    best_energy = 0  # Start with 0 (unfolded) as baseline, we want lower (more negative)
    
    # We will run initial growth and optimization in parallel? No, sequential.
    # We will iterate through num_initials.
    # We will distribute time_budget among them.
    
    time_per_initial = time_budget / num_initials
    
    rng = random.Random(12345 + n)
    
    for i in range(num_initials):
        # Check time
        if _time_left() < 1.0:
            break
            
        # Generate initial chain
        # If time is very tight, just do one greedy growth and no optimization?
        # Greedy growth is O(N^2) or O(N) with sets.
        
        chain = _initial_growth(sequence, rng)
        
        if chain is None:
            # Retry with different rng seed
            rng = random.Random()
            chain = _initial_growth(sequence, rng)
            
        if chain is None:
            # Fallback: straight line (if possible, else random walk?)
            # A straight line is always self-avoiding.
            chain = [(i, 0) for i in range(n)]
            
        # Calculate energy of initial chain
        e = _calculate_energy(sequence, chain)
        
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e
            
        # Optimize if we have time
        # We need to leave time for subsequent initials and other sequences.
        # Let's give 80% of time_per_initial to optimization.
        opt_time = time_per_initial * 0.8
        if opt_time > 0.1:
            optimized_chain = _optimize(sequence, chain, opt_time)
            if optimized_chain:
                e_opt = _calculate_energy(sequence, optimized_chain)
                if e_opt < best_energy:
                    best_chain = optimized_chain
                    best_energy = e_opt

    # Final fallback
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
