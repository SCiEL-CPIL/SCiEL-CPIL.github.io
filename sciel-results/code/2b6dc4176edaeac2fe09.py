"""HP lattice protein folding using simulated annealing with random restarts."""
import random
import time
import sys

# EVOLVE-BLOCK-START

def _neighbours(point):
    """Yield the 4 lattice neighbours of a point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute the HP contact energy of a conformation."""
    n = len(sequence)
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            for nb in _neighbours(chain[i]):
                j = pos_to_idx.get(nb)
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    if i < j:  # count each pair once
                        contacts += 1
    return -contacts


def _try_grow(sequence, dimension, rng):
    """
    Attempt to grow a self-avoiding walk using a randomized greedy strategy.
    Returns a list of lattice points or None if stuck.
    """
    n = len(sequence)
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score: prefer sites that create H-H contacts, then prefer free space
            score = rng.random() * 0.1
            if sequence[i] == 'H':
                # Count potential H-H contacts
                for nb in _neighbours(site):
                    if nb in occupied:
                        j = occupied[nb]
                        if j < i - 1 and sequence[j] == 'H':
                            score += 1.0
            # Slight preference for sites with more free neighbours (avoid trapping)
            free_count = sum(1 for nb in _neighbours(site) if nb not in occupied)
            score += 0.05 * free_count
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option (with some randomness via the score)
        best_score, best_site = max(options)
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain


def _simulated_annealing(sequence, dimension, rng, start_chain, time_limit):
    """
    Perform simulated annealing starting from start_chain.
    Returns the best chain found.
    """
    n = len(sequence)
    current_chain = list(start_chain)
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Build position lookup for fast move generation
    pos_set = set(current_chain)
    
    # Temperature schedule
    t_start = time.time()
    initial_temp = 2.0
    cooling_rate = 0.999
    
    temp = initial_temp
    
    while temp > 0.01:
        # Check time limit
        if time.time() - t_start > time_limit:
            break
        
        # Generate a random move: pick a residue (not ends) and try to move it
        if n <= 2:
            break
        
        i = rng.randint(1, n - 2)
        old_pos = current_chain[i]
        
        # Try each of the 4 neighbours
        moves = list(_neighbours(old_pos))
        rng.shuffle(moves)
        
        moved = False
        for new_pos in moves:
            # Check if new_pos is already occupied (excluding the residue itself)
            if new_pos in pos_set and new_pos != old_pos:
                continue
            
            # Check self-avoidance: new_pos must not be occupied by any other residue
            # We need to verify that moving residue i to new_pos doesn't create conflicts
            # Since we're moving one residue, we just need to check that new_pos is not occupied
            # by another residue. The old_pos will become free.
            
            # Also need to check that the move doesn't create a bond conflict:
            # The residue at i-1 and i+1 must still be adjacent to new_pos
            # Wait, no. The chain connectivity is preserved because we're moving a single residue.
            # But we need to ensure that after the move, the chain is still a valid SAW.
            # Since we're moving residue i, its neighbours are i-1 and i+1.
            # After the move, residue i is at new_pos. We need:
            # 1. new_pos is not occupied by any other residue
            # 2. The distance from new_pos to chain[i-1] is 1 (unit step)
            # 3. The distance from new_pos to chain[i+1] is 1 (unit step)
            
            # Check unit steps to neighbours
            prev_pos = current_chain[i-1]
            next_pos = current_chain[i+1]
            
            # Check distance to prev
            diff_prev = [abs(new_pos[d] - prev_pos[d]) for d in range(dimension)]
            if sum(diff_prev) != 1:
                continue
            
            # Check distance to next
            diff_next = [abs(new_pos[d] - next_pos[d]) for d in range(dimension)]
            if sum(diff_next) != 1:
                continue
            
            # Check that new_pos is not occupied by another residue
            # We remove old_pos from pos_set temporarily for the check
            pos_set.discard(old_pos)
            if new_pos in pos_set:
                pos_set.add(old_pos)
                continue
            
            # Compute energy difference
            # Remove old_pos, add new_pos
            # We need to recompute energy or compute delta
            # For simplicity, recompute energy
            new_chain = list(current_chain)
            new_chain[i] = new_pos
            new_energy = _compute_energy(sequence, new_chain)
            
            delta_e = new_energy - current_energy
            
            # Metropolis criterion
            if delta_e < 0 or rng.random() < pow(2.718281828, -delta_e / temp):
                # Accept the move
                current_chain[i] = new_pos
                pos_set.add(new_pos)
                # old_pos was already discarded
                current_energy = new_energy
                moved = True
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(new_chain)
                
                break
        
        if not moved:
            # No valid move found, continue to next iteration
            pass
        
        temp *= cooling_rate
    
    return best_chain


def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    Returns a list of lattice points (lists of length dimension).
    """
    n = len(sequence)
    
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    # Overall time budget
    total_deadline = time.time() + 100  # Leave some margin for the 120s budget
    
    # First, try to generate a good initial conformation
    best_chain = None
    best_energy = 0
    
    # Try multiple random greedy growths to find a good starting point
    rng = random.Random(42 + len(sequence))
    
    growth_attempts = 50
    for _ in range(growth_attempts):
        if time.time() > total_deadline:
            break
        chain = _try_grow(sequence, dimension, rng)
        if chain is None:
            continue
        energy = _compute_energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [[i, 0] if dimension == 2 else [i] + [0]*(dimension-1) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    
    # Now perform simulated annealing from the best starting point
    # Use multiple restarts with different initial conformations
    sa_time_per_restart = 8  # seconds per SA run
    
    # Try SA from the best chain found
    if time.time() < total_deadline - sa_time_per_restart:
        sa_result = _simulated_annealing(sequence, dimension, rng, best_chain, sa_time_per_restart)
        sa_energy = _compute_energy(sequence, sa_result)
        if sa_energy < best_energy:
            best_chain = sa_result
            best_energy = sa_energy
    
    # Try a few more random restarts if time permits
    restarts = 3
    for r in range(restarts):
        if time.time() > total_deadline - sa_time_per_restart:
            break
        
        # Generate a new random starting conformation
        chain = _try_grow(sequence, dimension, rng)
        if chain is None:
            continue
        energy = _compute_energy(sequence, chain)
        
        # Only do SA if this starting point is reasonably good
        if energy < best_energy + 2:  # Within 2 contacts of best
            sa_result = _simulated_annealing(sequence, dimension, rng, chain, sa_time_per_restart)
            sa_energy = _compute_energy(sequence, sa_result)
            if sa_energy < best_energy:
                best_chain = sa_result
                best_energy = sa_energy
    
    # Return the best chain found
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
