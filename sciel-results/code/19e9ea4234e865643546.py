"""Improved HP lattice protein folding using Multi-Start Simulated Annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global time management
_START_TIME = time.time()
_TOTAL_BUDGET = 290.0  # Leave 40s margin for 330s limit
_DEADLINE = _START_TIME + _TOTAL_BUDGET

# Counter for sequences processed
_CALL_COUNT = 0
_EXPECTED_CALLS = 26

def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _generate_saw(length, dimension, rng):
    """Generate a random self-avoiding walk."""
    if length == 0:
        return []
    if length == 1:
        return [(0, 0, 0)]
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    
    for _ in range(1, length):
        last = chain[-1]
        options = []
        for n in _neighbours(last):
            if n not in occupied:
                options.append(n)
        
        if not options:
            return None # Dead end
        
        next_pos = rng.choice(options)
        chain.append(next_pos)
        occupied.add(next_pos)
        
    return chain

def _calculate_energy(sequence, chain):
    """Calculate energy from scratch."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _sa_optimize(sequence, chain, dimension, rng, time_limit):
    """
    Optimize a single chain using Simulated Annealing.
    Returns the best chain found and its energy.
    """
    n = len(sequence)
    if n <= 1:
        return chain, 0

    current_chain = list(chain)
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    current_energy = _calculate_energy(sequence, current_chain)
    best_energy = current_energy
    best_chain = list(current_chain)
    
    start_t = time.time()
    T = 1.0
    cooling_rate = 0.9995 # Tune this
    
    while time.time() - start_t < time_limit:
        # Pick a random residue to move (not endpoints)
        i = rng.randint(1, n - 2)
        old_pos = current_chain[i]
        
        # Pick a random neighbor
        neighbors = list(_neighbours(old_pos))
        rng.shuffle(neighbors)
        
        moved = False
        for new_pos in neighbors:
            if new_pos in occupied:
                continue
            
            # Calculate delta E
            delta_e = 0
            if sequence[i] == 'H':
                # Lost contacts
                for q in _neighbours(old_pos):
                    j = occupied.get(q)
                    if j is not None and sequence[j] == 'H' and abs(j - i) > 1:
                        delta_e += 1
                # Gained contacts
                for q in _neighbours(new_pos):
                    j = occupied.get(q)
                    if j is not None and sequence[j] == 'H' and abs(j - i) > 1:
                        delta_e -= 1
            
            # Acceptance
            if delta_e <= 0 or rng.random() < math.exp(-delta_e / max(T, 1e-9)):
                # Accept move
                del occupied[old_pos]
                current_chain[i] = new_pos
                occupied[new_pos] = i
                current_energy += delta_e
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                
                moved = True
                break
        
        # Cool down
        T *= cooling_rate
        
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Best conformation found within the global time budget.
    """
    global _CALL_COUNT
    _CALL_COUNT += 1
    
    remaining_time = _DEADLINE - time.time()
    if remaining_time < 1.0:
        return [[i, 0, 0] for i in range(len(sequence))]
    
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0 # Energy is negative
    
    # Calculate time budget for this sequence
    # Heuristic: Distribute remaining time evenly among remaining sequences
    remaining_calls = max(1, _EXPECTED_CALLS - _CALL_COUNT + 1)
    time_per_call = remaining_time / remaining_calls
    # Cap it to prevent one sequence from taking too long if others are fast
    time_per_call = min(time_per_call, 15.0) 
    # Ensure at least 1s if possible
    time_per_call = max(time_per_call, 1.0)
    
    local_start = time.time()
    
    while time.time() - local_start < time_per_call:
        # Check global deadline
        if time.time() >= _DEADLINE - 1.0:
            break
        
        # Generate initial SAW
        init_chain = _generate_saw(n, dimension, rng)
        if init_chain is None:
            continue
        
        # Run SA for a slice of the time
        # Use 50% of the allocated time per restart? 
        # Or just run SA until the local time is up?
        # If we run SA until local time is up, we only do 1 restart.
        # Better to do multiple short restarts.
        
        # Let's do restarts of max 0.5s each.
        sa_time = min(0.5, time_per_call - (time.time() - local_start))
        if sa_time < 0.1:
            break
            
        opt_chain, opt_energy = _sa_optimize(sequence, init_chain, dimension, rng, sa_time)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_chain = opt_chain
            
    if best_chain is None:
        # Fallback: straight line
        best_chain = [[i, 0, 0] for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
