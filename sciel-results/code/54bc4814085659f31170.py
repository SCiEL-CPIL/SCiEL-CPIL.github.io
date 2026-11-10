"""HP lattice protein folding on the cubic lattice using Simulated Annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = time.time()
TIME_LIMIT = 320.0  # Safety margin against 330s kill

def _remaining_time():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _neighbours(point):
    """Yield 6 neighbors in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy_delta(sequence, chain, occupied, i, new_pos):
    """
    Compute the change in energy if residue i at chain[i] moves to new_pos.
    Energy = - (number of H-H contacts).
    We track contacts involving residue i.
    """
    if sequence[i] != 'H':
        # If P moves, it doesn't form H-H contacts, but it might block/unblock?
        # No, H-H contacts are only between H residues.
        # However, moving a P residue changes the self-avoidance constraints for others?
        # No, we are just checking if the move is valid (self-avoiding).
        # If valid, energy change is 0 because P doesn't contribute to H-H contacts.
        # Wait, does moving P break an H-H contact? No, P is not H.
        # Does moving P create an H-H contact? No.
        # So if sequence[i] == 'P', delta_energy is 0.
        return 0

    # If sequence[i] == 'H'
    old_pos = chain[i]
    delta = 0
    
    # 1. Remove contacts involving i at old_pos
    # A contact is between i and j if they are neighbors and not adjacent in chain.
    # We need to check all neighbors of old_pos that are H and not i-1, i+1.
    for q in _neighbours(old_pos):
        j = occupied.get(q)
        if j is not None and j != i and abs(j - i) > 1:
            if sequence[j] == 'H':
                # This contact was contributing -1 to energy. Removing it increases energy by +1.
                delta += 1

    # 2. Add contacts involving i at new_pos
    for q in _neighbours(new_pos):
        j = occupied.get(q)
        if j is not None and j != i and abs(j - i) > 1:
            if sequence[j] == 'H':
                # This contact will contribute -1 to energy. Adding it decreases energy by -1.
                delta -= 1
                
    return delta

def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n == 0:
        return []
    
    # Start at origin
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site not in occupied:
                options.append(site)
        
        if not options:
            return None # Stuck
        
        # Random choice
        site = rng.choice(options)
        chain.append(site)
        occupied[site] = i
        
    return chain

def _sa_fold(sequence, dimension, time_budget, rng):
    """Perform Simulated Annealing for a specific sequence."""
    n = len(sequence)
    if n == 0:
        return []
    
    # Initial random walk
    chain = _random_walk(sequence, dimension, rng)
    if chain is None:
        # Fallback: straight line (always valid)
        chain = [tuple([i] + [0]*(dimension-1)) for i in range(n)]
        
    occupied = {p: i for i, p in enumerate(chain)}
    
    # Initial energy
    current_energy = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = occupied.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    current_energy -= 1
                    
    best_chain = list(chain)
    best_energy = current_energy
    
    # SA Parameters
    # Temperature schedule: T = T0 * alpha^k
    # T0 should be high enough to accept most bad moves initially.
    # Max energy change is roughly n/2.
    T0 = max(1.0, n * 0.1)
    T_end = 0.01
    alpha = 0.995 # Cooling rate
    
    start_time = time.time()
    T = T0
    
    while time.time() - start_time < time_budget:
        if T < T_end:
            break
            
        # Pick a random residue to move
        i = rng.randint(1, n - 2) # Avoid ends to simplify boundary checks? 
        # Actually, moving ends is valid too, but let's stick to internal for simplicity 
        # or allow all? Moving ends changes the "fixed" nature of the walk? 
        # In lattice folding, all residues are part of the chain. 
        # Moving any residue i (1..n-2) is a standard "pivot" or "kink" move.
        # Moving ends (0 or n-1) is a "rotation" of the end segment, which is more complex.
        # Let's stick to single residue moves for i in 1..n-2.
        
        old_pos = chain[i]
        
        # Try to move to a random neighbor
        neighbors = list(_neighbours(old_pos))
        rng.shuffle(neighbors)
        
        moved = False
        for new_pos in neighbors:
            if new_pos in occupied:
                continue # Self-avoidance violated
            
            # Calculate delta E
            delta_e = _compute_energy_delta(sequence, chain, occupied, i, new_pos)
            
            # Acceptance criterion
            if delta_e <= 0:
                accept = True
            else:
                # Boltzmann probability
                # exp(-delta_e / T)
                # If T is small, exp is small.
                accept = rng.random() < math.exp(-delta_e / T)
                
            if accept:
                # Execute move
                del occupied[old_pos]
                occupied[new_pos] = i
                chain[i] = new_pos
                current_energy += delta_e
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                    
                moved = True
                break # Only one move per iteration to keep T schedule consistent with steps
        
        # Cool down
        T *= alpha
        
    return best_chain

def fold(sequence, dimension):
    """
    Fold a sequence using Simulated Annealing.
    Returns a list of lattice points.
    """
    # Time management
    # We have ~320s for all 26 sequences.
    # Average ~12s per sequence.
    # But some sequences are longer (136 residues) and harder.
    # Let's allocate time based on length or just a fixed budget per call.
    # The prompt says "fold is called once per benchmark sequence".
    # So we have a global budget, but we don't know which sequence is which.
    # However, we can track total elapsed time.
    
    total_elapsed = time.time() - _START_TIME
    # Assume 26 sequences.
    # If we've used 50% of time, we should use less per remaining sequence.
    # Simple heuristic: allocate 10% of remaining time to this call, 
    # but cap it to avoid starving later sequences if this one is easy?
    # Actually, better to just use a fixed fraction of the *remaining* time 
    # assuming we don't know how many are left.
    # But we *do* know there are 26.
    # Let's estimate remaining sequences.
    # We can't easily count calls without global state, but we can use a static counter.
    
    global _CALL_COUNT
    if not hasattr(fold, '_count'):
        fold._count = 0
    fold._count += 1
    calls_made = fold._count
    total_expected = 26
    remaining_calls = max(1, total_expected - calls_made + 1)
    
    time_left = _remaining_time()
    if time_left <= 0:
        # Emergency: return straight line
        n = len(sequence)
        return [list([i] + [0]*(dimension-1)) for i in range(n)]
        
    # Allocate time for this sequence
    # If it's the last one, use all remaining time.
    # Otherwise, use time_left / remaining_calls.
    # Add a safety margin.
    budget = time_left / remaining_calls * 0.9
    
    # Minimum budget to do *something*
    if budget < 0.1:
        budget = 0.1
        
    rng = random.Random(12345 + len(sequence) + fold._count)
    
    chain = _sa_fold(sequence, dimension, budget, rng)
    
    # Convert to list of lists
    return [list(p) for p in chain]
# EVOLVE-BLOCK-END
