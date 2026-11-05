"""HP lattice protein folding using simulated annealing with compact initial conformations."""
import random
import time
import math


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _build_occupied(chain):
    return {p: i for i, p in enumerate(chain)}


def _compute_energy(sequence, chain):
    where = _build_occupied(chain)
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _try_move(chain, move_idx, delta, occupied, sequence, dimension):
    """Try to move residue at move_idx to a neighboring site.
    Returns (new_chain, new_occupied, energy_delta) or None if invalid."""
    pos = chain[move_idx]
    for q in _neighbours(pos):
        if q in occupied:
            continue
        # Check self-avoidance: new position must not be occupied
        # and must not break adjacency with neighbors in chain
        # Move residue i from pos to q
        # Check that q is not occupied by any other residue
        # (already checked above)
        # Check that the chain remains connected: 
        # residue i-1 and i+1 must still be adjacent to their new positions
        # Actually, we're moving residue i, so we need to check:
        # - If i > 0, chain[i-1] must be adjacent to q OR we keep chain[i-1] and move i
        #   Wait, we're moving residue i, so chain[i-1] stays, chain[i+1] stays.
        #   We need chain[i-1] to be adjacent to q (if i > 0) and chain[i+1] to be adjacent to q (if i < len-1)
        
        valid = True
        if move_idx > 0:
            prev = chain[move_idx - 1]
            if abs(prev[0] - q[0]) + abs(prev[1] - q[1]) + (abs(prev[2] - q[2]) if dimension > 2 else 0) != 1:
                valid = False
        if valid and move_idx < len(chain) - 1:
            nxt = chain[move_idx + 1]
            if abs(nxt[0] - q[0]) + abs(nxt[1] - q[1]) + (abs(nxt[2] - q[2]) if dimension > 2 else 0) != 1:
                valid = False
        
        if not valid:
            continue
        
        # Compute energy delta
        # Remove contacts involving residue i at old position
        old_contacts = 0
        if sequence[move_idx] == "H":
            for nb in _neighbours(pos):
                j = occupied.get(nb)
                if j is not None and j != move_idx and sequence[j] == "H":
                    # Count each contact once: only if j > move_idx or we use a consistent rule
                    # Let's count all H-H pairs where the first index is smaller
                    if j > move_idx:
                        old_contacts += 1
        
        # Add contacts at new position
        new_contacts = 0
        if sequence[move_idx] == "H":
            for nb in _neighbours(q):
                j = occupied.get(nb)
                if j is not None and j != move_idx and sequence[j] == "H":
                    if j > move_idx:
                        new_contacts += 1
        
        energy_delta = -(new_contacts - old_contacts)
        
        # Apply move
        new_chain = chain[:]
        new_chain[move_idx] = q
        new_occupied = occupied.copy()
        del new_occupied[pos]
        new_occupied[q] = move_idx
        
        return (new_chain, new_occupied, energy_delta)
    
    return None


def _initial_spiral(sequence, dimension):
    """Create a compact initial conformation using a spiral pattern."""
    n = len(sequence)
    if n <= 1:
        return [(0,)] * dimension
    
    # Try to fit in a small box
    # Use a simple spiral in 2D, extended to 3D if needed
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    # Direction: start going +x, then +y, then -x, then -y, then +x with larger step, etc.
    # This creates a spiral
    directions = [
        (1, 0, 0),   # +x
        (0, 1, 0),   # +y
        (-1, 0, 0),  # -x
        (0, -1, 0),  # -y
    ]
    
    step_counts = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10]
    dir_idx = 0
    step_idx = 0
    steps_in_current = 1
    steps_done = 0
    
    for i in range(1, n):
        placed = False
        max_attempts = 100
        for attempt in range(max_attempts):
            # Get current direction
            dx, dy, dz = directions[dir_idx % 4]
            
            # Try to move in current direction
            if steps_done < steps_in_current:
                new_pos = (chain[-1][0] + dx, chain[-1][1] + dy, chain[-1][2] + dz)
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied[new_pos] = i
                    placed = True
                    steps_done += 1
                    break
            
            # If can't place or steps exhausted, try next direction
            dir_idx += 1
            steps_done = 0
            # Update step count based on direction index
            # Every 2 direction changes, increment step count
            if dir_idx % 2 == 0:
                step_idx = min(step_idx + 1, len(step_counts) - 1)
                steps_in_current = step_counts[step_idx]
        
        if not placed:
            # Try all 6 neighbors
            for nb in _neighbours(chain[-1]):
                if nb not in occupied:
                    chain.append(nb)
                    occupied[nb] = i
                    placed = True
                    break
        
        if not placed:
            # Give up, just extend in a straight line
            chain.append((chain[-1][0] + 1, 0, 0))
            occupied[(chain[-1][0], 0, 0)] = i
    
    return chain


def _simulated_annealing(sequence, dimension, time_budget):
    """Run simulated annealing to optimize the conformation."""
    n = len(sequence)
    if n <= 1:
        return [(0,)] * dimension
    
    # Initial conformation
    chain = _initial_spiral(sequence, dimension)
    occupied = _build_occupied(chain)
    current_energy = _compute_energy(sequence, chain)
    
    best_chain = chain[:]
    best_energy = current_energy
    
    start_time = time.time()
    temperature = 1.0
    cooling_rate = 0.999
    
    # Number of iterations based on time budget
    # Estimate: each iteration takes ~1-5 microseconds for small chains
    max_iterations = max(1000, int(time_budget * 100000))
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_budget * 0.9:
            break
        
        # Choose a random residue to move
        move_idx = random.randint(0, n - 1)
        
        result = _try_move(chain, move_idx, 0, occupied, sequence, dimension)
        if result is None:
            temperature *= cooling_rate
            continue
        
        new_chain, new_occupied, energy_delta = result
        
        # Accept or reject based on Metropolis criterion
        if energy_delta < 0 or random.random() < math.exp(-energy_delta / temperature):
            chain = new_chain
            occupied = new_occupied
            current_energy += energy_delta
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
        
        # Cool down
        temperature *= cooling_rate
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using simulated annealing with time budget."""
    n = len(sequence)
    
    # Allocate time budget based on sequence length
    # Total budget is 240s for all sequences, but we don't know how many
    # Assume we get a reasonable share. Let's use a fixed budget per sequence.
    # For short chains (< 20), use less time; for long chains, use more.
    if n < 20:
        time_budget = 5.0
    elif n < 50:
        time_budget = 15.0
    elif n < 100:
        time_budget = 40.0
    else:
        time_budget = 60.0
    
    # Run multiple restarts
    best_chain = None
    best_energy = 0
    
    num_restarts = 3 if n < 30 else 2
    
    for restart in range(num_restarts):
        # Use different random seeds for diversity
        random.seed(12345 + len(sequence) + restart * 1000)
        
        chain = _simulated_annealing(sequence, dimension, time_budget / num_restarts)
        energy = _compute_energy(sequence, chain)
        
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
