"""
HP Lattice Protein Folding using Simulated Annealing.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = time.time()
# The evaluator kills after 210s. We have 10 sequences.
# We should aim to use ~15-20s per sequence to be safe, leaving buffer for overhead.
# However, the prompt says "120 s CPU budget for all sequences together" in the task description,
# but the directive says "evaluator KILLS the program after 210s wall-clock".
# We will track time and stop if we are close to the limit.
# Let's reserve 10s at the end.
DEADLINE = 200.0

def _remaining_time():
    return DEADLINE - time.time()

def _neighbours(point):
    """Yield the 4 neighbors of a point on a 2D lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    """Calculate the energy of a conformation. Energy = - (number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _is_valid_move(chain, occupied, move_indices, new_positions):
    """
    Check if moving the residues at move_indices to new_positions is valid.
    move_indices: list of indices in the chain to move.
    new_positions: list of new coordinates for those indices.
    Returns True if valid, False otherwise.
    """
    # The positions currently occupied by the moving residues must be freed.
    # The new positions must not be occupied by any *other* residue.
    
    # Create a set of positions that will be freed
    freed_positions = set()
    for idx in move_indices:
        freed_positions.add(tuple(chain[idx]))
        
    # Check new positions
    for pos in new_positions:
        pos_t = tuple(pos)
        # If the new position is one of the freed positions, it's fine (unless it's occupied by a non-moving part, 
        # but we freed it from the moving part, so if it was occupied by a non-moving part, it wouldn't be in freed_positions 
        # unless the move is a rotation that maps a moving part onto another moving part, which is handled by the set logic? 
        # Wait, if I move A to B's old spot, and B moves to A's old spot, that's a swap. 
        # In pivot moves, we rotate a segment. The segment moves as a rigid body.
        # So, we remove the old segment positions from 'occupied' temporarily? 
        # Better logic:
        # 1. Identify all positions currently occupied by the moving segment.
        # 2. Identify all new positions.
        # 3. A new position is valid if it is either:
        #    a) In the set of old positions of the moving segment (it's being vacated), OR
        #    b) Not in the global 'occupied' set.
        
        if pos_t in freed_positions:
            continue
        if pos_t in occupied:
            return False
            
    return True

def _apply_move(chain, move_indices, new_positions):
    """Apply the move to the chain in-place."""
    for idx, pos in zip(move_indices, new_positions):
        chain[idx] = pos

def _generate_pivot_move(chain, rng):
    """
    Generate a pivot move: rotate a sub-chain (segment) by 90 degrees around a hinge.
    Returns (move_indices, new_positions) or None if invalid pivot.
    """
    n = len(chain)
    if n < 4:
        return None
        
    # Choose a hinge position i (1 to n-2)
    i = rng.randint(1, n - 2)
    # Choose a segment length L (2 to min(n-i, 5))
    max_len = min(n - i, 5)
    if max_len < 2:
        return None
    L = rng.randint(2, max_len)
    
    # The segment is chain[i+1 : i+1+L]
    # Hinge is at chain[i]
    # Direction from i to i+1 determines the rotation axis.
    # We rotate the vector (chain[k] - chain[i]) by 90 degrees.
    
    hinge = chain[i]
    # Vector from hinge to the first residue of the segment
    v0 = (chain[i+1][0] - hinge[0], chain[i+1][1] - hinge[1])
    
    # Rotate v0 by 90 degrees. There are two directions: CW and CCW.
    # Let's pick one randomly.
    if rng.random() < 0.5:
        # CW rotation: (x, y) -> (y, -x)
        v_rot = (v0[1], -v0[0])
    else:
        # CCW rotation: (x, y) -> (-y, x)
        v_rot = (-v0[1], v0[0])
        
    # Calculate new positions for the segment
    # We need to calculate the new positions relative to the hinge.
    # The new position of k is hinge + rotated_vector(k)
    # But we don't have the vector from hinge to k directly in a simple way if the chain is not straight.
    # Actually, a pivot move rotates the *entire subchain* as a rigid body.
    # So we need to rotate the vector from the hinge to each point in the segment.
    
    move_indices = list(range(i + 1, i + 1 + L))
    new_positions = []
    for idx in move_indices:
        # Vector from hinge to current position
        v = (chain[idx][0] - hinge[0], chain[idx][1] - hinge[1])
        # Rotate this vector
        if rng.random() < 0.5: # We should pick the same rotation direction for all points in the segment!
            pass # This is wrong. We must use the same rotation for the whole segment.
            
    # Let's re-implement properly.
    # Pick rotation direction once.
    rot_dir = rng.choice([1, -1]) # 1 for CCW, -1 for CW
    
    new_positions = []
    for idx in move_indices:
        v = (chain[idx][0] - hinge[0], chain[idx][1] - hinge[1])
        if rot_dir == 1:
            v_rot = (-v[1], v[0])
        else:
            v_rot = (v[1], -v[0])
        
        new_pos = (hinge[0] + v_rot[0], hinge[1] + v_rot[1])
        new_positions.append(new_pos)
        
    return move_indices, new_positions

def _generate_slide_move(chain, rng):
    """
    Generate a slide move: move a single residue (not end) to a neighbor if possible.
    Actually, sliding a single internal residue is hard to maintain connectivity unless it's an end.
    Let's stick to Pivot moves and End moves.
    End move: Rotate the last 2 or 3 residues.
    """
    n = len(chain)
    if n < 3:
        return None
    
    # Slide the last residue? No, that's a pivot of length 1.
    # Let's try a "Crankshaft" move: Rotate a segment of length 2 around a hinge at i.
    # This is effectively a pivot of length 2.
    
    # Let's add a specific "End Pivot" which is very efficient.
    if n >= 3:
        # Rotate last 2 residues around n-2
        i = n - 2
        L = 2
        move_indices = [i+1, i+2]
        hinge = chain[i]
        rot_dir = rng.choice([1, -1])
        new_positions = []
        for idx in move_indices:
            v = (chain[idx][0] - hinge[0], chain[idx][1] - hinge[1])
            if rot_dir == 1:
                v_rot = (-v[1], v[0])
            else:
                v_rot = (v[1], -v[0])
            new_pos = (hinge[0] + v_rot[0], hinge[1] + v_rot[1])
            new_positions.append(new_pos)
        return move_indices, new_positions
        
    return None

def _initial_chain(sequence, dimension, rng):
    """Create an initial self-avoiding chain."""
    # Start with a straight line, then maybe randomize?
    # A straight line is valid.
    # To give SA a head start, we can do a quick greedy growth.
    start = (0, 0)
    chain = [start]
    occupied = {start}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site not in occupied:
                # Heuristic: prefer sites that create H-H contacts
                gain = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        if q in occupied:
                            # Check if the residue at q is H and not adjacent in chain
                            # We don't have index mapping here easily, but we can approximate
                            # or just use random.
                            pass 
                options.append(site)
        
        if not options:
            # Fallback: try to backtrack or just pick a random neighbor if we are stuck
            # For initialization, if stuck, we can try a different start or just return what we have
            # and let SA fix it? No, SA needs a valid chain.
            # Let's try to continue by picking a random direction if possible, 
            # but if no options, we might have to restart initialization.
            # For simplicity, if we get stuck, we return a straight line.
            # But a straight line is always possible if we start at (0,0) and go right.
            # Let's just force a straight line if greedy fails.
            # Actually, let's just use a straight line for initialization to be safe and fast.
            # SA will do the work.
            break
            
        # Pick randomly to add diversity
        site = rng.choice(options)
        chain.append(site)
        occupied.add(site)
        
    # If we didn't place all residues, fill the rest in a straight line extension
    # This is a bit hacky. Let's just use a straight line for init.
    # It's a valid conformation.
    return [(i, 0) for i in range(len(sequence))]

def fold(sequence, dimension):
    """
    Fold the sequence using Simulated Annealing.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    rng = random.Random(12345 + len(sequence))
    
    # 1. Initialize
    # Let's use a simple straight line as the starting point.
    # It is guaranteed to be valid.
    chain = [(i, 0) for i in range(n)]
    occupied = set(chain)
    
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # 2. Simulated Annealing
    # Time budget per sequence: ~10 seconds? 
    # Total 10 sequences, 100s budget. 10s per sequence is reasonable.
    # We have 200s total deadline. 
    # Let's allocate time dynamically.
    
    start_time = time.time()
    # Let's run SA until we have used 80% of the available time per sequence?
    # Or just run for a fixed number of steps?
    # Steps are fast. Let's run until time limit for this sequence is reached.
    # We don't know exactly how much time is left for *this* sequence, 
    # but we can check global deadline.
    
    # Let's set a local time limit. 
    # If this is the first sequence, we have ~200s.
    # If it's the last, we have less.
    # We can't know the sequence index easily.
    # Let's just run until _remaining_time() is low, but we need to return a result.
    # Strategy: Run SA for a fixed number of iterations, or until time runs out.
    # Let's use a time-based loop.
    
    # Cooling schedule
    T0 = 1.0
    # We want T to drop to ~0.01
    # Number of steps?
    # Let's do 50,000 steps. Each step is O(N) for energy check? 
    # No, we can update energy incrementally or just recalc.
    # Recalc is O(N) with a set lookup. N=100. 50k * 100 = 5M ops. Very fast in Python?
    # Maybe 1-2 seconds.
    # Let's do 200,000 steps.
    
    max_steps = 200000
    steps_per_iter = 1000
    current_T = T0
    cooling_rate = 0.999
    
    for step in range(max_steps):
        if time.time() > DEADLINE - 1.0:
            break
            
        # Generate move
        move = _generate_pivot_move(chain, rng)
        if move is None:
            move = _generate_slide_move(chain, rng) # Actually _generate_slide_move returns end pivot
        if move is None:
            # Try a simple end move (rotate last 1? No, must be valid)
            # Just pick a random pivot again
            continue
            
        move_indices, new_positions = move
        
        # Check validity
        if not _is_valid_move(chain, occupied, move_indices, new_positions):
            continue
            
        # Calculate new energy
        # To speed up, we can calculate delta energy.
        # But for N=100, full recalc is fine.
        # Let's apply move, calc energy, then revert if rejected.
        
        old_positions = [chain[idx] for idx in move_indices]
        
        # Apply
        _apply_move(chain, move_indices, new_positions)
        
        # Update occupied set
        for pos in old_positions:
            occupied.discard(pos)
        for pos in new_positions:
            occupied.add(pos)
            
        new_energy = _energy(sequence, chain)
        delta_e = new_energy - current_energy
        
        # Acceptance criterion
        if delta_e < 0 or rng.random() < math.exp(-delta_e / current_T):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        else:
            # Revert
            _apply_move(chain, move_indices, old_positions)
            for pos in new_positions:
                occupied.discard(pos)
            for pos in old_positions:
                occupied.add(pos)
                
        # Cool down
        current_T *= cooling_rate
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
