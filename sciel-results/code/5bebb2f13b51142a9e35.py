import time
import random
import math

_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _calc_delta_energy(sequence, occupied, i, q):
    """Calculate new H-H contacts when placing residue i at position q."""
    if sequence[i] != 'H':
        return 0
    x, y, z = q
    delta = 0
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        r = (x + dx, y + dy, z + dz)
        if r in occupied:
            j = occupied[r]
            if j != i - 1 and sequence[j] == 'H':
                delta += 1
    return delta

def _compute_energy(sequence, chain):
    """Compute the total energy of a complete chain."""
    if not chain:
        return 0
    occupied = {pos: idx for idx, pos in enumerate(chain)}
    contacts = 0
    for idx, pos in enumerate(chain):
        if sequence[idx] == 'H':
            x, y, z = pos
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (x + dx, y + dy, z + dz)
                if r in occupied:
                    j = occupied[r]
                    if j != idx - 1 and j != idx + 1 and sequence[j] == 'H':
                        contacts += 1
    return -contacts // 2

def _greedy_complete(sequence, chain, occupied, n, deadline, noise=0.0):
    """Greedily complete a partial chain with some stochastic noise."""
    for j in range(len(chain), n):
        if time.time() > deadline:
            return None
        last_pos = chain[-1]
        valid_mvs = []
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            q = (last_pos[0] + dx, last_pos[1] + dy, last_pos[2] + dz)
            if q not in occupied:
                valid_mvs.append(q)
        if not valid_mvs:
            return None
        
        best_q = None
        best_score = -1
        for q in valid_mvs:
            score = _calc_delta_energy(sequence, occupied, j, q)
            # Add noise to encourage diversity
            score += random.uniform(0, noise)
            # Slight preference for free space to avoid trapping
            free = 0
            x, y, z = q
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (x + dx, y + dy, z + dz)
                if r not in occupied:
                    free += 1
            score += free * 0.05
            if score > best_score:
                best_score = score
                best_q = q
                
        if best_q is None:
            return None
        chain.append(best_q)
        occupied[best_q] = j
    return chain

def _beam_search_fold(sequence, n, deadline, beam_width_max=40, noise=0.0):
    """
    Perform a beam search folding.
    Returns the best complete chain found or None.
    """
    init_chain = [(0, 0, 0)]
    init_occupied = {(0, 0, 0): 0}
    init_energy = 0
    
    # Start with a small beam, grow it
    current_width = min(beam_width_max, 10)
    beam = [(init_energy, init_chain, init_occupied)]
    
    best_complete_chain = None
    best_complete_energy = 0
    
    for i in range(1, n):
        new_beam = []
        
        for energy, chain, occupied in beam:
            last_pos = chain[-1]
            
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (last_pos[0] + dx, last_pos[1] + dy, last_pos[2] + dz)
                if q not in occupied:
                    delta = _calc_delta_energy(sequence, occupied, i, q)
                    new_energy = energy - delta
                    
                    # Add noise to energy to break ties and explore
                    noisy_energy = new_energy - random.uniform(0, noise)
                    
                    new_chain = chain + [q]
                    new_occupied = occupied.copy()
                    new_occupied[q] = i
                    
                    new_beam.append((noisy_energy, new_energy, new_chain, new_occupied))
        
        if not new_beam:
            break
            
        # Sort by noisy energy (ascending)
        new_beam.sort(key=lambda x: x[0])
        
        # Adaptive beam width: start wide, narrow down
        # We want to keep more options early on
        target_width = min(current_width, len(new_beam))
        beam = []
        for noisy_e, real_e, chain, occ in new_beam[:target_width]:
            beam.append((real_e, chain, occ))
            
        # Increase beam width slightly for next step if we have time
        if time.time() < deadline - 10.0:
            current_width = min(beam_width_max, current_width + 2)
        else:
            current_width = max(5, current_width - 1)
            
        if time.time() > deadline:
            break
            
        if i == n - 1:
            # We have complete chains
            best_e, best_chain, _ = beam[0]
            # Verify energy
            real_e = _compute_energy(sequence, best_chain)
            if best_complete_chain is None or real_e < best_complete_energy:
                best_complete_chain = best_chain
                best_complete_energy = real_e
            # Check if any other in beam is better (unlikely with sort, but just in case)
            return best_complete_chain
    
    # If we broke out early, try to complete the best partial
    if beam:
        best_e, best_chain, best_occupied = beam[0]
        completed = _greedy_complete(sequence, best_chain.copy(), best_occupied.copy(), n, deadline, noise=noise)
        if completed is not None and len(completed) == n:
            return completed
            
    return None

def _pivot_move(chain, pivot_idx, direction):
    """
    Attempt a pivot move at pivot_idx.
    direction: 0 for +90, 1 for -90, 2 for 180, 3 for +90 around different axis, etc.
    For simplicity, we'll implement a few standard pivot moves.
    """
    n = len(chain)
    if pivot_idx <= 0 or pivot_idx >= n - 1:
        return None
    
    p0 = chain[pivot_idx - 1]
    p1 = chain[pivot_idx]
    
    # Vector from p0 to p1
    v = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
    
    # Generate 5 possible new directions (excluding the reverse of v)
    # This is a simplified pivot. A full pivot rotates the entire sub-chain.
    # For efficiency, we will just try moving the head of the sub-chain.
    # Actually, a true pivot rotates the tail.
    # Let's stick to a simpler local move: Shift the tail by 1 step in a valid direction.
    # This is less powerful than pivot but easier to implement correctly.
    # Let's implement a "Head Move" (move the last residue) and "Tail Move" (move the first residue).
    # And a "Pivot" at a random point.
    
    # For now, let's implement a simple "Shift" move on a random segment.
    # Or better, just move the last residue (Head move) which is easy to check.
    pass

def _sa_refine(sequence, chain, deadline, temp_start=1.0, temp_end=0.01):
    """
    Simulated Annealing refinement.
    Moves: 
    1. Move the last residue (Head move) to a valid neighbor.
    2. Move the first residue (Tail move) to a valid neighbor.
    3. Pivot: Rotate a sub-chain around a pivot point. (Complex to implement correctly for self-avoidance)
    
    For simplicity and robustness, we will use Head and Tail moves.
    """
    n = len(chain)
    if n < 3:
        return chain
    
    current_chain = chain
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = current_chain
    best_energy = current_energy
    
    # Precompute occupied set
    current_occupied = {pos: idx for idx, pos in enumerate(current_chain)}
    
    steps = 0
    start_time = time.time()
    elapsed = 0.0
    total_steps_target = 20000 # Adjust based on time
    
    while time.time() - start_time < deadline:
        # Cooling schedule
        progress = (time.time() - start_time) / deadline
        temp = temp_start * math.exp(-5 * progress)
        
        # Choose a move
        move_type = random.randint(0, 1)
        new_chain = None
        new_energy = None
        
        if move_type == 0:
            # Head move
            last_idx = n - 1
            last_pos = current_chain[-1]
            prev_pos = current_chain[-2]
            
            # Try moving last residue to a neighbor of prev_pos
            candidates = []
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (prev_pos[0] + dx, prev_pos[1] + dy, prev_pos[2] + dz)
                # Must be different from current last pos
                if q == last_pos:
                    continue
                # Must not be occupied by any other residue
                # Remove last pos from occupied for check
                if q in current_occupied:
                    if current_occupied[q] == last_idx:
                        continue # It's the one we are moving
                    else:
                        continue # Collision
                # Check if q is valid (not occupied by 0..n-2)
                # We removed last_pos from check, so if q is in occupied, it's a collision
                # Wait, current_occupied still has last_pos.
                # Let's do it properly:
                pass
            
            # Efficient Head Move Check
            # Remove last pos
            del current_occupied[last_pos]
            
            valid_moves = []
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (prev_pos[0] + dx, prev_pos[1] + dy, prev_pos[2] + dz)
                if q not in current_occupied:
                    valid_moves.append(q)
            
            if valid_moves:
                q = random.choice(valid_moves)
                new_chain = current_chain[:-1] + [q]
                
                # Calculate delta energy
                # Old energy contribution of last_pos
                old_contacts = 0
                if sequence[last_idx] == 'H':
                    x, y, z = last_pos
                    for dx, dy, dz in _NEIGHBOR_OFFSETS:
                        r = (x + dx, y + dy, z + dz)
                        if r in current_occupied: # current_occupied has last_pos removed
                            j = current_occupied[r]
                            if j != last_idx - 1 and sequence[j] == 'H':
                                old_contacts += 1
                # Note: current_occupied does NOT contain last_pos anymore.
                # We need to check neighbors of last_pos in the OLD state.
                # This is getting complicated. Let's just recompute energy for simplicity in SA?
                # No, that's too slow.
                
                # Let's use a helper to compute delta energy for head move.
                # Delta E = E(new) - E(old)
                
                # Restore last_pos to occupied for the "old" state context if needed, 
                # but actually we just need to know the contacts of the OLD last_pos 
                # and the NEW last_pos.
                
                # Let's just recompute energy for the whole chain? 
                # For n=136, O(N) is fast. 20000 steps * 136 * 6 neighbors = ~16M ops. Python might be slow.
                # Let's stick to incremental.
                
                # Old contacts for last_pos (index n-1):
                # Neighbors in current_occupied (which has n-1 removed)
                old_c = 0
                if sequence[last_idx] == 'H':
                    x, y, z = last_pos
                    for dx, dy, dz in _NEIGHBOR_OFFSETS:
                        r = (x + dx, y + dy, z + dz)
                        if r in current_occupied:
                            j = current_occupied[r]
                            if j != last_idx - 1 and sequence[j] == 'H':
                                old_c += 1
                            
                # New contacts for q (index n-1):
                new_c = 0
                if sequence[last_idx] == 'H':
                    x, y, z = q
                    for dx, dy, dz in _NEIGHBOR_OFFSETS:
                        r = (x + dx, y + dy, z + dz)
                        if r in current_occupied:
                            j = current_occupied[r]
                            if j != last_idx - 1 and sequence[j] == 'H':
                                new_c += 1
                            
                delta_e = -(new_c - old_c)
                new_energy = current_energy + delta_e
                
                # Update occupied for next iteration if accepted
                # For now, store the move
                current_occupied[q] = last_idx
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = new_chain
                    
                # Metropolis criterion
                if new_energy < current_energy or random.random() < math.exp(-delta_e / temp):
                    current_chain = new_chain
                    current_energy = new_energy
                    # current_occupied is already updated with q
                else:
                    # Revert occupied
                    del current_occupied[q]
                    
        else:
            # Tail move
            first_idx = 0
            first_pos = current_chain[0]
            next_pos = current_chain[1]
            
            del current_occupied[first_pos]
            
            valid_moves = []
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (next_pos[0] + dx, next_pos[1] + dy, next_pos[2] + dz)
                if q not in current_occupied:
                    valid_moves.append(q)
            
            if valid_moves:
                q = random.choice(valid_moves)
                new_chain = [q] + current_chain[1:]
                
                old_c = 0
                if sequence[first_idx] == 'H':
                    x, y, z = first_pos
                    for dx, dy, dz in _NEIGHBOR_OFFSETS:
                        r = (x + dx, y + dy, z + dz)
                        if r in current_occupied:
                            j = current_occupied[r]
                            if j != first_idx + 1 and sequence[j] == 'H':
                                old_c += 1
                                
                new_c = 0
                if sequence[first_idx] == 'H':
                    x, y, z = q
                    for dx, dy, dz in _NEIGHBOR_OFFSETS:
                        r = (x + dx, y + dy, z + dz)
                        if r in current_occupied:
                            j = current_occupied[r]
                            if j != first_idx + 1 and sequence[j] == 'H':
                                new_c += 1
                                
                delta_e = -(new_c - old_c)
                new_energy = current_energy + delta_e
                
                current_occupied[q] = first_idx
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = new_chain
                    
                if new_energy < current_energy or random.random() < math.exp(-delta_e / temp):
                    current_chain = new_chain
                    current_energy = new_energy
                else:
                    del current_occupied[q]
                    
        steps += 1
        
    return best_chain

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    start_time = time.time()
    # 240s budget, leave 15s margin
    deadline = start_time + 225.0
    
    # Phase 1: Multi-Start Beam Search
    # Allocate 70% of time to beam search
    beam_deadline = start_time + (deadline - start_time) * 0.7
    
    best_chain = None
    best_energy = 0
    
    # Number of restarts depends on chain length
    # Shorter chains can afford more restarts
    if n <= 50:
        restarts = 5
        beam_width = 30
    elif n <= 100:
        restarts = 3
        beam_width = 40
    else:
        restarts = 2
        beam_width = 50
        
    for r in range(restarts):
        if time.time() > beam_deadline:
            break
            
        # Randomize noise for diversity
        noise = 0.5 if r > 0 else 0.0
        
        chain = _beam_search_fold(sequence, n, beam_deadline, beam_width, noise)
        
        if chain is not None and len(chain) == n:
            e = _compute_energy(sequence, chain)
            if best_chain is None or e < best_energy:
                best_chain = chain
                best_energy = e
                
    # If no chain found, fallback to greedy
    if best_chain is None:
        best_chain = _greedy_complete(sequence, [(0,0,0)], {(0,0,0):0}, n, deadline, noise=0.0)
        if best_chain:
            best_energy = _compute_energy(sequence, best_chain)
        else:
            # Straight line fallback
            return [[i, 0, 0] for i in range(n)]

    # Phase 2: Simulated Annealing Refinement
    # Allocate 30% of time to SA
    sa_deadline = deadline - 5.0 # Leave 5s safety margin
    
    if time.time() < sa_deadline:
        # Run SA on the best chain
        refined_chain = _sa_refine(sequence, best_chain, sa_deadline - time.time())
        
        if refined_chain:
            refined_energy = _compute_energy(sequence, refined_chain)
            if refined_energy < best_energy:
                best_chain = refined_chain
                best_energy = refined_energy

    return [list(p) for p in best_chain]

if __name__ == "__main__":
    # Simple test
    seq = "HPPPPH"
    res = fold(seq, 3)
    print(res)
    print(_compute_energy(seq, res))
