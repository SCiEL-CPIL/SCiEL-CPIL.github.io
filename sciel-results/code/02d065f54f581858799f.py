"""
HP lattice protein folding using Beam Search initialization followed by Simulated Annealing refinement.
This strategy combines the ability of Beam Search to find valid, low-energy initial conformations
with the global exploration capabilities of Simulated Annealing to escape local minima.
"""
# EVOLVE-BLOCK-START
import time
import random
import math

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Safety margin: 210s limit, leave 15s buffer -> 195s usable
    return 195.0 - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _compute_energy(sequence, chain):
    """Compute the energy of a complete conformation."""
    n = len(chain)
    if n == 0:
        return 0
    is_h = [c == 'H' for c in sequence]
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    energy = 0
    for i, p in enumerate(chain):
        if is_h[i]:
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None and j > i + 1 and is_h[j]:
                    energy -= 1
    return energy

def _is_valid_conformation(chain):
    """Check if a conformation is self-avoiding with unit steps."""
    occupied = set()
    for i, p in enumerate(chain):
        if p in occupied:
            return False
        occupied.add(p)
        if i > 0:
            prev = chain[i-1]
            if abs(p[0]-prev[0]) + abs(p[1]-prev[1]) != 1:
                return False
    return True

def _beam_search(sequence, dimension, beam_width=50, time_limit=5.0, rng=None):
    """
    Perform Beam Search for the given sequence.
    Returns the best full conformation found.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    is_h = [c == 'H' for c in sequence]
    
    # Initial state
    start_pos = (0, 0)
    initial_chain = [start_pos]
    initial_occupied = {start_pos}
    initial_h = {start_pos} if is_h[0] else set()
    initial_energy = 0
    
    beam = [(initial_energy, initial_chain, initial_occupied, initial_h)]
    
    start_time = time.time()
    
    for i in range(1, n):
        new_beam = []
        is_current_h = is_h[i]
        
        for state in beam:
            energy, chain, occupied, h_pos = state
            last_pos = chain[-1]
            pred_pos = chain[-2] if len(chain) > 1 else None
            
            for nb in _neighbours(last_pos):
                if nb in occupied:
                    continue
                
                new_energy = energy
                new_h_pos = h_pos
                new_occupied = occupied | {nb}
                
                if is_current_h:
                    new_h_pos = h_pos | {nb}
                    for q in _neighbours(nb):
                        if q in h_pos:
                            if q != pred_pos:
                                new_energy -= 1
                
                new_chain = chain + [nb]
                new_beam.append((new_energy, new_chain, new_occupied, new_h_pos))
                
        if not new_beam:
            break
            
        # Prune with randomization for ties
        new_beam.sort(key=lambda x: x[0])
        
        if len(new_beam) > beam_width:
            cutoff_energy = new_beam[beam_width - 1][0]
            keep = []
            maybe = []
            for state in new_beam:
                if state[0] < cutoff_energy:
                    keep.append(state)
                elif state[0] == cutoff_energy:
                    maybe.append(state)
                else:
                    break
            
            slots_needed = beam_width - len(keep)
            if slots_needed > 0:
                if rng:
                    rng.shuffle(maybe)
                else:
                    random.shuffle(maybe)
                keep.extend(maybe[:slots_needed])
            
            beam = keep
        else:
            beam = new_beam
            
        if time.time() - start_time > time_limit:
            break
            
    if not beam:
        return None
        
    best_state = min(beam, key=lambda x: x[0])
    chain = best_state[1]
    occupied = best_state[2]
    h_pos = best_state[3]
    
    # If we broke early, extend greedily
    if len(chain) < n:
        for j in range(len(chain), n):
            is_j_h = is_h[j]
            last_pos = chain[-1]
            pred_pos = chain[-2] if len(chain) > 1 else None
            best_nb = None
            best_e = float('inf')
            
            candidates = []
            for nb in _neighbours(last_pos):
                if nb in occupied:
                    continue
                e = 0
                if is_j_h:
                    for q in _neighbours(nb):
                        if q in h_pos and q != pred_pos:
                            e -= 1
                candidates.append((e, nb))
                
            if candidates:
                candidates.sort(key=lambda x: x[0])
                best_e_val = candidates[0][0]
                best_opts = [nb for e, nb in candidates if e == best_e_val]
                if rng:
                    best_nb = rng.choice(best_opts)
                else:
                    best_nb = random.choice(best_opts)
                    
                chain.append(best_nb)
                occupied.add(best_nb)
                if is_j_h:
                    h_pos.add(best_nb)
            else:
                return None
                
    return chain

def _simulated_annealing(sequence, initial_chain, rng, time_limit=5.0):
    """
    Refine the chain using Simulated Annealing with Kink and Reversal moves.
    """
    n = len(initial_chain)
    if n < 3:
        return initial_chain, _compute_energy(sequence, initial_chain)
    
    chain = list(initial_chain)
    occupied = set(chain)
    is_h = [c == 'H' for c in sequence]
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    
    # Cooling schedule parameters
    # Start temp high enough to accept most moves, end temp low for fine tuning
    T_start = 1.0
    T_end = 0.01
    # Estimate number of steps we can do in time_limit
    # Roughly 10k steps/sec for Python SA on N=100? Let's assume 5k/sec to be safe.
    # We will just loop until time runs out, adjusting T dynamically.
    
    steps_done = 0
    total_steps_target = 20000 # Arbitrary, we will stop based on time
    
    while time.time() - start_time < time_limit:
        # Dynamic temperature
        progress = min(1.0, steps_done / total_steps_target)
        T = T_start * (T_end / T_start) ** progress
        if T < 0.001:
            T = 0.001
            
        move_type = rng.random()
        
        if move_type < 0.7:
            # Kink Move
            if n < 3:
                continue
            i = rng.randint(1, n - 2)
            prev = chain[i-1]
            curr = chain[i]
            next_ = chain[i+1]
            
            # Find valid new positions for curr
            candidates = set()
            for p in _neighbours(prev):
                if p == curr:
                    continue
                dist = abs(p[0] - next_[0]) + abs(p[1] - next_[1])
                if dist == 1:
                    if p not in occupied or p == curr:
                        candidates.add(p)
            
            if not candidates:
                continue
                
            new_pos = rng.choice(list(candidates))
            
            if new_pos == curr:
                continue
            
            # Apply move
            occupied.remove(curr)
            occupied.add(new_pos)
            chain[i] = new_pos
            
            new_energy = _compute_energy(sequence, chain)
            delta = new_energy - current_energy
            
            if delta < 0 or (T > 0.001 and rng.random() < math.exp(-delta / T)):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            else:
                # Revert
                chain[i] = curr
                occupied.remove(new_pos)
                occupied.add(curr)
        else:
            # Reversal Move
            # Pick a sub-chain of length L >= 2 and reverse it
            # Indices i, j such that 1 <= i < j <= n-1
            # To ensure the move is valid, the reversed segment must not collide with the rest.
            # The endpoints of the segment (i-1 and j+1) remain connected to i and j respectively?
            # No, reversal swaps the direction.
            # Original: ... A - S1 - S2 - ... - Sk - B ...
            # Reversed: ... A - Sk - ... - S2 - S1 - B ...
            # This is valid if the new positions of S1...Sk do not collide with {A, B} or the rest.
            
            if n < 4:
                continue
            
            # Choose segment length
            max_len = n - 2
            L = rng.randint(2, min(10, max_len)) # Limit length for efficiency
            if L > max_len:
                L = max_len
                
            i = rng.randint(1, n - L - 1)
            j = i + L - 1
            
            # Segment is chain[i:j+1]
            # We need to check if reversing it causes collisions.
            # The positions involved are chain[i], ..., chain[j].
            # The neighbors are chain[i-1] and chain[j+1].
            
            # Remove segment from occupied temporarily for check?
            # No, we can just check if any new position is in occupied \ {segment positions}.
            
            seg_positions = set(chain[i:j+1])
            rest_occupied = occupied - seg_positions
            
            # Generate reversed segment
            new_seg = chain[i:j+1][::-1]
            
            # Check validity
            valid = True
            for idx, pos in enumerate(new_seg):
                global_idx = i + idx
                # pos is the new position for chain[global_idx]
                # It must not be in rest_occupied
                if pos in rest_occupied:
                    valid = False
                    break
                # Also, consecutive positions in the new segment must be neighbors
                # Since it's a reversal of a valid path, they are neighbors.
            
            if not valid:
                continue
            
            # Check connectivity at boundaries
            # chain[i-1] must be neighbor of new_seg[0] (which is old chain[j])
            # chain[j+1] must be neighbor of new_seg[-1] (which is old chain[i])
            if abs(chain[i-1][0] - new_seg[0][0]) + abs(chain[i-1][1] - new_seg[0][1]) != 1:
                continue
            if abs(chain[j+1][0] - new_seg[-1][0]) + abs(chain[j+1][1] - new_seg[-1][1]) != 1:
                continue
            
            # Apply move
            # Update occupied set
            # The set of positions is the same, just indices changed.
            # So occupied set doesn't change!
            # Wait, if occupied set doesn't change, we don't need to update it.
            # But we need to update the chain list.
            
            old_seg = chain[i:j+1]
            chain[i:j+1] = new_seg
            
            new_energy = _compute_energy(sequence, chain)
            delta = new_energy - current_energy
            
            if delta < 0 or (T > 0.001 and rng.random() < math.exp(-delta / T)):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            else:
                # Revert
                chain[i:j+1] = old_seg
                
        steps_done += 1
        
        # Periodic check for time
        if steps_done % 100 == 0:
            if time.time() - start_time >= time_limit:
                break

    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold the sequence using Beam Search + Simulated Annealing.
    """
    _init_time()
    n = len(sequence)
    
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    # Adjust parameters based on sequence length
    if n <= 20:
        base_beam_width = 100
        base_time_limit = 2.0
        sa_time = 5.0
        num_restarts = 5
    elif n <= 50:
        base_beam_width = 50
        base_time_limit = 3.0
        sa_time = 8.0
        num_restarts = 3
    else:
        base_beam_width = 30
        base_time_limit = 4.0
        sa_time = 10.0
        num_restarts = 2
        
    best_chain = None
    best_energy = 0
    
    for run in range(num_restarts):
        if _time_remaining() < 10:
            break
            
        seed = 12345 + run * 7
        rng = random.Random(seed)
        
        # 1. Beam Search Initialization
        chain = _beam_search(sequence, dimension, base_beam_width, base_time_limit, rng)
        
        if chain is None or len(chain) != n:
            # Fallback to straight line if beam search fails completely
            chain = [(i, 0) for i in range(n)]
            
        if not _is_valid_conformation(chain):
            # Should not happen, but just in case
            continue
            
        # 2. Simulated Annealing Refinement
        polished_chain, polished_energy = _simulated_annealing(
            sequence, chain, rng, sa_time
        )
        
        if best_chain is None or polished_energy < best_energy:
            best_chain = polished_chain
            best_energy = polished_energy
            
    if best_chain is None:
        # Final fallback
        best_chain = [(i, 0) for i in range(n)]
        
    # Convert to lists of integers as required
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
