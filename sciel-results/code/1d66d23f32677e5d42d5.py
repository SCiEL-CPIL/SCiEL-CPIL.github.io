# EVOLVE-BLOCK-START
import time
import random
import math
import heapq
from collections import defaultdict

def _neighbours(point):
    """Yield the 6 neighbours of a point on the cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute the energy of a chain (negative number of H-H contacts)."""
    n = len(chain)
    if n == 0:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _compute_energy_delta(sequence, chain, i, j):
    """
    Compute the change in energy if we move residue j to a new position,
    given the rest of the chain is fixed.
    This is used for fast evaluation of moves.
    """
    pass # Not used directly in the main loop, but good to have for clarity

def _local_optimize_sa(sequence, chain, time_limit=10.0, start_temp=1.0, cool_rate=0.995):
    """
    Simulated Annealing local optimization.
    Uses pivot moves and kink moves.
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    start_time = time.time()
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    temp = start_temp
    
    while time.time() - start_time < time_limit:
        # Choose a move type: pivot or kink
        if random.random() < 0.7:
            move_type = 'pivot'
        else:
            move_type = 'kink'
        
        if move_type == 'pivot':
            # Pivot move: rotate the tail around a pivot point
            if n < 4:
                continue
            pivot_idx = random.randint(1, n - 2)
            p_prev = chain[pivot_idx - 1]
            p_pivot = chain[pivot_idx]
            
            v = [p_pivot[i] - p_prev[i] for i in range(3)]
            bond_axis = -1
            for i in range(3):
                if v[i] != 0:
                    bond_axis = i
                    break
            
            rot_axes = [i for i in range(3) if i != bond_axis]
            if not rot_axes:
                continue
            r_axis = random.choice(rot_axes)
            angle = random.choice([1, -1])
            
            def rotate_point(point, center, axis, angle):
                p = list(point)
                c = list(center)
                p[0] -= c[0]
                p[1] -= c[1]
                p[2] -= c[2]
                
                if axis == 0:
                    if angle == 1:
                        p[1], p[2] = -p[2], p[1]
                    else:
                        p[1], p[2] = p[2], -p[1]
                elif axis == 1:
                    if angle == 1:
                        p[0], p[2] = p[2], -p[0]
                    else:
                        p[0], p[2] = -p[2], p[0]
                elif axis == 2:
                    if angle == 1:
                        p[0], p[1] = -p[1], p[0]
                    else:
                        p[0], p[1] = p[1], -p[0]
                
                p[0] += c[0]
                p[1] += c[1]
                p[2] += c[2]
                return tuple(p)
            
            head_set = set(chain[:pivot_idx])
            valid = True
            new_tail_points = []
            occupied_tail = set()
            
            for i in range(pivot_idx + 1, n):
                orig_point = chain[i]
                new_point = rotate_point(orig_point, p_pivot, r_axis, angle)
                
                if new_point in head_set:
                    valid = False
                    break
                if new_point in occupied_tail:
                    valid = False
                    break
                if new_tail_points:
                    prev_new = new_tail_points[-1]
                    dist = sum(abs(new_point[k] - prev_new[k]) for k in range(3))
                    if dist != 1:
                        valid = False
                        break
                occupied_tail.add(new_point)
                new_tail_points.append(new_point)
            
            if not valid:
                continue
            
            candidate_chain = chain[:pivot_idx + 1] + new_tail_points
            new_energy = _energy(sequence, candidate_chain)
            
        else: # kink move
            # Kink move: move a single residue to a neighbouring site
            if n < 3:
                continue
            idx = random.randint(1, n - 2)
            p_prev = chain[idx - 1]
            p_curr = chain[idx]
            p_next = chain[idx + 1]
            
            # Find a valid neighbour for p_curr that is not p_prev, not p_next, and not occupied by others
            occupied = set(chain)
            occupied.remove(p_curr)
            
            valid_sites = []
            for site in _neighbours(p_curr):
                if site == p_prev or site == p_next:
                    continue
                if site in occupied:
                    continue
                # Check distance to p_next (must be 1)
                dist = sum(abs(site[k] - p_next[k]) for k in range(3))
                if dist == 1:
                    valid_sites.append(site)
            
            if not valid_sites:
                continue
            
            new_site = random.choice(valid_sites)
            candidate_chain = list(chain)
            candidate_chain[idx] = new_site
            new_energy = _energy(sequence, candidate_chain)
        
        delta = new_energy - current_energy
        if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-6)):
            chain = candidate_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        temp *= cool_rate
    
    return best_chain, best_energy

def _beam_search(sequence, dimension, beam_width=50, time_limit=30.0):
    """
    Beam search for initial conformation.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    start = (0, 0, 0)
    # State: (last_pos, occupied_set, h_positions_set, path, energy)
    start_state = (start, {start}, {start} if sequence[0] == 'H' else set(), [start], 0)
    beam = [start_state]
    
    start_time = time.time()
    
    for i in range(1, n):
        next_beam = []
        for state in beam:
            last_pos, occupied, h_positions, path, energy = state
            
            for neighbor in _neighbours(last_pos):
                if neighbor in occupied:
                    continue
                
                new_occupied = set(occupied)
                new_occupied.add(neighbor)
                
                new_h_positions = set(h_positions)
                new_energy = energy
                
                if sequence[i] == 'H':
                    new_h_positions.add(neighbor)
                    for q in _neighbours(neighbor):
                        if q in h_positions:
                            new_energy -= 1
                
                new_path = path + [neighbor]
                next_beam.append((neighbor, new_occupied, new_h_positions, new_path, new_energy))
        
        if not next_beam:
            break
        
        # Sort by energy (most negative is best)
        next_beam.sort(key=lambda x: x[4])
        beam = next_beam[:beam_width]
        
        if time.time() - start_time > time_limit:
            break
    
    if beam:
        best_state = min(beam, key=lambda x: x[4])
        return best_state[3]
    return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

def _multi_restart_fold(sequence, dimension, total_time_limit=240.0):
    """
    Multi-restart strategy:
    1. Generate diverse initial conformations using beam search with different random seeds/widths.
    2. Optimize each with Simulated Annealing.
    3. Keep the best result.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    start_time = time.time()
    best_chain = None
    best_energy = 0
    
    # Strategy: Use beam search to get a good start, then SA to refine.
    # We can vary the beam width or use different random seeds for the beam search if we add randomness.
    # Currently beam search is deterministic. Let's add randomness to the beam search selection.
    
    def _random_beam_search(beam_width=20, time_limit=5.0):
        # Modified beam search that keeps top K but with some randomness
        # Or just use the deterministic one and rely on SA to explore.
        # Let's just use the deterministic beam search and then run SA for a while.
        return _beam_search(sequence, dimension, beam_width, time_limit)
    
    num_restarts = 10
    time_per_restart = (total_time_limit - 10) / num_restarts
    
    for i in range(num_restarts):
        if time.time() - start_time > total_time_limit - 5:
            break
        
        # Generate initial chain
        chain = _random_beam_search(beam_width=20, time_limit=2.0)
        
        if chain is None or len(chain) != n:
            # Fallback to a straight line if beam search fails
            chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
        # Optimize with SA
        sa_time = max(1.0, time_per_restart - 2.0)
        optimized_chain, opt_energy = _local_optimize_sa(sequence, chain, time_limit=sa_time)
        
        if optimized_chain is not None:
            if best_chain is None or opt_energy < best_energy:
                best_chain = optimized_chain
                best_energy = opt_energy
    
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]

def fold(sequence, dimension):
    """
    Main entry point.
    Uses a multi-restart strategy with beam search initialization and simulated annealing refinement.
    """
    start_time = time.time()
    deadline = start_time + 300 # Safety margin for the 330s limit
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    # Allocate time based on chain length
    # Shorter chains can be solved more exactly or with more restarts
    # Longer chains need more time per restart
    
    if n <= 50:
        time_limit = 220
    elif n <= 80:
        time_limit = 230
    else:
        time_limit = 240
    
    # Ensure we don't exceed the global deadline
    time_left = deadline - time.time()
    if time_limit > time_left - 10:
        time_limit = max(10, time_left - 10)
    
    result = _multi_restart_fold(sequence, dimension, total_time_limit=time_limit)
    
    # Final safety check: ensure the result is a valid list of lists of ints
    if not isinstance(result, list) or len(result) != n:
        # Fallback
        result = [[i, 0, 0] for i in range(n)]
    else:
        # Convert to lists of ints
        result = [[int(p[0]), int(p[1]), int(p[2])] for p in result]
        
    return result
# EVOLVE-BLOCK-END
