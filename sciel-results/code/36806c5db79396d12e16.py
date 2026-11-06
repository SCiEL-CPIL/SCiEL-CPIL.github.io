# EVOLVE-BLOCK-START
import random
import time
import math
import sys
import heapq

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """
    Compute the energy (negative number of H-H contacts) of a chain.
    """
    if not chain:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None and abs(i - j) > 1:
                    if sequence[j] == 'H':
                        contacts += 1
    return -contacts

def _beam_search_fold(sequence, dimension, time_limit, rng, beam_width=100):
    """
    Beam search with look-ahead heuristic.
    This is a different structural approach from SA: it builds the chain step-by-step
    keeping a set of partial chains (the beam) and pruning based on a heuristic score
    that balances current energy and future potential.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    start_time = time.time()
    
    # Initial state: first residue at origin
    # State: (chain, occupied_set, current_energy, heuristic_score)
    # We store chains as tuples of points for hashing/efficiency in sets if needed, 
    # but lists are fine for the beam itself if we don't hash them directly.
    
    initial_chain = [(0, 0, 0)]
    initial_occupied = {(0, 0, 0)}
    
    beam = [(initial_chain, initial_occupied, 0, 0)] # (chain, occupied, energy, score)
    
    for step in range(1, n):
        if time.time() - start_time > time_limit - 1.0:
            break
            
        next_beam = []
        
        for chain, occupied, energy, score in beam:
            last_pos = chain[-1]
            neighbors = list(_neighbours(last_pos))
            random.shuffle(neighbors)
            
            for nb in neighbors:
                if nb in occupied:
                    continue
                
                new_chain = chain + [nb]
                new_occupied = occupied | {nb}
                
                # Calculate energy change
                # Only need to check contacts involving the new residue (step)
                # and potentially existing H residues that are now in contact
                # Since we add one residue, we only need to check its neighbors.
                delta_e = 0
                if sequence[step] == 'H':
                    for q in _neighbours(nb):
                        # q is a neighbor of the new position
                        # Check if q is occupied by an H residue
                        # We need to know the index of q. 
                        # Reconstructing index map every step is expensive.
                        # Optimization: Keep a pos_to_idx map in the state? 
                        # That increases memory. 
                        # Alternative: Just check if the position is in the chain.
                        # Since n is up to 136, linear search is O(N).
                        # Better: Maintain a set of H positions.
                        # Let's maintain a set of H positions in the state for O(1) check.
                        pass
                
                # The above energy calculation is tricky without a global index map.
                # Let's use a simpler heuristic for the beam:
                # Score = current_energy + alpha * (number of H residues placed so far that are isolated)
                # Actually, let's just compute the full energy for candidates if the beam is small,
                # or use a fast approximation.
                
                # For a true structural change, let's use a specific heuristic:
                # Heuristic = current_energy - beta * (potential future contacts)
                # Potential future contacts can be estimated by counting H neighbors of the current tip
                # that are unoccupied, assuming they might be filled by H later.
                
                # Let's stick to a robust heuristic:
                # 1. Current energy (exact)
                # 2. Number of H residues in the chain that have 0 H neighbors (penalty)
                
                # To make it fast, we'll compute the exact energy for the new chain.
                # With beam width 100 and N=136, 100*136*6 checks = ~80k ops per step.
                # 136 steps * 80k = 10M ops. This is fast in Python if optimized.
                
                new_energy = _energy(sequence, new_chain)
                
                # Heuristic: Prefer chains that are "compact" or have high current energy.
                # Also, avoid dead ends.
                # Check if the new position has any available neighbors (simple lookahead)
                has_moves = False
                for q in _neighbours(nb):
                    if q not in new_occupied:
                        has_moves = True
                        break
                
                if not has_moves and step < n - 1:
                    continue # Dead end
                
                # Score: Higher is better.
                # We want to maximize energy (minimize -energy).
                # So score = -new_energy (which is contacts)
                # Add a small bonus for compactness?
                # Let's just use energy.
                score = -new_energy
                
                next_beam.append((new_chain, new_occupied, new_energy, score))
        
        if not next_beam:
            break
            
        # Sort by score (descending) and keep top beam_width
        next_beam.sort(key=lambda x: x[3], reverse=True)
        beam = next_beam[:beam_width]
        
        # Early termination check
        if len(beam) == 0:
            break
            
    if not beam:
        # Fallback to linear
        return [(i, 0, 0) for i in range(n)]
        
    # Return the best chain from the final beam
    # The beam is already sorted by score, so the first one is the best
    return beam[0][0]

def _local_optimize(chain, sequence, dimension, time_limit, rng):
    """
    Perform local optimization using pivot moves to refine the beam search result.
    This combines the global structure of beam search with local refinement.
    """
    if not chain:
        return chain
        
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    
    # Simple random restarts of local search
    while time.time() - start_time < time_limit - 0.5:
        pivot_idx = rng.randint(1, n - 2)
        
        # Perform a pivot move
        p_prev = current_chain[pivot_idx - 1]
        p_pivot = current_chain[pivot_idx]
        
        v = [p_pivot[i] - p_prev[i] for i in range(dimension)]
        bond_axis = -1
        for i in range(dimension):
            if v[i] != 0:
                bond_axis = i
                break
        if bond_axis == -1:
            continue
            
        rot_axes = [i for i in range(dimension) if i != bond_axis]
        r_axis = rng.choice(rot_axes)
        angle = rng.choice([1, -1])
        
        head_set = set(current_chain[:pivot_idx])
        occupied_tail = set()
        new_tail = []
        valid = True
        
        for i in range(pivot_idx + 1, n):
            orig = current_chain[i]
            dx = orig[0] - p_pivot[0]
            dy = orig[1] - p_pivot[1]
            dz = orig[2] - p_pivot[2]
            
            if r_axis == 0:
                if angle == 1:
                    nx, ny, nz = dx, -dz, dy
                else:
                    nx, ny, nz = dx, dz, -dy
            elif r_axis == 1:
                if angle == 1:
                    nx, ny, nz = dz, dy, -dx
                else:
                    nx, ny, nz = -dz, dy, dx
            else:
                if angle == 1:
                    nx, ny, nz = -dy, dx, dz
                else:
                    nx, ny, nz = dy, -dx, dz
                    
            new_point = (p_pivot[0] + nx, p_pivot[1] + ny, p_pivot[2] + nz)
            
            if new_point in head_set or new_point in occupied_tail:
                valid = False
                break
                
            occupied_tail.add(new_point)
            new_tail.append(new_point)
            
        if not valid:
            continue
            
        new_chain = current_chain[:pivot_idx + 1] + new_tail
        new_energy = _energy(sequence, new_chain)
        
        if new_energy < current_energy:
            current_chain = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(current_chain)
                
    return best_chain

def fold(sequence, dimension):
    """
    Hybrid Beam Search + Local Optimization.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    total_time_limit = 8.5
    start_time = time.time()
    
    best_chain_overall = None
    best_energy_overall = float('inf')
    
    # Strategy:
    # 1. Run Beam Search to get a good initial structure.
    # 2. Run Local Optimization to refine it.
    # 3. Repeat with different seeds/parameters if time permits.
    
    # Beam search is deterministic given the seed for shuffling.
    # We can run it multiple times with different shuffles.
    
    num_restarts = 3
    time_per_run = total_time_limit / num_restarts
    
    for r in range(num_restarts):
        remaining = total_time_limit - (time.time() - start_time)
        if remaining < 0.5:
            break
            
        rng = random.Random(999 + len(sequence) * 100 + r)
        
        # Allocate 60% of time to Beam Search, 40% to Local Optimization
        bs_time = time_per_run * 0.6 if r < num_restarts - 1 else remaining * 0.6
        lo_time = time_per_run * 0.4 if r < num_restarts - 1 else remaining * 0.4
        
        # Run Beam Search
        chain = _beam_search_fold(sequence, dimension, bs_time, rng)
        
        if chain:
            # Run Local Optimization
            chain = _local_optimize(chain, sequence, dimension, lo_time, rng)
            
            energy = _energy(sequence, chain)
            if energy < best_energy_overall:
                best_energy_overall = energy
                best_chain_overall = chain
                
    if best_chain_overall is None:
        # Fallback
        best_chain_overall = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain_overall]

# EVOLVE-BLOCK-END
