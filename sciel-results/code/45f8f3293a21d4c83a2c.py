"""HP lattice protein folding with contact-guided growth and simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import defaultdict


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _build_contact_index(sequence, chain):
    """Build mapping from lattice point to residue index, and H position set."""
    where = {}
    for i, p in enumerate(chain):
        where[tuple(p)] = i
    h_positions = set()
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            h_positions.add(tuple(p))
    return where, h_positions


def _compute_energy(sequence, chain):
    """Compute energy: minus number of H-H contacts on non-adjacent lattice neighbours."""
    where = {}
    for i, p in enumerate(chain):
        where[tuple(p)] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    if j > i:  # count each pair once
                        contacts += 1
    return -contacts


def _grow_constrained(sequence, dimension, rng, time_limit=None):
    """
    Constrained chain growth: prioritize sites that create H-H contacts,
    especially with H residues close in sequence (small |i-j|).
    """
    n = len(sequence)
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    # Pre-compute H positions in sequence
    h_indices = [i for i in range(n) if sequence[i] == 'H']
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            # Compute gain: number of new H-H contacts if we place residue i here
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        # Weight by inverse sequence distance (closer = more valuable)
                        seq_dist = i - j
                        gain += 1.0 / (1.0 + 0.1 * seq_dist)
            
            # Free space heuristic
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Add random noise for diversity
            noise = rng.random() * 0.5
            score = gain + 0.1 * free + noise
            options.append((score, site))
        
        if not options:
            return None
        
        # Select best option
        site = max(options, key=lambda x: x[0])[1]
        occupied[site] = i
        chain.append(site)
        
        # Time check
        if time_limit is not None and time.time() > time_limit:
            return None
    
    return chain


def _try_reversal(chain, i, j, occupied):
    """
    Try to reverse segment chain[i:j+1]. Returns new_chain or None if invalid.
    A reversal is valid if the reversed segment fits in the bounding box
    without colliding with fixed parts.
    """
    n = len(chain)
    if i >= j:
        return None
    
    segment = chain[i:j+1]
    # Check if reversed segment collides with fixed parts
    # The reversed segment occupies the same set of lattice points, just in reverse order
    # So we need to check that consecutive points in the reversed segment are adjacent
    # and that the endpoints connect properly
    
    # Actually, for a simple reversal, the set of occupied points is the same.
    # We just need to verify that the chain remains self-avoiding with unit steps.
    # Since the original segment had unit steps, the reversed segment also has unit steps.
    # The only concern is the connections at the boundaries:
    # chain[i-1] to segment[-1] (which becomes segment[0] reversed... wait)
    
    # Let me reconsider. Reversal: chain[i..j] becomes chain[j], chain[j-1], ..., chain[i]
    # The new chain is: chain[0..i-1] + reversed(chain[i..j]) + chain[j+1..]
    # We need:
    # 1. chain[i-1] adjacent to chain[j] (if i > 0)
    # 2. chain[i] adjacent to chain[j+1] (if j < n-1)
    # 3. No self-intersection (but the set of points is the same, so this is automatic
    #    as long as the internal steps are valid, which they are)
    
    # Check boundary conditions
    if i > 0:
        a, b = chain[i-1], chain[j]
        if sum(abs(x-y) for x, y in zip(a, b)) != 1:
            return None
    if j < n - 1:
        a, b = chain[i], chain[j+1]
        if sum(abs(x-y) for x, y in zip(a, b)) != 1:
            return None
    
    # Build new chain
    new_chain = chain[:i] + list(reversed(chain[i:j+1])) + chain[j+1:]
    return new_chain


def _try_move(chain, i, new_pos, occupied):
    """
    Try to move residue i to new_pos. Returns new_chain or None.
    """
    n = len(chain)
    old_pos = chain[i]
    
    # new_pos must be adjacent to old_pos? No, it must be adjacent to neighbors in chain
    # Actually, for a single-residue move, new_pos must be adjacent to chain[i-1] and chain[i+1]
    # if they exist, and not occupied by any other residue
    
    if new_pos in occupied and occupied[new_pos] != i:
        return None
    
    # Check adjacency with neighbors
    if i > 0:
        a, b = chain[i-1], new_pos
        if sum(abs(x-y) for x, y in zip(a, b)) != 1:
            return None
    if i < n - 1:
        a, b = chain[i+1], new_pos
        if sum(abs(x-y) for x, y in zip(a, b)) != 1:
            return None
    
    new_chain = chain[:]
    new_chain[i] = new_pos
    return new_chain


def _simulated_annealing(sequence, chain, rng, time_limit):
    """
    Simulated annealing with reversals and single-residue moves.
    """
    n = len(chain)
    current = list(chain)
    current_energy = _compute_energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    start_temp = 5.0
    end_temp = 0.01
    start_time = time.time()
    
    while time.time() < time_limit:
        elapsed = time.time() - start_time
        # Linear cooling
        progress = min(1.0, elapsed / 10.0)  # cool over ~10 seconds
        temp = start_temp * math.exp(-3.0 * progress)
        
        if temp < end_temp:
            break
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.6:
            # Reversal
            if n < 4:
                continue
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            # Limit segment length for efficiency
            if j - i > n // 2:
                continue
            
            new_chain = _try_reversal(current, i, j, None)
            if new_chain is None:
                continue
            
            new_energy = _compute_energy(sequence, new_chain)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / temp):
                current = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best = list(current)
                    best_energy = current_energy
        else:
            # Single residue move
            if n < 3:
                continue
            i = rng.randint(1, n - 2)  # not endpoints for simplicity
            old_pos = current[i]
            
            # Try moving to a neighbor of old_pos
            for new_pos in _neighbours(old_pos):
                # Check if new_pos is free (not occupied by another residue)
                occupied_set = set(tuple(p) for p in current)
                if new_pos in occupied_set:
                    continue
                
                new_chain = _try_move(current, i, new_pos, None)
                if new_chain is None:
                    continue
                
                new_energy = _compute_energy(sequence, new_chain)
                delta_e = new_energy - current_energy
                
                if delta_e < 0 or rng.random() < math.exp(-delta_e / temp):
                    current = new_chain
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best = list(current)
                        best_energy = current_energy
                    break  # accept first valid move
    
    return best, best_energy


def fold(sequence, dimension):
    """
    Best conformation found by constrained growth + simulated annealing.
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    # Deadline: leave 15s margin before 210s
    deadline = time.time() + 195.0
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0  # energy is negative, so 0 is "unfolded"
    
    # Phase 1: Multiple constrained growths
    num_growths = 50
    for k in range(num_growths):
        if time.time() > deadline - 60.0:
            break
        
        chain = _grow_constrained(sequence, dimension, rng, time_limit=deadline - 60.0)
        if chain is None:
            continue
        
        energy = _compute_energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    
    # Phase 2: Simulated annealing polish on the best chain
    if time.time() < deadline - 5.0:
        polished, polished_energy = _simulated_annealing(
            sequence, best_chain, rng, deadline - 5.0
        )
        if polished_energy < best_energy:
            best_chain = polished
            best_energy = polished_energy
    
    # Return as list of lists
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
