"""HP lattice protein folding using Simulated Annealing with Pivot and Crankshaft moves."""
# EVOLVE-BLOCK-START
import random
import math
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _initial_chain(sequence, dimension, rng):
    """Generate an initial self-avoiding walk using randomized greedy growth."""
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0)
    chain = [start]
    occupied = {start}
    
    # Directions: 0: (0,1), 1: (1,0), 2: (0,-1), 3: (-1,0)
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for d in dirs:
            nx, ny = last[0] + d[0], last[1] + d[1]
            site = (nx, ny)
            if site not in occupied:
                # Heuristic: prefer sites that might lead to H-H contacts
                # Simple: just random
                options.append(site)
        
        if not options:
            # Dead end, try to backtrack or just fail (SA will restart)
            # For simplicity in init, return None to signal failure
            return None
        
        site = rng.choice(options)
        chain.append(site)
        occupied.add(site)
        
    return chain


def _calc_energy(sequence, chain):
    """Calculate the energy of a conformation."""
    n = len(chain)
    if n == 0:
        return 0
    
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _get_delta_energy_pivot(sequence, chain, occupied, k, L, rotation):
    """
    Calculate delta energy for a pivot move.
    Pivot at k, segment [k+1, k+L] rotates around k.
    rotation: 0, 1, 2, 3 (90 deg steps).
    Returns (delta_energy, new_chain, new_occupied, valid)
    """
    n = len(chain)
    if k + L >= n:
        return 0, None, None, False
        
    # Identify moved indices
    moved_indices = list(range(k + 1, k + L + 1))
    fixed_indices = set(range(n)) - set(moved_indices)
    
    # Precompute H sets
    H_moved = [i for i in moved_indices if sequence[i] == 'H']
    H_fixed = [i for i in fixed_indices if sequence[i] == 'H']
    
    # Calculate old contacts between H_moved and H_fixed
    old_contacts = 0
    pos_map = {p: i for i, p in enumerate(chain)}
    for i in H_moved:
        p = chain[i]
        for q in _neighbours(p):
            j = pos_map.get(q)
            if j is not None and j in H_fixed:
                old_contacts += 1
                
    # Apply rotation to find new positions
    # Pivot is at chain[k] = (px, py)
    px, py = chain[k]
    new_chain = list(chain)
    new_occupied = set(occupied)
    
    # Remove old positions of moved segment
    for i in moved_indices:
        new_occupied.remove(chain[i])
        
    # Calculate new positions
    # Rotation around (px, py)
    # dx, dy = p - pivot
    # new_dx, new_dy = rotate(dx, dy)
    # new_p = pivot + (new_dx, new_dy)
    
    valid = True
    for i in moved_indices:
        ox, oy = chain[i]
        dx = ox - px
        dy = oy - py
        
        if rotation == 1: # 90 deg CCW
            ndx, ndy = -dy, dx
        elif rotation == 2: # 180
            ndx, ndy = -dx, -dy
        elif rotation == 3: # 90 deg CW
            ndx, ndy = dy, -dx
        else:
            ndx, ndy = dx, dy
            
        nx, ny = px + ndx, py + ndy
        site = (nx, ny)
        
        # Check collision with fixed part
        # Note: site might collide with another moved site? 
        # Since it's a rigid rotation, if the original was SAW, the rotated is SAW internally.
        # But it might collide with fixed part.
        if site in new_occupied:
            valid = False
            break
            
        new_chain[i] = site
        new_occupied.add(site)
        
    if not valid:
        return 0, None, None, False
        
    # Calculate new contacts between H_moved and H_fixed
    new_contacts = 0
    # Rebuild pos_map for new_chain? Or just use new_chain directly.
    # We only need to check neighbors of H_moved in the new configuration.
    # The fixed part hasn't moved, so their positions are chain[j].
    # We can just check if new_chain[i] is neighbor to chain[j].
    
    for i in H_moved:
        p = new_chain[i]
        for q in _neighbours(p):
            # Check if q is occupied by a fixed H
            # We need to know which index j is at q.
            # Since fixed part is unchanged, we can use a map for fixed part.
            # But building a full map is O(N). 
            # Alternative: just check if q is in new_occupied and corresponds to a fixed H.
            # We don't have a direct pos->idx map for the new state without rebuilding.
            # However, we can just check if q is in the set of fixed H positions.
            pass
            
    # Optimization: Precompute set of fixed H positions
    fixed_H_positions = set(chain[j] for j in H_fixed)
    
    for i in H_moved:
        p = new_chain[i]
        for q in _neighbours(p):
            if q in fixed_H_positions:
                new_contacts += 1
                
    delta_e = -(new_contacts - old_contacts)
    
    return delta_e, new_chain, new_occupied, True


def _get_delta_energy_crankshaft(sequence, chain, occupied, k):
    """
    Crankshaft move at k. Rotates residue k+1 around axis k-k+2.
    Returns (delta_energy, new_chain, new_occupied, valid)
    """
    n = len(chain)
    if k < 1 or k + 2 >= n:
        return 0, None, None, False
        
    # Residues involved: k, k+1, k+2
    # k and k+2 are fixed. k+1 moves.
    # The "other" position for k+1 is the one that is not chain[k+1] and is adjacent to both chain[k] and chain[k+2].
    # In 2D square lattice, if k and k+2 are not opposite (which would make it a straight line, impossible for SAW of len 3? No, 0-1-2 can be straight), 
    # wait, if k, k+1, k+2 are straight, k and k+2 are distance 2 apart. 
    # The neighbors of k and k+2 that are common are 2 points. One is k+1. The other is the target.
    
    p_k = chain[k]
    p_k2 = chain[k + 2]
    
    # Find common neighbors of p_k and p_k2
    # n_k = neighbors of p_k
    # n_k2 = neighbors of p_k2
    # common = intersection
    # target = the one in common that is not p_k+1
    
    n_k = set(_neighbours(p_k))
    n_k2 = set(_neighbours(p_k + 2)) # Typo in thought, p_k2
    n_k2 = set(_neighbours(p_k2))
    
    common = n_k.intersection(n_k2)
    
    if len(common) != 2:
        # Should not happen for valid SAW step k->k+1->k+2
        # Unless it's a U-turn? 
        # If k, k+1, k+2 form a U-turn, p_k and p_k2 are adjacent.
        # Then they have 2 common neighbors. One is k+1. The other is the "inside" of the U.
        # If they are straight, p_k and p_k2 are distance 2. They have 2 common neighbors (the "sides").
        # So always 2 common neighbors.
        return 0, None, None, False
        
    p_k1 = chain[k + 1]
    targets = [p for p in common if p != p_k1]
    
    if not targets:
        return 0, None, None, False
        
    target = targets[0]
    
    # Check if target is occupied
    if target in occupied:
        return 0, None, None, False
        
    # Calculate delta energy
    # Only k+1 moves.
    # Old contacts involving k+1 with others.
    # New contacts involving k+1 with others.
    
    H_k1 = (sequence[k + 1] == 'H')
    
    # If k+1 is P, delta E is 0 (unless it affects neighbors? No, P doesn't contribute to energy).
    # If k+1 is H, we check its neighbors.
    
    if not H_k1:
        return 0, None, None, True # Valid move, no energy change
        
    # Old contacts
    old_contacts = 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    for q in _neighbours(p_k1):
        j = pos_to_idx.get(q)
        if j is not None and j != k and j != k + 2 and sequence[j] == 'H':
            old_contacts += 1
            
    # New contacts
    new_contacts = 0
    for q in _neighbours(target):
        # Check if q is occupied by an H
        # We can use pos_to_idx, but target is new.
        # Just check if q is in the set of H positions (excluding k+1's old pos).
        # Actually, just check if q corresponds to an H index.
        j = pos_to_idx.get(q)
        if j is not None and j != k + 1 and sequence[j] == 'H':
            new_contacts += 1
            
    delta_e = -(new_contacts - old_contacts)
    
    new_chain = list(chain)
    new_chain[k + 1] = target
    new_occupied = set(occupied)
    new_occupied.remove(p_k1)
    new_occupied.add(target)
    
    return delta_e, new_chain, new_occupied, True


def fold(sequence, dimension):
    """
    Fold the sequence using Simulated Annealing with Pivot and Crankshaft moves.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
        
    # Time budget management
    # Assume 120s total for 10 sequences -> 12s each.
    # We will use 10s per sequence to be safe.
    start_time = time.time()
    deadline = start_time + 10.0
    
    best_chain = None
    best_energy = float('inf')
    
    rng = random.Random(12345 + len(sequence))
    
    # Precompute H indices for speed
    H_indices = [i for i in range(n) if sequence[i] == 'H']
    
    # Multiple restarts
    # Let's try 5 restarts, each getting 2s.
    num_restarts = 5
    time_per_restart = 2.0
    
    for r in range(num_restarts):
        restart_start = time.time()
        restart_deadline = restart_start + time_per_restart
        
        # Init
        chain = _initial_chain(sequence, dimension, rng)
        if chain is None:
            # Fallback: straight line (might not be valid if n > grid size? No, infinite grid)
            # But straight line has energy 0.
            chain = [(i, 0) for i in range(n)]
            
        occupied = set(chain)
        energy = _calc_energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
            
        # SA Parameters
        T = 1.0
        alpha = 0.99
        T_min = 0.01
        
        while time.time() < restart_deadline:
            # Choose move
            if rng.random() < 0.5:
                # Pivot
                # Choose pivot k and length L
                # L should be small to medium
                k = rng.randint(1, n - 2)
                max_L = min(10, n - k - 1)
                if max_L <= 0: continue
                L = rng.randint(1, max_L)
                rotation = rng.randint(1, 3)
                
                delta_e, new_chain, new_occupied, valid = _get_delta_energy_pivot(
                    sequence, chain, occupied, k, L, rotation
                )
            else:
                # Crankshaft
                k = rng.randint(1, n - 2)
                delta_e, new_chain, new_occupied, valid = _get_delta_energy_crankshaft(
                    sequence, chain, occupied, k
                )
                
            if not valid:
                continue
                
            # Metropolis
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                chain = new_chain
                occupied = new_occupied
                energy += delta_e
                
                if energy < best_energy:
                    best_energy = energy
                    best_chain = list(chain)
                    
            # Cool
            T *= alpha
            if T < T_min:
                # Reheat or break
                T = T_min
                
    if best_chain is None:
        # Fallback
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
