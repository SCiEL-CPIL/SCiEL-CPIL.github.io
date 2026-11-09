"""Rewritten program for HP lattice protein folding; uses simulated annealing with moves."""
# EVOLVE-BLOCK-START
import random
import math


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute HP contact energy: -number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _try_move(chain, move_idx, direction, occupied):
    """
    Attempt a local move: rotate residue at move_idx by 90 degrees in given direction.
    Returns new_chain and new_occupied if valid, else None.
    direction: 0,1,2,3 for four possible rotations around the previous residue.
    """
    n = len(chain)
    if move_idx <= 0 or move_idx >= n:
        return None
    
    prev = chain[move_idx - 1]
    curr = chain[move_idx]
    
    # Generate all 4 neighbors of prev
    neighbors = list(_neighbours(prev))
    
    # Find which neighbor is current position
    try:
        curr_idx = neighbors.index(curr)
    except ValueError:
        return None
    
    # Try the 3 other directions (skip staying put)
    new_pos = neighbors[(curr_idx + direction) % 4]
    
    if new_pos == curr:
        return None
    
    if new_pos in occupied and occupied[new_pos] != move_idx:
        return None
    
    # Check self-avoidance: new_pos must not be occupied by any other residue
    # (we already checked it's not occupied by move_idx, and if it's in occupied 
    #  it must be another residue, which we reject)
    
    # Actually, let's be more careful. The occupied set contains all residues.
    # We need to remove curr, add new_pos, and check no conflicts.
    # But since we're only moving one residue, we just need new_pos not to be
    # occupied by any residue other than move_idx itself.
    
    if new_pos in occupied:
        occ_idx = occupied[new_pos]
        if occ_idx != move_idx:
            return None
    
    # Also need to check that the new position doesn't create a conflict
    # with the next residue (if any)
    if move_idx + 1 < n:
        next_res = chain[move_idx + 1]
        if new_pos == next_res:
            return None
    
    new_chain = list(chain)
    new_chain[move_idx] = new_pos
    new_occupied = dict(occupied)
    del new_occupied[curr]
    new_occupied[new_pos] = move_idx
    
    return new_chain, new_occupied


def _fold_sa(sequence, dimension, max_iter=50000, t0=10.0, t_final=0.1):
    """Simulated annealing to find low-energy conformation."""
    n = len(sequence)
    if n <= 2:
        return [(i, 0) for i in range(n)] if dimension == 2 else [(i,) + (0,) * (dimension - 1) for i in range(n)]
    
    # Initialize with a spiral or random walk
    # Use a simple random walk initialization
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = float('inf')
    
    for init_attempt in range(5):
        # Try different initializations
        chain = []
        occupied = {}
        
        # Start at origin
        chain.append((0, 0))
        occupied[(0, 0)] = 0
        
        success = True
        for i in range(1, n):
            # Try all 4 directions
            last = chain[-1]
            neighbors = list(_neighbours(last))
            rng.shuffle(neighbors)
            
            placed = False
            for nb in neighbors:
                if nb not in occupied:
                    chain.append(nb)
                    occupied[nb] = i
                    placed = True
                    break
            
            if not placed:
                success = False
                break
        
        if not success:
            continue
        
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
        
        # Run SA from this starting point
        current_chain = list(chain)
        current_occupied = dict(occupied)
        current_energy = energy
        
        n_iter = max_iter // 5
        cooling = (t_final / t0) ** (1.0 / n_iter)
        
        for it in range(n_iter):
            t = t0 * (cooling ** it)
            
            # Pick a random residue to move (not endpoints)
            move_idx = rng.randint(1, n - 2)
            direction = rng.randint(0, 3)
            
            result = _try_move(current_chain, move_idx, direction, current_occupied)
            if result is None:
                continue
            
            new_chain, new_occupied = result
            new_energy = _energy(sequence, new_chain)
            
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < math.exp(-delta / max(t, 1e-10)):
                current_chain = new_chain
                current_occupied = new_occupied
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        
        if best_chain is not None:
            break
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation found by simulated annealing."""
    if dimension != 2:
        # Fallback for non-2D (shouldn't happen per spec)
        return [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    chain = _fold_sa(sequence, dimension)
    return [list(p) for p in chain]
# EVOLVE-BLOCK-END
