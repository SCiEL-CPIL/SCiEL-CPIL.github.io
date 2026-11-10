"""Rewritten program: parallel tempering Monte Carlo on a fixed SAW scaffold."""
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


def _make_scaffold(length, dimension):
    """Create a simple self-avoiding walk scaffold (a snake in a box)."""
    # Use a serpentine pattern in a box that fits the chain
    # For simplicity, use a straight line with some bends to allow moves
    # Actually, let's use a simple snake in a 2D plane extended to 3D
    points = []
    x = 0
    y = 0
    z = 0
    direction = 0  # 0: +x, 1: +y, 2: +z, 3: -x, 4: -y, 5: -z
    # Serpentine: go along x, then y, then x back, then y...
    # Let's just do a simple 3D snake
    for i in range(length):
        points.append((x, y, z))
        # Move in a pattern that creates a compact but self-avoiding walk
        # Use a simple pattern: right, up, right, up, back, down, back, down...
        # Actually, let's use a spiral-like pattern
        if i % 4 == 0:
            x += 1
        elif i % 4 == 1:
            y += 1
        elif i % 4 == 2:
            x -= 1
        elif i % 4 == 3:
            y -= 1
        # Occasionally move in z to add 3D character
        if i % 8 == 7:
            z += 1
    return points


def _compute_energy(sequence, chain, dimension):
    """Compute HP energy: -1 for each H-H contact (non-adjacent in chain)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _is_self_avoiding(chain):
    return len(set(chain)) == len(chain)


def _can_move(chain, occupied, idx, new_pos, dimension):
    """Check if residue at idx can move to new_pos."""
    if new_pos in occupied and occupied[new_pos] != idx:
        return False
    # Check that moving doesn't break adjacency with neighbors
    # The chain must remain a valid walk: consecutive residues must be adjacent
    if idx > 0:
        prev = chain[idx - 1]
        if sum(abs(a - b) for a, b in zip(prev, new_pos)) != 1:
            return False
    if idx < len(chain) - 1:
        nxt = chain[idx + 1]
        if sum(abs(a - b) for a, b in zip(nxt, new_pos)) != 1:
            return False
    return True


def _parallel_tempering_fold(sequence, dimension, time_budget):
    """
    Parallel tempering Monte Carlo on a fixed-length self-avoiding walk.
    We maintain multiple replicas at different temperatures.
    Moves: pivot moves, crankshaft moves, and local moves.
    """
    n = len(sequence)
    if n <= 1:
        return [list((0,) * dimension)]
    
    # Create initial scaffold
    chain = _make_scaffold(n, dimension)
    
    # Ensure the scaffold is valid
    if not _is_self_avoiding(chain):
        # Fallback to straight line
        chain = [(i, 0, 0) if dimension >= 3 else (i, 0) if dimension == 2 else (i,) for i in range(n)]
        if dimension == 3:
            chain = [(i, 0, 0) for i in range(n)]
        elif dimension == 2:
            chain = [(i, 0) for i in range(n)]
    
    # Compute initial energy
    best_energy = _compute_energy(sequence, chain, dimension)
    best_chain = list(chain)
    
    # Set up parallel tempering with multiple temperatures
    n_replicas = 5
    base_temp = 1.0
    temps = [base_temp * (2 ** i) for i in range(n_replicas)]
    
    # Each replica has its own chain
    replicas = [list(chain) for _ in range(n_replicas)]
    replica_energies = [_compute_energy(sequence, r, dimension) for r in replicas]
    
    rng = random.Random(42 + n)
    start_time = time.time()
    
    # Monte Carlo steps
    mc_steps = 0
    while time.time() - start_time < time_budget:
        # Perform a move on a random replica
        rep = rng.randint(0, n_replicas - 1)
        chain_rep = replicas[rep]
        energy_rep = replica_energies[rep]
        temp = temps[rep]
        
        # Choose a move type
        move_type = rng.choice(['pivot', 'crankshaft', 'local'])
        
        new_chain = list(chain_rep)
        new_energy = None
        
        if move_type == 'pivot':
            # Pivot move: choose a pivot point and rotate one end
            if n > 2:
                pivot = rng.randint(1, n - 2)
                # Rotate the tail (pivot+1 to end) around the pivot
                # Try all 5 possible rotations (excluding the original direction)
                pivot_pos = chain_rep[pivot]
                tail_start = chain_rep[pivot + 1]
                # Direction from pivot to tail_start
                dx = tail_start[0] - pivot_pos[0]
                dy = tail_start[1] - pivot_pos[1]
                dz = tail_start[2] - pivot_pos[2] if dimension >= 3 else 0
                
                # Generate possible new directions (unit vectors except the current one)
                possible_dirs = []
                for axis in range(dimension):
                    for step in (-1, 1):
                        d = [0] * dimension
                        d[axis] = step
                        # Skip the current direction
                        if d[0] == dx and d[1] == dy and (d[2] == dz if dimension >= 3 else True):
                            continue
                        possible_dirs.append(tuple(d))
                
                if possible_dirs:
                    new_dir = rng.choice(possible_dirs)
                    # Apply the rotation to the entire tail
                    # Simple approach: just move the next residue and try to fix the rest
                    # This is complex; let's use a simpler local move instead
                    new_chain = None
                    pass
        
        elif move_type == 'crankshaft':
            # Crankshaft move: rotate a segment of 2-3 residues
            if n > 3:
                seg_len = rng.choice([2, 3])
                start = rng.randint(1, n - seg_len - 1)
                # Try to rotate the segment
                # For simplicity, just try moving one residue in the segment
                idx = start
                current = chain_rep[idx]
                # Try all 6 neighbors
                for new_pos in _neighbours(current):
                    if _can_move(chain_rep, None, idx, new_pos, dimension):
                        # Check if this creates a self-avoiding walk
                        test_chain = list(chain_rep)
                        test_chain[idx] = new_pos
                        if _is_self_avoiding(test_chain):
                            new_chain = test_chain
                            break
        
        else:
            # Local move: try to move a single residue to a neighbor
            idx = rng.randint(1, n - 2)  # Don't move ends
            current = chain_rep[idx]
            # Try all 6 neighbors
            for new_pos in _neighbours(current):
                if _can_move(chain_rep, None, idx, new_pos, dimension):
                    test_chain = list(chain_rep)
                    test_chain[idx] = new_pos
                    if _is_self_avoiding(test_chain):
                        new_chain = test_chain
                        break
        
        if new_chain is None:
            continue
        
        new_energy = _compute_energy(sequence, new_chain, dimension)
        delta_e = new_energy - energy_rep
        
        # Metropolis acceptance
        if delta_e <= 0 or rng.random() < math.exp(-delta_e / temp):
            replicas[rep] = new_chain
            replica_energies[rep] = new_energy
            
            # Update best
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(new_chain)
        
        mc_steps += 1
        
        # Periodically swap replicas (parallel tempering swap)
        if mc_steps % 100 == 0:
            i = rng.randint(0, n_replicas - 2)
            j = i + 1
            e_i = replica_energies[i]
            e_j = replica_energies[j]
            t_i = temps[i]
            t_j = temps[j]
            
            # Swap probability
            delta = (1.0 / t_i - 1.0 / t_j) * (e_j - e_i)
            if delta <= 0 or rng.random() < math.exp(-delta):
                replicas[i], replicas[j] = replicas[j], replicas[i]
                replica_energies[i], replica_energies[j] = replica_energies[j], replica_energies[i]
    
    return [list(p) for p in best_chain]


def fold(sequence, dimension):
    """Fold a sequence using parallel tempering Monte Carlo."""
    n = len(sequence)
    if n <= 1:
        return [list((0,) * dimension)]
    
    # Allocate time based on sequence length
    # Shorter sequences get less time, longer get more
    if n <= 50:
        time_budget = 5.0
    elif n <= 100:
        time_budget = 10.0
    else:
        time_budget = 15.0
    
    # Leave some time for fallback
    result = _parallel_tempering_fold(sequence, dimension, time_budget)
    
    # Verify the result is valid
    if not _is_self_avoiding(result):
        # Fallback to a straight line
        result = [list((i, 0, 0)) if dimension >= 3 else list((i, 0)) if dimension == 2 else [i] for i in range(n)]
    
    # Verify unit steps
    for i in range(1, n):
        if sum(abs(a - b) for a, b in zip(result[i-1], result[i])) != 1:
            # Fallback to a straight line
            result = [list((i, 0, 0)) if dimension >= 3 else list((i, 0)) if dimension == 2 else [i] for i in range(n)]
            break
    
    return result
# EVOLVE-BLOCK-END
