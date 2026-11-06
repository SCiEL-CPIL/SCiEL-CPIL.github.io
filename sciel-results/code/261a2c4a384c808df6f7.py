"""HP lattice protein folding: multi-start simulated annealing with efficient moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

_T0 = time.time()
_DEADLINE = 195.0  # wall-clock limit with safety margin (210s kill, leave 15s buffer)


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _compute_energy(sequence, chain):
    """Compute energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    if n < 3:
        return 0
    
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
    
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (x + dx, y + dy)
            j = pos_to_idx.get(nb)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _generate_initial(sequence, rng):
    """Generate a random self-avoiding walk as initial conformation."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0)}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (last[0] + dx, last[1] + dy)
            if nb not in occupied:
                options.append(nb)
        
        if not options:
            return None
        
        next_pos = rng.choice(options)
        chain.append(next_pos)
        occupied.add(next_pos)
    
    return chain


def _pivot_move(chain, rng):
    """Perform a pivot move: rotate a suffix of the chain around a pivot point."""
    n = len(chain)
    if n < 4:
        return None
    
    pivot = rng.randint(1, n - 2)
    pivot_pos = chain[pivot]
    suffix = chain[pivot + 1:]
    
    # Try 90-degree CCW rotation
    rotated = []
    valid = True
    for (x, y) in suffix:
        rx, ry = x - pivot_pos[0], y - pivot_pos[1]
        new_x, new_y = -ry, rx
        pos = (pivot_pos[0] + new_x, pivot_pos[1] + new_y)
        rotated.append(pos)
    
    new_chain = chain[:pivot + 1] + rotated
    
    # Check self-avoidance
    if len(set(new_chain)) != n:
        return None
    
    # Check connectivity
    for i in range(n - 1):
        if abs(new_chain[i][0] - new_chain[i+1][0]) + abs(new_chain[i][1] - new_chain[i+1][1]) != 1:
            return None
    
    return new_chain


def _kink_move(chain, rng):
    """Perform a kink move: try to flip a short segment (length 2-3)."""
    n = len(chain)
    if n < 5:
        return None
    
    # Choose segment length 2 or 3
    seg_len = rng.choice([2, 3])
    
    if n <= seg_len + 2:
        return None
    
    start = rng.randint(1, n - seg_len - 1)
    
    before = chain[:start]
    segment = chain[start:start + seg_len]
    after = chain[start + seg_len:]
    
    # Try to reverse the segment
    reversed_segment = list(reversed(segment))
    
    new_chain = before + reversed_segment + after
    
    # Check self-avoidance
    if len(set(new_chain)) != n:
        return None
    
    # Check connectivity
    for i in range(n - 1):
        if abs(new_chain[i][0] - new_chain[i+1][0]) + abs(new_chain[i][1] - new_chain[i+1][1]) != 1:
            return None
    
    return new_chain


def _local_search(sequence, chain, rng, max_iters):
    """Simulated annealing local search."""
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    T = 5.0
    cooling = 0.9995
    
    for it in range(max_iters):
        if _remaining() < 3.0:
            break
        
        # Choose a move type
        move_type = rng.random()
        if move_type < 0.6:
            new_chain = _pivot_move(chain, rng)
        elif move_type < 0.8:
            new_chain = _kink_move(chain, rng)
        else:
            # Simple single-residue move
            idx = rng.randint(1, n - 1)
            x, y = chain[idx]
            moved = False
            new_chain = list(chain)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_pos = (x + dx, y + dy)
                occupied_without = set(chain[:idx] + chain[idx+1:])
                if new_pos not in occupied_without:
                    # Check adjacency to neighbors
                    if idx > 0:
                        prev = chain[idx - 1]
                        if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
                            continue
                    if idx < n - 1:
                        next_p = chain[idx + 1]
                        if abs(new_pos[0] - next_p[0]) + abs(new_pos[1] - next_p[1]) != 1:
                            continue
                    new_chain[idx] = new_pos
                    moved = True
                    break
            
            if not moved:
                continue
        
        if new_chain is None:
            continue
        
        # Verify validity
        if len(set(new_chain)) != n:
            continue
        valid = True
        for i in range(n - 1):
            if abs(new_chain[i][0] - new_chain[i+1][0]) + abs(new_chain[i][1] - new_chain[i+1][1]) != 1:
                valid = False
                break
        if not valid:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Acceptance criterion
        if delta < 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(new_chain)
        
        T *= cooling
        if T < 0.05:
            T = 0.05
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold sequence into low-energy conformation using multi-start simulated annealing."""
    if dimension != 2:
        return [[i] + [0] * (dimension - 1) for i in range(len(sequence))]
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Time budget
    start_time = time.time()
    time_budget = min(_remaining() - 5.0, 180.0)
    
    # Determine number of restarts and iterations based on sequence length
    if n <= 30:
        num_restarts = 60
        iters_per_restart = 8000
    elif n <= 50:
        num_restarts = 40
        iters_per_restart = 6000
    elif n <= 70:
        num_restarts = 25
        iters_per_restart = 4000
    else:
        num_restarts = 15
        iters_per_restart = 3000
    
    for restart in range(num_restarts):
        if _remaining() < 10.0:
            break
        
        # Generate initial conformation
        chain = _generate_initial(sequence, rng)
        if chain is None or len(chain) != n:
            continue
        
        # Verify validity
        if len(set(chain)) != n:
            continue
        valid = True
        for i in range(n - 1):
            if abs(chain[i][0] - chain[i+1][0]) + abs(chain[i][1] - chain[i+1][1]) != 1:
                valid = False
                break
        if not valid:
            continue
        
        # Run local search
        chain, energy = _local_search(sequence, chain, rng, max_iters=iters_per_restart)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
        
        # Check time
        if time.time() - start_time > time_budget:
            break
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [[x, y] for x, y in best_chain]
# EVOLVE-BLOCK-END
