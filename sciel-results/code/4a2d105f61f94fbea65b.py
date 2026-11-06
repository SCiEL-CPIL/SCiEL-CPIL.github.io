"""HP lattice protein folding using simulated annealing with reversals and moves."""
# EVOLVE-BLOCK-START
import random
import math
from collections import defaultdict


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _build_contact_index(sequence, chain):
    """Build a mapping from lattice point to residue index, and a set of H positions."""
    where = {}
    for i, p in enumerate(chain):
        where[tuple(p)] = i
    h_positions = set()
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            h_positions.add(tuple(p))
    return where, h_positions


def _compute_energy(sequence, chain, where=None, h_positions=None):
    """Compute energy = -number of H-H contacts (non-consecutive)."""
    if where is None or h_positions is None:
        where, h_positions = _build_contact_index(sequence, chain)
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            pt = tuple(p)
            for q in _neighbours(pt):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2  # each contact counted twice


def _initial_compact_walk(n, dimension, rng):
    """Create a compact initial self-avoiding walk using a spiral-like pattern."""
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    # Try to make a compact square spiral
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_idx = 0
    step_len = 1
    steps_taken = 0
    for _ in range(4):
        for _ in range(step_len):
            if len(chain) >= n:
                break
            dx, dy = directions[dir_idx]
            new_pt = (chain[-1][0] + dx, chain[-1][1] + dy)
            if new_pt not in occupied:
                occupied[new_pt] = len(chain)
                chain.append(new_pt)
                steps_taken += 1
            else:
                # Try to find any free neighbor
                for d in directions:
                    alt = (chain[-1][0] + d[0], chain[-1][1] + d[1])
                    if alt not in occupied:
                        occupied[alt] = len(chain)
                        chain.append(alt)
                        steps_taken += 1
                        break
        if len(chain) >= n:
            break
        dir_idx = (dir_idx + 1) % 4
        if dir_idx % 2 == 0:
            step_len += 1
    # If we didn't reach n, fill with random growth
    while len(chain) < n:
        options = []
        for site in _neighbours(tuple(chain[-1])):
            if site not in occupied:
                options.append(site)
        if not options:
            break
        site = rng.choice(options)
        occupied[site] = len(chain)
        chain.append(site)
    return chain[:n]


def _initial_random_walk(n, dimension, rng):
    """Create a random self-avoiding walk."""
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    for _ in range(1, n):
        options = []
        for site in _neighbours(tuple(chain[-1])):
            if site not in occupied:
                options.append(site)
        if not options:
            return None
        site = rng.choice(options)
        occupied[site] = len(chain)
        chain.append(site)
    return chain


def _reverse_subchain(chain, i, j):
    """Reverse the subchain from index i to j (inclusive)."""
    new_chain = chain[:i] + list(reversed(chain[i:j+1])) + chain[j+1:]
    return new_chain


def _try_move(chain, i, j, occupied):
    """Try to move subchain [i..j] to a new location adjacent to chain[i-1] or chain[j+1].
    Returns new_chain if successful, None otherwise."""
    n = len(chain)
    # Determine which end we're moving from
    # We'll try moving the subchain so that it connects at the same endpoint
    # Move subchain [i..j] to be adjacent to chain[i-1] (if i > 0)
    if i > 0:
        anchor = tuple(chain[i-1])
        # The subchain starts at chain[i], so we need to find a free neighbor of anchor
        # where we can place the start of the subchain
        for site in _neighbours(anchor):
            if site in occupied:
                continue
            # Check if we can place the entire subchain starting from site
            # with the same relative positions
            new_chain = list(chain)
            # Remove old positions
            old_positions = set(tuple(chain[k]) for k in range(i, j+1))
            # New positions: shift the subchain so that chain[i] goes to site
            dx = site[0] - chain[i][0]
            dy = site[1] - chain[i][1]
            # Check if the shifted subchain is self-avoiding and doesn't collide
            valid = True
            new_positions = []
            for k in range(i, j+1):
                np = (chain[k][0] + dx, chain[k][1] + dy)
                if np in occupied and np not in old_positions:
                    valid = False
                    break
                new_positions.append(np)
            if not valid:
                continue
            # Also check that the new positions don't collide with each other
            if len(set(new_positions)) != len(new_positions):
                continue
            # Check connectivity: chain[i-1] to new_chain[i] must be adjacent
            if abs(new_positions[0][0] - anchor[0]) + abs(new_positions[0][1] - anchor[1]) != 1:
                continue
            # Check connectivity: new_chain[j] to chain[j+1] must be adjacent
            if j + 1 < n:
                next_pt = tuple(chain[j+1])
                if abs(new_positions[-1][0] - next_pt[0]) + abs(new_positions[-1][1] - next_pt[1]) != 1:
                    continue
            # Valid move
            for k in range(i, j+1):
                new_chain[k] = list(new_positions[k-i])
            return new_chain
    # If i == 0, try moving from the other end
    elif j < n - 1:
        anchor = tuple(chain[j+1])
        for site in _neighbours(anchor):
            if site in occupied:
                continue
            dx = site[0] - chain[j][0]
            dy = site[1] - chain[j][1]
            valid = True
            new_positions = []
            old_positions = set(tuple(chain[k]) for k in range(i, j+1))
            for k in range(j, i-1, -1):
                np = (chain[k][0] + dx, chain[k][1] + dy)
                if np in occupied and np not in old_positions:
                    valid = False
                    break
                new_positions.append(np)
            if not valid:
                continue
            if len(set(new_positions)) != len(new_positions):
                continue
            # new_positions[0] corresponds to chain[j], should be adjacent to anchor
            if abs(new_positions[0][0] - anchor[0]) + abs(new_positions[0][1] - anchor[1]) != 1:
                continue
            # chain[i] to new_chain[i] (which is new_positions[-1]) must be adjacent
            if i > 0:
                prev_pt = tuple(chain[i-1])
                if abs(new_positions[-1][0] - prev_pt[0]) + abs(new_positions[-1][1] - prev_pt[1]) != 1:
                    continue
            new_chain = list(chain)
            for k in range(i, j+1):
                idx = j - k
                new_chain[k] = list(new_positions[idx])
            return new_chain
    return None


def _sa_fold(sequence, n, dimension, rng, time_budget=1.5):
    """Simulated annealing for one sequence."""
    import time
    start_time = time.time()
    
    # Try multiple initializations
    best_chain = None
    best_energy = 0
    
    for init_method in ['compact', 'random']:
        if time.time() - start_time > time_budget * 0.3:
            break
        
        if init_method == 'compact':
            chain = _initial_compact_walk(n, dimension, rng)
        else:
            chain = _initial_random_walk(n, dimension, rng)
        
        if chain is None or len(chain) != n:
            continue
        
        # Compute initial energy
        where, h_positions = _build_contact_index(sequence, chain)
        current_energy = _compute_energy(sequence, chain, where, h_positions)
        
        if best_chain is None or current_energy < best_energy:
            best_chain = [list(p) for p in chain]
            best_energy = current_energy
        
        # Simulated annealing
        T_start = 10.0
        T_end = 0.1
        cooling_rate = 0.995
        
        T = T_start
        no_improve_count = 0
        max_no_improve = 200
        
        while T > T_end and (time.time() - start_time) < time_budget * 0.7:
            # Choose a move: reversal or move
            move_type = rng.random()
            
            if move_type < 0.7:
                # Reversal
                if n < 3:
                    break
                i = rng.randint(0, n - 2)
                j = rng.randint(i + 1, n - 1)
                new_chain = _reverse_subchain(chain, i, j)
                
                # Check self-avoidance
                new_occupied = set()
                valid = True
                for p in new_chain:
                    if tuple(p) in new_occupied:
                        valid = False
                        break
                    new_occupied.add(tuple(p))
                
                if not valid:
                    continue
                
                # Check connectivity
                for k in range(1, n):
                    if abs(new_chain[k][0] - new_chain[k-1][0]) + abs(new_chain[k][1] - new_chain[k-1][1]) != 1:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Compute new energy
                new_where, new_h_positions = _build_contact_index(sequence, new_chain)
                new_energy = _compute_energy(sequence, new_chain, new_where, new_h_positions)
                
                delta = new_energy - current_energy
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    chain = new_chain
                    current_energy = new_energy
                    where, h_positions = new_where, new_h_positions
                    no_improve_count = 0
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = [list(p) for p in chain]
                else:
                    no_improve_count += 1
            else:
                # Move
                if n < 4:
                    break
                length = rng.randint(1, min(n // 2, 8))
                i = rng.randint(0, n - length)
                j = i + length - 1
                
                # Build occupied set
                occupied = set(tuple(p) for p in chain)
                new_chain = _try_move(chain, i, j, occupied)
                
                if new_chain is None:
                    no_improve_count += 1
                    continue
                
                # Verify self-avoidance and connectivity
                new_occupied = set()
                valid = True
                for p in new_chain:
                    if tuple(p) in new_occupied:
                        valid = False
                        break
                    new_occupied.add(tuple(p))
                
                if not valid:
                    no_improve_count += 1
                    continue
                
                for k in range(1, n):
                    if abs(new_chain[k][0] - new_chain[k-1][0]) + abs(new_chain[k][1] - new_chain[k-1][1]) != 1:
                        valid = False
                        break
                
                if not valid:
                    no_improve_count += 1
                    continue
                
                new_where, new_h_positions = _build_contact_index(sequence, new_chain)
                new_energy = _compute_energy(sequence, new_chain, new_where, new_h_positions)
                
                delta = new_energy - current_energy
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    chain = new_chain
                    current_energy = new_energy
                    where, h_positions = new_where, new_h_positions
                    no_improve_count = 0
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = [list(p) for p in chain]
                else:
                    no_improve_count += 1
            
            T *= cooling_rate
            
            if no_improve_count > max_no_improve:
                break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using simulated annealing with multiple restarts."""
    n = len(sequence)
    if n <= 2:
        return [[0, 0], [1, 0]][:n]
    
    total_budget = 10.0  # seconds per sequence
    import time
    start_time = time.time()
    
    best_chain = None
    best_energy = 0
    
    # Try multiple restarts with different seeds
    n_restarts = 5
    time_per_restart = total_budget / n_restarts
    
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        remaining = total_budget - elapsed
        if remaining < 0.5:
            break
        
        rng = random.Random(12345 + len(sequence) * 100 + restart * 7)
        time_budget = min(time_per_restart, remaining)
        
        chain, energy = _sa_fold(sequence, n, dimension, rng, time_budget)
        
        if chain is not None:
            if best_chain is None or energy < best_energy:
                best_chain = chain
                best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [[i, 0] for i in range(n)]
    
    return best_chain
# EVOLVE-BLOCK-END
