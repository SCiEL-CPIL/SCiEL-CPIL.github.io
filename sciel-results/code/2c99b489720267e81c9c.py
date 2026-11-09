"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _grow_greedy(sequence, dimension, rng):
    """One randomized greedy chain growth."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_search(sequence, chain, rng, max_iters=5000, deadline=None):
    """Simulated annealing local search with pivot moves."""
    n = len(chain)
    if n < 2:
        return chain
    
    def _compute_energy_fast(chain, seq):
        where = {p: i for i, p in enumerate(chain)}
        contacts = 0
        for i, p in enumerate(chain):
            if seq[i] == 'H':
                for q in _neighbours(p):
                    j = where.get(q)
                    if j is not None and j > i + 1 and seq[j] == 'H':
                        contacts += 1
        return -contacts
    
    current = list(chain)
    current_energy = _compute_energy_fast(current, sequence)
    best = list(current)
    best_energy = current_energy
    
    # Simulated annealing parameters
    T_start = 5.0
    T_end = 0.1
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        T = T_start * (cooling ** it)
        
        # Try a pivot move: reverse a segment of the chain
        if n < 3:
            break
            
        # Choose a random segment to reverse
        # The segment must be at least length 2 and the reversal must be valid
        # We'll try several random segments
        found_valid_move = False
        for attempt in range(10):
            # Choose start and end of segment
            # Ensure the segment is not too close to ends to avoid issues
            min_len = 2
            if n <= min_len:
                break
            start = rng.randint(0, n - 1 - min_len)
            length = rng.randint(min_len, n - 1 - start)
            end = start + length - 1
            
            # Check if reversing the segment is valid
            # The segment is current[start:end+1]
            # After reversal, the new segment is reversed
            # We need to check:
            # 1. The new positions don't collide with non-adjacent chain parts
            # 2. The connections at the boundaries are valid
            
            # The new chain would be:
            # current[:start] + reversed(current[start:end+1]) + current[end+1:]
            
            # Check boundary connections
            # Left boundary: if start > 0, current[start-1] must be adjacent to new current[start] (which was current[end])
            if start > 0:
                left_neighbor = current[start - 1]
                new_start_pos = current[end]
                if abs(left_neighbor[0] - new_start_pos[0]) + abs(left_neighbor[1] - new_start_pos[1]) != 1:
                    continue
            
            # Right boundary: if end < n-1, current[end+1] must be adjacent to new current[end] (which was current[start])
            if end < n - 1:
                right_neighbor = current[end + 1]
                new_end_pos = current[start]
                if abs(right_neighbor[0] - new_end_pos[0]) + abs(right_neighbor[1] - new_end_pos[1]) != 1:
                    continue
            
            # Check self-avoidance within the reversed segment
            # The reversed segment occupies the same set of positions, so internal collisions
            # are not a new issue, but we need to check against the rest of the chain
            # Actually, the positions are the same, just in different order.
            # The only new collisions could be with the parts outside the segment.
            # But wait, the positions are the same set, so if there was no collision before,
            # there won't be a collision with the same set of positions.
            # However, the adjacency changes.
            # Let's just construct the new chain and check self-avoidance properly.
            
            new_chain = current[:start] + current[start:end+1][::-1] + current[end+1:]
            
            # Check self-avoidance
            seen = set()
            valid = True
            for i, p in enumerate(new_chain):
                if p in seen:
                    valid = False
                    break
                seen.add(p)
            
            if not valid:
                continue
            
            # Check that consecutive points are adjacent
            valid = True
            for i in range(len(new_chain) - 1):
                if abs(new_chain[i][0] - new_chain[i+1][0]) + abs(new_chain[i][1] - new_chain[i+1][1]) != 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Compute new energy
            new_energy = _compute_energy_fast(new_chain, sequence)
            delta_e = new_energy - current_energy
            
            # Accept or reject
            if delta_e < 0 or (T > 0 and rng.random() < math.exp(-delta_e / T)):
                current = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best = list(current)
                    best_energy = current_energy
                found_valid_move = True
                break
        
        if not found_valid_move:
            # No valid move found in this iteration, continue
            continue
    
    return best


def fold(sequence, dimension):
    """Best conformation using greedy growth followed by local search."""
    n = len(sequence)
    
    # Deadline: use 100 seconds for all sequences combined, but we don't know the total
    # We'll use a per-call deadline of about 15 seconds to be safe
    # Actually, the task says 120s CPU budget for all sequences. Let's use a fraction.
    # Since we don't know how many sequences will be called, let's be conservative.
    # Use 10 seconds per sequence.
    start_time = time.time()
    deadline = start_time + 10.0
    
    rng = random.Random(12345 + len(sequence))
    
    # Phase 1: Generate initial conformations using greedy growth
    initial_chains = []
    num_greedy = 50
    for i in range(num_greedy):
        if time.time() > deadline - 2.0:
            break
        chain = _grow_greedy(sequence, dimension, rng)
        if chain is not None:
            initial_chains.append(chain)
    
    if not initial_chains:
        # Fallback: straight line
        return [list(p) for p in [tuple([i] + [0] * (dimension - 1)) for i in range(n)]]
    
    # Phase 2: Local search on each initial conformation
    best_chain = initial_chains[0]
    best_energy = _energy(sequence, best_chain)
    
    num_local = min(20, len(initial_chains))
    for i in range(num_local):
        if time.time() > deadline - 1.0:
            break
        
        chain = initial_chains[i]
        current_energy = _energy(sequence, chain)
        
        # Local search
        improved_chain = _local_search(sequence, chain, rng, max_iters=3000, deadline=deadline - 0.5)
        improved_energy = _energy(sequence, improved_chain)
        
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    # Phase 3: If we have time, do more local search on the best chain
    if time.time() < deadline - 0.5:
        improved_chain = _local_search(sequence, best_chain, rng, max_iters=5000, deadline=deadline - 0.2)
        improved_energy = _energy(sequence, improved_chain)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
