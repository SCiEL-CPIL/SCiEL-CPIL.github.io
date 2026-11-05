"""HP lattice protein folding using multi-start local search with efficient energy updates."""
import time
import random
import math
import sys

def _neighbours(point):
    """Yield the 4 neighbours of a point on the square lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    """Compute the HP contact energy of a chain conformation.
    Energy = - (number of H-H contacts) where contacts are non-adjacent H residues on neighbouring lattice sites."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _energy_fast(sequence, chain):
    """Faster energy computation using a set for occupied positions."""
    occupied = set(chain)
    contacts = 0
    n = len(chain)
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                j = None
                # Check if neighbour position is in chain
                # We need to find which index is at that position
                # This is O(n) per check, so let's use a dict
                pass
    # Actually, let's use the dict approach but with local variables for speed
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                q = (x + dx, y + dy)
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _generate_compact_walk(n, rng, compactness=1.0):
    """Generate a self-avoiding walk of length n with varying compactness."""
    if n <= 1:
        return [(0, 0)] if n == 1 else []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    direction = (1, 0)
    
    # Choose a bias: how often to turn
    turn_prob = 0.3 + compactness * 0.3
    
    for i in range(1, n):
        found = False
        # Try current direction first, then turn left, then right, then back
        dirs_to_try = [direction]
        # Left turn
        dx, dy = direction
        left = (-dy, dx)
        right = (dy, -dx)
        back = (-dx, -dy)
        
        # Shuffle with bias toward current direction
        if rng.random() < turn_prob:
            dirs_to_try = [left, right, direction, back]
        else:
            dirs_to_try = [direction, left, right, back]
        
        for d in dirs_to_try:
            x, y = chain[-1]
            new_pos = (x + d[0], y + d[1])
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied[new_pos] = i
                direction = d
                found = True
                break
        
        if not found:
            # Dead end: try to backtrack or just fail
            # For now, return what we have if it's long enough, else None
            if len(chain) >= n * 0.8:
                # Pad with a straight line if possible
                last = chain[-1]
                dx, dy = direction
                for k in range(len(chain), n):
                    new_pos = (last[0] + dx, last[1] + dy)
                    if new_pos in occupied:
                        # Try other directions
                        found_pad = False
                        for pd in [(1,0),(-1,0),(0,1),(0,-1)]:
                            np2 = (last[0] + pd[0], last[1] + pd[1])
                            if np2 not in occupied:
                                chain.append(np2)
                                occupied[np2] = k
                                last = np2
                                direction = pd
                                found_pad = True
                                break
                        if not found_pad:
                            return None
                    else:
                        chain.append(new_pos)
                        occupied[new_pos] = k
                        last = new_pos
                return chain
            return None
    
    return chain

def _local_search(sequence, chain, rng, max_iter=5000, time_limit=10.0):
    """Local search: try moving individual residues to adjacent empty positions."""
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    start_time = time.time()
    no_improve_count = 0
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        # Try moving each interior residue
        for i in range(1, n - 1):
            old_pos = best_chain[i]
            
            for new_pos in _neighbours(old_pos):
                if new_pos in occupied:
                    continue
                
                # Check connectivity
                prev_pos = best_chain[i - 1]
                next_pos = best_chain[i + 1]
                
                dist_prev = abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1])
                dist_next = abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1])
                
                if dist_prev != 1 or dist_next != 1:
                    continue
                
                # Perform move
                del occupied[old_pos]
                occupied[new_pos] = i
                best_chain[i] = new_pos
                
                new_energy = _energy(sequence, best_chain)
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    # Revert
                    del occupied[new_pos]
                    occupied[old_pos] = i
                    best_chain[i] = old_pos
            
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 50:
                break
    
    return best_chain, best_energy

def _simulated_annealing(sequence, chain, rng, time_limit=5.0):
    """Simulated annealing for HP folding."""
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    current_chain = list(chain)
    current_energy = best_energy
    
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    T = 2.0
    cooling = 0.999
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Pick a random interior residue
        i = rng.randint(1, n - 2)
        old_pos = current_chain[i]
        
        # Pick a random adjacent empty position
        candidates = []
        for new_pos in _neighbours(old_pos):
            if new_pos not in occupied:
                prev_pos = current_chain[i - 1]
                next_pos = current_chain[i + 1]
                dist_prev = abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1])
                dist_next = abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1])
                if dist_prev == 1 and dist_next == 1:
                    candidates.append(new_pos)
        
        if not candidates:
            T *= cooling
            continue
        
        new_pos = rng.choice(candidates)
        
        # Perform move
        del occupied[old_pos]
        occupied[new_pos] = i
        current_chain[i] = new_pos
        
        new_energy = _energy(sequence, current_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(current_chain)
        else:
            # Revert
            del occupied[new_pos]
            occupied[old_pos] = i
            current_chain[i] = old_pos
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation.
    Uses multi-start local search with simulated annealing."""
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin
    n = len(sequence)
    
    if n <= 1:
        return [[0, 0]] if n == 1 else []
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0  # Start with unfolded (energy 0)
    
    # Generate diverse initial conformations
    initial_chains = []
    num_starts = 20
    
    for s in range(num_starts):
        if time.time() > deadline - 30.0:
            break
        
        # Vary compactness
        compactness = 0.2 + 0.6 * (s / num_starts)
        chain = _generate_compact_walk(n, rng, compactness)
        
        if chain is not None and len(chain) == n:
            initial_chains.append(chain)
    
    # Also try a simple spiral
    def _simple_spiral(n):
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        direction = (1, 0)
        for i in range(1, n):
            found = False
            for _ in range(4):
                x, y = chain[-1]
                dx, dy = direction
                new_pos = (x + dx, y + dy)
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied[new_pos] = i
                    found = True
                    break
                # Turn right
                direction = (dy, -dx)
            if not found:
                return None
        return chain
    
    spiral = _simple_spiral(n)
    if spiral is not None:
        initial_chains.append(spiral)
    
    # Evaluate all initial chains and refine the best ones
    for chain in initial_chains:
        if time.time() > deadline - 20.0:
            break
        
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
    
    # Refine the best chain with local search
    if best_chain is not None and time.time() < deadline - 15.0:
        refined_chain, refined_energy = _local_search(sequence, best_chain, rng, max_iter=2000, time_limit=10.0)
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain
    
    # Try simulated annealing on the best chain
    if best_chain is not None and time.time() < deadline - 10.0:
        sa_chain, sa_energy = _simulated_annealing(sequence, best_chain, rng, time_limit=5.0)
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = sa_chain
    
    # Final local search polish
    if best_chain is not None and time.time() < deadline - 5.0:
        final_chain, final_energy = _local_search(sequence, best_chain, rng, max_iter=1000, time_limit=5.0)
        if final_energy < best_energy:
            best_energy = final_energy
            best_chain = final_chain
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
