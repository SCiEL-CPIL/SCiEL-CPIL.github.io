"""
HP lattice protein folding with diverse initialization and multi-stage refinement.
"""
# EVOLVE-BLOCK-START
import random
import time

def _neighbours(point):
    """Yield the 4 lattice neighbors of a 2D point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    """Compute HP energy: -number of non-consecutive H-H contacts."""
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] != 'H':
            continue
        for q in _neighbours(p):
            j = pos_to_idx.get(q)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts

def _generate_spiral(n, rng):
    """Generate a spiral-shaped initial conformation."""
    if n <= 1:
        return [(0, 0)]
    chain = [(0, 0)]
    occupied = {(0, 0)}
    # Spiral directions: right, down, left, up (cycling)
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    di = 0
    steps_in_dir = 1
    steps_done = 0
    for i in range(1, n):
        # Try to continue in current direction
        dx, dy = directions[di]
        new_pos = (chain[-1][0] + dx, chain[-1][1] + dy)
        if new_pos in occupied:
            # Turn right
            di = (di + 1) % 4
            if steps_done >= steps_in_dir:
                steps_in_dir += 1
                steps_done = 0
            dx, dy = directions[di]
            new_pos = (chain[-1][0] + dx, chain[-1][1] + dy)
            if new_pos in occupied:
                # Try other directions
                found = False
                for d in range(4):
                    ndi = (di + d) % 4
                    ddx, ddy = directions[ndi]
                    np_ = (chain[-1][0] + ddx, chain[-1][1] + ddy)
                    if np_ not in occupied:
                        new_pos = np_
                        di = ndi
                        steps_done = 0
                        found = True
                        break
                if not found:
                    # Can't place, return what we have
                    break
            else:
                steps_done = 0
        else:
            steps_done += 1
        chain.append(new_pos)
        occupied.add(new_pos)
    # If we couldn't place all, extend linearly
    while len(chain) < n:
        last = chain[-1]
        placed = False
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            np_ = (last[0] + dx, last[1] + dy)
            if np_ not in occupied:
                chain.append(np_)
                occupied.add(np_)
                placed = True
                break
        if not placed:
            # Place far away
            chain.append((len(chain) * 2, 0))
            occupied.add((len(chain) * 2, 0))
    return chain

def _grow_random(sequence, dimension, rng, bias=0.3):
    """Randomized greedy growth with contact bias."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * (1 + bias) + 0.1 * free + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        options.sort(key=lambda x: -x[0])
        # Pick from top options with some randomness
        top_k = min(3, len(options))
        site = options[rng.randint(0, top_k - 1)][1]
        occupied[site] = i
        chain.append(site)
    return chain

def _pivot_move(chain, i, direction, occupied):
    """Try to pivot the chain at position i in the given direction.
    Returns new chain if valid, None otherwise."""
    if i <= 0 or i >= len(chain) - 1:
        return None
    # The segment from i to end needs to be moved
    # For simplicity, only try moving single residues (i is the moving position)
    # Actually, let's do a proper pivot: move residue i to a new position
    old_pos = chain[i]
    new_pos = (old_pos[0] + direction[0], old_pos[1] + direction[1])
    
    # Check if new position is free
    if new_pos in occupied and occupied[new_pos] != i:
        return None
    
    # Check that the move doesn't create a collision with adjacent chain positions
    # The residue i-1 and i+1 must not be at new_pos (they're already checked by occupied)
    # But we need to ensure the path from i-1 to new_pos to i+1 is valid
    # Since we're only moving one residue, we just need new_pos to be free and adjacent to i-1
    # Actually, for a simple move, new_pos just needs to be free and adjacent to old_pos
    # But we also need it to be adjacent to chain[i-1] or chain[i+1] to maintain connectivity
    # For a single-residue move, we require new_pos to be adjacent to chain[i-1] OR chain[i+1]
    # This is a restriction. Let's do a proper pivot instead.
    
    # For now, let's just do single-residue moves where the new position is adjacent to the previous position
    if i > 0:
        prev = chain[i - 1]
        if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
            return None
    
    # Check no collision
    if new_pos in occupied:
        occ_idx = occupied[new_pos]
        if occ_idx != i:
            return None
    
    # Perform the move
    del occupied[old_pos]
    chain[i] = new_pos
    occupied[new_pos] = i
    return chain

def _local_search(sequence, chain, max_iters=500, rng=None):
    """Simple local search using single-residue moves."""
    if rng is None:
        rng = random.Random()
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for it in range(max_iters):
        i = rng.randint(1, len(chain) - 2)  # Don't move ends
        old_pos = chain[i]
        moved = False
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            new_pos = (old_pos[0] + dx, old_pos[1] + dy)
            # Check if new position is free
            if new_pos in occupied and occupied[new_pos] != i:
                continue
            # Check adjacency to previous
            if i > 0:
                prev = chain[i - 1]
                if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
                    continue
            # Try the move
            del occupied[old_pos]
            chain[i] = new_pos
            occupied[new_pos] = i
            new_energy = _energy(sequence, chain)
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
                current_energy = new_energy
                moved = True
                break
            else:
                # Revert
                del occupied[new_pos]
                chain[i] = old_pos
                occupied[old_pos] = i
        if not moved:
            # No improving move found, try a random move to escape
            i = rng.randint(1, len(chain) - 2)
            old_pos = chain[i]
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                new_pos = (old_pos[0] + dx, old_pos[1] + dy)
                if new_pos in occupied and occupied[new_pos] != i:
                    continue
                if i > 0:
                    prev = chain[i - 1]
                    if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
                        continue
                del occupied[old_pos]
                chain[i] = new_pos
                occupied[new_pos] = i
                current_energy = _energy(sequence, chain)
                break
    return best_chain, best_energy

def _simulated_annealing(sequence, chain, max_iters=1000, t_start=1.0, t_end=0.01, rng=None):
    """Simulated annealing for refinement."""
    if rng is None:
        rng = random.Random()
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for it in range(max_iters):
        t = t_start * (t_end / t_start) ** (it / max_iters)
        i = rng.randint(1, len(chain) - 2)
        old_pos = chain[i]
        moved = False
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            new_pos = (old_pos[0] + dx, old_pos[1] + dy)
            if new_pos in occupied and occupied[new_pos] != i:
                continue
            if i > 0:
                prev = chain[i - 1]
                if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
                    continue
            # Try the move
            del occupied[old_pos]
            chain[i] = new_pos
            occupied[new_pos] = i
            new_energy = _energy(sequence, chain)
            delta = new_energy - current_energy
            if delta < 0 or rng.random() < __import__('math').exp(-delta / max(t, 1e-10)):
                current_energy = new_energy
                moved = True
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                break
            else:
                # Revert
                del occupied[new_pos]
                chain[i] = old_pos
                occupied[old_pos] = i
        if not moved:
            continue
    return best_chain, best_energy

def fold(sequence, dimension):
    """Find the best conformation using diverse initialization and refinement."""
    start_time = time.time()
    deadline = start_time + 180  # Leave 30s margin
    
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    # Generate diverse initial conformations
    initial_chains = []
    
    # Spiral templates
    for seed in range(5):
        rng = random.Random(12345 + seed * 1000 + n)
        chain = _generate_spiral(n, rng)
        if len(chain) == n:
            initial_chains.append(chain)
    
    # Randomized greedy growth
    for seed in range(10):
        rng = random.Random(54321 + seed * 777 + n)
        bias = 0.2 + 0.1 * (seed % 5)
        chain = _grow_random(sequence, dimension, rng, bias)
        if chain is not None and len(chain) == n:
            initial_chains.append(chain)
    
    # Random walks
    for seed in range(5):
        rng = random.Random(98765 + seed * 333 + n)
        chain = [(0, 0)]
        occupied = {(0, 0)}
        for i in range(1, n):
            options = []
            for site in _neighbours(chain[-1]):
                if site not in occupied:
                    options.append(site)
            if not options:
                break
            site = rng.choice(options)
            chain.append(site)
            occupied.add(site)
        if len(chain) == n:
            initial_chains.append(chain)
    
    if not initial_chains:
        # Fallback: straight line
        return [[i, 0] for i in range(n)]
    
    # Evaluate initial conformations
    scored_chains = []
    for chain in initial_chains:
        e = _energy(sequence, chain)
        scored_chains.append((e, chain))
    scored_chains.sort(key=lambda x: x[0])
    
    best_chain = scored_chains[0][1]
    best_energy = scored_chains[0][0]
    
    # Refine the best initial conformations
    for e, chain in scored_chains[:min(5, len(scored_chains))]:
        if time.time() > deadline - 30:
            break
        
        # Local search
        refined, e_ls = _local_search(sequence, chain, max_iters=200, rng=random.Random())
        if e_ls < best_energy:
            best_energy = e_ls
            best_chain = refined
        
        # Simulated annealing
        if time.time() > deadline - 15:
            break
        refined, e_sa = _simulated_annealing(sequence, refined, max_iters=300, t_start=0.5, t_end=0.01, rng=random.Random())
        if e_sa < best_energy:
            best_energy = e_sa
            best_chain = refined
        
        # Final hill-climb polish
        if time.time() > deadline - 5:
            break
        refined, e_hc = _local_search(sequence, refined, max_iters=100, rng=random.Random())
        if e_hc < best_energy:
            best_energy = e_hc
            best_chain = refined
    
    # Ensure valid output
    result = [list(p) for p in best_chain]
    return result
# EVOLVE-BLOCK-END
