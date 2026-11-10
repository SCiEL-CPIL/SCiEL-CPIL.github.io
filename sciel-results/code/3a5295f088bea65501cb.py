"""HP lattice protein folding using Monte Carlo End-to-Tail (E2T) moves with simulated annealing."""
import random
import time
import math


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute the HP energy: minus the number of H-H non-bonded contacts."""
    n = len(sequence)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _is_valid_walk(chain, dimension):
    """Check that the chain is self-avoiding with unit steps."""
    n = len(chain)
    if n == 0:
        return True
    occupied = set()
    for i, p in enumerate(chain):
        if p in occupied:
            return False
        occupied.add(p)
        if i > 0:
            prev = chain[i - 1]
            dist = sum(abs(p[k] - prev[k]) for k in range(dimension))
            if dist != 1:
                return False
    return True


def _initial_spiral(sequence, dimension):
    """Generate an initial self-avoiding walk using a spiral/snake pattern."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)]
    
    # Try a simple snake pattern first
    chain = [(0, 0)]
    occupied = {(0, 0)}
    direction = (0, 1)  # start going right
    dir_index = 0
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    for i in range(1, n):
        placed = False
        for try_dir in range(4):
            d = dirs[(dir_index + try_dir) % 4]
            new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1])
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied.add(new_pos)
                dir_index = (dir_index + try_dir) % 4
                placed = True
                break
        if not placed:
            # Fallback: just go in a straight line (shouldn't happen for reasonable n)
            chain = [(i, 0) for i in range(n)]
            break
    
    return chain


def _grow_tail(sequence, chain, cut_point, dimension, rng, bias=0.5):
    """Regrow the tail from cut_point onward, preferring H-H contacts."""
    n = len(sequence)
    if cut_point >= n:
        return list(chain)
    
    new_chain = list(chain[:cut_point])
    occupied = set(chain[:cut_point])
    
    for i in range(cut_point, n):
        last = new_chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score: prefer H-H contacts
            score = rng.random()
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        # Find which residue is at q
                        for j in range(len(new_chain)):
                            if new_chain[j] == q and j < i - 1 and sequence[j] == 'H':
                                score += bias
                                break
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option with some randomness
        options.sort(key=lambda x: -x[0])
        # Pick from top few with some probability
        k = min(3, len(options))
        if rng.random() < 0.7:
            site = options[0][1]
        else:
            site = options[rng.randint(0, k - 1)][1]
        
        new_chain.append(site)
        occupied.add(site)
    
    return new_chain


def _e2t_move(sequence, chain, dimension, rng, cut_min=None, cut_max=None):
    """Perform an End-to-Tail move: pick a cut point and regrow the tail."""
    n = len(sequence)
    if n < 5:
        return None
    
    if cut_min is None:
        cut_min = max(1, n // 4)
    if cut_max is None:
        cut_max = n - 1
    
    cut_point = rng.randint(cut_min, cut_max)
    new_chain = _grow_tail(sequence, chain, cut_point, dimension, rng)
    return new_chain


def _local_refine(sequence, chain, dimension, rng, max_steps=1000):
    """Local refinement using single-residue moves (pivot-like)."""
    n = len(sequence)
    current = list(chain)
    current_energy = _compute_energy(sequence, current)
    
    for step in range(max_steps):
        # Pick a random internal residue to pivot
        i = rng.randint(1, n - 2)
        # Try rotating the segment from i to i+2
        # Actually, let's do a simpler move: try to move a single residue
        # But that's tricky. Let's use E2T with small tails instead.
        
        # Use E2T with tail length 2-5
        tail_len = rng.randint(2, min(5, n - 1))
        cut_point = n - tail_len
        new_chain = _grow_tail(sequence, current, cut_point, dimension, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        if new_energy <= current_energy:
            current = new_chain
            current_energy = new_energy
    
    return current, current_energy


def fold(sequence, dimension):
    """Fold the HP sequence using E2T moves with simulated annealing."""
    start_time = time.time()
    deadline = start_time + 180  # Leave 30s safety margin before 210s kill
    
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = 0  # We want to minimize energy (more negative is better)
    
    # Generate initial conformation
    initial = _initial_spiral(sequence, dimension)
    if not _is_valid_walk(initial, dimension):
        # Fallback: straight line
        initial = [(i, 0) for i in range(n)]
    
    best_chain = list(initial)
    best_energy = _compute_energy(sequence, best_chain)
    
    # Simulated annealing with E2T moves
    temperature = 5.0
    cooling_rate = 0.9995
    
    current_chain = list(best_chain)
    current_energy = best_energy
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Perform E2T move
        new_chain = _e2t_move(sequence, current_chain, dimension, rng)
        
        if new_chain is None:
            # Try a smaller tail
            new_chain = _grow_tail(sequence, current_chain, n - 3, dimension, rng)
            if new_chain is None:
                new_chain = _grow_tail(sequence, current_chain, n - 2, dimension, rng)
        
        if new_chain is None:
            # Regrow from further back
            new_chain = _grow_tail(sequence, current_chain, max(1, n // 2), dimension, rng)
        
        if new_chain is None:
            # Skip this step
            temperature *= cooling_rate
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Metropolis acceptance
        if delta_e <= 0 or (temperature > 0.01 and rng.random() < math.exp(-delta_e / temperature)):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        # Cool down
        temperature *= cooling_rate
        
        # Periodically do local refinement
        if step % 500 == 0:
            refined, refined_energy = _local_refine(sequence, current_chain, dimension, rng, max_steps=200)
            if refined_energy < best_energy:
                best_energy = refined_energy
                best_chain = list(refined)
                current_chain = refined
                current_energy = refined_energy
        
        # Reset temperature occasionally to escape deep minima
        if step % 2000 == 0:
            temperature = max(temperature, 3.0)
    
    # Final local refinement
    if time.time() < deadline - 5:
        refined, refined_energy = _local_refine(sequence, best_chain, dimension, rng, max_steps=500)
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined
    
    # Convert to list of lists
    return [list(p) for p in best_chain]
