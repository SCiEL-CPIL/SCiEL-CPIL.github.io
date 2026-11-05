"""
HP lattice protein folding using a Hamiltonian path pre-computation strategy.

For each sequence length, we pre-compute a set of low-contact-energy self-avoiding walks
(geometric skeletons) via simulated annealing on the lattice topology. At runtime, we use
a fast greedy assignment to map the specific H/P sequence onto these pre-optimized
geometric skeletons to guarantee valid, high-contact conformations.
"""

import random
import math
import time
from functools import lru_cache

# EVOLVE-BLOCK-START

# Pre-computed skeletons for each sequence length
# We'll generate them on first use and cache them
_skeleton_cache = {}
_START_TIME = time.time()
_TIME_BUDGET = 100  # Leave 20s buffer for evaluation overhead

def _initial_walk(n, rng):
    """Generate a deterministic spiral walk for fast initialization."""
    if n == 1:
        return [(0, 0)]
    
    walk = [(0, 0)]
    occupied = {(0, 0)}
    
    # Spiral pattern: right, up, left, down, repeat with increasing steps
    dx, dy = 1, 0
    steps = 1
    step_count = 0
    turn_count = 0
    
    while len(walk) < n:
        for _ in range(steps):
            if len(walk) >= n:
                break
            x, y = walk[-1]
            nx, ny = x + dx, y + dy
            if (nx, ny) not in occupied:
                walk.append((nx, ny))
                occupied.add((nx, ny))
            else:
                # Spiral is blocked, try a simple random walk continuation
                while len(walk) < n:
                    last = walk[-1]
                    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
                    rng.shuffle(dirs)
                    moved = False
                    for ddx, ddy in dirs:
                        nnx, nny = last[0] + ddx, last[1] + ddy
                        if (nnx, nny) not in occupied:
                            walk.append((nnx, nny))
                            occupied.add((nnx, nny))
                            moved = True
                            break
                    if not moved:
                        return None
                break
        # Turn clockwise
        dx, dy = -dy, dx
        turn_count += 1
        if turn_count % 2 == 0:
            steps += 1
    
    return walk

def _random_walk(n, rng):
    """Generate a random self-avoiding walk of length n."""
    walk = [(0, 0)]
    occupied = {(0, 0)}
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for _ in range(1, n):
        last = walk[-1]
        rng.shuffle(directions)
        moved = False
        for dx, dy in directions:
            nx, ny = last[0] + dx, last[1] + dy
            if (nx, ny) not in occupied:
                walk.append((nx, ny))
                occupied.add((nx, ny))
                moved = True
                break
        if not moved:
            return None
    
    return walk

def _walk_contacts(walk):
    """Compute the number of geometric contacts (adjacent non-consecutive pairs) for a walk."""
    if len(walk) < 3:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(walk)}
    contacts = 0
    for i, (x, y) in enumerate(walk):
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            j = pos_to_idx.get((x+dx, y+dy))
            if j is not None and j > i + 1:
                contacts += 1
    return contacts

def _pivot_move_fast(walk, occupied, rng):
    """
    Perform a pivot move with fast validation.
    Returns (new_walk, new_occupied) if valid, else (None, None).
    """
    n = len(walk)
    if n < 4:
        return None, None
    
    # Choose pivot position (at least 2 so we have a head and tail)
    i = rng.randint(2, n - 1)
    
    # The tail is walk[i:], pivot point is walk[i-1]
    pivot = walk[i-1]
    tail = walk[i:]
    
    # Pre-calculate relative coordinates of tail
    rel_tail = [(px - pivot[0], py - pivot[1]) for px, py in tail]
    
    # Define rotation functions as tuples of (dx, dy) transformations
    # rot90: (dx, dy) -> (-dy, dx)
    # rot180: (dx, dy) -> (-dx, -dy)  
    # rot270: (dx, dy) -> (dy, -dx)
    
    rotations = [
        lambda dx, dy: (-dy, dx),   # 90
        lambda dx, dy: (-dx, -dy),  # 180
        lambda dx, dy: (dy, -dx),   # 270
    ]
    rng.shuffle(rotations)
    
    # Build head set once (positions not in tail)
    tail_set = set(tail)
    head_set = occupied - tail_set
    
    for rot_func in rotations:
        # Check if the first element of the rotated tail is adjacent to pivot
        new_head_rel = rot_func(rel_tail[0][0], rel_tail[0][1])
        new_head = (pivot[0] + new_head_rel[0], pivot[1] + new_head_rel[1])
        
        # The new head must be adjacent to pivot (Manhattan dist 1)
        if abs(new_head[0] - pivot[0]) + abs(new_head[1] - pivot[1]) != 1:
            continue
        
        # Build new tail and check for collisions
        new_tail = []
        new_occupied_tail = set()
        valid = True
        
        for dx, dy in rel_tail:
            ndx, ndy = rot_func(dx, dy)
            pos = (pivot[0] + ndx, pivot[1] + ndy)
            if pos in new_occupied_tail:
                valid = False
                break
            new_occupied_tail.add(pos)
            new_tail.append(pos)
        
        if not valid:
            continue
        
        # Check collision with head
        if new_occupied_tail & head_set:
            continue
        
        # If we reach here, the move is valid
        new_walk = walk[:i] + new_tail
        new_occupied = head_set | new_occupied_tail
        
        return new_walk, new_occupied
    
    return None, None

def _anneal_walk(walk, n, rng, iterations=5000):
    """Optimize a walk using simulated annealing with pivot moves to maximize contacts."""
    if walk is None or len(walk) != n:
        return None
    
    best_walk = list(walk)
    best_contacts = _walk_contacts(best_walk)
    current_walk = list(walk)
    current_occupied = set(current_walk)
    current_contacts = best_contacts
    
    T = 2.0
    T_min = 0.01
    cooling = (T_min / T) ** (1.0 / max(iterations, 1))
    
    for _ in range(iterations):
        # Check time budget
        if time.time() - _START_TIME > _TIME_BUDGET * 0.6:
            break
        
        new_walk, new_occupied = _pivot_move_fast(current_walk, current_occupied, rng)
        if new_walk is None:
            T *= cooling
            if T < T_min:
                break
            continue
        
        new_contacts = _walk_contacts(new_walk)
        delta = new_contacts - current_contacts
        
        if delta > 0 or rng.random() < math.exp(delta / T):
            current_walk = new_walk
            current_occupied = new_occupied
            current_contacts = new_contacts
            if current_contacts > best_contacts:
                best_contacts = current_contacts
                best_walk = list(current_walk)
        
        T *= cooling
        if T < T_min:
            break
    
    return best_walk

def _generate_skeletons(n, num_skeletons=15, seed=42):
    """
    Generate a set of self-avoiding walks of length n using simulated annealing.
    Each skeleton is a list of (x, y) coordinates optimized for high geometric contacts.
    Reduced number of skeletons and iterations for speed.
    """
    if n <= 1:
        return [[(0, 0)]]
    
    rng = random.Random(seed + n)
    skeletons = []
    seen = set()
    
    # Generate initial skeletons using various strategies
    for s in range(num_skeletons):
        # Check time budget
        if time.time() - _START_TIME > _TIME_BUDGET * 0.5:
            break
        
        walk = None
        if s % 2 == 0:
            walk = _initial_walk(n, rng)
        else:
            walk = _random_walk(n, rng)
        
        if walk is None:
            continue
        
        # Normalize
        min_x = min(p[0] for p in walk)
        min_y = min(p[1] for p in walk)
        walk = [(p[0] - min_x, p[1] - min_y) for p in walk]
        
        # Check if we've seen this shape before
        shape = tuple(walk)
        if shape in seen:
            continue
        seen.add(shape)
        
        # Simulated annealing to optimize compactness - reduced iterations
        walk = _anneal_walk(walk, n, rng, iterations=1500)
        if walk is not None and len(walk) == n:
            min_x = min(p[0] for p in walk)
            min_y = min(p[1] for p in walk)
            walk = [(p[0] - min_x, p[1] - min_y) for p in walk]
            skeletons.append(walk)
        
        if len(skeletons) >= num_skeletons:
            break
    
    return skeletons if skeletons else [None]

def _get_skeletons(n):
    """Get pre-computed skeletons for sequence length n."""
    if n not in _skeleton_cache:
        _skeleton_cache[n] = _generate_skeletons(n)
    return _skeleton_cache[n]

def _compute_energy(sequence, chain):
    """Compute the HP contact energy of a conformation."""
    if len(chain) < 3:
        return 0
    
    pos_to_idx = {tuple(p): i for i, p in enumerate(chain)}
    contacts = 0
    
    for i, (x, y) in enumerate(chain):
        if sequence[i] != 'H':
            continue
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            j = pos_to_idx.get((x+dx, y+dy))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    
    return -contacts

def _sequence_aware_local_search(sequence, chain, rng, iterations=2000):
    """
    Local search on a specific conformation: use pivot moves to minimize HP energy
    for the given sequence.
    """
    n = len(sequence)
    if n < 4:
        return chain
    
    best_chain = [list(p) for p in chain]
    best_energy = _compute_energy(sequence, best_chain)
    current_chain = [list(p) for p in chain]
    current_occupied = set(tuple(p) for p in current_chain)
    current_energy = best_energy
    
    T = 1.0
    T_min = 0.01
    cooling = (T_min / T) ** (1.0 / max(iterations, 1))
    
    for _ in range(iterations):
        # Check time budget
        if time.time() - _START_TIME > _TIME_BUDGET:
            break
        
        new_chain, new_occupied = _pivot_move_fast([(p[0], p[1]) for p in current_chain], current_occupied, rng)
        if new_chain is None:
            T *= cooling
            if T < T_min:
                break
            continue
        
        new_chain = [list(p) for p in new_chain]
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_chain = new_chain
            current_occupied = new_occupied
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = [list(p) for p in current_chain]
        
        T *= cooling
        if T < T_min:
            break
    
    return best_chain

def _try_direct_growth(sequence, dimension, rng, max_restarts=50):
    """
    Try direct randomized greedy growth as a fallback.
    """
    start = (0,) * dimension
    best_chain = None
    best_energy = 0
    
    for restart in range(max_restarts):
        chain = [start]
        occupied = {start: 0}
        
        for i in range(1, len(sequence)):
            last = chain[-1]
            options = []
            
            for axis in range(dimension):
                for step in (-1, 1):
                    q = list(last)
                    q[axis] += step
                    q = tuple(q)
                    if q in occupied:
                        continue
                    
                    # Score: prefer positions that create H-H contacts
                    gain = 0
                    if sequence[i] == 'H':
                        for ax in range(dimension):
                            for st in (-1, 1):
                                nq = list(q)
                                nq[ax] += st
                                nq = tuple(nq)
                                if nq in occupied:
                                    j = occupied[nq]
                                    if j < i - 1 and sequence[j] == 'H':
                                        gain += 1
                    
                    # Add some randomness
                    score = gain + rng.random() * 0.5
                    options.append((score, q))
            
            if not options:
                break
            
            # Pick the best option
            options.sort(key=lambda x: -x[0])
            # Sometimes pick a random option for diversity
            if rng.random() < 0.1 and len(options) > 1:
                idx = rng.randint(0, min(3, len(options)-1))
                site = options[idx][1]
            else:
                site = options[0][1]
            
            occupied[site] = i
            chain.append(site)
        
        if len(chain) == len(sequence):
            energy = _compute_energy(sequence, chain)
            if best_chain is None or energy < best_energy:
                best_chain = chain
                best_energy = energy
    
    if best_chain is None:
        return None
    
    return [list(p) for p in best_chain]

def fold(sequence, dimension):
    """
    Fold an HP sequence into a low-energy conformation on the square lattice.
    
    Uses a combination of:
    1. Pre-computed geometric skeletons (Hamiltonian paths) optimized for compactness
    2. Sequence-aware local search on each skeleton
    3. Direct randomized greedy growth as a fallback
    4. Returns the best conformation found
    """
    n = len(sequence)
    
    if n <= 1:
        return [[0] * dimension for _ in range(n)]
    
    if dimension != 2:
        # Fallback for non-2D: just use greedy growth
        rng = random.Random(12345 + n)
        return _try_direct_growth(sequence, dimension, rng)
    
    # Get pre-computed skeletons
    skeletons = _get_skeletons(n)
    
    best_chain = None
    best_energy = 0
    
    # Try each skeleton with sequence-aware local search
    for skeleton in skeletons:
        # Check time budget
        if time.time() - _START_TIME > _TIME_BUDGET:
            break
        
        if skeleton is None:
            continue
        if len(skeleton) != n:
            continue
        
        chain = [list(p) for p in skeleton]
        energy = _compute_energy(sequence, chain)
        
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
        
        # Apply sequence-aware local search to improve this skeleton - reduced iterations
        rng = random.Random(99999 + n + hash(tuple(skeleton)) % 10000)
        improved = _sequence_aware_local_search(sequence, chain, rng, iterations=1000)
        improved_energy = _compute_energy(sequence, improved)
        
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved
    
    # Also try direct growth as a complement if time permits
    if time.time() - _START_TIME < _TIME_BUDGET * 0.7:
        rng = random.Random(12345 + n)
        grown_chain = _try_direct_growth(sequence, dimension, rng, max_restarts=30)
        
        if grown_chain is not None:
            grown_energy = _compute_energy(sequence, grown_chain)
            if best_chain is None or grown_energy < best_energy:
                best_chain = grown_chain
                best_energy = grown_energy
            
            # Also apply local search to the grown chain if time permits
            if time.time() - _START_TIME < _TIME_BUDGET * 0.85:
                rng2 = random.Random(77777 + n)
                improved_grown = _sequence_aware_local_search(sequence, grown_chain, rng2, iterations=800)
                improved_energy = _compute_energy(sequence, improved_grown)
                if improved_energy < best_energy:
                    best_energy = improved_energy
                    best_chain = improved_grown
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [[i, 0] for i in range(n)]
    
    return best_chain

# EVOLVE-BLOCK-END
