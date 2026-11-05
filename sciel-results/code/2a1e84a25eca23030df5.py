"""
HP lattice protein folding using a Hamiltonian path pre-computation strategy.

For each sequence length, we pre-compute a set of low-contact-energy self-avoiding walks
(geometric skeletons) via simulated annealing on the lattice topology. At runtime, we use
a fast greedy assignment to map the specific H/P sequence onto these pre-optimized
geometric skeletons to guarantee valid, high-contact conformations.
"""

import random
import math
from functools import lru_cache

# EVOLVE-BLOCK-START

# Pre-computed skeletons for each sequence length
# We'll generate them on first use and cache them
_skeleton_cache = {}

def _initial_walk(n, rng):
    """Generate a simple space-filling walk (spiral-like)."""
    walk = [(0, 0)]
    occupied = {(0, 0)}
    x, y = 0, 0
    dx, dy = 1, 0
    for i in range(1, n):
        moved = False
        for _ in range(4):
            nx, ny = x + dx, y + dy
            if (nx, ny) not in occupied:
                x, y = nx, ny
                occupied.add((x, y))
                walk.append((x, y))
                moved = True
                break
            else:
                dx, dy = -dy, dx
        if not moved:
            # Backtrack
            if len(walk) >= 2:
                px, py = walk.pop()
                occupied.discard((px, py))
                x, y = walk[-1]
                # Try random direction
                dirs = [(1,0),(-1,0),(0,1),(0,-1)]
                rng.shuffle(dirs)
                for dx, dy in dirs:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) not in occupied:
                        x, y = nx, ny
                        occupied.add((x, y))
                        walk.append((x, y))
                        break
                else:
                    return None
            else:
                return None
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
    Perform a pivot move with O(1) amortized validation using an occupied set.
    Returns (new_walk, new_occupied) if valid, else (None, None).
    """
    n = len(walk)
    if n < 4:
        return None, None
    
    # Choose pivot position (at least 2 so we have a head and tail)
    # Bias toward shorter tails (more recent positions) for better mixing
    i = rng.randint(2, n - 1)
    
    # The tail is walk[i:], pivot point is walk[i-1]
    pivot = walk[i-1]
    tail = walk[i:]
    
    # Pre-calculate relative coordinates of tail
    rel_tail = [(px - pivot[0], py - pivot[1]) for px, py in tail]
    
    # Define rotation functions
    def rot90(dx, dy): return (-dy, dx)
    def rot180(dx, dy): return (-dx, -dy)
    def rot270(dx, dy): return (dy, -dx)
    
    rotations = [rot90, rot180, rot270]
    rng.shuffle(rotations)
    
    for rot_func in rotations:
        # Check if the first element of the rotated tail is adjacent to pivot
        new_head_rel = rot_func(rel_tail[0][0], rel_tail[0][1])
        new_head = (pivot[0] + new_head_rel[0], pivot[1] + new_head_rel[1])
        
        # The new head must be adjacent to pivot (Manhattan dist 1)
        if abs(new_head[0] - pivot[0]) + abs(new_head[1] - pivot[1]) != 1:
            continue
        
        # Build new tail and check for collisions
        new_tail = []
        valid = True
        new_occupied_tail = set()
        
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
            
        # Check collision with head (walk[:i])
        head_set = set(walk[:i])
        
        collision = False
        for pos in new_tail:
            if pos in head_set:
                collision = True
                break
        
        if collision:
            continue
            
        # Valid move
        new_walk = walk[:i] + new_tail
        new_occupied = occupied.copy()
        for pos in tail:
            new_occupied.discard(pos)
        for pos in new_tail:
            new_occupied.add(pos)
            
        return new_walk, new_occupied

    return None, None

def _pivot_move_with_score(walk, occupied, rng, sequence):
    """
    Perform a pivot move, returning the move with the best estimated energy impact.
    Tries all valid rotations and picks the one with the most negative energy delta.
    Returns (new_walk, new_occupied, delta_energy) or (None, None, 0).
    """
    n = len(walk)
    if n < 4:
        return None, None, 0
    
    i = rng.randint(2, n - 1)
    pivot = walk[i-1]
    tail = walk[i:]
    
    rel_tail = [(px - pivot[0], py - pivot[1]) for px, py in tail]
    
    def rot90(dx, dy): return (-dy, dx)
    def rot180(dx, dy): return (-dx, -dy)
    def rot270(dx, dy): return (dy, -dx)
    
    rotations = [rot90, rot180, rot270]
    rng.shuffle(rotations)
    
    best_move = None
    best_delta = 0
    
    head_set = set(walk[:i])
    
    for rot_func in rotations:
        new_head_rel = rot_func(rel_tail[0][0], rel_tail[0][1])
        new_head = (pivot[0] + new_head_rel[0], pivot[1] + new_head_rel[1])
        
        if abs(new_head[0] - pivot[0]) + abs(new_head[1] - pivot[1]) != 1:
            continue
        
        new_tail = []
        valid = True
        new_occupied_tail = set()
        
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
            
        collision = False
        for pos in new_tail:
            if pos in head_set:
                collision = True
                break
        
        if collision:
            continue
        
        # Valid move - compute energy delta
        new_walk = walk[:i] + new_tail
        delta = _energy_delta_for_move(sequence, walk, new_walk, i)
        
        if best_move is None or delta < best_delta:
            best_delta = delta
            new_occupied = occupied.copy()
            for pos in tail:
                new_occupied.discard(pos)
            for pos in new_tail:
                new_occupied.add(pos)
            best_move = (new_walk, new_occupied)
    
    if best_move is None:
        return None, None, 0
    
    return best_move[0], best_move[1], best_delta

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

def _generate_skeletons(n, num_skeletons=30, seed=42):
    """
    Generate a set of self-avoiding walks of length n using simulated annealing.
    Each skeleton is a list of (x, y) coordinates optimized for high geometric contacts.
    """
    if n <= 1:
        return [[(0, 0)]]
    
    rng = random.Random(seed + n)
    skeletons = []
    seen = set()
    
    # Generate initial skeletons using various strategies
    for s in range(num_skeletons):
        walk = None
        if s % 4 == 0:
            walk = _initial_walk(n, rng)
        elif s % 4 == 1:
            walk = _random_walk(n, rng)
        elif s % 4 == 2:
            walk = _initial_walk(n, random.Random(seed + n + s))
        else:
            # Try a different random seed for variety
            walk = _random_walk(n, random.Random(seed + n + s * 7))
        
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
        
        # Simulated annealing to optimize compactness
        walk = _anneal_walk(walk, n, rng, iterations=4000)
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

def _energy_delta_for_move(sequence, old_chain, new_chain, pivot_idx):
    """
    Compute the energy difference when applying a pivot move at pivot_idx.
    Only residues near the pivot (within 2 of the tail boundary) can have changed contacts.
    Returns new_energy - old_energy (negative is better).
    """
    n = len(old_chain)
    # The tail starts at pivot_idx, so positions pivot_idx..n-1 moved.
    # Contacts involving these positions may change.
    # We need to check positions from max(0, pivot_idx-2) to n-1.
    
    old_pos_to_idx = {tuple(p): i for i, p in enumerate(old_chain)}
    new_pos_to_idx = {tuple(p): i for i, p in enumerate(new_chain)}
    
    # Build set of H positions for quick lookup
    old_h_set = set(i for i in range(n) if sequence[i] == 'H')
    new_h_set = set(i for i in range(n) if sequence[i] == 'H')  # same as old_h_set
    
    # Contacts that could change: any contact where at least one endpoint is in the moved tail
    # or is adjacent to the moved tail (within 1 step)
    
    # Efficient approach: compute contacts for the affected region
    # A contact (i, j) with i < j is affected if i >= pivot_idx-2 or j >= pivot_idx
    # But we need to be careful: a contact between two head residues is unchanged.
    
    # Let's just compute the full energy delta by checking all contacts involving moved positions
    # For each position k in the moved tail (pivot_idx to n-1), check its 4 neighbors
    # in both old and new configurations
    
    delta = 0
    
    # Contacts LOST: contacts in old config where at least one endpoint is in tail
    for k in range(pivot_idx, n):
        x, y = old_chain[k]
        if sequence[k] != 'H':
            continue
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            j = old_pos_to_idx.get((x+dx, y+dy))
            if j is not None and j > k + 1 and sequence[j] == 'H':
                delta += 1  # lost contact (energy increases by 1, i.e., delta_energy = +1)
    
    # Contacts GAINED: contacts in new config where at least one endpoint is in tail
    for k in range(pivot_idx, n):
        x, y = new_chain[k]
        if sequence[k] != 'H':
            continue
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            j = new_pos_to_idx.get((x+dx, y+dy))
            if j is not None and j > k + 1 and sequence[j] == 'H':
                delta -= 1  # gained contact (energy decreases by 1)
    
    return delta

def _sequence_aware_local_search(sequence, chain, rng, iterations=2000):
    """
    Local search on a specific conformation: use pivot moves to minimize HP energy
    for the given sequence. Uses incremental energy delta for speed.
    """
    n = len(sequence)
    if n < 4:
        return chain
    
    best_chain = [list(p) for p in chain]
    best_energy = _compute_energy(sequence, best_chain)
    current_chain = [list(p) for p in chain]
    current_occupied = set(tuple(p) for p in current_chain)
    current_energy = best_energy
    
    T = 1.5
    T_min = 0.05
    cooling = (T_min / T) ** (1.0 / max(iterations, 1))
    
    no_improve = 0
    max_no_improve = iterations // 5
    
    for _ in range(iterations):
        # Use scored pivot move that picks the best rotation
        walk_tuples = [(p[0], p[1]) for p in current_chain]
        new_chain, new_occupied, delta = _pivot_move_with_score(walk_tuples, current_occupied, rng, sequence)
        
        if new_chain is None:
            T *= cooling
            if T < T_min:
                break
            continue
        
        new_chain = [list(p) for p in new_chain]
        new_energy = current_energy + delta
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_chain = new_chain
            current_occupied = new_occupied
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = [list(p) for p in current_chain]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        T *= cooling
        if T < T_min or no_improve > max_no_improve:
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
    2. Sequence-aware local search on each skeleton with incremental energy
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
    for idx, skeleton in enumerate(skeletons):
        if skeleton is None:
            continue
        if len(skeleton) != n:
            continue
        
        chain = [list(p) for p in skeleton]
        energy = _compute_energy(sequence, chain)
        
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
        
        # Apply sequence-aware local search to improve this skeleton
        # Use more iterations for earlier skeletons (they're more diverse)
        iters = 3000 if idx < 10 else 2000
        rng = random.Random(99999 + n + idx * 1000)
        improved = _sequence_aware_local_search(sequence, chain, rng, iterations=iters)
        improved_energy = _compute_energy(sequence, improved)
        
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved
    
    # Also try direct growth as a complement
    rng = random.Random(12345 + n)
    grown_chain = _try_direct_growth(sequence, dimension, rng, max_restarts=50)
    
    if grown_chain is not None:
        grown_energy = _compute_energy(sequence, grown_chain)
        if best_chain is None or grown_energy < best_energy:
            best_chain = grown_chain
            best_energy = grown_energy
        
        # Also apply local search to the grown chain
        rng2 = random.Random(77777 + n)
        improved_grown = _sequence_aware_local_search(sequence, grown_chain, rng2, iterations=2000)
        improved_energy = _compute_energy(sequence, improved_grown)
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_grown
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [[i, 0] for i in range(n)]
    
    return best_chain

# EVOLVE-BLOCK-END
