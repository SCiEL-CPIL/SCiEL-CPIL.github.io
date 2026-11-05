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

def _generate_skeletons(n, num_skeletons=50, seed=42):
    """
    Generate a set of self-avoiding walks of length n using simulated annealing.
    Each skeleton is a list of (x, y) coordinates.
    
    We use a simple spiral-like or space-filling curve approach combined with
    simulated annealing to find compact, self-avoiding walks.
    """
    if n <= 1:
        return [[(0, 0)]]
    
    rng = random.Random(seed + n)
    skeletons = []
    
    # Start with a simple snake/spiral pattern as initial guess
    def _initial_walk(n):
        """Generate a simple space-filling walk."""
        walk = [(0, 0)]
        x, y = 0, 0
        dx, dy = 1, 0
        for i in range(1, n):
            # Try to continue in current direction, else turn
            nx, ny = x + dx, y + dy
            if (nx, ny) not in set(walk) and i < n:
                x, y = nx, ny
            else:
                # Turn right
                dx, dy = -dy, dx
                nx, ny = x + dx, y + dy
                if (nx, ny) in set(walk):
                    # Turn again
                    dx, dy = -dy, dx
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in set(walk):
                        # Turn again
                        dx, dy = -dy, dx
                        nx, ny = x + dx, y + dy
                        if (nx, ny) in set(walk):
                            # Backtrack
                            x, y = x - dx, y - dy
                            # This is a fallback - just try random
                            for _ in range(100):
                                dx, dy = rng.choice([(1,0),(-1,0),(0,1),(0,-1)])
                                nx, ny = x + dx, y + dy
                                if (nx, ny) not in set(walk):
                                    break
                            else:
                                return None
            x, y = nx, ny
            walk.append((x, y))
        return walk
    
    # Generate initial skeletons
    for _ in range(num_skeletons):
        walk = _initial_walk(n)
        if walk is None:
            continue
        
        # Simulated annealing to optimize compactness
        walk = _anneal_walk(walk, n, rng, iterations=2000)
        if walk is not None and len(walk) == n:
            # Normalize to start at (0,0)
            min_x = min(p[0] for p in walk)
            min_y = min(p[1] for p in walk)
            walk = [(p[0] - min_x, p[1] - min_y) for p in walk]
            skeletons.append(walk)
    
    # If we didn't get enough, add more variations
    if len(skeletons) < num_skeletons:
        for i in range(num_skeletons - len(skeletons)):
            walk = _random_walk(n, rng)
            if walk is not None:
                min_x = min(p[0] for p in walk)
                min_y = min(p[1] for p in walk)
                walk = [(p[0] - min_x, p[1] - min_y) for p in walk]
                skeletons.append(walk)
    
    return skeletons if skeletons else [None]

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

def _walk_energy(walk):
    """Compute the number of contacts (adjacent pairs not consecutive in chain) for a walk."""
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

def _anneal_walk(walk, n, rng, iterations=2000):
    """Optimize a walk using simulated annealing to maximize contacts."""
    if walk is None or len(walk) != n:
        return None
    
    best_walk = list(walk)
    best_energy = _walk_energy(best_walk)
    current_walk = list(walk)
    current_energy = best_energy
    
    T = 1.0
    T_min = 0.01
    cooling = (T_min / T) ** (1.0 / iterations)
    
    for _ in range(iterations):
        # Pick a random position to try moving
        i = rng.randint(1, n - 2)  # Don't move first or last
        x, y = current_walk[i]
        directions = [(1,0),(-1,0),(0,1),(0,-1)]
        rng.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # Check if the new position is valid (not occupied by another part of the chain)
            # We need to check that (nx, ny) is not in the walk except at position i
            # Also, the move must not break the chain connectivity
            
            # Simple check: is (nx, ny) already occupied?
            occupied = set(current_walk)
            if (nx, ny) in occupied and (nx, ny) != (x, y):
                continue
            
            # Try the move
            new_walk = current_walk[:]
            new_walk[i] = (nx, ny)
            
            # Check self-avoidance
            if len(set(new_walk)) != n:
                continue
            
            # Check connectivity: each consecutive pair must be adjacent
            valid = True
            for j in range(n - 1):
                ax, ay = new_walk[j]
                bx, by = new_walk[j+1]
                if abs(ax - bx) + abs(ay - by) != 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            new_energy = _walk_energy(new_walk)
            delta = new_energy - current_energy
            
            if delta > 0 or rng.random() < math.exp(delta / T):
                current_walk = new_walk
                current_energy = new_energy
                if current_energy > best_energy:
                    best_energy = current_energy
                    best_walk = list(current_walk)
                break
        
        T *= cooling
        if T < T_min:
            break
    
    return best_walk

def _get_skeletons(n):
    """Get pre-computed skeletons for sequence length n."""
    if n not in _skeleton_cache:
        _skeleton_cache[n] = _generate_skeletons(n)
    return _skeleton_cache[n]

def _assign_sequence_to_skeleton(sequence, skeleton):
    """
    Assign an H/P sequence to a fixed geometric skeleton.
    The skeleton defines the positions; we just return the skeleton as the conformation.
    The energy is determined by which positions have H residues.
    
    Since the geometry is fixed, the only thing that matters is the sequence itself.
    We return the skeleton coordinates as the conformation.
    """
    if skeleton is None or len(skeleton) != len(sequence):
        return None
    
    # The conformation is just the skeleton positions
    # Energy is computed from H-H contacts in this fixed geometry
    return [list(p) for p in skeleton]

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
    2. Direct randomized greedy growth as a fallback
    3. Returns the best conformation found
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
    
    # Try each skeleton
    for skeleton in skeletons:
        if skeleton is None:
            continue
        if len(skeleton) != n:
            continue
        
        chain = [list(p) for p in skeleton]
        energy = _compute_energy(sequence, chain)
        
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    # Also try direct growth as a complement
    rng = random.Random(12345 + n)
    grown_chain = _try_direct_growth(sequence, dimension, rng, max_restarts=100)
    
    if grown_chain is not None:
        grown_energy = _compute_energy(sequence, grown_chain)
        if best_chain is None or grown_energy < best_energy:
            best_chain = grown_chain
            best_energy = grown_energy
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [[i, 0] for i in range(n)]
    
    return best_chain

# EVOLVE-BLOCK-END
