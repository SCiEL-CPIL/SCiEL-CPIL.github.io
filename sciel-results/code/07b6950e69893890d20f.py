"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math

_DEADLINE = time.time() + 315  # safety margin before 330s kill


def _neighbours(point):
    """Yield the 6 face-adjacent neighbours of a point in 3D."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the HP energy: -number of non-consecutive H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk with the given sequence."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,) * dimension]
    
    # Start with a random initial direction
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    # First step
    dirs = list(_neighbours(start))
    rng.shuffle(dirs)
    if not dirs:
        return None
    chain.append(dirs[0])
    occupied[dirs[0]] = 1
    
    for i in range(2, n):
        last = chain[-1]
        # Get unoccupied neighbours
        options = []
        for site in _neighbours(last):
            if site not in occupied:
                options.append(site)
        
        if not options:
            return None
        
        site = rng.choice(options)
        chain.append(site)
        occupied[site] = i
    
    return chain


def _local_search(sequence, chain, rng, max_steps=5000, temperature=1.0):
    """Simulated annealing local search to improve a conformation."""
    n = len(sequence)
    if n < 4:
        return chain
    
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    where = {p: i for i, p in enumerate(chain)}
    
    step = 0
    start_time = time.time()
    
    while step < max_steps:
        if time.time() > _DEADLINE - 5:
            break
        
        # Pick a random residue to try moving (not the first or last)
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        
        # Try all 6 neighbours
        neighbours = list(_neighbours(old_pos))
        rng.shuffle(neighbours)
        
        moved = False
        for new_pos in neighbours:
            if new_pos in where:
                continue
            
            # Check that moving this residue doesn't create conflicts
            # Remove old position from where
            del where[old_pos]
            
            # Check if new position conflicts with any other residue
            conflict = False
            for q in _neighbours(new_pos):
                if q in where:
                    conflict = True
                    break
            
            if conflict:
                where[old_pos] = i
                continue
            
            # Compute energy change
            # Remove old contacts involving residue i
            old_contacts = 0
            if sequence[i] == 'H':
                for q in _neighbours(old_pos):
                    j = where.get(q)
                    if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                        old_contacts += 1
            
            # Add new contacts involving residue i
            new_contacts = 0
            if sequence[i] == 'H':
                for q in _neighbours(new_pos):
                    j = where.get(q)
                    if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                        new_contacts += 1
            
            delta_e = -(new_contacts - old_contacts)
            
            # Also need to check if this move breaks any existing contacts
            # between other residues that were connected through old_pos
            # Actually, since we're only moving one residue, the only contacts
            # that change are those involving residue i itself.
            
            # Accept or reject
            if delta_e < 0 or rng.random() < math.exp(-delta_e / temperature):
                # Accept the move
                chain[i] = new_pos
                where[new_pos] = i
                current_energy += delta_e
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                
                moved = True
                break
        
        step += 1
        
        # Cool down
        temperature *= 0.999
    
    return best_chain


def _greedy_grow(sequence, dimension, rng):
    """Greedy chain growth: prefer sites with new H-H contacts."""
    start = (0,) * dimension
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
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        
        if not options:
            return None
        
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def fold(sequence, dimension):
    """Best conformation using multiple strategies and local search."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = float('inf')
    
    # Strategy 1: Greedy growth with local search
    for _ in range(30):
        if time.time() > _DEADLINE - 10:
            break
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            continue
        chain = _local_search(sequence, chain, rng, max_steps=2000, temperature=0.5)
        e = _energy(sequence, chain)
        if e < best_energy:
            best_energy = e
            best_chain = chain
    
    # Strategy 2: Random walks with local search
    for _ in range(20):
        if time.time() > _DEADLINE - 10:
            break
        chain = _random_walk(sequence, dimension, rng)
        if chain is None:
            continue
        chain = _local_search(sequence, chain, rng, max_steps=3000, temperature=1.0)
        e = _energy(sequence, chain)
        if e < best_energy:
            best_energy = e
            best_chain = chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
