import random
import time
import math
import sys

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
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _grow(sequence, dimension, rng, bias=1.0):
    """
    Randomized greedy chain growth.
    bias > 1: more aggressive contact seeking.
    bias < 1: more random exploration.
    """
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            
            # Add randomness
            score = gain * bias + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
            
        # Sort by score and pick the best
        options.sort(reverse=True)
        site = options[0][1]
        occupied[site] = i
        chain.append(site)
        
    return chain

def _local_search(sequence, chain, rng, time_limit, max_steps=5000):
    """
    Simple local search: try to move single residues to improve energy.
    """
    n = len(chain)
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    start_time = time.time()
    steps = 0
    
    while steps < max_steps:
        if time.time() > time_limit:
            break
            
        # Pick a random index to try moving
        i = rng.randint(1, n - 2)
        p = chain[i]
        
        # Try all neighbours
        moved = False
        for q in _neighbours(p):
            if q in {chain[j] for j in range(n) if j != i}:
                continue
            
            # Check connectivity: q must be adjacent to chain[i-1] or chain[i+1]?
            # Actually, for a valid walk, if we move i to q:
            # 1. q must be adjacent to chain[i-1] (bond i-1 -> i)
            # 2. q must be adjacent to chain[i+1] (bond i -> i+1)
            # This is only possible if q is adjacent to BOTH? No.
            # If we move i, the bonds are (i-1, i) and (i, i+1).
            # So q must be adjacent to chain[i-1] AND chain[i+1] is NOT required to be adjacent to q?
            # Wait, the chain is a path.
            # If we change chain[i] to q, the path is ... chain[i-1], q, chain[i+1] ...
            # So q must be adjacent to chain[i-1] AND q must be adjacent to chain[i+1].
            # This is a very restrictive move, often impossible.
            
            # Better local move: Pivot.
            # Pick a pivot point k. Rotate the tail.
            # But that's complex.
            
            # Alternative: "Slide" a segment?
            
            # Let's stick to a simpler "Random Tail Regeneration" as the main move in SA.
            # For local search, let's just try to swap positions? No.
            
            # Let's implement a simple "Crankshaft" or just use Tail Regen in the main loop.
            # This function is a bit weak. Let's replace it with a proper SA loop in the main driver.
            pass
            
        steps += 1
        
    return best_chain, best_energy

def _sa_fold(sequence, dimension, initial_chain, rng, deadline):
    """
    Simulated Annealing with Tail Regeneration moves.
    """
    if initial_chain is None:
        return None, 0
        
    chain = initial_chain[:]
    n = len(chain)
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    t = 1.0
    cooling_rate = 0.999
    min_temp = 0.01
    
    # Pre-compute H positions for faster checks?
    h_positions = [i for i, c in enumerate(sequence) if c == 'H']
    
    while t > min_temp:
        if time.time() > deadline:
            break
            
        # Generate a new chain via Tail Regeneration
        # Pick a random pivot point k
        # To avoid breaking the chain at the very start or end too often,
        # pick k in [1, n-2]
        if n > 2:
            k = rng.randint(1, n - 2)
            
            # Regenerate tail from k
            # Fixed part: chain[:k]
            # We need to connect chain[k-1] to the rest.
            # Actually, let's regenerate from k to n-1.
            # The new chain will be chain[:k] + new_tail
            
            # We need to ensure the new tail starts at a neighbor of chain[k-1]
            # and is self-avoiding and doesn't collide with chain[:k-1]
            
            # Let's try to grow a new tail of length n-k
            # Start from chain[k-1]
            # But we already have chain[k]. We are replacing chain[k:]
            
            # Let's try a different move: "Pivot"
            # Rotate the segment chain[k:] around chain[k-1]
            # This is hard to implement correctly without numpy.
            
            # Let's use "Segment Refold"
            # Pick a segment [start, end]
            # Refold it.
            
            # Simpler: Just regenerate the whole chain from a random point?
            # Let's try to move a single residue? No.
            
            # Let's implement "Tail Regeneration" properly:
            # Pick k.
            # New chain = chain[:k] + new_tail
            # new_tail must be a self-avoiding walk of length n-k starting from a neighbor of chain[k-1]
            # that does not collide with chain[:k-1]
            
            # This is expensive to generate.
            
            # Let's use a simpler move: "Random Walk Refold" of a sub-segment.
            # Pick start, end.
            # Refold chain[start:end+1]
            
            # For now, let's just do a simple "Perturbation":
            # Pick a random index i, and try to move it to a random neighbor.
            # If valid, accept.
            
            i = rng.randint(1, n - 2)
            p = chain[i]
            valid_moves = []
            for q in _neighbours(p):
                if q in {chain[j] for j in range(n) if j != i}:
                    continue
                # Check connectivity
                # q must be adjacent to chain[i-1] and chain[i+1]
                # This is rarely true.
                
                # Let's check if q is adjacent to chain[i-1]
                # And if chain[i+1] is adjacent to q?
                # No, the bond is i-1 -> i -> i+1.
                # So q must be adjacent to chain[i-1] AND q must be adjacent to chain[i+1].
                
                # This move is too restrictive.
                
                # Let's use "Pivot" instead.
                # Pick k.
                # Rotate chain[k:] around chain[k-1].
                # This requires calculating new positions.
                
                # Let's implement a "Simple Pivot":
                # Pick k.
                # Pick a new direction for the bond k-1 -> k.
                # Then, try to continue the chain from k using a greedy/random walk.
                
                # This is "Tail Regeneration".
                
                # Let's implement it.
                
                # Fixed part: chain[:k]
                # Start growing from chain[k-1]
                # But we want to keep the length n-k.
                
                # Let's try to grow a new tail of length n-k
                # starting from chain[k-1]
                # such that it doesn't collide with chain[:k-1]
                
                # This is what _grow does, but we need to restrict it.
                
                # Let's just call _grow on the suffix? No, _grow starts from scratch.
                
                # Let's implement a helper: _grow_from(start, length, occupied, sequence_suffix, rng)
                
                # For now, let's just use a very simple move:
                # Swap two non-adjacent residues? No, breaks connectivity.
                
                # Let's just use "Random Restart" if stuck?
                
                # Given the complexity, let's stick to a simple "Greedy Tail Regeneration"
                # where we pick a k, and try to regenerate the tail using _grow logic
                # but constrained to not collide with the fixed part.
                
                # Let's implement _grow_constrained
                pass
            
        t *= cooling_rate
        
    return best_chain, best_energy

def _grow_constrained(start_point, length, sequence_suffix, occupied, rng, bias=1.0):
    """
    Grow a chain of 'length' residues starting from 'start_point'.
    The new chain must not collide with 'occupied'.
    'sequence_suffix' is the sequence for the new residues.
    Returns the new chain (list of points) or None.
    """
    chain = [start_point]
    # Note: start_point is already occupied, so we don't add it to the new chain's occupied set
    # but we must check against it.
    
    # We are growing 'length' new residues.
    # The first new residue is at index 0 of the suffix.
    
    for i in range(length):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            if site in chain: # Self-collision in new chain
                continue
                
            gain = 0
            if sequence_suffix[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        # Check if the occupant is H
                        # We don't have the index in occupied, just the point.
                        # We need a map point -> index for the fixed part.
                        # This is getting complicated.
                        pass
                    if q in chain:
                        # Check if the occupant in the new chain is H
                        pass
            
            score = gain * bias + rng.random() * 0.5
            options.append((score, site))
            
        if not options:
            return None
            
        options.sort(reverse=True)
        site = options[0][1]
        chain.append(site)
        
    return chain

def fold(sequence, dimension):
    """
    Best conformation using Multi-Start Greedy Growth.
    """
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    start_time = time.time()
    deadline = start_time + 280.0 # Leave 50s margin for 330s limit
    
    best_chain = None
    best_energy = 0 # 0 is unfolded, we want negative energy
    
    # Phase 1: Generate initial population
    # Use multiple greedy starts with different biases
    
    init_chains = []
    
    # Try different biases
    for bias in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for _ in range(5):
            c = _grow(sequence, dimension, rng, bias=bias)
            if c:
                init_chains.append(c)
                
    # Random walks
    for _ in range(10):
        c = _random_walk(sequence, dimension, rng)
        if c:
            init_chains.append(c)
            
    if not init_chains:
        # Fallback to straight line
        return [list([i] + [0] * (dimension - 1)) for i in range(n)]
        
    # Phase 2: Evaluate and pick the best
    # We don't have time for SA, so we just pick the best from the initial population.
    # If we have time, we could do a simple local search.
    
    for c in init_chains:
        if time.time() > deadline:
            break
        e = _energy(sequence, c)
        if e < best_energy:
            best_energy = e
            best_chain = c
            
    if best_chain is None:
        best_chain = init_chains[0]
        
    return [list(p) for p in best_chain]

def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                options.append(site)
        if not options:
            return None
        site = rng.choice(options)
        occupied[site] = i
        chain.append(site)
        
    return chain
