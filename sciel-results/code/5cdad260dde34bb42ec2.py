"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _neighbours(point):
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _energy(sequence, chain):
    """Calculate total energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    h_positions = [i for i, c in enumerate(sequence) if c == 'H']
    contacts = 0
    for i in h_positions:
        p = chain[i]
        x, y, z = p
        for q in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
            j = where.get(q)
            if j is not None and j > i + 1:
                if sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _grow_greedy(sequence, dimension, rng, bias=1.0):
    """
    Generate a self-avoiding walk with a greedy bias towards H-H contacts.
    bias controls the strength of the preference for creating contacts.
    """
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    h_set = set(i for i, c in enumerate(sequence) if c == 'H')
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            score = 0
            if i in h_set:
                x, y, z = site
                for q in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and j in h_set:
                            score += 1
            # Add some randomness to avoid getting stuck in local minima of growth
            score += rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
            
        # Sort by score descending, pick from top few or best
        options.sort(reverse=True)
        # With high bias, always pick best. With low bias, pick randomly from top 3.
        if bias > 0.8:
            site = options[0][1]
        else:
            top_k = min(3, len(options))
            site = options[rng.randint(0, top_k-1)][1]
            
        chain.append(site)
        occupied[site] = i
        
    return chain

def _tail_regrowth(sequence, chain, occupied, k, rng, temperature=1.0):
    """
    Regrow the tail from index k to end.
    Returns new_chain, new_occupied, or None if stuck.
    Uses a greedy-with-randomness approach to choose the next step.
    """
    if k >= len(chain):
        return chain, occupied
    
    # Remove tail from k to end
    new_occupied = dict(occupied)
    for i in range(k, len(chain)):
        del new_occupied[chain[i]]
    
    new_chain = chain[:k]
    h_set = set(i for i, c in enumerate(sequence) if c == 'H')
    
    for i in range(k, len(sequence)):
        last = new_chain[-1]
        options = []
        for site in _neighbours(last):
            if site in new_occupied:
                continue
            
            score = 0
            if i in h_set:
                x, y, z = site
                for q in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
                    if q in new_occupied:
                        j = new_occupied[q]
                        if j < i - 1 and j in h_set:
                            score += 1
            # Add noise
            score += rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None, None
            
        # Boltzmann selection based on temperature
        # Higher temperature -> more random choices
        # Lower temperature -> more greedy
        if temperature < 0.1:
            # Greedy
            options.sort(reverse=True)
            site = options[0][1]
        else:
            # Softmax selection
            scores = [s for s, _ in options]
            max_score = max(scores)
            # Scale scores to avoid overflow
            exp_scores = [math.exp((s - max_score) / temperature) for s in scores]
            total = sum(exp_scores)
            r = rng.random() * total
            cumulative = 0
            site = options[0][1]
            for idx, (score, s) in enumerate(options):
                cumulative += exp_scores[idx]
                if r <= cumulative:
                    site = s
                    break
        
        new_chain.append(site)
        new_occupied[site] = i
        
    return new_chain, new_occupied

def _sa_fold(sequence, dimension, rng, time_limit):
    """
    Simulated Annealing for a single sequence.
    """
    n = len(sequence)
    if n < 2:
        return [(i, 0, 0) for i in range(n)]
        
    # Initialize with a few random greedy grows, pick the best
    best_init = None
    best_init_energy = float('inf')
    
    for _ in range(15):
        init_chain = _grow_greedy(sequence, dimension, rng, bias=0.4)
        if init_chain is None:
            continue
        e = _energy(sequence, init_chain)
        if e < best_init_energy:
            best_init_energy = e
            best_init = init_chain
            
    if best_init is None:
        # Fallback to a simple line
        best_init = [(i, 0, 0) for i in range(n)]
        best_init_energy = _energy(sequence, best_init)
        
    current_chain = best_init
    current_occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = best_init_energy
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    
    # Annealing schedule
    # T0 should be high enough to accept moves, but not too high to be random
    # Energy range is roughly -n/2 to 0. Let's use T0 = 3.0
    T = 3.0
    T_min = 0.02
    # Number of iterations depends on time limit
    # Let's just loop until time runs out.
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random cut point k
        # Biased towards the end to allow local moves, but sometimes deeper
        # k from 1 to n-1
        # Weight towards larger k (shorter tails) for faster moves
        # Or uniform? Uniform allows global rearrangements.
        # Let's use a mix: 40% uniform, 60% biased towards end
        if rng.random() < 0.4:
            k = rng.randint(1, n - 1)
        else:
            # Biased towards end
            k = n - 1 - int(n * rng.expovariate(1.0 / (n * 0.08)))
            k = max(1, min(n - 1, k))
            
        new_chain, new_occupied = _tail_regrowth(sequence, current_chain, current_occupied, k, rng, T)
        
        if new_chain is None:
            T *= 0.998
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current_chain = new_chain
            current_occupied = new_occupied
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        T *= 0.998
        
    return best_chain

def fold(sequence, dimension):
    """
    Best conformation using multiple SA restarts.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
        
    # Total time budget is 330s for 26 sequences.
    # Approx 12s per sequence.
    # We will run multiple SA restarts.
    
    # Estimate time per sequence
    # Let's allocate 11 seconds per sequence to be safe (26 * 11 = 286s)
    seq_time_limit = 11.0
    
    best_chain = None
    best_energy = float('inf')
    
    # Number of restarts
    # Each restart has a time slice
    # Let's do 4 restarts, each with 1/4 of the time
    num_restarts = 4
    slice_time = seq_time_limit / num_restarts
    
    # Use a fixed seed for reproducibility in evaluation?
    # The task says "sandboxed subprocess", so randomness is fine.
    # But to be safe, let's use a seed based on the sequence.
    rng = random.Random(hash(sequence) % (2**32))
    
    for restart in range(num_restarts):
        # Run SA
        sa_chain = _sa_fold(sequence, dimension, rng, slice_time)
        
        if sa_chain is not None:
            e = _energy(sequence, sa_chain)
            if e < best_energy:
                best_energy = e
                best_chain = sa_chain
                
    if best_chain is None:
        # Fallback
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
