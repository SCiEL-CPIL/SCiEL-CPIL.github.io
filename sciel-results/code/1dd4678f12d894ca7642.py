# EVOLVE-BLOCK-START
import random
import time
import math
import sys

sys.setrecursionlimit(10000)

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D cubic lattice."""
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y+1, z) # Wait, typo in original, fixing standard neighbors
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

# Correct neighbor generator
def _get_neighbors(point):
    x, y, z = point
    return [
        (x+1, y, z), (x-1, y, z),
        (x, y+1, z), (x, y-1, z),
        (x, y, z+1), (x, y, z-1)
    ]

def _compute_energy(sequence, chain):
    """Calculate total energy of a conformation."""
    if not chain:
        return 0
    n = len(chain)
    if n == 0:
        return 0
    
    # Map position to index
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    contacts = 0
    # Only H residues contribute to energy
    for i, res in enumerate(sequence):
        if res == 'H':
            p = chain[i]
            for q in _get_neighbors(p):
                j = pos_to_idx.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _grow_greedy(sequence, rng, bias=1.0):
    """
    Generate a self-avoiding walk with a greedy bias towards H-H contacts.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
        
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    h_set = set(i for i, c in enumerate(sequence) if c == 'H')
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _get_neighbors(last):
            if site in occupied:
                continue
            score = 0
            if i in h_set:
                for q in _get_neighbors(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and j in h_set:
                            score += 1
            # Add randomness
            score += rng.random() * bias
            options.append((score, site))
        
        if not options:
            return None
            
        # Sort by score descending
        options.sort(key=lambda x: x[0], reverse=True)
        
        # Pick best or random from top
        if bias > 0.5:
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
    """
    n = len(sequence)
    if k >= n:
        return chain, occupied
    
    # Copy current state to allow rollback if needed (though we return new objects)
    # We only modify from k onwards
    new_occupied = dict(occupied)
    # Remove old tail
    for i in range(k, n):
        del new_occupied[chain[i]]
        
    new_chain = chain[:k]
    h_set = set(i for i, c in enumerate(sequence) if c == 'H')
    
    for i in range(k, n):
        last = new_chain[-1]
        options = []
        for site in _get_neighbors(last):
            if site in new_occupied:
                continue
            
            score = 0
            if i in h_set:
                for q in _get_neighbors(site):
                    if q in new_occupied:
                        j = new_occupied[q]
                        if j < i - 1 and j in h_set:
                            score += 1
                            
            # Boltzmann score
            # We want to pick sites with higher scores more often
            # score is non-negative integer + noise? 
            # Let's use raw contact count for the exponential
            options.append((score, site))
        
        if not options:
            return None, None
            
        # Softmax selection
        if temperature < 0.01:
            # Greedy
            options.sort(key=lambda x: x[0], reverse=True)
            site = options[0][1]
        else:
            scores = [s for s, _ in options]
            max_score = max(scores)
            # Avoid overflow
            exp_scores = [math.exp((s - max_score) / temperature) for s in scores]
            total = sum(exp_scores)
            if total == 0:
                site = options[0][1]
            else:
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
        
    # Initialize with multiple greedy grows
    best_init = None
    best_init_energy = float('inf')
    
    # Try different biases
    biases = [0.1, 0.5, 1.0, 2.0]
    for bias in biases:
        # Multiple tries per bias
        for _ in range(5):
            init_chain = _grow_greedy(sequence, rng, bias=bias)
            if init_chain is not None and len(init_chain) == n:
                e = _compute_energy(sequence, init_chain)
                if e < best_init_energy:
                    best_init_energy = e
                    best_init = init_chain
    
    if best_init is None:
        # Fallback: random walk or line
        # Try random walk
        for _ in range(50):
             init_chain = _grow_greedy(sequence, rng, bias=0.0)
             if init_chain is not None:
                 best_init = init_chain
                 best_init_energy = _compute_energy(sequence, init_chain)
                 break
        if best_init is None:
             # Straight line fallback (might be invalid if n > space, but usually ok for short)
             # Actually, straight line is always valid SAW.
             best_init = [(i, 0, 0) for i in range(n)]
             best_init_energy = _compute_energy(sequence, best_init)

    current_chain = best_init
    current_occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = best_init_energy
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    
    # Annealing schedule
    T = 2.0  # Start temperature
    T_min = 0.05
    cooling_rate = 0.999
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random cut point k
        # Biased towards the end to allow local moves, but sometimes deeper
        # k from 1 to n-1
        # Use exponential distribution biased towards small tails (large k)
        # tail_length = n - k
        # We want tail_length to be small on average, but allow large.
        
        # 50% chance: local move (tail length 1-10)
        # 30% chance: medium move (tail length 10-30)
        # 20% chance: global move (tail length 30-n)
        
        r = rng.random()
        if r < 0.5:
            tail_len = rng.randint(1, min(10, n-1))
        elif r < 0.8:
            tail_len = rng.randint(10, min(30, n-1))
        else:
            tail_len = rng.randint(30, n-1)
            
        k = n - tail_len
        if k < 1:
            k = 1
            
        new_chain, new_occupied = _tail_regrowth(sequence, current_chain, current_occupied, k, rng, T)
        
        if new_chain is None:
            T *= cooling_rate
            continue
            
        new_energy = _compute_energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current_chain = new_chain
            current_occupied = new_occupied
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        T *= cooling_rate
        
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
        
    # Time budget allocation
    # 26 sequences, 330s total. ~12s per sequence.
    # We use 10s for search to be safe.
    time_limit = 10.0
    
    # For very short chains, exact search or more restarts is possible.
    # For long chains, fewer restarts but longer runs.
    
    if n <= 30:
        num_restarts = 5
        time_per_restart = 1.5
    elif n <= 60:
        num_restarts = 3
        time_per_restart = 3.0
    else:
        num_restarts = 2
        time_per_restart = 4.5
        
    best_chain = None
    best_energy = float('inf')
    
    # Use a deterministic seed for reproducibility? 
    # The task says "random is fine for diversity". We'll use a fixed base seed 
    # plus restart index to ensure different paths, but consistent across runs if needed.
    base_seed = hash(sequence) % (2**32)
    
    global_start = time.time()
    max_total_time = 11.0 # Safety margin
    
    for restart in range(num_restarts):
        if time.time() - global_start > max_total_time:
            break
            
        rng = random.Random(base_seed + restart * 12345)
        
        # Adjust time per restart if we are running out of time
        remaining = max_total_time - (time.time() - global_start)
        current_time_limit = min(time_per_restart, remaining * 0.9)
        
        if current_time_limit <= 0.1:
            break
            
        chain = _sa_fold(sequence, dimension, rng, current_time_limit)
        
        if chain is not None and len(chain) == n:
            e = _compute_energy(sequence, chain)
            if e < best_energy:
                best_energy = e
                best_chain = chain
                
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
