# EVOLVE-BLOCK-START
import random
import math


def _neighbours(point):
    """Yield the 6 neighbours of a 3D lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the negative H-H contact count for a given chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng):
    """Greedy chain growth with random tie-breaking. Returns chain or None if trapped."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score: prefer H-H contacts, then more free neighbours
            score = 0.0
            if sequence[i] == 'H':
                for nb in _neighbours(site):
                    if nb in occupied:
                        idx = occupied[nb]
                        if idx <= i - 2 and sequence[idx] == 'H':
                            score += 1.0
            # Free space heuristic
            free = sum(1 for nb in _neighbours(site) if nb not in occupied)
            score += 0.1 * free
            score += rng.random() * 0.5  # small randomization to avoid cycles
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Pick the best candidate
        candidates.sort(reverse=True)
        site = candidates[0][1]
        chain.append(site)
        occupied[site] = i
    
    return chain


def _metropolis_improve(sequence, chain, dimension, rng, num_steps):
    """
    Local search via Metropolis moves.
    Moves: flip a sub-chain segment (reverse a contiguous portion).
    Energy delta computed efficiently using precomputed contact sets.
    """
    n = len(chain)
    if n < 4:
        return chain
    
    # Precompute neighbor relationships and contacts
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    # For each position, store the set of neighbors that are occupied
    # We'll recompute energy from scratch for simplicity in this version,
    # but track contacts for delta calculation.
    
    current_energy = _energy(sequence, chain)
    
    for step in range(num_steps):
        # Choose a sub-chain to reverse
        # Length of sub-chain: random between 3 and min(n//2, 20)
        max_len = min(n // 2, 25)
        if max_len < 3:
            max_len = 3 if n >= 3 else 2
        seg_len = rng.randint(3, max_len)
        
        # Choose start position
        if n - seg_len <= 0:
            continue
        start_idx = rng.randint(0, n - seg_len)
        end_idx = start_idx + seg_len
        
        # Check if the reversed segment would cause collisions
        # The segment is chain[start_idx:end_idx+1]
        # After reversal, position start_idx+k gets what was at end_idx-k
        # We need to check that the new positions are not occupied by positions outside [start_idx, end_idx]
        
        # Create a temporary copy of the chain
        new_chain = chain[:]
        
        # Reverse the segment
        for k in range(seg_len):
            new_chain[start_idx + k] = chain[end_idx - k]
        
        # Check self-avoidance
        occupied_set = set(new_chain)
        if len(occupied_set) != n:
            continue
        
        # Compute new energy
        new_energy = _energy(sequence, new_chain)
        
        # Metropolis acceptance
        delta = new_energy - current_energy  # We want more negative, so delta < 0 is good
        if delta < 0 or rng.random() < math.exp(-delta / max(1.0, abs(current_energy) * 0.1)):
            chain = new_chain
            current_energy = new_energy
    
    return chain


def _random_walk_fold(sequence, dimension, rng, num_attempts=50):
    """
    Perform multiple random greedy growths followed by Metropolis improvement.
    """
    best_chain = None
    best_energy = 0
    
    for attempt in range(num_attempts):
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Improve with Metropolis
        num_met_steps = max(50, len(sequence) * 2)
        chain = _metropolis_improve(sequence, chain, dimension, rng, num_met_steps)
        
        energy = _energy(sequence, chain)
        
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    return best_chain


def fold(sequence, dimension):
    """
    Fold the given sequence on a d-dimensional cubic lattice.
    Returns a list of lattice points (each a list of length dimension).
    """
    rng = random.Random(42 + len(sequence) * 7 + sum(ord(c) for c in sequence))
    
    # For short sequences, use more attempts
    n = len(sequence)
    if n <= 30:
        num_attempts = 100
    elif n <= 60:
        num_attempts = 50
    else:
        num_attempts = 25
    
    best_chain = _random_walk_fold(sequence, dimension, rng, num_attempts)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
