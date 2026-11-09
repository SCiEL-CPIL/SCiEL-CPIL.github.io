"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp

_START_TIME = time.time()
_DEADLINE = 205.0  # Safety margin before 210s kill

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            # Heuristic: Count potential H-H contacts this site would form
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 1
            
            # Heuristic: Avoid dead ends (unless it's the last residue)
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            if i < len(sequence) - 1 and free == 0:
                continue
            
            # Score: Gain is dominant, then free space (to avoid trapping), then random
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
            
        if not options:
            return None
            
        # Pick the best option
        options.sort(key=lambda x: x[0], reverse=True)
        # Occasionally pick a random option to allow exploration? 
        # For initialization, greedy is usually fine, but let's stick to best for speed.
        # Actually, let's add a bit of randomness to avoid deterministic traps.
        # If top score is much higher, pick it. Else, pick randomly from top few.
        # Simple: pick best.
        site = options[0][1]
        occupied[site] = i
        chain.append(site)
    return chain

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _rotate_point(point, center, rot):
    x, y = point[0] - center[0], point[1] - center[1]
    if rot == 0:
        nx, ny = x, y
    elif rot == 90:
        nx, ny = -y, x
    elif rot == 180:
        nx, ny = -x, -y
    else: # 270
        nx, ny = y, -x
    return (center[0] + nx, center[1] + ny)

def _incremental_energy_update(sequence, chain, pivot_idx, new_segment, current_energy):
    """
    Fast energy calculation for a pivot move.
    Only re-evaluates contacts involving the moved segment and its boundary with the fixed part.
    """
    N = len(chain)
    k = pivot_idx
    
    # Identify H positions in fixed part (indices < k)
    # And H positions in segment (indices >= k)
    
    # To be safe and simple, let's just recalculate energy for the whole chain if N is small, 
    # but for N up to 100, incremental is faster.
    
    # Let's do a hybrid: 
    # 1. Identify all H indices in fixed part (0 to k-1)
    # 2. Identify all H indices in segment (k to N-1)
    
    # Old crossing contacts:
    # For each position in the old segment (chain[k:N]), if it's H, check if any of its neighbors are in fixed_h_positions.
    # But we must ensure the index difference > 1.
    # Since i < k and j >= k, the max index in fixed is k-1. The min index in segment is k.
    # So |i-j| >= 1.
    # The only case where |i-j|=1 is when i=k-1 and j=k.
    # So we just need to exclude the pair (k-1, k) if both are H? No, (k-1,k) is adjacent in chain, so dist=1.
    # But energy definition excludes adjacent residues.
    # So if chain[k-1] is H and chain[k] is H, they are neighbors in lattice but adjacent in chain -> NOT a contact.
    # My check `j > i + 1` in _energy handles this.
    # In incremental:
    # If i = k-1 and j = k, they are adjacent in chain. Exclude.
    # All other pairs (i < k-1, j >= k) have |i-j| >= 2.
    
    old_crossing = 0
    # Iterate over segment positions
    for j in range(k, N):
        if sequence[j] == 'H':
            pos = chain[j]
            for nb in _neighbours(pos):
                if nb in set(chain[:k]): # This is O(k) per step, so O(N*k). 
                    # Wait, building a set each time is expensive.
                    # Let's precompute fixed_h_positions.
                    pass
    
    # Let's optimize this properly.
    fixed_h_positions = set()
    for i in range(k):
        if sequence[i] == 'H':
            fixed_h_positions.add(chain[i])
            
    old_crossing = 0
    for j in range(k, N):
        if sequence[j] == 'H':
            pos = chain[j]
            for nb in _neighbours(pos):
                if nb in fixed_h_positions:
                    if nb == chain[k-1] and j == k:
                        continue
                    old_crossing += 1
                    
    new_crossing = 0
    for idx, pos in enumerate(new_segment):
        j = k + idx
        if sequence[j] == 'H':
            for nb in _neighbours(pos):
                if nb in fixed_h_positions:
                    if nb == chain[k-1] and j == k:
                        continue
                    new_crossing += 1
                    
    delta_e = -(new_crossing - old_crossing)
    return current_energy + delta_e

def fold(sequence, dimension):
    """
    Best conformation using greedy initialization and Incremental Simulated Annealing.
    """
    rng = random.Random(12345 + len(sequence))
    N = len(sequence)
    
    # 1. Generate initial valid chain using greedy growth
    best_chain = None
    best_energy = 0
    
    # Try a few initializations to get a good start
    for _ in range(10):
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        e = _energy(sequence, chain)
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e
        if best_energy <= -len(sequence) * 0.5: # Good enough
            break

    if best_chain is None:
        # Fallback: straight line
        return [list(p) for p in [(i, 0) for i in range(N)]]

    chain = best_chain
    current_energy = best_energy
    
    # SA Parameters
    T = 5.0
    T_min = 0.01
    cooling = 0.995 # Slower cooling for more exploration
    
    # Time check
    while time.time() < _DEADLINE:
        # Pick a pivot
        # Bias towards shorter segments? Or random?
        # Random is standard.
        k = rng.randint(1, N - 2)
        rot = rng.choice([90, 180, 270])
        
        # Rotate segment k..N-1 around chain[k-1]
        pivot = chain[k-1]
        new_segment = []
        for i in range(k, N):
            new_segment.append(_rotate_point(chain[i], pivot, rot))
        
        # Check collision with fixed part (0..k-2)
        # fixed_set = set(chain[:k-1])
        # To speed up, we can check only the new_segment against the fixed part.
        # Since N is small, building the set is O(k).
        fixed_set = set(chain[:k-1])
        
        valid = True
        for p in new_segment:
            if p in fixed_set:
                valid = False
                break
        
        if not valid:
            continue
            
        # Calculate new energy incrementally
        # Note: _incremental_energy_update expects the OLD chain to compute old_crossing.
        # It uses `chain` for the old positions.
        new_energy = _incremental_energy_update(sequence, chain, k, new_segment, current_energy)
        
        delta_e = new_energy - current_energy
        
        # Acceptance
        if delta_e < 0 or (T > 0 and rng.random() < exp(-delta_e / T)):
            chain = chain[:k] + new_segment
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain
        
        T *= cooling
        if T < T_min:
            T = 5.0 # Restart temperature (Re-annealing)
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
