"""
HP Lattice Protein Folding: Hybrid Simulated Annealing with Tail Regrowth
and Segment Insertion moves, plus global deadline guard.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard: 330s wall-clock budget for all 26 sequences.
# Leave a 10s safety margin.
_GLOBAL_DEADLINE = time.time() + 320.0

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
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y, z = p
            for q in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
                j = where.get(q)
                if j is not None and j > i + 1:
                    if sequence[j] == 'H':
                        contacts += 1
    return -contacts

def _grow_greedy(sequence, dimension, rng, bias=0.5):
    """
    Generate a self-avoiding walk with a greedy bias towards H-H contacts.
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
            score += rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
            
        options.sort(reverse=True)
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
    Returns (new_chain, new_occupied) or (None, None) if stuck.
    """
    if k >= len(chain):
        return chain, occupied
    
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
            score += rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None, None
            
        if temperature < 0.1:
            options.sort(reverse=True)
            site = options[0][1]
        else:
            scores = [s for s, _ in options]
            max_score = max(scores)
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

def _segment_insertion(sequence, chain, occupied, rng, temperature=1.0):
    """
    Structural move: lift a short segment (length 2-4) and reinsert it elsewhere
    in the chain. This allows global rearrangements that tail regrowth cannot.
    Returns (new_chain, new_occupied) or (None, None).
    """
    n = len(chain)
    if n < 5:
        return None, None
    
    # Pick a segment length (short segments are more likely to fit)
    seg_len = rng.choice([2, 3, 4])
    
    # Pick start of segment (not at the very ends to allow reinsertion)
    seg_start = rng.randint(1, n - seg_len - 1)
    seg_end = seg_start + seg_len  # exclusive
    
    # Pick insertion point (where to reinsert the segment)
    # Must be different from original position
    insert_at = rng.randint(0, n - seg_len)
    if insert_at == seg_start:
        return None, None
    
    # Extract segment
    segment = chain[seg_start:seg_end]
    
    # Remove segment from chain
    remaining = chain[:seg_start] + chain[seg_end:]
    
    # Build occupied set for remaining
    rem_occupied = {p: i for i, p in enumerate(remaining)}
    
    # Determine new insertion index in the remaining chain
    if insert_at > seg_start:
        # Insertion point shifts because we removed seg_len elements before it
        new_insert_idx = insert_at - seg_len
    else:
        new_insert_idx = insert_at
    
    # Try to insert the segment at new_insert_idx
    # The segment must connect: remaining[new_insert_idx-1] -> segment[0] and segment[-1] -> remaining[new_insert_idx]
    
    if new_insert_idx == 0:
        # Insert at beginning: segment[0] must be at origin or we just check self-avoidance
        # Actually, the chain is a walk, so the first element has no predecessor.
        # We just need segment to be self-avoiding and not collide with rest.
        # But the walk must be contiguous. So segment[0] is the new start.
        # The connection is segment[-1] -> remaining[0].
        if len(remaining) == 0:
            new_chain = segment
        else:
            # Check connection segment[-1] to remaining[0]
            if sum(abs(a-b) for a,b in zip(segment[-1], remaining[0])) != 1:
                return None, None
            new_chain = segment + remaining
    elif new_insert_idx == len(remaining):
        # Insert at end: remaining[-1] -> segment[0]
        if len(remaining) == 0:
            new_chain = segment
        else:
            if sum(abs(a-b) for a,b in zip(remaining[-1], segment[0])) != 1:
                return None, None
            new_chain = remaining + segment
    else:
        # Insert in middle: remaining[new_insert_idx-1] -> segment[0] and segment[-1] -> remaining[new_insert_idx]
        if sum(abs(a-b) for a,b in zip(remaining[new_insert_idx-1], segment[0])) != 1:
            return None, None
        if sum(abs(a-b) for a,b in zip(segment[-1], remaining[new_insert_idx])) != 1:
            return None, None
        new_chain = remaining[:new_insert_idx] + segment + remaining[new_insert_idx:]
    
    # Check self-avoidance
    new_occupied = {}
    for i, p in enumerate(new_chain):
        if p in new_occupied:
            return None, None
        new_occupied[p] = i
    
    return new_chain, new_occupied

def _sa_fold(sequence, dimension, rng, time_limit):
    """
    Simulated Annealing with Tail Regrowth and Segment Insertion.
    """
    n = len(sequence)
    if n < 2:
        return [(i, 0, 0) for i in range(n)]
        
    # Initialize with multiple greedy grows
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
        best_init = [(i, 0, 0) for i in range(n)]
        best_init_energy = _energy(sequence, best_init)
        
    current_chain = best_init
    current_occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = best_init_energy
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    
    T = 3.0
    T_min = 0.02
    cooling = 0.998
    
    while T > T_min:
        # Check global deadline and local time limit
        if time.time() > _GLOBAL_DEADLINE - 5.0:
            break
        if time.time() - start_time > time_limit:
            break
            
        # Choose move type: 70% tail regrowth, 30% segment insertion
        if rng.random() < 0.7:
            # Tail regrowth
            if rng.random() < 0.4:
                k = rng.randint(1, n - 1)
            else:
                k = n - 1 - int(n * rng.expovariate(1.0 / (n * 0.08)))
                k = max(1, min(n - 1, k))
            
            new_chain, new_occupied = _tail_regrowth(sequence, current_chain, current_occupied, k, rng, T)
        else:
            # Segment insertion
            new_chain, new_occupied = _segment_insertion(sequence, current_chain, current_occupied, rng, T)
        
        if new_chain is None:
            T *= cooling
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
                
        T *= cooling
        
    return best_chain

def fold(sequence, dimension):
    """
    Best conformation using SA with tail regrowth and segment insertion.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
        
    # Per-sequence time limit: 320s / 26 ≈ 12.3s, use 11.5s to be safe
    seq_time_limit = 11.5
    
    # Check if we already have little time left
    remaining = _GLOBAL_DEADLINE - time.time()
    if remaining < 2.0:
        # Not enough time, return a quick greedy solution
        chain = _grow_greedy(sequence, dimension, random.Random(42), bias=0.8)
        if chain is None:
            chain = [(i, 0, 0) for i in range(n)]
        return [list(p) for p in chain]
    
    # Use a fixed seed based on sequence for reproducibility
    rng = random.Random(hash(sequence) % (2**32))
    
    best_chain = _sa_fold(sequence, dimension, rng, seq_time_limit)
    
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
