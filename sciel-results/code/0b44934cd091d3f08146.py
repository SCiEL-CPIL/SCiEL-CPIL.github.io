"""
HP lattice protein folding using fragment-based Monte Carlo search.
Uses random self-avoiding walk initialization and segment moves with
efficient energy delta computation.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Safety margin: 210s limit, leave 15s buffer -> 195s usable
    return 195.0 - (time.time() - _START_TIME)

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _is_adjacent(p, q):
    return sum(abs(a - b) for a, b in zip(p, q)) == 1

def _energy(sequence, chain):
    """Calculate energy: -number of H-H contacts (non-bonded neighbors)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(sequence)
    is_h = [c == 'H' for c in sequence]
    for i, p in enumerate(chain):
        if is_h[i]:
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and is_h[j]:
                    contacts += 1
    return -contacts

def _generate_random_saw(n, rng, max_attempts=200):
    """Generate a random self-avoiding walk of length n on the square lattice."""
    for _ in range(max_attempts):
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        success = True
        for i in range(1, n):
            neighbors = list(_neighbours(chain[-1]))
            rng.shuffle(neighbors)
            placed = False
            for site in neighbors:
                if site not in occupied:
                    chain.append(site)
                    occupied[site] = i
                    placed = True
                    break
            if not placed:
                success = False
                break
        if success and len(chain) == n:
            return chain
    return None

def _compute_contact_map(sequence, chain):
    """
    Compute a contact map: for each residue i, the set of residue indices j
    that are in contact with i (non-bonded neighbors).
    Returns (contacts_count, contact_sets) where contact_sets[i] is a set of j > i+1.
    """
    where = {p: i for i, p in enumerate(chain)}
    n = len(chain)
    is_h = [c == 'H' for c in sequence]
    contact_sets = [set() for _ in range(n)]
    contacts = 0
    for i, p in enumerate(chain):
        if is_h[i]:
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and is_h[j]:
                    contact_sets[i].add(j)
                    contact_sets[j].add(i)
                    contacts += 1
    return -contacts, contact_sets

def _segment_move(chain, i, j, rng, fixed_head, fixed_tail):
    """
    Try to move segment [i+1, j] (inclusive) around pivot at position i.
    Returns new_chain if successful, None otherwise.
    fixed_head: set of positions in chain[0:i+1]
    fixed_tail: set of positions in chain[j+1:]
    """
    n = len(chain)
    if i < 0 or j >= n or i >= j:
        return None
    
    pivot_pos = chain[i]
    has_tail = (j + 1 < n)
    tail_pos = chain[j + 1] if has_tail else None
    
    # Old positions of the moving segment
    old_seg = set(chain[i+1:j+1])
    
    # Fixed positions
    fixed = fixed_head | fixed_tail
    
    # Try each of the 4 neighbor directions for the first step
    neighbors = list(_neighbours(pivot_pos))
    rng.shuffle(neighbors)
    
    seg_len = j - i
    for first_pos in neighbors:
        if first_pos in fixed:
            continue
        
        new_seg = [first_pos]
        valid = True
        
        for k in range(1, seg_len):
            last = new_seg[-1]
            opts = list(_neighbours(last))
            rng.shuffle(opts)
            placed = False
            for nb in opts:
                if nb in fixed:
                    continue
                if nb in new_seg:
                    continue
                # If this is the last step and there's a tail, must connect to tail
                if k == seg_len - 1 and has_tail:
                    if not _is_adjacent(nb, tail_pos):
                        continue
                new_seg.append(nb)
                placed = True
                break
            if not placed:
                valid = False
                break
        
        if not valid:
            continue
        
        # Construct new chain
        new_chain = chain[:i+1] + new_seg + (chain[j+1:] if has_tail else [])
        
        # Check if it's actually different
        if new_chain == chain:
            continue
        
        return new_chain
    
    return None

def _local_search_fragment(sequence, chain, rng, max_iters, start_temp=2.0, end_temp=0.05):
    """
    Local search using segment moves with simulated annealing.
    Uses efficient energy delta by recomputing only affected contacts.
    """
    n = len(chain)
    current = list(chain)
    current_energy = _energy(sequence, current)
    is_h = [c == 'H' for c in sequence]
    
    # Precompute position lookup
    where = {p: i for i, p in enumerate(current)}
    
    def recompute_energy(new_chain):
        """Recompute full energy for a new chain."""
        return _energy(sequence, new_chain)
    
    def try_segment_move(chain, rng):
        """Try a random segment move."""
        if len(chain) < 4:
            return None
        
        # Choose segment length: bias towards shorter segments
        max_seg = min(len(chain) - 2, 12)
        seg_len = rng.randint(2, max_seg)
        
        # Choose pivot index i (segment is i+1 to i+seg_len)
        i = rng.randint(0, len(chain) - seg_len - 1)
        j = i + seg_len
        
        # Build fixed sets
        fixed_head = set(chain[:i+1])
        fixed_tail = set(chain[j+1:]) if j + 1 < len(chain) else set()
        
        return _segment_move(chain, i, j, rng, fixed_head, fixed_tail)
    
    def try_kink_move(chain, rng):
        """Kink move: move two consecutive residues."""
        if len(chain) < 4:
            return None
        
        i = rng.randint(1, len(chain) - 3)
        p0 = chain[i-1]
        p1 = chain[i]
        p2 = chain[i+1]
        p3 = chain[i+2]
        
        # Try all possible new positions for p1 and p2
        for p1_new in _neighbours(p0):
            if p1_new == p2:
                continue
            for p2_new in _neighbours(p1_new):
                if p2_new == p0:
                    continue
                if not _is_adjacent(p2_new, p3):
                    continue
                # Check self-avoidance
                occupied = set(chain)
                occupied.discard(p1)
                occupied.discard(p2)
                if p1_new in occupied or p2_new in occupied:
                    continue
                
                new_chain = chain.copy()
                new_chain[i] = p1_new
                new_chain[i+1] = p2_new
                return new_chain
        
        return None
    
    for it in range(max_iters):
        if _time_remaining() < 5:
            break
        
        # Annealing schedule
        progress = it / max(1, max_iters)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Choose move type
        move_type = rng.random()
        if move_type < 0.5:
            new_chain = try_segment_move(current, rng)
        else:
            new_chain = try_kink_move(current, rng)
        
        if new_chain is None:
            continue
        
        new_energy = recompute_energy(new_chain)
        delta = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta/temp)
        if delta < 0:
            accept = True
        else:
            if temp > 0.01:
                prob = math.exp(-delta / temp)
                accept = rng.random() < prob
            else:
                accept = False
        
        if accept:
            current = new_chain
            current_energy = new_energy
    
    return current

def _grow(sequence, dimension, rng):
    """Grow a chain greedily, preferring moves that create H-H contacts."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    is_h = [c == 'H' for c in sequence]
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if is_h[i]:
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and is_h[occupied[q]])
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain * 10 + free + rng.random() * 0.1, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def fold(sequence, dimension):
    """
    Fold the sequence using fragment-based Monte Carlo search with
    random SAW initialization and segment moves.
    """
    _init_time()
    n = len(sequence)
    
    # Determine search parameters based on sequence length
    if n <= 20:
        num_restarts = 50
        iters_per_restart = 5000
    elif n <= 50:
        num_restarts = 30
        iters_per_restart = 3000
    else:
        num_restarts = 20
        iters_per_restart = 2000
    
    best_chain = None
    best_energy = 0  # 0 is neutral, we want negative (lower is better)
    
    for restart in range(num_restarts):
        if _time_remaining() < 15:
            break
        
        rng = random.Random(12345 + len(sequence) + restart * 7919)
        
        # Try to initialize with random SAW first, fall back to greedy growth
        chain = _generate_random_saw(n, rng)
        if chain is None:
            chain = _grow(sequence, dimension, rng)
            if chain is None:
                # Try a few more times with different seeds
                success = False
                for attempt in range(5):
                    rng2 = random.Random(9999 + restart * 101 + attempt)
                    chain = _grow(sequence, dimension, rng2)
                    if chain is not None:
                        success = True
                        break
                if not success:
                    continue
        
        # Refine with local search
        refined_chain = _local_search_fragment(sequence, chain, rng, max_iters=iters_per_restart)
        
        energy = _energy(sequence, refined_chain)
        if best_chain is None or energy < best_energy:
            best_chain = refined_chain
            best_energy = energy
    
    if best_chain is None:
        # Final fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
