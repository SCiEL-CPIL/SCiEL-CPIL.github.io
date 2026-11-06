"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = None
_DEADLINE = 200.0  # Safety margin before 210s wall-clock kill


def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_remaining():
    _init_time()
    return _DEADLINE - (time.time() - _START_TIME)


def _neighbours(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _compute_energy(sequence, chain):
    """Compute the total energy (negative number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _init_chain(sequence, rng):
    """Initialize a random self-avoiding walk on the square lattice."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)]
    
    # Try to build a compact random walk
    for attempt in range(100):
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        success = True
        for i in range(1, n):
            neighbors = [p for p in _neighbours(chain[-1]) if p not in occupied]
            if not neighbors:
                success = False
                break
            # Bias towards moving inward (toward origin) to encourage compactness
            # but with randomness to explore
            def dist_sq(p):
                return p[0] * p[0] + p[1] * p[1]
            
            # Weight by inverse distance to encourage compactness
            weights = [1.0 / (1.0 + dist_sq(p)) for p in neighbors]
            total = sum(weights)
            r = rng.random() * total
            cum = 0
            chosen = neighbors[-1]
            for p, w in zip(neighbors, weights):
                cum += w
                if r <= cum:
                    chosen = p
                    break
            
            chain.append(chosen)
            occupied[chosen] = i
        
        if success:
            return chain
    
    # Fallback: straight line (always valid)
    return [(i, 0) for i in range(n)]


def _local_search(sequence, chain, rng, time_limit):
    """Simulated annealing with kink moves and segment shifts."""
    n = len(sequence)
    where = {p: i for i, p in enumerate(chain)}
    energy = _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = energy
    
    # Precompute H positions for fast contact checks
    h_positions = [i for i in range(n) if sequence[i] == 'H']
    
    # Temperature schedule
    t_start = 0.5
    t_end = 0.001
    steps_per_second = 50000  # Estimate
    
    elapsed = 0.0
    last_time = time.time()
    
    while _time_remaining() > 5.0:
        # Check time periodically
        now = time.time()
        elapsed = now - last_time
        if elapsed > 0.1:
            last_time = now
        
        # Determine temperature based on progress
        # We don't know total steps, so use a cooling schedule based on iterations
        # Use a simple exponential cooling with a fixed number of iterations estimate
        # For now, use a dynamic approach: cool based on time spent
        
        # Choose a move
        move_type = rng.random()
        
        if move_type < 0.6:
            # Kink move: pick a residue and try to flip its direction
            if n < 3:
                continue
            i = rng.randint(1, n - 2)
            p_prev = chain[i - 1]
            p_cur = chain[i]
            p_next = chain[i + 1]
            
            # The two possible kink positions are the two neighbors of p_cur
            # that are not p_prev or p_next
            all_neighbors = list(_neighbours(p_cur))
            options = [p for p in all_neighbors if p != p_prev and p != p_next]
            
            if not options:
                continue
            
            new_p = rng.choice(options)
            
            if new_p in where:
                continue
            
            # Compute energy delta
            # Remove contacts involving p_cur, add contacts for new_p
            old_contacts = 0
            if sequence[i] == 'H':
                for q in _neighbours(p_cur):
                    j = where.get(q)
                    if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                        old_contacts += 1
            
            new_contacts = 0
            if sequence[i] == 'H':
                for q in _neighbours(new_p):
                    j = where.get(q)
                    if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                        new_contacts += 1
            
            delta = (new_contacts - old_contacts) * -1  # Energy is -contacts
            
            # Acceptance criterion
            if delta < 0 or rng.random() < pow(2.718281828, -delta / 0.1):
                # Apply move
                del where[p_cur]
                where[new_p] = i
                chain[i] = new_p
                energy += delta
                
                if energy < best_energy:
                    best_energy = energy
                    best_chain = list(chain)
        
        elif move_type < 0.85:
            # Segment shift: move a short segment (length 1-4) to a nearby position
            seg_len = rng.randint(1, min(4, n - 2))
            i = rng.randint(1, n - seg_len - 1)
            
            # Extract segment
            seg = chain[i:i + seg_len]
            
            # Try to shift the segment by one step in a random direction
            dx, dy = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            
            new_seg = [(x + dx, y + dy) for x, y in seg]
            
            # Check validity: no self-overlap
            valid = True
            for p in new_seg:
                # Check if this position is occupied by a non-segment residue
                if p in where:
                    j = where[p]
                    if not (i <= j < i + seg_len):
                        valid = False
                        break
                # Check adjacency to neighboring residues
                # The first residue of the segment must be adjacent to chain[i-1]
                # The last residue of the segment must be adjacent to chain[i+seg_len]
            
            if not valid:
                continue
            
            # Check adjacency constraints
            if i > 0:
                prev_res = chain[i - 1]
                first_new = new_seg[0]
                if abs(prev_res[0] - first_new[0]) + abs(prev_res[1] - first_new[1]) != 1:
                    continue
            
            if i + seg_len < n:
                next_res = chain[i + seg_len]
                last_new = new_seg[-1]
                if abs(next_res[0] - last_new[0]) + abs(next_res[1] - last_new[1]) != 1:
                    continue
            
            # Check internal adjacency of the segment
            for k in range(seg_len - 1):
                if abs(new_seg[k][0] - new_seg[k + 1][0]) + abs(new_seg[k][1] - new_seg[k + 1][1]) != 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Compute energy delta
            # This is complex; for simplicity, recompute energy for affected region
            # For now, skip segment shifts if they're too expensive
            
            continue  # Skip segment shifts for now due to complexity
        
        else:
            # Perturbation: make a random large move
            if n < 5:
                continue
            i = rng.randint(1, n - 2)
            p_cur = chain[i]
            neighbors = [p for p in _neighbours(p_cur) if p not in where]
            if not neighbors:
                continue
            new_p = rng.choice(neighbors)
            
            # Recompute energy (expensive but rare)
            del where[p_cur]
            where[new_p] = i
            chain[i] = new_p
            energy = _compute_energy(sequence, chain)
            
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
    
    return best_chain, best_energy


def _refold(sequence, rng):
    """Complete refolding with local search."""
    chain = _init_chain(sequence, rng)
    chain, energy = _local_search(sequence, chain, rng, 10.0)
    return chain, energy


def fold(sequence, dimension):
    """Fold a protein sequence into a low-energy conformation."""
    _init_time()
    n = len(sequence)
    
    if n <= 1:
        return [[0, 0]]
    
    rng = random.Random(42)
    
    best_chain = None
    best_energy = float('inf')
    
    # Multiple restarts with local search
    while _time_remaining() > 10.0:
        chain, energy = _refold(sequence, rng)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
        
        # Check if we've hit a good energy
        if best_energy <= -n // 3:
            # Good enough, do one more refinement
            break
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
