"""HP lattice protein folding with Hamiltonian template pre-computation and efficient local search."""
# EVOLVE-BLOCK-START
import random
import time
import math
import sys

# Pre-computed Hamiltonian path templates for various lengths
_TEMPLATE_CACHE = {}

def _serpentine_template(n):
    """Generate a serpentine (zigzag) self-avoiding walk of length n."""
    if n <= 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0)}
    
    # Serpentine: go right, down, left, down, right, down, ...
    row = 0
    col = 0
    direction = 1  # 1 = right, -1 = left
    width = 1
    
    for i in range(1, n):
        if direction == 1:
            col += 1
        else:
            col -= 1
        
        if col == width or col == 0:
            # Turn: go down and reverse direction
            row += 1
            direction = -direction
            if direction == 1:
                col = 0
            else:
                col = width - 1
            # Expand width if needed
            if i >= 1 and (i % 2 == 0) and col == width:
                width += 1
        
        new_pos = (col, row)
        if new_pos in occupied:
            # Fallback: try a simple linear path
            return [(i, 0) for i in range(n)]
        chain.append(new_pos)
        occupied.add(new_pos)
    
    return chain

def _spiral_template(n):
    """Generate a spiral-shaped self-avoiding walk of length n."""
    if n <= 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0)}
    
    # Spiral: right 1, down 1, left 2, up 2, right 3, down 3, ...
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    di = 0
    steps_in_dir = 1
    steps_taken = 0
    
    while len(chain) < n:
        dx, dy = dirs[di % 4]
        for _ in range(steps_in_dir):
            if len(chain) >= n:
                break
            last = chain[-1]
            new = (last[0] + dx, last[1] + dy)
            if new in occupied:
                break
            chain.append(new)
            occupied.add(new)
            steps_taken += 1
        
        if len(chain) >= n:
            break
        
        # Check if we need to advance direction
        last = chain[-1]
        next_pos = (last[0] + dx, last[1] + dy)
        if next_pos in occupied:
            di += 1
            if di % 2 == 0:
                steps_in_dir += 1
    
    return chain[:n]

def _get_templates(n):
    """Get pre-computed templates for sequence length n."""
    if n in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[n]
    
    templates = []
    
    # Serpentine
    serp = _serpentine_template(n)
    if len(serp) == n:
        templates.append(serp)
    
    # Spiral
    spiral = _spiral_template(n)
    if len(spiral) == n:
        templates.append(spiral)
    
    # Simple linear (fallback)
    templates.append([(i, 0) for i in range(n)])
    
    # Vertical linear
    templates.append([(0, i) for i in range(n)])
    
    _TEMPLATE_CACHE[n] = templates
    return templates

def _neighbours(point):
    """Yield the 4 neighbours of a point on the square lattice."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))

def _energy_fast(sequence, chain, occupied=None):
    """Fast energy computation using pre-built occupied dict if provided."""
    if occupied is None:
        occupied = {p: i for i, p in enumerate(chain)}
    else:
        # Rebuild if chain changed
        occupied = {p: i for i, p in enumerate(chain)}
    
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y = p
            for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                j = occupied.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _energy(sequence, chain):
    """Compute the HP contact energy of a chain conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, rng, temperature=1.0):
    """Randomized greedy chain growth with temperature-based selection."""
    n = len(sequence)
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Calculate gain: number of new H-H contacts
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 1
            # Count free neighbours
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Score with randomness
            score = gain * 5.0 + free * 0.1 - rng.random() * temperature
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Boltzmann-like selection for diversity
        if temperature > 0.1 and len(candidates) > 1:
            max_score = max(s for s, _ in candidates)
            weights = [math.exp((s - max_score) / (temperature * 0.5 + 0.01)) for s, _ in candidates]
            total = sum(weights)
            r = rng.random() * total
            cum = 0
            selected = candidates[-1][1]
            for (s, site), w in zip(candidates, weights):
                cum += w
                if r <= cum:
                    selected = site
                    break
        else:
            selected = max(candidates)[1]
        
        chain.append(selected)
        occupied[selected] = i
    
    return chain

def _build_occupied_set(chain):
    """Build a set of occupied positions from a chain."""
    return set(chain)

def _try_single_move(sequence, chain, occupied, i, new_pos):
    """Try moving residue i to new_pos. Returns (new_chain, new_occupied, energy_delta) or None."""
    n = len(sequence)
    old_pos = chain[i]
    
    # Check adjacency to neighbors
    if i > 0:
        if abs(new_pos[0] - chain[i-1][0]) + abs(new_pos[1] - chain[i-1][1]) != 1:
            return None
    if i < n - 1:
        if abs(new_pos[0] - chain[i+1][0]) + abs(new_pos[1] - chain[i+1][1]) != 1:
            return None
    
    # Check self-avoidance: new_pos must not be occupied (except by itself)
    if new_pos in occupied and new_pos != old_pos:
        return None
    
    # Compute energy delta using occupied dict for O(1) lookup
    old_contacts = 0
    new_contacts = 0
    
    if sequence[i] == 'H':
        for q in _neighbours(old_pos):
            j = occupied.get(q)
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                old_contacts += 1
        for q in _neighbours(new_pos):
            j = occupied.get(q)
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                new_contacts += 1
    
    energy_delta = -(new_contacts - old_contacts)
    
    new_chain = list(chain)
    new_chain[i] = new_pos
    new_occupied = set(occupied)
    new_occupied.discard(old_pos)
    new_occupied.add(new_pos)
    
    return (new_chain, new_occupied, energy_delta)

def _local_search_sa(sequence, chain, max_iters=2000, rng=None, time_limit=15.0):
    """Simulated annealing local search with single-residue and segment-reversal moves."""
    if rng is None:
        rng = random.Random()
    
    n = len(sequence)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    t0 = time.time()
    t_initial = 5.0
    t_final = 0.01
    cooling_rate = (t_final / t_initial) ** (1.0 / max(1, max_iters - 1))
    
    for it in range(max_iters):
        if time.time() - t0 > time_limit:
            break
        
        temperature = t_initial * (cooling_rate ** it)
        
        # Choose move type: 70% single-residue, 30% segment reversal
        if rng.random() < 0.7 or n < 6:
            # Single-residue move
            i = rng.randint(1, n - 2)
            old_pos = current_chain[i]
            
            best_move = None
            best_delta = float('inf')
            
            for new_pos in _neighbours(old_pos):
                result = _try_single_move(sequence, current_chain, current_occupied, i, new_pos)
                if result is None:
                    continue
                _, _, delta = result
                if delta < best_delta:
                    best_delta = delta
                    best_move = result
            
            if best_move is None:
                continue
            
            new_chain, new_occupied_dict, delta = best_move
            new_occupied = set(new_occupied_dict)
        else:
            # Segment reversal
            seg_len = rng.randint(2, min(6, n - 2))
            i = rng.randint(1, n - 1 - seg_len)
            j = i + seg_len - 1
            if j > n - 2:
                j = n - 2
            if i >= j:
                continue
            
            new_chain = list(current_chain)
            for k in range((j - i + 1) // 2):
                new_chain[i + k], new_chain[j - k] = new_chain[j - k], new_chain[i + k]
            
            # Validate: self-avoidance and unit steps
            seen = set()
            valid = True
            for p in new_chain:
                if p in seen:
                    valid = False
                    break
                seen.add(p)
            if not valid:
                continue
            for k in range(n - 1):
                if abs(new_chain[k][0] - new_chain[k+1][0]) + abs(new_chain[k][1] - new_chain[k+1][1]) != 1:
                    valid = False
                    break
            if not valid:
                continue
            
            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy
            new_occupied = set(new_chain)
        
        new_energy = current_energy + delta
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or (temperature > 0.001 and rng.random() < math.exp(-max(0, -delta) / max(temperature, 0.001))):
            current_chain = new_chain
            current_occupied = {p: i for i, p in enumerate(current_chain)}
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
    
    return best_chain, best_energy

def _segment_reversal(sequence, chain, i, j, rng):
    """Try reversing the segment from i to j. Returns new chain or None."""
    n = len(sequence)
    if i >= j or i < 1 or j > n - 2:
        return None
    
    new_chain = list(chain)
    # Reverse positions i..j
    for k in range((j - i + 1) // 2):
        new_chain[i + k], new_chain[j - k] = new_chain[j - k], new_chain[i + k]
    
    # Check self-avoidance
    seen = set()
    for p in new_chain:
        if p in seen:
            return None
        seen.add(p)
    
    # Check unit steps
    for k in range(n - 1):
        if abs(new_chain[k][0] - new_chain[k+1][0]) + abs(new_chain[k][1] - new_chain[k+1][1]) != 1:
            return None
    
    return new_chain

def _polish(sequence, chain, rng, time_limit=5.0):
    """Final polish: deterministic hill-climbing on single moves and segment reversals."""
    n = len(sequence)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_occupied = _build_occupied_set(current_chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    t0 = time.time()
    improved = True
    while improved and time.time() - t0 < time_limit:
        improved = False
        # Single-residue moves
        for i in range(1, n - 1):
            if time.time() - t0 > time_limit:
                break
            old_pos = current_chain[i]
            for new_pos in _neighbours(old_pos):
                result = _try_single_move(sequence, current_chain, current_occupied, i, new_pos)
                if result is None:
                    continue
                new_chain, new_occupied, delta = result
                if delta < 0:
                    current_chain = new_chain
                    current_occupied = new_occupied
                    current_energy += delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    improved = True
                    break
        # Segment reversals (short segments)
        if time.time() - t0 < time_limit:
            for seg_len in range(2, min(5, n - 1)):
                if time.time() - t0 > time_limit:
                    break
                for i in range(1, n - seg_len):
                    if time.time() - t0 > time_limit:
                        break
                    j = i + seg_len - 1
                    if j > n - 2:
                        break
                    new_chain = list(current_chain)
                    for k in range((j - i + 1) // 2):
                        new_chain[i + k], new_chain[j - k] = new_chain[j - k], new_chain[i + k]
                    # Validate
                    seen = set()
                    valid = True
                    for p in new_chain:
                        if p in seen:
                            valid = False
                            break
                        seen.add(p)
                    if not valid:
                        continue
                    for k in range(n - 1):
                        if abs(new_chain[k][0] - new_chain[k+1][0]) + abs(new_chain[k][1] - new_chain[k+1][1]) != 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    new_energy = _energy(sequence, new_chain)
                    if new_energy < current_energy:
                        current_chain = new_chain
                        current_occupied = set(new_chain)
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(current_chain)
                        improved = True
    
    return best_chain, best_energy

def _spiral_template(n, direction=1):
    """Generate a spiral-shaped self-avoiding walk of length n."""
    if n <= 0:
        return []
    chain = [(0, 0)]
    occupied = {(0, 0)}
    # Spiral: right, up, left, down, right, up, ...
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    di = 0
    steps_in_dir = 1
    steps_taken = 0
    while len(chain) < n:
        dx, dy = dirs[di % 4]
        for _ in range(steps_in_dir):
            if len(chain) >= n:
                break
            last = chain[-1]
            new = (last[0] + dx, last[1] + dy)
            if new in occupied:
                break
            chain.append(new)
            occupied.add(new)
            steps_taken += 1
        if len(chain) >= n:
            break
        # Check if we need to advance
        last = chain[-1]
        next_pos = (last[0] + dx, last[1] + dy)
        if next_pos in occupied:
            di += 1
            if di % 2 == 0:
                steps_in_dir += 1
    return chain[:n]

def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation."""
    start_time = time.time()
    deadline = start_time + 165.0  # Safety margin before 180s limit
    n = len(sequence)
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0
    candidate_chains = []  # Store multiple good chains for multi-start
    
    # Phase 1a: Use pre-computed Hamiltonian templates as starting points
    templates = _get_templates(n)
    for template in templates:
        if time.time() > deadline - 50.0:
            break
        if len(template) == n:
            energy = _energy(sequence, template)
            if energy < best_energy:
                best_energy = energy
                best_chain = list(template)
            candidate_chains.append((list(template), energy))
    
    # Phase 1b: Multiple greedy growths with varying temperatures
    grows_done = 0
    max_grows = 2000
    while grows_done < max_grows:
        if time.time() > deadline - 45.0:
            break
        
        temperature = 0.3 + rng.random() * 1.5
        chain = _greedy_grow(sequence, rng, temperature)
        if chain is not None:
            grows_done += 1
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
            # Keep top candidates
            if len(candidate_chains) < 12 or energy < candidate_chains[-1][1]:
                candidate_chains.append((list(chain), energy))
                candidate_chains.sort(key=lambda x: x[1])
                candidate_chains = candidate_chains[:12]
    
    # Phase 2: Multi-start local search on top candidates
    if best_chain is not None and time.time() < deadline - 20.0:
        # Refine the best chain with longer SA
        remaining = deadline - 20.0 - time.time()
        improved_chain, improved_energy = _local_search_sa(
            sequence, best_chain, max_iters=5000, rng=rng, time_limit=min(remaining, 30.0)
        )
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
        
        # Refine other candidates if time permits
        for chain_cand, _ in candidate_chains[1:4]:
            if time.time() > deadline - 15.0:
                break
            remaining = deadline - 15.0 - time.time()
            if remaining < 5.0:
                break
            improved_chain, improved_energy = _local_search_sa(
                sequence, chain_cand, max_iters=2000, rng=rng, time_limit=min(remaining, 12.0)
            )
            if improved_energy < best_energy:
                best_chain = improved_chain
                best_energy = improved_energy
    
    # Phase 3: Polish
    if best_chain is not None and time.time() < deadline - 5.0:
        remaining = deadline - 5.0 - time.time()
        polished_chain, polished_energy = _polish(
            sequence, best_chain, rng, time_limit=min(remaining, 10.0)
        )
        if polished_energy < best_energy:
            best_chain = polished_chain
            best_energy = polished_energy
    
    # Fallback
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
