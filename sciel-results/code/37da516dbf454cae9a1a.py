"""HP lattice protein folding: combined diverse init + SA with greedy polish
and efficient incremental energy tracking."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _neighbours(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))

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

def _compute_contacts(sequence, chain):
    """Return dict mapping (i,j) -> 1 for each H-H contact pair (i<j, |i-j|>1)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts.add((i, j))
    return contacts

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
            # Look-ahead: can the next residue also make a contact?
            future_gain = 0
            if i + 1 < n and sequence[i + 1] == 'H':
                for q in _neighbours(site):
                    if q not in occupied:
                        for r in _neighbours(q):
                            if r in occupied:
                                j = occupied[r]
                                if j < i - 1 and sequence[j] == 'H':
                                    future_gain += 1
                        break
            # Count free neighbours
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Score with randomness
            score = gain * 10.0 + future_gain * 3.0 + free * 0.3 + rng.random() * temperature
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

def _serpentine_template(n):
    """Compact self-avoiding serpentine walk of length n."""
    pts = []
    x, y = 0, 0
    pts.append((x, y))
    direction = (1, 0)
    for _ in range(1, n):
        x += direction[0]
        y += direction[1]
        pts.append((x, y))
        if (len(pts) % 2) == 0:
            direction = (0, 1) if direction[0] == 1 else (-1, 0)
        else:
            direction = (1, 0) if direction[0] == -1 else (0, 1)
    if len(set(pts)) != n:
        pts = [(i, 0) for i in range(n)]
    return pts

def _diverse_init(sequence, n, rng):
    chains = []
    for temp in [0.1, 0.5, 1.0, 2.0]:
        c = _greedy_grow(sequence, rng, temperature=temp)
        if c is not None and len(c) == n:
            chains.append(c)
    for _ in range(3):
        chain = [(0, 0)]
        occupied = {(0, 0)}
        ok = True
        for i in range(1, n):
            last = chain[-1]
            opts = [s for s in _neighbours(last) if s not in occupied]
            if not opts:
                ok = False
                break
            site = rng.choice(opts)
            chain.append(site)
            occupied.add(site)
        if ok and len(chain) == n:
            chains.append(chain)
    serp = _serpentine_template(n)
    chains.append(serp)
    return chains

def _sa_local_search(sequence, chain, max_iters=2000, rng=None, time_limit=15.0):
    """Simulated annealing with single-residue moves and segment reversals."""
    if rng is None:
        rng = random.Random()
    
    n = len(sequence)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_occupied = set(current_chain)
    current_energy = _energy(sequence, current_chain)
    current_contacts = _compute_contacts(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    best_contacts = set(current_contacts)
    
    t0 = time.time()
    t_initial = 5.0
    t_final = 0.01
    cooling_rate = (t_final / t_initial) ** (1.0 / max(1, max_iters - 1))
    
    for it in range(max_iters):
        if time.time() - t0 > time_limit:
            break
        
        temperature = t_initial * (cooling_rate ** it)
        
        # Pick a random residue to move
        i = rng.randint(1, n - 2)
        old_pos = current_chain[i]
        
        # Try all 4 possible moves
        best_move = None
        best_delta = float('inf')
        
        for new_pos in _neighbours(old_pos):
            # Check adjacency to neighbors
            if i > 0:
                if abs(new_pos[0] - current_chain[i-1][0]) + abs(new_pos[1] - current_chain[i-1][1]) != 1:
                    continue
            if i < n - 1:
                if abs(new_pos[0] - current_chain[i+1][0]) + abs(new_pos[1] - current_chain[i+1][1]) != 1:
                    continue
            
            # Check self-avoidance
            if new_pos in current_occupied and new_pos != old_pos:
                continue
            
            # Compute energy delta incrementally
            old_energy = 0
            new_energy = 0
            
            if sequence[i] == 'H':
                # Remove old contacts involving residue i
                for q in _neighbours(old_pos):
                    j = None
                    for k, p in enumerate(current_chain):
                        if p == q:
                            j = k
                            break
                    if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                        old_energy += 1
                
                # Add new contacts involving residue i at new_pos
                for q in _neighbours(new_pos):
                    j = None
                    for k, p in enumerate(current_chain):
                        if p == q:
                            j = k
                            break
                    if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                        new_energy += 1
            
            energy_delta = -(new_energy - old_energy)
            
            if energy_delta < best_delta:
                best_delta = energy_delta
                best_move = (new_pos, energy_delta)
        
        if best_move is None:
            # Try segment reversal occasionally
            if n >= 6 and rng.random() < 0.3:
                seg_len = rng.randint(3, min(9, n // 2))
                a = rng.randint(1, n - seg_len - 1)
                b = a + seg_len - 1
                if b + 1 < n:
                    left = current_chain[a - 1]
                    right = current_chain[b + 1]
                    if (abs(left[0] - current_chain[b][0]) + abs(left[1] - current_chain[b][1]) == 1 and
                            abs(current_chain[a][0] - right[0]) + abs(current_chain[a][1] - right[1]) == 1):
                        seg = current_chain[a:b + 1]
                        seg.reverse()
                        new_chain = current_chain[:a] + seg + current_chain[b+1:]
                        
                        # Check self-avoidance
                        if len(set(new_chain)) == n:
                            new_energy = _energy(sequence, new_chain)
                            delta = new_energy - current_energy
                            if delta < 0 or (temperature > 0.001 and rng.random() < math.exp(-max(0, -delta) / max(temperature, 0.001))):
                                current_chain = new_chain
                                current_occupied = set(current_chain)
                                current_energy = new_energy
                                current_contacts = _compute_contacts(sequence, current_chain)
                                if current_energy < best_energy:
                                    best_energy = current_energy
                                    best_chain = list(current_chain)
                                    best_contacts = set(current_contacts)
                        continue
        
        new_pos, delta = best_move
        new_energy = current_energy + delta
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or (temperature > 0.001 and rng.random() < math.exp(-max(0, -delta) / max(temperature, 0.001))):
            current_occupied.discard(old_pos)
            current_occupied.add(new_pos)
            current_chain[i] = new_pos
            current_energy = new_energy
            current_contacts = _compute_contacts(sequence, current_chain)
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                best_contacts = set(current_contacts)
    
    return best_chain, best_energy

def _greedy_polish(sequence, chain, rng, time_limit=5.0):
    """Deterministic hill-climbing: scan all pivot moves, accept any improving change."""
    n = len(sequence)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_occupied = set(current_chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    t0 = time.time()
    improved = True
    while improved and time.time() - t0 < time_limit:
        improved = False
        
        for i in range(1, n - 1):
            if time.time() - t0 > time_limit:
                break
            
            old_pos = current_chain[i]
            for new_pos in _neighbours(old_pos):
                # Check adjacency
                if abs(new_pos[0] - current_chain[i-1][0]) + abs(new_pos[1] - current_chain[i-1][1]) != 1:
                    continue
                if abs(new_pos[0] - current_chain[i+1][0]) + abs(new_pos[1] - current_chain[i+1][1]) != 1:
                    continue
                
                # Check self-avoidance
                if new_pos in current_occupied and new_pos != old_pos:
                    continue
                
                # Compute energy delta
                old_energy = 0
                new_energy = 0
                
                if sequence[i] == 'H':
                    for q in _neighbours(old_pos):
                        j = None
                        for k, p in enumerate(current_chain):
                            if p == q:
                                j = k
                                break
                        if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                            old_energy += 1
                    
                    for q in _neighbours(new_pos):
                        j = None
                        for k, p in enumerate(current_chain):
                            if p == q:
                                j = k
                                break
                        if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                            new_energy += 1
                
                delta = -(new_energy - old_energy)
                
                if delta < 0:
                    # Accept improving move
                    current_occupied.discard(old_pos)
                    current_occupied.add(new_pos)
                    current_chain[i] = new_pos
                    current_energy += delta
                    improved = True
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    break  # Restart scan from beginning after improvement
    
    return best_chain, best_energy

def fold(sequence, dimension):
    start_time = time.time()
    deadline = start_time + 170.0  # safety margin before 180s
    n = len(sequence)
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0
    
    inits = _diverse_init(sequence, n, rng)
    
    for init_chain in inits:
        e = _energy(sequence, init_chain)
        if e < best_energy:
            best_energy = e
            best_chain = list(init_chain)
    
    # SA local search on each diverse init
    for init_chain in inits:
        if time.time() > deadline - 25.0:
            break
        remaining = deadline - time.time()
        time_per_init = max(3.0, remaining / max(1, len(inits)))
        improved_chain, improved_energy = _sa_local_search(
            sequence, init_chain, max_iters=2000, rng=rng,
            time_limit=time_per_init)
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = list(improved_chain)
    
    # Fresh greedy grows + SA polish
    grows_done = 0
    while time.time() < deadline - 20.0:
        temperature = 0.3 + rng.random() * 2.5
        chain = _greedy_grow(sequence, rng, temperature=temperature)
        if chain is not None and len(chain) == n:
            grows_done += 1
            e = _energy(sequence, chain)
            if e < best_energy or e < -3:
                time_left = deadline - time.time() - 15.0
                if time_left > 5.0:
                    improved_chain, improved_energy = _sa_local_search(
                        sequence, chain, max_iters=1500, rng=rng,
                        time_limit=min(10.0, time_left * 0.4))
                    if improved_energy < best_energy:
                        best_energy = improved_energy
                        best_chain = list(improved_chain)
                elif e < best_energy:
                    best_energy = e
                    best_chain = list(chain)
        if grows_done > 2000:
            break
    
    # Final greedy polish on the best chain
    if best_chain is not None and time.time() < deadline - 3.0:
        final_chain, final_energy = _greedy_polish(
            sequence, best_chain, rng=rng, time_limit=deadline - 2.0)
        if final_energy < best_energy:
            best_chain = final_chain
            best_energy = final_energy
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
