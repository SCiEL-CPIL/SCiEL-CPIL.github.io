"""HP lattice protein folding using multi-strategy search with simulated annealing."""
import time
import random
import math

def _neighbours(point):
    """Yield the 4 neighbours of a point on the square lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    """Compute the HP contact energy of a chain conformation.
    Energy = - (number of H-H contacts) where contacts are non-adjacent H residues on neighbouring lattice sites."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _delta_energy(sequence, chain, i, new_pos, occupied):
    """Compute the change in energy if residue i moves from chain[i] to new_pos.
    Only considers contacts involving residue i and its neighbors."""
    old_pos = chain[i]
    if new_pos == old_pos:
        return 0
    
    # Remove old contacts involving residue i
    old_contacts = 0
    if sequence[i] == 'H':
        for q in _neighbours(old_pos):
            j = occupied.get(q)
            if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                old_contacts += 1
    
    # Add new contacts involving residue i
    new_contacts = 0
    if sequence[i] == 'H':
        for q in _neighbours(new_pos):
            j = occupied.get(q)
            if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                new_contacts += 1
    
    return -(new_contacts - old_contacts)

def _random_walk(sequence, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)] if n == 1 else []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for i in range(1, n):
        last = chain[-1]
        rng.shuffle(directions)
        found = False
        for dx, dy in directions:
            new_pos = (last[0] + dx, last[1] + dy)
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied[new_pos] = i
                found = True
                break
        if not found:
            return None
    
    return chain

def _greedy_grow(sequence, rng):
    """Grow a chain greedily, preferring positions that maximize H-H contacts."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)] if n == 1 else []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            score = 0.0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            score += 10.0
            
            free_count = sum(1 for q in _neighbours(site) if q not in occupied)
            score += free_count * 0.1
            score += rng.random() * 0.5
            
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: -x[0])
        if rng.random() < 0.2 and len(candidates) > 1:
            best_site = candidates[rng.randint(1, min(2, len(candidates) - 1))][1]
        else:
            best_site = candidates[0][1]
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain

def _spiral_template(n, rng):
    """Generate a compact spiral-shaped self-avoiding walk of length n."""
    if n <= 1:
        return [(0, 0)] if n == 1 else []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    direction = (1, 0)
    
    for i in range(1, n):
        found = False
        for step in range(4):
            x, y = chain[-1]
            dx, dy = direction
            new_pos = (x + dx, y + dy)
            
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied[new_pos] = i
                found = True
                break
            
            direction = (dy, -dx)
        
        if not found:
            return None
    
    return chain

def _local_search(sequence, chain, rng, max_iter=500):
    """Local search: try moving each interior residue to an adjacent empty position."""
    n = len(sequence)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    for _ in range(max_iter):
        improved = False
        
        for i in range(1, n - 1):
            old_pos = best_chain[i]
            
            for new_pos in _neighbours(old_pos):
                if new_pos in occupied:
                    continue
                
                prev_pos = best_chain[i - 1]
                next_pos = best_chain[i + 1]
                
                dist_prev = abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1])
                dist_next = abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1])
                
                if dist_prev != 1 or dist_next != 1:
                    continue
                
                delta_e = _delta_energy(sequence, best_chain, i, new_pos, occupied)
                
                if delta_e < 0:
                    del occupied[old_pos]
                    occupied[new_pos] = i
                    best_chain[i] = new_pos
                    best_energy += delta_e
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best_chain, best_energy

def _simulated_annealing(sequence, chain, rng, start_temp=1.0, end_temp=0.01, max_iter=2000):
    """Simulated annealing to escape local minima."""
    n = len(sequence)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    temp = start_temp
    cooling = (math.log(start_temp / end_temp) / max_iter) if max_iter > 0 else 0
    
    for _ in range(max_iter):
        temp *= math.exp(-cooling)
        if temp < end_temp:
            break
        
        i = rng.randint(1, n - 2)
        old_pos = current_chain[i]
        
        moves = []
        for new_pos in _neighbours(old_pos):
            if new_pos in occupied:
                continue
            
            prev_pos = current_chain[i - 1]
            next_pos = current_chain[i + 1]
            
            dist_prev = abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1])
            dist_next = abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1])
            
            if dist_prev == 1 and dist_next == 1:
                moves.append(new_pos)
        
        if not moves:
            continue
        
        new_pos = moves[rng.randint(0, len(moves) - 1)]
        delta_e = _delta_energy(sequence, current_chain, i, new_pos, occupied)
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(temp, 1e-10)):
            del occupied[old_pos]
            occupied[new_pos] = i
            current_chain[i] = new_pos
            current_energy += delta_e
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
    
    return best_chain, best_energy

def _generate_compact_templates(n, num_templates=20):
    """Pre-compute multiple compact templates of length n."""
    templates = []
    seen = set()
    
    for t in range(num_templates):
        rng = random.Random(1000 + t)
        
        start_dirs = [(1,0), (0,1), (-1,0), (0,-1)]
        start_dir = start_dirs[t % 4]
        
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        direction = start_dir
        
        for i in range(1, n):
            found = False
            for step in range(4):
                x, y = chain[-1]
                dx, dy = direction
                new_pos = (x + dx, y + dy)
                
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied[new_pos] = i
                    found = True
                    break
                
                if t % 3 == 0:
                    direction = (-dy, dx)
                else:
                    direction = (dy, -dx)
            
            if not found:
                break
        
        if len(chain) == n:
            min_x = min(p[0] for p in chain)
            min_y = min(p[1] for p in chain)
            normalized = tuple((p[0] - min_x, p[1] - min_y) for p in chain)
            if normalized not in seen:
                seen.add(normalized)
                templates.append(list(chain))
    
    return templates

def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation.
    Uses a portfolio of construction methods plus local refinement and simulated annealing."""
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin before 210s limit
    n = len(sequence)
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0  # Start with unfolded (energy 0)
    
    # Strategy 1: Use pre-computed compact templates
    templates = _generate_compact_templates(n, num_templates=30)
    
    for template in templates:
        if time.time() > deadline - 30.0:
            break
        
        chain, energy = template, _energy(sequence, template)
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
        
        # Try reversed sequence
        rev_sequence = sequence[::-1]
        energy_rev = _energy(rev_sequence, template)
        if energy_rev < best_energy:
            best_energy = energy_rev
            best_chain = list(template)
    
    # Strategy 2: Multiple randomized constructions
    num_attempts = 0
    max_attempts = 3000
    
    while num_attempts < max_attempts:
        if time.time() > deadline - 30.0:
            break
        
        strategy = rng.randint(0, 2)
        
        if strategy == 0:
            chain = _greedy_grow(sequence, rng)
        elif strategy == 1:
            chain = _spiral_template(n, rng)
        else:
            chain = _random_walk(sequence, rng)
        
        if chain is not None and len(chain) == n:
            num_attempts += 1
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
    
    # Strategy 3: Local refinement on the best chain
    if best_chain is not None and time.time() < deadline - 10.0:
        improved_chain, improved_energy = _local_search(sequence, best_chain, rng, max_iter=300)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    # Strategy 4: Simulated annealing on the best chain
    if best_chain is not None and time.time() < deadline - 5.0:
        sa_chain, sa_energy = _simulated_annealing(sequence, best_chain, rng, start_temp=1.0, end_temp=0.01, max_iter=1500)
        if sa_energy < best_energy:
            best_chain = sa_chain
            best_energy = sa_energy
    
    # Fallback: if no valid chain was found, return a straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    # Convert to list of lists as required
    return [list(p) for p in best_chain]
