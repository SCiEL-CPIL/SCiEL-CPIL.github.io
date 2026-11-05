"""HP lattice protein folding using pre-computed compact skeletons with contact-guided refinement."""
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

def _delta_energy(sequence, chain, i, new_pos, occupied):
    """Compute the change in energy if residue i moves from chain[i] to new_pos."""
    old_pos = chain[i]
    if new_pos == old_pos:
        return 0
    
    old_contacts = 0
    if sequence[i] == 'H':
        for q in _neighbours(old_pos):
            j = occupied.get(q)
            if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                old_contacts += 1
    
    new_contacts = 0
    if sequence[i] == 'H':
        for q in _neighbours(new_pos):
            j = occupied.get(q)
            if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                new_contacts += 1
    
    return -(new_contacts - old_contacts)

def _generate_compact_skeletons(n, num_templates=40):
    """Generate multiple compact self-avoiding walk skeletons of length n.
    Uses spiral patterns with variations to create diverse compact structures."""
    templates = []
    seen = set()
    
    start_dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    for t in range(num_templates):
        rng = random.Random(1000 + t)
        start_dir = start_dirs[t % 4]
        turn_pattern = t % 4  # 0: standard right, 1: alternate, 2: left, 3: random-ish
        
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        direction = start_dir
        
        for i in range(1, n):
            found = False
            x, y = chain[-1]
            
            if turn_pattern == 0:
                # Standard right-turn spiral
                for step in range(4):
                    dx, dy = direction
                    new_pos = (x + dx, y + dy)
                    if new_pos not in occupied:
                        chain.append(new_pos)
                        occupied[new_pos] = i
                        found = True
                        break
                    direction = (dy, -dx)
            elif turn_pattern == 1:
                # Alternate left/right turns
                for step in range(4):
                    dx, dy = direction
                    new_pos = (x + dx, y + dy)
                    if new_pos not in occupied:
                        chain.append(new_pos)
                        occupied[new_pos] = i
                        found = True
                        break
                    if step % 2 == 0:
                        direction = (dy, -dx)
                    else:
                        direction = (-dy, dx)
            elif turn_pattern == 2:
                # Left-turn spiral
                for step in range(4):
                    dx, dy = direction
                    new_pos = (x + dx, y + dy)
                    if new_pos not in occupied:
                        chain.append(new_pos)
                        occupied[new_pos] = i
                        found = True
                        break
                    direction = (-dy, dx)
            else:
                # Compactness-guided random walk
                candidates = []
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    new_pos = (x + dx, y + dy)
                    if new_pos not in occupied:
                        # Prefer positions closer to centroid
                        cx = sum(p[0] for p in chain) / len(chain)
                        cy = sum(p[1] for p in chain) / len(chain)
                        dist = abs(new_pos[0] - cx) + abs(new_pos[1] - cy)
                        candidates.append((dist, new_pos))
                
                if candidates:
                    candidates.sort(key=lambda c: c[0])
                    if len(candidates) > 1 and rng.random() < 0.3:
                        new_pos = candidates[1][1]
                    else:
                        new_pos = candidates[0][1]
                    chain.append(new_pos)
                    occupied[new_pos] = i
                    found = True
                else:
                    break
            
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

def _contact_guided_growth(sequence, rng):
    """Grow chain by explicitly targeting H-H contacts."""
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
            
            # Primary: direct H-H contact gain
            if sequence[i] == 'H':
                direct_contacts = 0
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            direct_contacts += 1
                score += direct_contacts * 15.0
            
            # Secondary: potential future contacts
            if sequence[i] == 'H':
                future_h_count = sum(1 for j in range(i + 2, n) if sequence[j] == 'H')
                if future_h_count > 0:
                    empty_neighbors = sum(1 for q in _neighbours(site) if q not in occupied)
                    score += empty_neighbors * 2.0
            
            # Tertiary: compactness
            if len(chain) > 3:
                cx = sum(p[0] for p in chain) / len(chain)
                cy = sum(p[1] for p in chain) / len(chain)
                dist_to_centroid = abs(site[0] - cx) + abs(site[1] - cy)
                score -= dist_to_centroid * 0.5
            
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

def _greedy_polish(sequence, chain, rng, max_iter=300):
    """Systematic greedy polish: scan all pivot moves and accept any energy-improving change."""
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
        
        if not improved:
            break
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation.
    Uses pre-computed compact skeletons, contact-guided growth, and multi-stage refinement."""
    start_time = time.time()
    deadline = start_time + 190.0
    n = len(sequence)
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0
    
    # Strategy 1: Pre-computed compact skeletons
    templates = _generate_compact_skeletons(n, num_templates=40)
    
    for template in templates:
        if time.time() > deadline - 30.0:
            break
        
        energy = _energy(sequence, template)
        if energy < best_energy:
            best_energy = energy
            best_chain = list(template)
        
        # Try reversed sequence
        rev_sequence = sequence[::-1]
        energy_rev = _energy(rev_sequence, template)
        if energy_rev < best_energy:
            best_energy = energy_rev
            best_chain = list(template)
    
    # Strategy 2: Contact-guided growth
    if time.time() < deadline - 30.0:
        for _ in range(100):
            if time.time() > deadline - 30.0:
                break
            chain = _contact_guided_growth(sequence, rng)
            if chain is not None and len(chain) == n:
                energy = _energy(sequence, chain)
                if energy < best_energy:
                    best_energy = energy
                    best_chain = list(chain)
    
    # Strategy 3: Local refinement on the best chain
    if best_chain is not None and time.time() < deadline - 15.0:
        improved_chain, improved_energy = _local_search(sequence, best_chain, rng, max_iter=300)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    # Strategy 4: Simulated annealing on the best chain
    if best_chain is not None and time.time() < deadline - 10.0:
        sa_chain, sa_energy = _simulated_annealing(sequence, best_chain, rng, start_temp=1.0, end_temp=0.01, max_iter=1500)
        if sa_energy < best_energy:
            best_chain = sa_chain
            best_energy = sa_energy
    
    # Strategy 5: Greedy polish
    if best_chain is not None and time.time() < deadline - 5.0:
        polished_chain, polished_energy = _greedy_polish(sequence, best_chain, rng, max_iter=300)
        if polished_energy < best_energy:
            best_chain = polished_chain
            best_energy = polished_energy
    
    # Fallback: if no valid chain was found, return a straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    # Convert to list of lists as required
    return [list(p) for p in best_chain]
