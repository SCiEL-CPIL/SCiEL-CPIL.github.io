# EVOLVE-BLOCK-START
import random
import time
import math

def _neighbours2d(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))

def _compute_energy(sequence, chain):
    """Compute HP energy: -number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours2d(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _fast_grow(sequence, rng):
    """Randomized greedy chain growth biased toward H-H contact formation."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours2d(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours2d(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours2d(site) if q not in occupied)
            options.append((gain * 2.0 + 0.1 * free + rng.random() * 0.5, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _energy_delta_for_move(sequence, chain, i, old_pos, new_pos, occupied):
    """Compute energy delta for moving residue i from old_pos to new_pos.
    Returns (delta_energy, valid) where delta_energy = new_energy - old_energy."""
    n = len(chain)
    if new_pos in occupied:
        return 0, False
    
    # Check self-avoidance: new_pos must not be adjacent to non-consecutive residues
    for q in _neighbours2d(new_pos):
        j = occupied.get(q)
        if j is not None and abs(j - i) > 1:
            return 0, False
    
    # Compute contact changes involving residue i
    delta = 0
    is_h_i = sequence[i] == "H"
    
    # Remove old contacts from old_pos
    if is_h_i:
        for q in _neighbours2d(old_pos):
            j = occupied.get(q)
            if j is not None and abs(j - i) > 1 and sequence[j] == "H":
                delta += 1  # removing a contact increases energy by 1
    
    # Add new contacts from new_pos
    if is_h_i:
        for q in _neighbours2d(new_pos):
            j = occupied.get(q)
            if j is not None and abs(j - i) > 1 and sequence[j] == "H":
                delta -= 1  # adding a contact decreases energy by 1
    
    return delta, True

def _local_improve_fast(sequence, chain, max_iter=2000):
    """Fast local search using incremental energy deltas."""
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    for _ in range(max_iter):
        improved = False
        # Try moving each interior residue
        for i in range(1, n - 1):
            current_pos = best_chain[i]
            for new_pos in _neighbours2d(current_pos):
                if new_pos in occupied:
                    continue
                
                # Quick validity check
                valid = True
                for q in _neighbours2d(new_pos):
                    j = occupied.get(q)
                    if j is not None and abs(j - i) > 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Compute delta
                delta, _ = _energy_delta_for_move(sequence, best_chain, i, current_pos, new_pos, occupied)
                
                if delta < 0:
                    # Accept the move
                    best_chain[i] = new_pos
                    occupied.pop(current_pos)
                    occupied[new_pos] = i
                    best_energy += delta
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_chain, best_energy

def _min_conflicts_fold(sequence, max_iter=5000):
    """Min-conflicts style local search: repeatedly move the residue with most 'conflicts'."""
    n = len(sequence)
    if n < 4:
        chain = [(i, 0) for i in range(n)]
        return chain, _compute_energy(sequence, chain)
    
    # Start with a random valid conformation from greedy growth
    rng = random.Random(42 + n)
    chain = _fast_grow(sequence, rng)
    if chain is None:
        chain = [(i, 0) for i in range(n)]
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    for _ in range(max_iter):
        # Find the residue with the most potential improvement
        best_i, best_delta, best_new_pos = -1, 0, None
        for i in range(1, n - 1):
            current_pos = best_chain[i]
            for new_pos in _neighbours2d(current_pos):
                if new_pos in occupied:
                    continue
                # Quick validity check
                valid = True
                for q in _neighbours2d(new_pos):
                    j = occupied.get(q)
                    if j is not None and abs(j - i) > 1:
                        valid = False
                        break
                if not valid:
                    continue
                delta, _ = _energy_delta_for_move(sequence, best_chain, i, current_pos, new_pos, occupied)
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_new_pos = new_pos
        
        if best_i == -1 or best_delta >= 0:
            break
        
        # Apply the best move
        old_pos = best_chain[best_i]
        best_chain[best_i] = best_new_pos
        occupied.pop(old_pos)
        occupied[best_new_pos] = best_i
        best_energy += best_delta
    
    return best_chain, best_energy

def _regrow_sa_fast(sequence, start_chain, rng, max_iter=1000):
    """Simulated annealing on chain regrowth: perturb a prefix and regrow the suffix."""
    n = len(sequence)
    if n < 6:
        return _fast_grow(sequence, rng)
    
    best_chain = list(start_chain)
    best_energy = _compute_energy(sequence, best_chain)
    
    for _ in range(max_iter):
        # Pick a random split point
        split = rng.randint(3, n - 3)
        
        # Keep prefix, regrow suffix with randomness
        prefix = best_chain[:split]
        occupied = {p: i for i, p in enumerate(prefix)}
        chain = list(prefix)
        
        success = True
        for i in range(split, n):
            options = []
            for site in _neighbours2d(chain[-1]):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == "H":
                    gain = sum(1 for q in _neighbours2d(site)
                               if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
                free = sum(1 for q in _neighbours2d(site) if q not in occupied)
                options.append((gain * 2.0 + 0.1 * free + rng.random(), site))
            if not options:
                success = False
                break
            site = max(options)[1]
            occupied[site] = i
            chain.append(site)
        
        if not success:
            continue
        
        energy = _compute_energy(sequence, chain)
        if energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    return best_chain

def _kmc_search(sequence, chain, occupied, max_iter=1000):
    """
    Kinetic Monte Carlo search using Kink moves and Segment Regrowth (Slither).
    """
    n = len(chain)
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    
    current_chain = list(chain)
    current_occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, current_chain)
    
    rng = random.Random(999)
    
    for _ in range(max_iter):
        # Randomly choose move type
        # 80% Kink, 20% Segment Regrowth (Slither)
        
        if rng.random() < 0.8:
            # Kink move
            i = rng.randint(1, n - 2)
            old_pos = current_chain[i]
            candidates = [p for p in _neighbours2d(old_pos) if p not in current_occupied]
            if not candidates:
                continue
            new_pos = rng.choice(candidates)
            
            # Check validity
            valid = True
            for q in _neighbours2d(new_pos):
                j = current_occupied.get(q)
                if j is not None and abs(j - i) > 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            delta, _ = _energy_delta_for_move(sequence, current_chain, i, old_pos, new_pos, current_occupied)
            
            # Metropolis criterion (T=0.5)
            if delta < 0 or rng.random() < math.exp(-delta / 0.5):
                # Accept
                current_chain[i] = new_pos
                del current_occupied[old_pos]
                current_occupied[new_pos] = i
                current_energy += delta
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        else:
            # Segment Regrowth (Slither)
            # Pick a segment length
            seg_len = rng.randint(2, min(10, n - 2))
            start = rng.randint(1, n - seg_len - 1)
            end = start + seg_len - 1
            
            # Remove segment
            segment = current_chain[start:end+1]
            for pos in segment:
                del current_occupied[pos]
            
            # Regrow segment
            # We need to connect left_anchor (start-1) to right_anchor (end+1)
            left_anchor = current_chain[start-1]
            right_anchor = current_chain[end+1] if end < n-1 else None
            
            # Try to regrow
            success = False
            # Simple regrowth from left
            new_segment = []
            current_pos = left_anchor
            # We need to place seg_len residues
            # The last one must be adjacent to right_anchor
            
            # This is a constraint satisfaction problem. 
            # For simplicity, we just try to regrow and check if it works.
            # If it fails, we restore.
            
            # To save time, we only try a few random paths.
            for attempt in range(3):
                new_segment = []
                cur = left_anchor
                valid_path = True
                local_occupied = set()
                
                for k in range(seg_len):
                    candidates = [s for s in _neighbours2d(cur) if s not in current_occupied and s not in local_occupied]
                    if not candidates:
                        valid_path = False
                        break
                    
                    # If this is the last residue, it must be adjacent to right_anchor
                    if k == seg_len - 1:
                        if right_anchor is not None:
                            # Filter candidates that are adjacent to right_anchor
                            filtered = [c for c in candidates if right_anchor in _neighbours2d(c)]
                            if not filtered:
                                valid_path = False
                                break
                            next_pos = rng.choice(filtered)
                        else:
                            next_pos = rng.choice(candidates)
                    else:
                        next_pos = rng.choice(candidates)
                    
                    new_segment.append(next_pos)
                    local_occupied.add(next_pos)
                    cur = next_pos
                
                if valid_path:
                    # Check if the new segment is valid (no self collision, no chain collision)
                    # Already checked via local_occupied and current_occupied
                    # But we must ensure the new segment doesn't collide with itself
                    # local_occupied ensures that.
                    
                    # Calculate new energy
                    # We can't easily do incremental delta for whole segment.
                    # We will just reconstruct the chain and compute energy.
                    # This is O(N) per successful slither.
                    # Since slithers are rare (20%), and we have 120s, it's acceptable.
                    
                    new_chain = current_chain[:start] + new_segment + current_chain[end+1:]
                    new_energy = _compute_energy(sequence, new_chain)
                    delta = new_energy - current_energy
                    
                    if delta < 0 or rng.random() < math.exp(-delta / 0.5):
                        # Accept
                        current_chain = new_chain
                        current_occupied = {p: i for i, p in enumerate(new_chain)}
                        current_energy = new_energy
                        success = True
                        
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(current_chain)
                        break
                # Else, try next random path
            
            # If we didn't accept, restore the segment
            if not success:
                for idx, pos in enumerate(segment):
                    current_occupied[pos] = start + idx
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Best conformation using KMC and fast local search."""
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin before 210s kill
    
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0  # 0 is better than any negative energy
    
    n = len(sequence)
    
    # Phase 1: Multiple fast growth + local improvement
    num_phase1 = max(50, 3000 // n)
    for attempt in range(num_phase1):
        if time.time() > deadline:
            break
        
        chain = _fast_grow(sequence, rng)
        if chain is None:
            continue
        
        chain, energy = _local_improve_fast(sequence, chain, max_iter=min(3000, max(200, 5000 // n)))
        
        if energy < best_energy:
            best, best_energy = chain, energy
    
    # Phase 2: KMC search on the best found so far
    if best is not None:
        if time.time() < deadline - 10:
            # Initialize KMC with best chain
            occupied = {p: i for i, p in enumerate(best)}
            kmc_chain, kmc_energy = _kmc_search(sequence, best, occupied, max_iter=min(2000, max(500, 10000 // n)))
            if kmc_energy < best_energy:
                best, best_energy = kmc_chain, kmc_energy
    
    # Phase 3: Regrowth with SA on the best found
    if best is not None and time.time() < deadline - 5:
        regrown = _regrow_sa_fast(sequence, best, rng, max_iter=min(500, max(50, 2000 // n)))
        if regrown is not None:
            regrown, regrown_energy = _local_improve_fast(sequence, regrown, max_iter=min(2000, max(100, 3000 // n)))
            if regrown_energy < best_energy:
                best, best_energy = regrown, regrown_energy
    
    if best is None:  # fallback: straight line
        best = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
