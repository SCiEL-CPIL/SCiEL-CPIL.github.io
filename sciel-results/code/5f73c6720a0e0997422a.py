"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import random
import time
import heapq

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - time.time()


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Energy = -(# of H-H non-adjacent lattice contacts)."""
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _beam_search(sequence, dimension, beam_width=60, max_time=None):
    """Beam search for initial chain conformation."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,) * dimension]
    
    if max_time is None:
        max_time = _time_left() * 0.3
    
    # Each beam entry: (score, chain, occupied_set, energy)
    # Score = energy + small tiebreaker
    start = (0,) * dimension
    occupied = {start}
    initial_chain = [start]
    initial_energy = 0
    
    # Beam: list of (neg_score, counter, chain, occupied, energy)
    # We maximize score = energy (more negative is better), so we minimize neg_score
    beam = []
    beam.append((0, 0, initial_chain, occupied, initial_energy))
    
    counter = 0
    for i in range(1, n):
        new_beam = []
        for entry in beam:
            neg_score, _, chain, occupied, energy = entry
            last_pos = chain[-1]
            
            for site in _neighbours(last_pos):
                if site in occupied:
                    continue
                
                # Compute energy gain
                gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        if q in occupied:
                            # Find which residue is at q
                            # We need to track position->index mapping
                            # For simplicity in beam search, just check if it's an H
                            # Actually we need the index. Let's maintain a pos_to_idx dict
                            pass
                
                # Simpler: just use energy computation on the extended chain
                # But that's expensive. Let's do incremental.
                
                # For now, use a simplified scoring
                # Count potential H-H contacts
                new_energy = energy
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        if q in occupied:
                            # Check if the residue at q is H
                            # We need to know which index is at q
                            # This is getting complex. Let's just extend and compute
                            pass
                
                # Simplified: just use greedy scoring with random tiebreak
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                
                # Build new entry
                new_chain = chain + [site]
                new_occupied = occupied | {site}
                
                # Compute actual energy for this extension
                # To be efficient, we'll compute incremental energy
                if sequence[i] == 'H':
                    # Count H-H contacts involving position i
                    for q in _neighbours(site):
                        if q in occupied:
                            # Find index of residue at q
                            # We need a position->index mapping
                            # For beam search, let's maintain this
                            pass
                
                # For simplicity and correctness, compute full energy
                # But that's O(n) per extension, which is expensive for large beams
                # Let's use a heuristic score instead
                
                score = -i * 0.1 + free * 0.05 + random.random() * 0.1
                
                # Actually, let's compute the energy properly but incrementally
                # We'll maintain a pos_to_idx mapping
                # For now, let's just use a simple heuristic
                
                new_beam.append((score, counter, new_chain, new_occupied, energy))
                counter += 1
        
        # Sort by score (we want lowest score = best)
        new_beam.sort(key=lambda x: (x[0], x[1]))
        
        # Keep top beam_width
        beam = new_beam[:beam_width]
        
        if not beam:
            return None
        
        if time.time() > _START_TIME + (max_time if max_time else 0):
            break
    
    if not beam:
        return None
    
    # Return the best chain
    best_entry = min(beam, key=lambda x: x[0])
    return best_entry[2]


def _beam_search_optimized(sequence, dimension, beam_width=60):
    """Optimized beam search with incremental energy tracking."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,) * dimension]
    
    # Each entry: (energy, chain, pos_to_idx, occupied)
    start = (0,) * dimension
    initial_chain = [start]
    initial_pos_to_idx = {start: 0}
    initial_occupied = {start}
    
    beam = [(0, initial_chain, initial_pos_to_idx, initial_occupied)]
    
    for i in range(1, n):
        new_beam = []
        for energy, chain, pos_to_idx, occupied in beam:
            last_pos = chain[-1]
            
            for site in _neighbours(last_pos):
                if site in occupied:
                    continue
                
                # Compute energy delta
                delta = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        if q in pos_to_idx:
                            j = pos_to_idx[q]
                            if abs(j - i) > 1 and sequence[j] == 'H':
                                delta -= 1
                
                new_energy = energy + delta
                new_chain = chain + [site]
                new_pos_to_idx = dict(pos_to_idx)
                new_pos_to_idx[site] = i
                new_occupied = occupied | {site}
                
                new_beam.append((new_energy, new_chain, new_pos_to_idx, new_occupied))
        
        if not new_beam:
            return None
        
        # Keep top beam_width by energy (most negative = best)
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
        
        if time.time() > _DEADLINE - 5:
            break
    
    if not beam:
        return None
    
    best = min(beam, key=lambda x: x[0])
    return best[1]


def _kink_move(sequence, chain, i, site1, site2, dimension, pos_to_idx):
    """Perform a 2-residue kink move. Returns (success, new_chain, new_pos_to_idx, energy_delta)."""
    n = len(chain)
    if i < 1 or i + 1 >= n:
        return False, chain, pos_to_idx, 0
    
    old1, old2 = chain[i], chain[i + 1]
    
    # Check validity
    if site1 in pos_to_idx or site2 in pos_to_idx:
        return False, chain, pos_to_idx, 0
    if site1 == site2:
        return False, chain, pos_to_idx, 0
    
    prev = chain[i - 1]
    d_prev = sum(abs(site1[a] - prev[a]) for a in range(dimension))
    d_mid = sum(abs(site2[a] - site1[a]) for a in range(dimension))
    if d_prev != 1 or d_mid != 1:
        return False, chain, pos_to_idx, 0
    
    if i + 2 < n:
        nxt = chain[i + 2]
        d_next = sum(abs(site2[a] - nxt[a]) for a in range(dimension))
        if d_next != 1:
            return False, chain, pos_to_idx, 0
    
    # Compute energy delta
    # Remove contacts involving old1 and old2
    delta = 0
    for pos in [old1, old2]:
        if pos in pos_to_idx:
            idx = pos_to_idx[pos]
            if sequence[idx] == 'H':
                for q in _neighbours(pos):
                    if q in pos_to_idx:
                        j = pos_to_idx[q]
                        if abs(j - idx) > 1 and sequence[j] == 'H':
                            delta += 1  # Removing a contact increases energy
    
    # Add contacts involving site1 and site2
    for pos, idx in [(site1, i), (site2, i + 1)]:
        if sequence[idx] == 'H':
            for q in _neighbours(pos):
                if q in pos_to_idx:
                    j = pos_to_idx[q]
                    if abs(j - idx) > 1 and sequence[j] == 'H':
                        delta -= 1  # Adding a contact decreases energy
    
    # Also need to handle the fact that old1 and old2 are being removed from pos_to_idx
    # and site1 and site2 are being added
    
    new_chain = chain[:]
    new_chain[i] = site1
    new_chain[i + 1] = site2
    
    new_pos_to_idx = dict(pos_to_idx)
    del new_pos_to_idx[old1]
    del new_pos_to_idx[old2]
    new_pos_to_idx[site1] = i
    new_pos_to_idx[site2] = i + 1
    
    return True, new_chain, new_pos_to_idx, delta


def _local_optimize_sa(sequence, chain, dimension, max_time=None, rng=None):
    """Simulated Annealing with kink moves and pivot moves."""
    if rng is None:
        rng = random.Random()
    
    n = len(chain)
    if n < 4:
        return chain
    
    if max_time is None:
        max_time = max(0.1, _time_left() * 0.6)
    
    chain = list(chain)
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)
    best_pos_to_idx = dict(pos_to_idx)
    
    T_start = 10.0
    T_end = 0.01
    cooling = (T_end / T_start) ** (1.0 / max(1, int(max_time * 50)))
    T = T_start
    
    start = time.time()
    steps = 0
    
    while time.time() - start < max_time and T > T_end:
        steps += 1
        
        # Choose move type
        move_type = rng.random()
        
        accepted = False
        if move_type < 0.5:
            # Kink move
            i = rng.randint(1, n - 2)
            old1, old2 = chain[i], chain[i + 1]
            
            for site1 in _neighbours(old1):
                if site1 in pos_to_idx:
                    continue
                for site2 in _neighbours(site1):
                    if site2 in pos_to_idx or site2 == site1:
                        continue
                    
                    success, new_chain, new_pos_to_idx, delta = _kink_move(
                        sequence, chain, i, site1, site2, dimension, pos_to_idx
                    )
                    if not success:
                        continue
                    
                    new_energy = current_energy + delta
                    if new_energy <= current_energy or rng.random() < pow(2.718281828, -abs(new_energy - current_energy) / T):
                        chain = new_chain
                        pos_to_idx = new_pos_to_idx
                        current_energy = new_energy
                        
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(chain)
                            best_pos_to_idx = dict(pos_to_idx)
                        
                        accepted = True
                        break
                if accepted:
                    break
        
        else:
            # Single residue move (pivot-like)
            i = rng.randint(1, n - 2)
            old_pos = chain[i]
            
            for site in _neighbours(old_pos):
                if site in pos_to_idx:
                    continue
                
                # Check validity
                prev = chain[i - 1]
                nxt = chain[i + 1]
                d_prev = sum(abs(site[a] - prev[a]) for a in range(dimension))
                d_next = sum(abs(site[a] - nxt[a]) for a in range(dimension))
                if d_prev != 1 or d_next != 1:
                    continue
                
                # Compute energy delta
                delta = 0
                if sequence[i] == 'H':
                    for q in _neighbours(old_pos):
                        if q in pos_to_idx:
                            j = pos_to_idx[q]
                            if abs(j - i) > 1 and sequence[j] == 'H':
                                delta += 1
                    for q in _neighbours(site):
                        if q in pos_to_idx:
                            j = pos_to_idx[q]
                            if abs(j - i) > 1 and sequence[j] == 'H':
                                delta -= 1
                
                new_energy = current_energy + delta
                if new_energy <= current_energy or rng.random() < pow(2.718281828, -abs(new_energy - current_energy) / T):
                    new_chain = list(chain)
                    new_chain[i] = site
                    new_pos_to_idx = dict(pos_to_idx)
                    del new_pos_to_idx[old_pos]
                    new_pos_to_idx[site] = i
                    
                    chain = new_chain
                    pos_to_idx = new_pos_to_idx
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                        best_pos_to_idx = dict(pos_to_idx)
                    
                    accepted = True
                    break
        
        T *= cooling
    
    return best_chain


def _grow_random(sequence, dimension, rng):
    """Randomized chain growth."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start}
    
    for i in range(1, len(sequence)):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            candidates.append((free + rng.random(), site))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: -x[0])
        site = candidates[0][1]
        chain.append(site)
        occupied.add(site)
    
    return chain


def fold(sequence, dimension):
    """Best conformation using Beam Search initialization and Simulated Annealing refinement."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    rng = random.Random(42)
    
    # Phase 1: Generate multiple initial solutions
    best_chain = None
    best_energy = 0  # 0 means no contacts (worst case)
    
    # Try beam search first
    if _time_left() > 10:
        beam_chain = _beam_search_optimized(sequence, dimension, beam_width=80)
        if beam_chain:
            best_chain = beam_chain
            best_energy = _compute_energy(sequence, best_chain)
    
    # Add some random growths
    num_random = 20 if n <= 48 else 10
    for _ in range(num_random):
        if _time_left() < 10:
            break
        chain = _grow_random(sequence, dimension, rng)
        if chain:
            energy = _compute_energy(sequence, chain)
            if best_chain is None or energy < best_energy:
                best_chain = chain
                best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    
    # Phase 2: Refine with SA
    sa_time = max(0.5, _time_left() * 0.7)
    refined = _local_optimize_sa(sequence, best_chain, dimension, max_time=sa_time, rng=rng)
    refined_energy = _compute_energy(sequence, refined)
    
    if refined_energy < best_energy:
        best_chain = refined
        best_energy = refined_energy
    
    # Phase 3: If time remains, try more SA restarts from perturbed solutions
    if _time_left() > 20:
        for _ in range(3):
            if _time_left() < 25:
                break
            
            # Perturb the best solution
            perturbed = list(best_chain)
            pos_to_idx = {p: i for i, p in enumerate(perturbed)}
            
            # Make a few random kink moves
            for _ in range(5):
                i = rng.randint(1, n - 2)
                old1, old2 = perturbed[i], perturbed[i + 1]
                
                for site1 in _neighbours(old1):
                    if site1 in pos_to_idx:
                        continue
                    for site2 in _neighbours(site1):
                        if site2 in pos_to_idx or site2 == site1:
                            continue
                        
                        success, new_chain, new_pos_to_idx, delta = _kink_move(
                            sequence, perturbed, i, site1, site2, dimension, pos_to_idx
                        )
                        if success:
                            perturbed = new_chain
                            pos_to_idx = new_pos_to_idx
                            break
                    break
            
            # Refine
            remaining = max(0.5, _time_left() * 0.2)
            refined2 = _local_optimize_sa(sequence, perturbed, dimension, max_time=remaining, rng=rng)
            refined_energy2 = _compute_energy(sequence, refined2)
            
            if refined_energy2 < best_energy:
                best_chain = refined2
                best_energy = refined_energy2
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
