"""HP lattice protein folding - structured local search with segment moves and restarts."""
import time
import random
import sys

# Global time tracking
_START_TIME = time.time()
_TIME_BUDGET = 280  # seconds, leaving 20s safety margin before 300s limit


def _remaining_time():
    return max(0.0, _TIME_BUDGET - (time.time() - _START_TIME))


def _neighbours(point):
    """Generate all 6 neighbours in 3D cubic lattice."""
    n = len(point)
    for axis in range(n):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute H-H contact energy for a chain. Energy = -number of H-H contacts."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    contacts = 0
    h_indices = [i for i in range(n) if sequence[i] == 'H']
    
    for i in h_indices:
        p = chain[i]
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    
    return -contacts


def _energy_delta_move(sequence, chain, occupied, i, old_pos, new_pos):
    """
    Compute the change in contact count when moving residue i from old_pos to new_pos.
    Returns (delta_contacts) where positive means more contacts (better, more negative energy).
    """
    # Count H-H contacts involving residue i
    # Before move: contacts between i and other H residues at old_pos
    # After move: contacts between i and other H residues at new_pos
    
    if sequence[i] != 'H':
        return 0
    
    delta = 0
    
    # Remove old contacts: count H neighbours of old_pos that are not i or adjacent in chain
    for q in _neighbours(old_pos):
        j = occupied.get(q)
        if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
            delta -= 1  # losing a contact
    
    # Add new contacts: count H neighbours of new_pos that are not i or adjacent in chain
    for q in _neighbours(new_pos):
        j = occupied.get(q)
        if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
            delta += 1  # gaining a contact
    
    return delta


def _random_grow(sequence, dimension, rng, bias=0.5):
    """Random chain growth with bias toward hydrophobic contacts."""
    n = len(sequence)
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Count potential H-H contacts
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            # Count free space
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 3.0 + free * bias + rng.random() * 0.5
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Weighted selection
        total = sum(c[0] for c in candidates)
        r = rng.random() * total
        cum = 0
        chosen = candidates[-1][1]
        for score, site in candidates:
            cum += score
            if r <= cum:
                chosen = site
                break
        
        chain.append(chosen)
        occupied[chosen] = i
    
    return chain


def _try_single_move(sequence, chain, occupied, i, time_limit):
    """
    Try to find a better single-residue move for position i.
    Returns (new_chain, new_energy, moved) or (chain, energy, False).
    """
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_new_energy = current_energy
    best_target = None
    
    pos = chain[i]
    
    # Try neighbours of pos
    targets = list(_neighbours(pos))
    # Also try neighbours of neighbours for longer jumps
    for q in list(_neighbours(pos)):
        if q not in occupied:
            for r in _neighbours(q):
                if r not in occupied and r != pos:
                    targets.append(r)
    
    rng_local = random.Random(int(time.time() * 10000) % 100000)
    rng_local.shuffle(targets)
    
    for target in targets:
        if target in occupied:
            continue
        
        # Check adjacency constraints with immediate neighbors
        if i > 0:
            d1 = sum(abs(a - b) for a, b in zip(target, chain[i-1]))
            if d1 != 1:
                continue
        if i < n - 1:
            d2 = sum(abs(a - b) for a, b in zip(target, chain[i+1]))
            if d2 != 1:
                continue
        
        # Compute energy delta
        delta = _energy_delta_move(sequence, chain, occupied, i, pos, target)
        new_energy = current_energy - delta
        
        if new_energy < best_new_energy:
            best_new_energy = new_energy
            best_target = target
    
    if best_target is not None and best_new_energy < current_energy:
        # Perform the move
        old_pos = chain[i]
        del occupied[old_pos]
        chain[i] = best_target
        occupied[best_target] = i
        return chain, best_new_energy, True
    
    return chain, current_energy, False


def _try_segment_reversal(sequence, chain, occupied, start, end, time_limit):
    """
    Try reversing a segment of the chain from start to end.
    Returns (new_chain, new_energy, moved) or (chain, energy, False).
    """
    n = len(chain)
    if end - start < 2:
        return chain, _compute_energy(sequence, chain), False
    
    current_energy = _compute_energy(sequence, chain)
    
    # Check if reversal is valid:
    # 1. The reversed segment must not self-intersect with itself or the rest of the chain
    # 2. Adjacency at the boundaries must hold
    
    # First, check self-avoidance of the reversed segment
    segment = chain[start:end+1][::-1]
    seg_occupied = set()
    for p in segment:
        if p in seg_occupied:
            return chain, current_energy, False
        seg_occupied.add(p)
    
    # Check boundary adjacency
    # After reversal:
    # - chain[start-1] must be adjacent to segment[0] = chain[end]
    # - chain[end+1] must be adjacent to segment[-1] = chain[start]
    # - Also, within the segment, consecutive positions must be adjacent
    
    if start > 0:
        d1 = sum(abs(a - b) for a, b in zip(segment[0], chain[start-1]))
        if d1 != 1:
            return chain, current_energy, False
    
    if end < n - 1:
        d2 = sum(abs(a - b) for a, b in zip(segment[-1], chain[end+1]))
        if d2 != 1:
            return chain, current_energy, False
    
    # Check internal adjacency
    for k in range(len(segment) - 1):
        d = sum(abs(a - b) for a, b in zip(segment[k], segment[k+1]))
        if d != 1:
            return chain, current_energy, False
    
    # Check no intersection with rest of chain
    rest_occupied = set(chain[:start]) | set(chain[end+1:])
    for p in segment:
        if p in rest_occupied:
            return chain, current_energy, False
    
    # Valid move - compute new energy
    new_chain = chain[:start] + segment + chain[end+1:]
    new_energy = _compute_energy(sequence, new_chain)
    
    if new_energy < current_energy:
        # Update occupied
        new_occupied = {p: i for i, p in enumerate(new_chain)}
        return new_chain, new_energy, True
    
    return chain, current_energy, False


def _local_search(sequence, chain, occupied, current_energy, time_limit=5.0):
    """Local search with single moves and segment reversals."""
    n = len(chain)
    start_time = time.time()
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        rng_local = random.Random(int(time.time() * 10000) % 100000)
        
        # Randomize order of positions
        order = list(range(1, n - 1))
        rng_local.shuffle(order)
        
        for i in order:
            if (time.time() - start_time) >= time_limit:
                break
            
            # Try single move
            new_chain, new_energy, moved = _try_single_move(sequence, chain, occupied, i, time_limit)
            if moved:
                # Update occupied
                occupied = {p: i for i, p in enumerate(new_chain)}
                chain = new_chain
                current_energy = new_energy
                improved = True
                continue
            
            # Try segment reversals around position i
            # Try reversing small segments that include position i
            for length in [2, 3, 4, 5]:
                for offset in range(-length // 2, length // 2 + 1):
                    s = i + offset
                    e = i + offset + length - 1
                    if s < 1 or e > n - 2:
                        continue
                    
                    new_chain, new_energy, moved = _try_segment_reversal(sequence, chain, occupied, s, e, time_limit)
                    if moved:
                        occupied = {p: i for i, p in enumerate(new_chain)}
                        chain = new_chain
                        current_energy = new_energy
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            # No improvement in this pass
            # Try a few random segment reversals as perturbation
            for _ in range(3):
                if (time.time() - start_time) >= time_limit:
                    break
                
                seg_len = rng_local.randint(3, min(8, n // 3))
                s = rng_local.randint(1, n - seg_len - 1)
                e = s + seg_len - 1
                
                new_chain, new_energy, moved = _try_segment_reversal(sequence, chain, occupied, s, e, time_limit)
                if moved:
                    occupied = {p: i for i, p in enumerate(new_chain)}
                    chain = new_chain
                    current_energy = new_energy
                    improved = True
                    break
    
    return chain, current_energy, occupied


def _anneal(sequence, chain, occupied, current_energy, time_limit=10.0):
    """Simulated annealing with single moves and occasional segment reversals."""
    n = len(chain)
    start_time = time.time()
    rng = random.Random(int(time.time() * 10000) % 100000)
    
    T = 5.0
    T_min = 0.01
    cooling = 0.998
    
    while T > T_min and (time.time() - start_time) < time_limit:
        # Decide move type
        if rng.random() < 0.8:
            # Single move
            i = rng.randint(1, n - 2)
            pos = chain[i]
            targets = list(_neighbours(pos))
            rng.shuffle(targets)
            
            moved = False
            for target in targets:
                if target in occupied:
                    continue
                
                if i > 0:
                    d1 = sum(abs(a - b) for a, b in zip(target, chain[i-1]))
                    if d1 != 1:
                        continue
                if i < n - 1:
                    d2 = sum(abs(a - b) for a, b in zip(target, chain[i+1]))
                    if d2 != 1:
                        continue
                
                delta = _energy_delta_move(sequence, chain, occupied, i, pos, target)
                new_energy = current_energy - delta
                
                if new_energy <= current_energy or rng.random() < pow(2.718281828, -(new_energy - current_energy) / T):
                    del occupied[pos]
                    chain[i] = target
                    occupied[target] = i
                    current_energy = new_energy
                    moved = True
                    break
        else:
            # Segment reversal
            seg_len = rng.randint(3, min(6, n // 4))
            s = rng.randint(1, n - seg_len - 1)
            e = s + seg_len - 1
            
            new_chain, new_energy, moved = _try_segment_reversal(sequence, chain, occupied, s, e, time_limit)
            if moved:
                chain = new_chain
                current_energy = new_energy
                occupied = {p: i for i, p in enumerate(chain)}
            else:
                # Check if reversal was structurally valid but worse
                # For simplicity, just skip
                pass
        
        T *= cooling
    
    return chain, current_energy, occupied


def fold(sequence, dimension):
    """Fold an HP sequence into a self-avoiding walk on the cubic lattice."""
    global _START_TIME
    n = len(sequence)
    
    if n <= 1:
        return [[0] * dimension]
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Multiple random growths
    num_growths = 40
    rng_main = random.Random(12345 + n)
    
    initial_chains = []
    for _ in range(num_growths):
        chain = _random_grow(sequence, dimension, rng_main, bias=0.5)
        if chain is not None:
            energy = _compute_energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
            initial_chains.append((chain, energy))
    
    if not initial_chains:
        return [[i] + [0] * (dimension - 1) for i in range(n)]
    
    # Sort by energy and take top candidates
    initial_chains.sort(key=lambda x: x[1])
    top_chains = initial_chains[:min(8, len(initial_chains))]
    
    # Phase 2: Local search + annealing on top candidates
    # Allocate time proportionally
    remaining = _remaining_time()
    if remaining < 5:
        if best_chain is None:
            return [[i] + [0] * (dimension - 1) for i in range(n)]
        return [list(p) for p in best_chain]
    
    num_to_optimize = min(len(top_chains), max(2, int(remaining / 20)))
    time_per_chain = (remaining * 0.8) / num_to_optimize
    
    for idx, (chain, energy) in enumerate(top_chains[:num_to_optimize]):
        if _remaining_time() < 5:
            break
        
        # Local search first
        ls_time = min(time_per_chain * 0.4, 5.0)
        chain, energy, occupied = _local_search(sequence, chain, {p: i for i, p in enumerate(chain)}, energy, time_limit=ls_time)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
        
        # Then anneal if time permits
        sa_time = min(time_per_chain * 0.6, 15.0)
        if sa_time > 2:
            chain, energy, occupied = _anneal(sequence, chain, occupied, energy, time_limit=sa_time)
            
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
    
    # Phase 3: Final polish on the best chain
    if _remaining_time() > 10 and best_chain is not None:
        polish_time = min(20.0, _remaining_time() * 0.6)
        occupied = {p: i for i, p in enumerate(best_chain)}
        chain, energy, occupied = _local_search(sequence, best_chain, occupied, best_energy, time_limit=polish_time * 0.4)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
        
        sa_time = polish_time * 0.6
        if sa_time > 2:
            chain, energy, occupied = _anneal(sequence, best_chain, occupied, best_energy, time_limit=sa_time)
            
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
        
        # One more local search
        if _remaining_time() > 3:
            chain, energy, occupied = _local_search(sequence, best_chain, occupied, best_energy, time_limit=3.0)
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
    
    # Final safety check
    if best_chain is None:
        return [[i] + [0] * (dimension - 1) for i in range(n)]
    
    return [list(p) for p in best_chain]
