"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp

_START_TIME = time.time()
_DEADLINE = 195.0  # Safety margin before 210s kill


def _remaining_time():
    return max(0.0, _DEADLINE - (time.time() - _START_TIME))


def _energy_delta_single(sequence, chain, where, idx):
    """Compute energy change when moving residue idx to a new site.
    Returns the delta (new_energy - old_energy)."""
    old_site = chain[idx]
    new_site = chain[idx]  # placeholder, caller will set
    
    # We need to compute contacts involving idx before and after move
    # But for simplicity in the move functions, we'll just recompute fully
    # This is a helper that's not used directly but shows intent
    pass


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute HP energy: -number of H-H contacts (non-adjacent in chain)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _energy_fast(sequence, chain):
    """Faster energy computation using precomputed neighbor indices."""
    n = len(chain)
    # Build position map
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    contacts = 0
    # Only check H residues, and only look forward to avoid double counting
    for i in range(n):
        if sequence[i] != "H":
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts += 1
    return -contacts


def _grow(sequence, dimension, rng, bias=1.0):
    """
    One randomized greedy chain growth.
    bias > 1 favors contacts, bias < 1 favors free space.
    """
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * bias + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_move(sequence, chain, where, current_energy, T, rng):
    """
    Attempt a single-residue move (pivot). Returns (new_chain, new_where, new_energy, accepted).
    Uses incremental energy update for speed.
    """
    n = len(chain)
    if n <= 2:
        return chain, where, current_energy, False

    idx = rng.randint(1, n - 2)
    old_site = chain[idx]
    prev_site = chain[idx - 1]
    next_site = chain[idx + 1]

    candidates = []
    for site in _neighbours(old_site):
        if site == old_site:
            continue
        if abs(site[0] - prev_site[0]) + abs(site[1] - prev_site[1]) != 1:
            continue
        if abs(site[0] - next_site[0]) + abs(site[1] - next_site[1]) != 1:
            continue
        if site in where:
            occ_idx = where[site]
            if occ_idx != idx:
                continue
        candidates.append(site)

    if not candidates:
        return chain, where, current_energy, False

    new_site = rng.choice(candidates)
    if new_site == old_site:
        return chain, where, current_energy, False

    # Incremental energy calculation
    # Remove contacts involving old_site for residue idx
    delta_e = 0
    if sequence[idx] == "H":
        # Count H-H contacts at old_site (excluding adjacent residues idx-1, idx+1)
        for q in _neighbours(old_site):
            j = where.get(q)
            if j is not None and j != idx - 1 and j != idx + 1 and sequence[j] == "H":
                delta_e -= 1  # Removing a contact increases energy (less negative)
        
        # Count H-H contacts at new_site
        for q in _neighbours(new_site):
            j = where.get(q)
            if j is not None and j != idx - 1 and j != idx + 1 and sequence[j] == "H":
                delta_e += 1  # Adding a contact decreases energy (more negative)
    
    # Also need to check if moving idx changes contacts for other H residues
    # This is complex, so for correctness we do a full recompute if delta_e != 0
    # But let's just do full recompute for safety - it's fast enough
    
    del where[old_site]
    where[new_site] = idx
    chain[idx] = new_site

    new_energy = _energy(sequence, chain)
    delta_e = new_energy - current_energy

    if delta_e <= 0 or rng.random() < exp(-delta_e / T):
        return chain, where, new_energy, True
    else:
        del where[new_site]
        where[old_site] = idx
        chain[idx] = old_site
        return chain, where, current_energy, False


def _reverse_move(sequence, chain, where, current_energy, T, rng):
    """
    Attempt a segment reversal. Returns (new_chain, new_where, new_energy, accepted).
    """
    n = len(chain)
    if n < 4:
        return chain, where, current_energy, False

    # Prefer shorter segments for more efficient exploration
    seg_len = rng.randint(2, min(6, n - 2))
    i = rng.randint(0, n - seg_len - 1)
    j = i + seg_len

    if i > 0:
        if abs(chain[i - 1][0] - chain[j][0]) + abs(chain[i - 1][1] - chain[j][1]) != 1:
            return chain, where, current_energy, False
    if j < n - 1:
        if abs(chain[i][0] - chain[j + 1][0]) + abs(chain[i][1] - chain[j + 1][1]) != 1:
            return chain, where, current_energy, False

    # Check self-avoidance before committing
    old_seg = chain[i:j + 1]
    new_seg = old_seg[::-1]
    
    # Create temporary where to check collisions
    temp_where = dict(where)
    for k in range(i, j + 1):
        del temp_where[chain[k]]
    
    for k, pt in enumerate(new_seg):
        idx_in_seg = i + k
        if pt in temp_where:
            return chain, where, current_energy, False
        temp_where[pt] = idx_in_seg

    chain[i:j + 1] = new_seg
    for k in range(i, j + 1):
        where[chain[k]] = k

    new_energy = _energy(sequence, chain)
    delta_e = new_energy - current_energy

    if delta_e <= 0 or rng.random() < exp(-delta_e / T):
        return chain, where, new_energy, True
    else:
        chain[i:j + 1] = old_seg
        for k in range(i, j + 1):
            where[chain[k]] = k
        return chain, where, current_energy, False


def _rotate_move(sequence, chain, where, current_energy, T, rng):
    """
    Attempt a 90-degree rotation of a segment. Returns (new_chain, new_where, new_energy, accepted).
    """
    n = len(chain)
    if n < 4:
        return chain, where, current_energy, False

    i = rng.randint(0, n - 3)
    j = rng.randint(i + 2, n - 1)

    # The segment is chain[i..j]. We rotate it 90 degrees around chain[i].
    # After rotation, chain[i] stays fixed, and the rest of the segment rotates.
    # We need to check self-avoidance and boundary connectivity.

    pivot = chain[i]
    seg = chain[i:j + 1]

    # Rotate each point in seg (except pivot) 90 degrees clockwise around pivot
    # In 2D: (dx, dy) -> (dy, -dx)
    rotated_seg = [pivot]
    for pt in seg[1:]:
        dx = pt[0] - pivot[0]
        dy = pt[1] - pivot[1]
        new_pt = (pivot[0] + dy, pivot[1] - dx)
        rotated_seg.append(new_pt)

    # Check self-avoidance for the rotated segment
    new_where = dict(where)
    # Remove old segment points (except pivot which stays)
    for pt in seg[1:]:
        if pt in new_where:
            del new_where[pt]

    # Check if rotated points collide
    for pt in rotated_seg[1:]:
        if pt in new_where:
            return chain, where, current_energy, False

    # Check boundary connectivity
    # Left boundary: chain[i-1] must be adjacent to chain[i] (pivot, unchanged) - OK
    # Right boundary: chain[j] (old) becomes rotated_seg[-1], must be adjacent to chain[j+1]
    if j < n - 1:
        if abs(rotated_seg[-1][0] - chain[j + 1][0]) + abs(rotated_seg[-1][1] - chain[j + 1][1]) != 1:
            return chain, where, current_energy, False

    # Perform rotation
    chain[i:j + 1] = rotated_seg
    for k in range(i, j + 1):
        where[chain[k]] = k

    new_energy = _energy(sequence, chain)
    delta_e = new_energy - current_energy

    if delta_e <= 0 or rng.random() < exp(-delta_e / T):
        return chain, where, new_energy, True
    else:
        chain[i:j + 1] = seg
        for k in range(i, j + 1):
            where[chain[k]] = k
        return chain, where, current_energy, False


def _anneal(sequence, dimension, chain, max_time=10.0, seed=12345):
    """Simulated annealing with local moves, reversals, and rotations."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    # Adaptive temperature schedule
    T = 4.0
    T_min = 0.05
    cooling = 0.9998
    rng = random.Random(seed)
    
    # Track acceptance rate to adapt cooling
    accept_count = 0
    total_moves = 0
    window_size = 100
    recent_accepts = []

    start = time.time()
    while T > T_min and (time.time() - start) < max_time:
        r = rng.random()
        if r < 0.5:
            chain, where, current_energy, accepted = _local_move(sequence, chain, where, current_energy, T, rng)
        elif r < 0.8:
            chain, where, current_energy, accepted = _reverse_move(sequence, chain, where, current_energy, T, rng)
        else:
            chain, where, current_energy, accepted = _rotate_move(sequence, chain, where, current_energy, T, rng)

        if accepted:
            accept_count += 1
            recent_accepts.append(1)
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        else:
            recent_accepts.append(0)
        
        total_moves += 1
        
        # Adapt cooling rate based on recent acceptance
        if len(recent_accepts) >= window_size:
            accept_rate = sum(recent_accepts[-window_size:]) / window_size
            # If acceptance too high, cool faster; if too low, cool slower
            if accept_rate > 0.5:
                cooling = min(0.9999, cooling * 1.01)
            elif accept_rate < 0.1:
                cooling = max(0.9995, cooling * 0.99)
            recent_accepts = recent_accepts[-window_size // 2:]

        T *= cooling

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold a sequence using multiple growth strategies and annealing.
    """
    n = len(sequence)
    rng = random.Random(12345 + n)
    best, best_energy = None, 1

    # Phase 1: Generate diverse initial conformations
    initial_chains = []
    growth_budget = 200
    biases = [1.2, 1.8, 2.5, 0.7, 1.0, 1.5]
    for trial in range(growth_budget):
        if _remaining_time() < 12:
            break
        bias = biases[trial % len(biases)]
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is not None:
            initial_chains.append(chain)
    
    # Also add a few random walks for diversity
    for _ in range(20):
        if _remaining_time() < 12:
            break
        chain = _grow(sequence, dimension, rng, bias=0.3)
        if chain is not None:
            initial_chains.append(chain)

    if not initial_chains:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in best]

    # Phase 2: Anneal each initial conformation
    # Sort by initial energy to prioritize better starts
    initial_chains.sort(key=lambda c: _energy(sequence, c))
    
    # Deduplicate: keep only unique chains
    seen = set()
    unique_chains = []
    for c in initial_chains:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_chains.append(c)
    initial_chains = unique_chains
    
    # Allocate time: top 3 get more, rest share remaining
    top_k = min(3, len(initial_chains))
    time_per_top = max(2.0, _remaining_time() * 0.25 / max(1, top_k))
    remaining_time = _remaining_time() - time_per_top * top_k
    num_rest = max(0, len(initial_chains) - top_k)
    time_per_rest = max(0.5, remaining_time / max(1, num_rest)) if num_rest > 0 else 0
    
    for idx, chain in enumerate(initial_chains):
        if _remaining_time() < 3:
            break
        
        if idx < top_k:
            t = time_per_top
        else:
            t = time_per_rest
        t = min(t, _remaining_time() - 1)
        if t <= 0:
            break
        
        # Use different seeds for diversity
        seed_val = 12345 + idx * 7
        annealed_chain, annealed_energy = _anneal(sequence, dimension, chain, max_time=t, seed=seed_val)
        if annealed_energy < best_energy:
            best, best_energy = annealed_chain, annealed_energy

    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
