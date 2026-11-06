"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math
import numpy as np

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 195.0  # Safety margin for 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours(point):
    """Return the 4 neighbours of a 2D lattice point as a tuple."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))

_NEIGHBOUR_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _energy(sequence, chain):
    """Compute the HP contact energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    contacts = 0
    for i in h_indices:
        x, y = chain[i]
        for dx, dy in _NEIGHBOUR_OFFSETS:
            j = where.get((x + dx, y + dy))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _energy_numpy(h_positions, chain_arr):
    """Fast energy computation using numpy. h_positions: indices of H residues,
    chain_arr: numpy array of shape (n, 2) with lattice coordinates."""
    n = len(chain_arr)
    # Create a mapping from position to index
    # Shift coordinates to be non-negative for array indexing
    min_x = chain_arr[:, 0].min()
    min_y = chain_arr[:, 1].min()
    max_x = chain_arr[:, 0].max()
    max_y = chain_arr[:, 1].max()
    
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    
    # Only use numpy if the grid is reasonably small
    if width * height > 10000:
        return None  # Fall back to Python
    
    pos_to_idx = np.full((height, width), -1, dtype=np.int32)
    for i in range(n):
        px = chain_arr[i, 0] - min_x
        py = chain_arr[i, 1] - min_y
        pos_to_idx[py, px] = i
    
    contacts = 0
    for i in h_positions:
        px = chain_arr[i, 0] - min_x
        py = chain_arr[i, 1] - min_y
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = px + dx, py + dy
            if 0 <= nx < width and 0 <= ny < height:
                j = pos_to_idx[ny, nx]
                if j > i + 1:
                    contacts += 1
    return -contacts


def _build_contact_priorities(sequence):
    """Build a core set of mutually compatible H-H contacts using a greedy
    maximum-weight independent set heuristic on the contact graph."""
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    n = len(sequence)
    
    # Candidate contacts: H-H pairs with gap >= 2 (not adjacent in chain)
    candidates = []
    for idx_a, i in enumerate(h_indices):
        for j in h_indices[idx_a + 1:]:
            gap = j - i
            if gap >= 2:
                weight = 1.0 / gap
                candidates.append((i, j, weight))
    
    if not candidates:
        return set()
    
    candidates.sort(key=lambda x: -x[2])
    
    def _compatible(a, b):
        i, j, _ = a
        k, l, _ = b
        if j == k or l == i:
            return True
        if max(i, k) < min(j, l):
            overlap = min(j, l) - max(i, k)
            if overlap > 3:
                return False
        return True
    
    selected = []
    selected_set = set()
    
    for candidate in candidates:
        i, j, w = candidate
        if (i, j) in selected_set:
            continue
        compatible = True
        for sel in selected:
            if not _compatible(candidate, sel):
                compatible = False
                break
        if compatible:
            selected.append(candidate)
            selected_set.add((i, j))
    
    return selected_set


def _compute_contact_gains(sequence, chain, pos_idx, site, occupied):
    """Compute how many new H-H contacts are formed by placing residue pos_idx at site."""
    if sequence[pos_idx] != 'H':
        return 0
    gains = 0
    x, y = site
    for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
        j = occupied.get(q)
        if j is not None and j < pos_idx - 1 and sequence[j] == 'H':
            gains += 1
    return gains


def _grow(sequence, dimension, rng, bias=1.0, contact_priorities=None):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    cx, cy = 0.0, 0.0
    
    core_targets = {}
    if contact_priorities:
        for (a, b) in contact_priorities:
            if b not in core_targets:
                core_targets[b] = []
            core_targets[b].append((a, 1.0 / (b - a)))
    
    for i in range(1, n):
        options = []
        last = chain[-1]
        cx = (cx * i + last[0]) / (i + 1)
        cy = (cy * i + last[1]) / (i + 1)
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            gain = _compute_contact_gains(sequence, chain, i, site, occupied)
            
            core_bonus = 0.0
            if sequence[i] == 'H' and i in core_targets:
                for (other_idx, weight) in core_targets[i]:
                    if other_idx < i:
                        ox, oy = chain[other_idx]
                        sx, sy = site
                        if abs(ox - sx) + abs(oy - sy) == 1:
                            core_bonus += 3.0 * weight
            
            any_contact_bonus = 0.0
            if sequence[i] == 'H':
                x, y = site
                for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        is_core = False
                        if i in core_targets:
                            for (other_idx, _) in core_targets[i]:
                                if other_idx == j:
                                    is_core = True
                                    break
                        if not is_core:
                            any_contact_bonus += 1.0
            
            dist = math.sqrt((site[0] - cx)**2 + (site[1] - cy)**2)
            
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            score = bias * gain + core_bonus + any_contact_bonus - 0.4 * dist + 0.05 * free + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _apply_pivot(chain, pivot_idx, angle):
    """Apply a pivot move and return the new chain if valid, else None."""
    n = len(chain)
    pivot = chain[pivot_idx]
    segment_start = pivot_idx + 1
    if segment_start >= n:
        return None
    
    fixed_set = set(chain[:pivot_idx + 1])
    segment = chain[segment_start:]
    
    rotated = []
    for p in segment:
        dx, dy = p[0] - pivot[0], p[1] - pivot[1]
        if angle == 90:
            ndx, ndy = -dy, dx
        elif angle == -90:
            ndx, ndy = dy, -dx
        else:  # 180
            ndx, ndy = -dx, -dy
        new_p = (pivot[0] + ndx, pivot[1] + ndy)
        if new_p in fixed_set:
            return None
        rotated.append(new_p)
    
    return chain[:segment_start] + rotated


def _pivot_move_fast(chain, rng):
    """Fast pivot move: rotate a subchain. Validity check is O(segment_length)."""
    n = len(chain)
    if n < 4:
        return None

    pivot_idx = rng.randint(1, n - 2)
    angle = rng.choice((90, -90, 180))
    return _apply_pivot(chain, pivot_idx, angle)


def _slither_move(chain, rng):
    """Try to move the end of the chain."""
    n = len(chain)
    if n < 4:
        return None
    
    last = chain[-1]
    prev = chain[-2]
    occupied = set(chain[:-1])
    
    candidates = [site for site in _neighbours(prev) if site not in occupied]
    if not candidates:
        return None
    site = rng.choice(candidates)
    return chain[:-1] + [site]


def _local_search(sequence, chain, max_iterations=5000, time_limit=5.0, rng=None):
    """Simulated annealing local search to improve a conformation."""
    if chain is None:
        return None

    if rng is None:
        rng = random.Random(hash(tuple(chain)) % (2**32))

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    # Precompute H positions for fast energy
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    
    start_time = time.time()
    temperature = 4.0
    cooling = 0.9995
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break

        r = rng.random()
        if r < 0.65:
            new_chain = _pivot_move_fast(current_chain, rng)
        else:
            new_chain = _slither_move(current_chain, rng)

        if new_chain is None:
            continue

        # Use fast energy if possible
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy

        if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 0.1)):
            current_chain = new_chain
            current_energy = new_energy

            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)

        temperature *= cooling

    return best_chain


def _polish(sequence, chain, max_iterations=2000, time_limit=3.0, rng=None):
    """Deterministic hill-climbing polish: accept only improving moves."""
    if chain is None:
        return None
    if rng is None:
        rng = random.Random(42)

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 15:
            break

        n = len(current_chain)
        best_move = None
        best_delta = 0

        # Try ALL pivot positions for thorough local search
        for pivot_idx in range(1, n - 1):
            for angle in (90, -90, 180):
                new_chain = _apply_pivot(current_chain, pivot_idx, angle)
                if new_chain is None:
                    continue
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                if delta < best_delta:
                    best_delta = delta
                    best_move = new_chain

        # Try slither
        slither = _slither_move(current_chain, rng)
        if slither is not None:
            new_energy = _energy(sequence, slither)
            delta = new_energy - current_energy
            if delta < best_delta:
                best_delta = delta
                best_move = slither

        if best_move is None:
            break

        current_chain = best_move
        current_energy += best_delta
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(current_chain)

    return best_chain


def fold(sequence, dimension):
    """Best conformation using multiple strategies: greedy growth + local search."""
    seq_len = len(sequence)

    # Build contact priorities
    contact_priorities = _build_contact_priorities(sequence)

    # Adjust strategy based on sequence length - increase restarts
    if seq_len <= 30:
        num_grow = 500
        local_search_time = 8.0
        polish_time = 4.0
    elif seq_len <= 50:
        num_grow = 320
        local_search_time = 6.5
        polish_time = 3.5
    elif seq_len <= 70:
        num_grow = 230
        local_search_time = 5.5
        polish_time = 3.0
    else:
        num_grow = 160
        local_search_time = 4.5
        polish_time = 2.5

    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0

    # Pass 1: Greedy Growth with contact priorities
    for i in range(num_grow):
        if _time_left() < 20:
            break

        bias = 0.5 + rng.random() * 2.0
        chain = _grow(sequence, dimension, rng, bias=bias, contact_priorities=contact_priorities)
        if chain is None:
            continue

        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain

    # Pass 2: Local Search on the best chain found
    if best_chain is not None and _time_left() > 25:
        improved = _local_search(sequence, best_chain, max_iterations=10000,
                                  time_limit=local_search_time, rng=rng)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved

        # Pass 3: Deterministic polish
        if _time_left() > 20:
            polished = _polish(sequence, best_chain, max_iterations=5000,
                               time_limit=polish_time, rng=rng)
            if polished is not None:
                polished_energy = _energy(sequence, polished)
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_chain = polished

        # Pass 4: Additional local search with different seed
        if _time_left() > 30:
            rng2 = random.Random(99999 + seq_len)
            improved2 = _local_search(sequence, best_chain, max_iterations=5000,
                                       time_limit=local_search_time * 0.6, rng=rng2)
            if improved2 is not None:
                e2 = _energy(sequence, improved2)
                if e2 < best_energy:
                    best_energy = e2
                    best_chain = improved2

            if _time_left() > 20:
                final_polish = _polish(sequence, best_chain, max_iterations=3000,
                                       time_limit=polish_time * 0.5, rng=rng2)
                if final_polish is not None:
                    fp_energy = _energy(sequence, final_polish)
                    if fp_energy < best_energy:
                        best_energy = fp_energy
                        best_chain = final_polish

        # Pass 5: Second round of growth with refined priorities
        if _time_left() > 35:
            satisfied = set()
            where = {p: i for i, p in enumerate(best_chain)}
            h_indices = [i for i, c in enumerate(sequence) if c == 'H']
            for i in h_indices:
                x, y = best_chain[i]
                for dx, dy in _NEIGHBOUR_OFFSETS:
                    j = where.get((x + dx, y + dy))
                    if j is not None and j > i + 1 and sequence[j] == 'H':
                        satisfied.add((min(i, j), max(i, j)))
            
            unsatisfied_core = contact_priorities - satisfied
            
            if unsatisfied_core:
                num_refine = min(80, num_grow // 3)
                for i in range(num_refine):
                    if _time_left() < 25:
                        break
                    bias = 1.0 + rng.random() * 1.5
                    chain = _grow(sequence, dimension, rng, bias=bias, 
                                  contact_priorities=unsatisfied_core)
                    if chain is None:
                        continue
                    value = _energy(sequence, chain)
                    if value < best_energy:
                        best_energy = value
                        best_chain = chain
                        if _time_left() > 20:
                            p = _polish(sequence, best_chain, max_iterations=2000,
                                        time_limit=2.0, rng=rng)
                            if p is not None:
                                pe = _energy(sequence, p)
                                if pe < best_energy:
                                    best_energy = pe
                                    best_chain = p

    # Final fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(seq_len)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
