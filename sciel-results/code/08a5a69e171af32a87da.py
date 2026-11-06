"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = None
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_remaining():
    _init_time()
    return _DEADLINE - (time.time() - _START_TIME)


def _neighbours(point):
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy_fast(seq, chain):
    """Fast energy using a position->index map and precomputed H indices."""
    where = {p: i for i, p in enumerate(chain)}
    h_set = set(i for i, c in enumerate(seq) if c == "H")
    contacts = 0
    for i in h_set:
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = where.get((x + dx, y + dy))
            if j is not None and j > i + 1 and j in h_set:
                contacts += 1
    return -contacts


def _is_valid_chain(chain):
    """Check self-avoidance and unit steps."""
    n = len(chain)
    if n < 2:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            dx = abs(chain[i][0] - chain[i - 1][0])
            dy = abs(chain[i][1] - chain[i - 1][1])
            if dx + dy != 1:
                return False
    return True


def _grow_beam(sequence, dimension, rng, beam_width=8):
    """
    Beam search growth. Keeps top beam_width partial chains scored by
    immediate H-H contacts plus a free-space heuristic.
    """
    n = len(sequence)
    if n == 0:
        return []
    start = (0, 0)
    beam = [([start], {start: 0}, 0.0)]
    
    for i in range(1, n):
        next_beam = []
        for chain, occupied, score in beam:
            cur = chain[-1]
            x, y = cur
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                site = (x + dx, y + dy)
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == 'H':
                    sx, sy = site
                    for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        q = (sx + adx, sy + ady)
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1
                # Free space heuristic: prefer sites with more open neighbors
                free = 0
                sx, sy = site
                for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if (sx + adx, sy + ady) not in occupied:
                        free += 1
                new_score = score - 2.0 * gain - 0.1 * free
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                next_beam.append((new_chain, new_occupied, new_score))
        
        if not next_beam:
            return None
        
        next_beam.sort(key=lambda x: (x[2], rng.random()))
        beam = next_beam[:beam_width]
        
        if _time_remaining() < 2.0:
            break
    
    if not beam:
        return None
    return beam[0][0]


def _try_reinsert_fragments(chain, sequence, rng, max_frag_len=5):
    """
    Attempt to re-insert a fragment: cut out chain[a:b], then try to regrow it
    from chain[a-1] connecting to chain[b].
    """
    n = len(chain)
    if n < 5:
        return None

    L = rng.randint(2, min(max_frag_len, n - 3))
    a = rng.randint(1, n - L - 1)
    b = a + L

    left_neighbor = chain[a - 1]
    right_neighbor = chain[b]

    rest = chain[:a] + chain[b:]
    rest_occupied = set(rest)

    lx, ly = left_neighbor
    rx, ry = right_neighbor
    manhattan = abs(rx - lx) + abs(ry - ly)
    if manhattan > L or (L - manhattan) % 2 != 0:
        return None

    def _find_path(start, end, length, occupied_set):
        if length == 0:
            return [start] if start == end else None

        def _dfs(current, remaining, path, occupied):
            if remaining == 0:
                if current == end:
                    return list(path)
                return None
            x, y = current
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                site = (x + dx, y + dy)
                if site in occupied:
                    continue
                dist_to_end = abs(end[0] - site[0]) + abs(end[1] - site[1])
                if dist_to_end > remaining or (remaining - dist_to_end) % 2 != 0:
                    continue
                path.append(site)
                occupied.add(site)
                result = _dfs(site, remaining - 1, path, occupied)
                if result is not None:
                    return result
                path.pop()
                occupied.discard(site)
            return None

        return _dfs(start, length, [start], {start} | occupied_set)

    path = _find_path(left_neighbor, right_neighbor, L, rest_occupied)
    if path is None:
        return None
    if path[-1] != right_neighbor:
        return None

    new_frag = path[1:]
    if len(new_frag) != L:
        return None

    new_chain = chain[:a] + new_frag + chain[b:]
    if not _is_valid_chain(new_chain):
        return None
    return new_chain


def _reversal_move(chain, rng, max_len=None):
    """Random reversal move."""
    n = len(chain)
    if n < 4:
        return None
    if max_len is None:
        max_len = min(8, n - 2)
    L = rng.randint(2, max_len)
    a = rng.randint(0, n - L - 1)
    b = a + L
    new_chain = chain[:a] + chain[a:b + 1][::-1] + chain[b + 1:]
    if _is_valid_chain(new_chain):
        return new_chain
    return None


def _kink_move(chain, rng):
    """Kink move: pick a residue i and try to move it to an adjacent free position."""
    n = len(chain)
    if n < 3:
        return None
    i = rng.randint(1, n - 2)
    x, y = chain[i]
    options = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        site = (x + dx, y + dy)
        if site in chain and site != chain[i]:
            continue
        px, py = chain[i - 1]
        if abs(site[0] - px) + abs(site[1] - py) != 1:
            continue
        nx, ny = chain[i + 1]
        if abs(site[0] - nx) + abs(site[1] - ny) != 1:
            continue
        options.append(site)
    if not options:
        return None
    site = options[rng.randint(0, len(options) - 1)]
    new_chain = chain[:]
    new_chain[i] = site
    if _is_valid_chain(new_chain):
        return new_chain
    return None


def _local_search(sequence, chain, rng, max_steps=200):
    """
    Local search using kink, reversal, and fragment re-insertion moves.
    Accepts moves that improve energy.
    """
    n = len(chain)
    if n < 3:
        return chain, _energy_fast(sequence, chain)
    
    current_energy = _energy_fast(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    for step in range(max_steps):
        if _time_remaining() < 2.0:
            break
        
        move_type = rng.random()
        new_chain = None
        
        if move_type < 0.5:
            # Kink move
            new_chain = _kink_move(chain, rng)
        elif move_type < 0.8:
            # Reversal move
            new_chain = _reversal_move(chain, rng)
        else:
            # Fragment re-insertion
            new_chain = _try_reinsert_fragments(chain, sequence, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _energy_fast(sequence, new_chain)
        
        if new_energy < current_energy:
            current_energy = new_energy
            chain = new_chain
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = chain[:]
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold an HP sequence on the square lattice.
    Uses beam search growth followed by multi-move local search refinement.
    """
    _init_time()
    
    n = len(sequence)
    if n == 0:
        return []
    
    # Emergency fallback if no time left
    if _time_remaining() < 1.0:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Perform restarts until time runs out
    max_iterations = 5000
    iterations = 0
    
    while _time_remaining() > 5.0 and iterations < max_iterations:
        iterations += 1
        
        # Generate a new conformation using beam search
        chain = _grow_beam(sequence, dimension, rng, beam_width=8)
        if chain is None:
            continue
        
        # Refine with local search
        chain, energy = _local_search(sequence, chain, rng, max_steps=150)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
