"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

# Global deadline: evaluator kills after 210s wall-clock. Use a safety margin.
_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 195.0

def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)

def _neighbours(point):
    x, y = point
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

def _energy(sequence, chain):
    """Compute the HP energy: -1 for each non-consecutive H-H contact."""
    if chain is None:
        return 1
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y = p
            for q in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _contact_map(sequence, chain):
    """Build a map from position to list of H indices at that position's neighbors."""
    where = {p: i for i, p in enumerate(chain)}
    h_positions = {}
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            h_positions[i] = p
    return where, h_positions

def _energy_delta_single(sequence, chain, pos_map, i, new_pos):
    """Fast energy delta for moving residue i to new_pos."""
    n = len(chain)
    old_pos = chain[i]
    
    # Check occupancy
    if new_pos in pos_map and pos_map[new_pos] != i:
        return None
    
    # Check unit steps to neighbors
    if i > 0:
        prev = chain[i-1]
        if abs(new_pos[0]-prev[0]) + abs(new_pos[1]-prev[1]) != 1:
            return None
    if i < n - 1:
        nxt = chain[i+1]
        if abs(new_pos[0]-nxt[0]) + abs(new_pos[1]-nxt[1]) != 1:
            return None
    
    if sequence[i] != 'H':
        return 0
    
    delta = 0
    # Remove old contacts
    for q in _neighbours(old_pos):
        j = pos_map.get(q)
        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
            delta += 1
    # Add new contacts
    for q in _neighbours(new_pos):
        j = pos_map.get(q)
        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
            delta -= 1
    return delta

def _grow(sequence, dimension, rng, temperature=0.5, contact_bonus=5.0):
    """
    Grow a chain on the lattice with strong contact bias.
    """
    n = len(sequence)
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}

    for i in range(1, n):
        res = sequence[i]
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if res == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * contact_bonus + 0.3 * free + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        options.sort(key=lambda x: -x[0])
        if temperature > 0 and len(options) > 1:
            top = options[0][0]
            probs = []
            total = 0.0
            for sc, _ in options[:min(4, len(options))]:
                p = math.exp((sc - top) / max(temperature, 0.1))
                probs.append(p)
                total += p
            if total > 0:
                probs = [p / total for p in probs]
                r = rng.random()
                cum = 0.0
                chosen_idx = len(probs) - 1
                for idx, p in enumerate(probs):
                    cum += p
                    if r <= cum:
                        chosen_idx = idx
                        break
                site = options[chosen_idx][1]
            else:
                site = options[0][1]
        else:
            site = options[0][1]

        occupied[site] = i
        chain.append(site)

    return chain

def _pivot_move_fast(sequence, chain, pos_map, i):
    """
    Try all valid pivot moves for residue i. Returns list of (delta, new_pos).
    """
    n = len(chain)
    p = chain[i]
    results = []
    for q in _neighbours(p):
        if q == chain[i-1] or q == chain[i+1]:
            continue
        if q in pos_map and pos_map[q] != i:
            continue
        delta = _energy_delta_single(sequence, chain, pos_map, i, q)
        if delta is not None:
            results.append((delta, q))
    return results

def _slither_move(sequence, chain, pos_map, i):
    """
    Try slithering: move residue i and i+1 together (2-residue segment shift).
    """
    n = len(chain)
    if i + 1 >= n - 1:
        return []
    p1 = chain[i]
    p2 = chain[i+1]
    results = []
    # Try shifting the pair in each direction
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        n1 = (p1[0]+dx, p1[1]+dy)
        n2 = (p2[0]+dx, p2[1]+dy)
        # Check connections
        if i > 0:
            prev = chain[i-1]
            if abs(n1[0]-prev[0]) + abs(n1[1]-prev[1]) != 1:
                continue
        if i + 2 < n:
            nxt = chain[i+2]
            if abs(n2[0]-nxt[0]) + abs(n2[1]-nxt[1]) != 1:
                continue
        # Check occupancy
        if n1 in pos_map and pos_map[n1] not in (i, i+1):
            continue
        if n2 in pos_map and pos_map[n2] not in (i, i+1):
            continue
        # Internal step
        if abs(n2[0]-n1[0]) + abs(n2[1]-n1[1]) != 1:
            continue
        
        # Compute energy delta
        delta = 0
        for k in (i, i+1):
            if sequence[k] == 'H':
                old_p = chain[k]
                new_p = n1 if k == i else n2
                for q in _neighbours(old_p):
                    j = pos_map.get(q)
                    if j is not None and j != k and j not in (i, i+1) and sequence[j] == 'H' and abs(j - k) > 1:
                        delta += 1
                for q in _neighbours(new_p):
                    j = pos_map.get(q)
                    if j is not None and j != k and j not in (i, i+1) and sequence[j] == 'H' and abs(j - k) > 1:
                        delta -= 1
        results.append((delta, (n1, n2)))
    return results

def _local_search(sequence, chain, dimension, rng, max_iter=200):
    """
    Improve a valid chain via local moves with simulated annealing.
    """
    if chain is None:
        return None, 1

    current_energy = _energy(sequence, chain)
    current_chain = list(chain)
    n = len(current_chain)
    pos_map = {p: i for i, p in enumerate(current_chain)}

    for it in range(max_iter):
        if _time_left() < 0.5:
            break

        # Try multiple move types
        best_delta = 0
        best_move = None  # (type, i, new_positions)

        # Single residue move
        i = rng.randint(1, n - 2)
        moves = _pivot_move_fast(sequence, current_chain, pos_map, i)
        for delta, q in moves:
            if delta < best_delta:
                best_delta = delta
                best_move = ('single', i, q)

        # Slither move
        if n > 4:
            i = rng.randint(1, n - 3)
            moves = _slither_move(sequence, current_chain, pos_map, i)
            for delta, (n1, n2) in moves:
                if delta < best_delta:
                    best_delta = delta
                    best_move = ('slither', i, (n1, n2))

        # Apply or perturb
        if best_move is not None and best_delta < 0:
            mtype, i, newpos = best_move
            if mtype == 'single':
                old_p = current_chain[i]
                del pos_map[old_p]
                current_chain[i] = newpos
                pos_map[newpos] = i
            else:
                old_p1 = current_chain[i]
                old_p2 = current_chain[i+1]
                del pos_map[old_p1]
                del pos_map[old_p2]
                current_chain[i] = newpos[0]
                current_chain[i+1] = newpos[1]
                pos_map[newpos[0]] = i
                pos_map[newpos[1]] = i + 1
            current_energy += best_delta
        else:
            # Simulated annealing perturbation
            temp = max(0.5, 3.0 * (max_iter - it) / max_iter)
            i = rng.randint(1, n - 2)
            moves = _pivot_move_fast(sequence, current_chain, pos_map, i)
            if moves:
                delta, q = moves[rng.randint(0, len(moves)-1)]
                if delta <= 0 or rng.random() < math.exp(-delta / temp):
                    old_p = current_chain[i]
                    del pos_map[old_p]
                    current_chain[i] = q
                    pos_map[q] = i
                    current_energy += delta

    return current_chain, current_energy

def _polish(sequence, chain, dimension):
    """
    Deterministic polish: repeatedly try all single-residue moves until no improvement.
    """
    if chain is None:
        return None, 1
    current_chain = list(chain)
    current_energy = _energy(sequence, chain)
    n = len(current_chain)
    pos_map = {p: i for i, p in enumerate(current_chain)}
    improved = True
    max_passes = 50

    for _ in range(max_passes):
        if not improved:
            break
        if _time_left() < 0.5:
            break
        improved = False
        for i in range(1, n - 1):
            if _time_left() < 0.5:
                break
            p = current_chain[i]
            for q in _neighbours(p):
                if q == current_chain[i-1] or q == current_chain[i+1]:
                    continue
                if q in pos_map and pos_map[q] != i:
                    continue
                delta = _energy_delta_single(sequence, current_chain, pos_map, i, q)
                if delta is not None and delta < 0:
                    old_p = current_chain[i]
                    del pos_map[old_p]
                    current_chain[i] = q
                    pos_map[q] = i
                    current_energy += delta
                    improved = True
                    break  # restart scan after a move

    return current_chain, current_energy

def fold(sequence, dimension):
    """
    Find a low-energy conformation for the given HP sequence on a square lattice.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    rng = random.Random(42 + len(sequence))

    best_chain = None
    best_energy = 1

    temps = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
    bonuses = [3.0, 5.0, 7.0, 10.0]

    if n <= 25:
        max_restarts = 800
        ls_iter = 120
    elif n <= 40:
        max_restarts = 400
        ls_iter = 80
    elif n <= 60:
        max_restarts = 200
        ls_iter = 55
    else:
        max_restarts = 100
        ls_iter = 40

    for restart in range(max_restarts):
        if _time_left() < 2.0:
            break

        temp = temps[restart % len(temps)] + rng.random() * 0.3
        temp = min(temp, 2.5)
        bonus = bonuses[restart % len(bonuses)]

        chain = _grow(sequence, dimension, rng, temperature=temp, contact_bonus=bonus)
        if chain is None:
            continue

        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_chain = chain
            best_energy = energy

        if _time_left() < 3.0:
            break

        improved_chain, improved_energy = _local_search(
            sequence, chain, dimension, rng,
            max_iter=ls_iter
        )

        if improved_chain is not None and improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy

        # Final polish on best chain
        if best_chain is not None and _time_left() > 5.0:
            polished_chain, polished_energy = _polish(sequence, best_chain, dimension)
            if polished_chain is not None and polished_energy < best_energy:
                best_chain = polished_chain
                best_energy = polished_energy

    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
