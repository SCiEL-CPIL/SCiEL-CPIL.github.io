"""
HP lattice protein folding on the simple cubic lattice.
Strategy: Monte Carlo (Metropolis) with Pivot and Slide moves,
seeded from multiple greedy growths. Time-bounded with explicit deadline.
"""
import random
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 310.0  # 20s safety margin before 330s kill


def _is_deadline():
    return time.time() >= _DEADLINE


def _energy(sequence, chain):
    """Compute negative H-H contact count for a chain."""
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    n = len(chain)
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            x, y, z = chain[i]
            for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                j = where.get((x+dx, y+dy, z+dz))
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng, max_iter=200):
    """Randomized greedy chain growth preferring H-H contacts."""
    n = len(sequence)
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, n):
        x, y, z = chain[-1]
        candidates = []
        for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            site = (x+dx, y+dy, z+dz)
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for adx, ady, adz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                    q = (site[0]+adx, site[1]+ady, site[2]+adz)
                    if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H':
                        gain += 1
            free = 0
            for adx, ady, adz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                if (site[0]+adx, site[1]+ady, site[2]+adz) not in occupied:
                    free += 1
            score = gain + 0.1 * free + rng.random()
            candidates.append((score, site))
        if not candidates:
            return None
        site = max(candidates)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move(chain, rng):
    """Attempt a pivot move: rotate a subchain about a bond axis. Returns new chain or None."""
    n = len(chain)
    if n < 3:
        return None
    # Pick pivot index (bond between k-1 and k)
    k = rng.randint(1, n - 1)
    # Pick rotation: axis (0,1,2) and angle (+90, -90)
    axis = rng.randint(0, 2)
    sign = rng.choice((-1, 1))
    # Compute rotation
    pivot_point = chain[k - 1]
    subchain = chain[k:]
    # Rotate subchain about pivot_point around axis by sign*90 degrees
    rotated = []
    for p in subchain:
        dx = p[0] - pivot_point[0]
        dy = p[1] - pivot_point[1]
        dz = p[2] - pivot_point[2]
        if axis == 0:  # x-axis: y,z rotate
            if sign == 1:
                ndx, ndy, ndz = dx, -dz, dy
            else:
                ndx, ndy, ndz = dx, dz, -dy
        elif axis == 1:  # y-axis: x,z rotate
            if sign == 1:
                ndx, ndy, ndz = dz, dy, -dx
            else:
                ndx, ndy, ndz = -dz, dy, dx
        else:  # z-axis: x,y rotate
            if sign == 1:
                ndx, ndy, ndz = -dy, dx, dz
            else:
                ndx, ndy, ndz = dy, -dx, dz
        rotated.append((pivot_point[0] + ndx, pivot_point[1] + ndy, pivot_point[2] + ndz))
    new_chain = chain[:k] + rotated
    # Check self-avoidance
    seen = set()
    for p in new_chain:
        if p in seen:
            return None
        seen.add(p)
    return new_chain


def _slide_move(chain, rng):
    """Attempt a slide move: move an internal segment by one step. Returns new chain or None."""
    n = len(chain)
    if n < 4:
        return None
    # Pick a segment [a, b] (inclusive) to slide, not including endpoints
    a = rng.randint(1, n - 3)
    b = rng.randint(a + 1, n - 2)
    # Pick a direction to slide
    dx, dy, dz = ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))[rng.randint(0, 5)]
    new_chain = list(chain)
    for i in range(a, b + 1):
        p = chain[i]
        new_chain[i] = (p[0] + dx, p[1] + dy, p[2] + dz)
    # Check self-avoidance
    seen = set()
    for p in new_chain:
        if p in seen:
            return None
        seen.add(p)
    return new_chain


def _monte_carlo(sequence, chain, rng, max_steps=20000, temperature=1.0):
    """Run Metropolis Monte Carlo with pivot and slide moves. Returns best chain found."""
    n = len(chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    steps_done = 0
    while steps_done < max_steps and not _is_deadline():
        # Choose move type
        if rng.random() < 0.7:
            new_chain = _pivot_move(chain, rng)
        else:
            new_chain = _slide_move(chain, rng)
        if new_chain is None:
            steps_done += 1
            continue
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        if delta <= 0 or rng.random() < (2.718281828 ** (-delta / temperature)):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        steps_done += 1
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold the sequence and return the best conformation found within time budget."""
    n = len(sequence)
    rng = random.Random(42 + len(sequence))
    
    # Generate initial candidates via greedy growth
    candidates = []
    for i in range(30):
        if _is_deadline():
            break
        c = _greedy_grow(sequence, dimension, rng)
        if c is not None:
            e = _energy(sequence, c)
            candidates.append((e, c))
    
    if not candidates:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]
    
    # Sort by energy, take best
    candidates.sort(key=lambda x: x[0])
    best_chain = candidates[0][1]
    best_energy = candidates[0][0]
    
    # Run Monte Carlo on the best candidates
    num_mc = min(len(candidates), 8)
    for i in range(num_mc):
        if _is_deadline():
            break
        e, c = candidates[i]
        # Adjust temperature based on how many steps we've done
        mc_chain, mc_energy = _monte_carlo(sequence, c, rng, max_steps=50000, temperature=0.5)
        if mc_energy < best_energy:
            best_energy = mc_energy
            best_chain = mc_chain
    
    # Final local refinement: try multiple small MC runs from the best
    if not _is_deadline():
        mc_chain, mc_energy = _monte_carlo(sequence, best_chain, rng, max_steps=100000, temperature=0.3)
        if mc_energy < best_energy:
            best_energy = mc_energy
            best_chain = mc_chain
    
    return [list(p) for p in best_chain]
