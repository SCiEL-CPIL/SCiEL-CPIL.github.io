"""HP lattice protein folding: global stochastic local search (SA) on random initial SAWs."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = time.time() + 175.0


def _neighbours2d(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(sequence, chain):
    """Compute HP contact energy (negative = better)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours2d(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _random_saw(n, rng, max_tries=200):
    """Generate a random self-avoiding walk of length n on the square lattice."""
    for _ in range(max_tries):
        chain = [(0, 0)]
        occupied = {(0, 0)}
        success = True
        for _ in range(n - 1):
            x, y = chain[-1]
            options = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
            random.shuffle(options, rng)
            moved = False
            for site in options:
                if site not in occupied:
                    chain.append(site)
                    occupied.add(site)
                    moved = True
                    break
            if not moved:
                success = False
                break
        if success:
            return chain
    # Fallback: simple L-shape (should not happen for n up to 100)
    return [(i, 0) for i in range(n)]


def _random_saw_compact(n, rng, max_tries=200):
    """Generate a random SAW biased toward compactness (prefer turns)."""
    for _ in range(max_tries):
        chain = [(0, 0)]
        occupied = {(0, 0)}
        success = True
        last_dir = None
        for _ in range(n - 1):
            x, y = chain[-1]
            options = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
            # Filter out the reverse direction
            if last_dir is not None:
                rev = (x - last_dir[0], y - last_dir[1])
                options = [o for o in options if o != rev]
            random.shuffle(options, rng)
            # Bias toward turns: weight by 1 for straight, 2 for turn
            scored = []
            for site in options:
                if site in occupied:
                    continue
                sx, sy = site
                dx, dy = sx - x, sy - y
                if last_dir is not None and (dx, dy) == last_dir:
                    score = 1.0
                else:
                    score = 2.0
                scored.append((score, site))
            if not scored:
                success = False
                break
            # Weighted selection
            total = sum(s for s, _ in scored)
            r = rng.random() * total
            cum = 0
            chosen = scored[0][1]
            for s, site in scored:
                cum += s
                if r <= cum:
                    chosen = site
                    break
            chain.append(chosen)
            occupied.add(chosen)
            last_dir = (chosen[0] - x, chosen[1] - y)
        if success:
            return chain
    return [(i, 0) for i in range(n)]


def _sa_optimize(sequence, chain, time_budget):
    """Simulated annealing with single-residue, kink, and segment-reversal moves."""
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)

    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)

    T = max(1.0, n * 0.1)
    T_min = 0.01
    cooling = 0.998

    start_time = time.time()
    iters = 0

    while time.time() - start_time < time_budget and time.time() < _DEADLINE:
        iters += 1
        move_type = random.random()

        if move_type < 0.4:
            # Single-residue move
            i = random.randint(1, n - 2)
            x, y = chain[i]
            candidates = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
            random.shuffle(candidates)
            for new_site in candidates:
                if new_site in occupied and occupied[new_site] != i:
                    continue
                # Connectivity check
                if i > 0:
                    px, py = chain[i-1]
                    if abs(new_site[0]-px) + abs(new_site[1]-py) != 1:
                        continue
                if i < n-1:
                    px, py = chain[i+1]
                    if abs(new_site[0]-px) + abs(new_site[1]-py) != 1:
                        continue

                # Compute delta
                old_site = chain[i]
                delta = 0
                if sequence[i] == 'H':
                    # Remove old contacts
                    for q in _neighbours2d(old_site):
                        j = occupied.get(q)
                        if j is not None and j != i and sequence[j] == 'H':
                            delta += 1
                    # Add new contacts
                    for q in _neighbours2d(new_site):
                        j = occupied.get(q)
                        if j is not None and j != i and sequence[j] == 'H':
                            delta -= 1

                if delta < 0 or random.random() < math.exp(-delta / T):
                    occupied[old_site] = None
                    del occupied[old_site]
                    occupied[new_site] = i
                    chain[i] = new_site
                    current_energy += delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                break

        elif move_type < 0.7:
            # Kink move: move residues i and i+1 together
            i = random.randint(1, n - 3)
            p1, p2 = chain[i], chain[i+1]
            x1, y1 = p1
            x2, y2 = p2
            dx, dy = x2 - x1, y2 - y1

            if dx != 0:
                candidates = [(x1, y1+1), (x1, y1-1)]
            else:
                candidates = [(x1+1, y1), (x1-1, y1)]

            for sign_pos in candidates:
                sx, sy = sign_pos - (x1, y1)
                np1 = (x1 + sx, y1 + sy)
                np2 = (x2 + sx, y2 + sy)

                if np1 in occupied and occupied[np1] not in (i, i+1):
                    continue
                if np2 in occupied and occupied[np2] not in (i, i+1):
                    continue
                if i > 0:
                    px, py = chain[i-1]
                    if abs(np1[0]-px) + abs(np1[1]-py) != 1:
                        continue
                if i+1 < n-1:
                    px, py = chain[i+2]
                    if abs(np2[0]-px) + abs(np2[1]-py) != 1:
                        continue

                # Compute delta
                old_occ = {k: v for k, v in occupied.items() if v not in (i, i+1)}
                delta = 0
                for idx, old_pos, new_pos in [(i, p1, np1), (i+1, p2, np2)]:
                    if sequence[idx] == 'H':
                        for q in _neighbours2d(old_pos):
                            j = old_occ.get(q)
                            if j is not None and sequence[j] == 'H':
                                delta += 1
                        for q in _neighbours2d(new_pos):
                            j = old_occ.get(q)
                            if j is not None and sequence[j] == 'H':
                                delta -= 1

                if delta < 0 or random.random() < math.exp(-delta / T):
                    del occupied[p1]
                    del occupied[p2]
                    occupied[np1] = i
                    occupied[np2] = i + 1
                    chain[i] = np1
                    chain[i+1] = np2
                    current_energy += delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                break

        else:
            # Segment reversal move
            i = random.randint(1, n - 3)
            length = random.randint(2, min(n - i - 1, 10))
            j = i + length - 1
            if j >= n - 1:
                j = n - 2

            # Check if reversal is valid
            seg = chain[i:j+1]
            reversed_seg = list(reversed(seg))
            
            # Check connectivity at boundaries
            if i > 0:
                px, py = chain[i-1]
                if abs(reversed_seg[0][0]-px) + abs(reversed_seg[0][1]-py) != 1:
                    continue
            if j < n-1:
                px, py = chain[j+1]
                if abs(reversed_seg[-1][0]-px) + abs(reversed_seg[-1][1]-py) != 1:
                    continue

            # Check self-avoidance: reversed segment must not overlap with rest
            rest = set(chain[:i]) | set(chain[j+1:])
            valid = True
            for p in reversed_seg:
                if p in rest:
                    valid = False
                    break
            if not valid:
                continue

            # Compute delta
            old_occ = {p: idx for idx, p in enumerate(chain) if idx < i or idx > j}
            delta = 0
            for idx in range(i, j+1):
                if sequence[idx] == 'H':
                    old_pos = chain[idx]
                    new_pos = reversed_seg[idx - i]
                    for q in _neighbours2d(old_pos):
                        jj = old_occ.get(q)
                        if jj is not None and sequence[jj] == 'H':
                            delta += 1
                    for q in _neighbours2d(new_pos):
                        jj = old_occ.get(q)
                        if jj is not None and sequence[jj] == 'H':
                            delta -= 1
                    # Also check contacts within the segment itself
                    # (these are preserved in reversal, so no net change)

            if delta < 0 or random.random() < math.exp(-delta / T):
                # Apply reversal
                for idx in range(i, j+1):
                    del occupied[chain[idx]]
                for k, p in enumerate(reversed_seg):
                    chain[i+k] = p
                    occupied[p] = i+k
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)

        T *= cooling
        if T < T_min:
            T = T_min

    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold sequence using global stochastic local search with multiple restarts."""
    n = len(sequence)
    
    # Time budget per sequence
    if n <= 30:
        seq_time = 10.0
    elif n <= 50:
        seq_time = 18.0
    elif n <= 70:
        seq_time = 28.0
    else:
        seq_time = 40.0

    seq_deadline = time.time() + seq_time

    best_chain = None
    best_energy = 0

    # Multiple restarts with different initial SAWs
    rng = random.Random(42 + n)
    restarts_done = 0
    max_restarts = 50

    while time.time() < seq_deadline and time.time() < _DEADLINE and restarts_done < max_restarts:
        restarts_done += 1

        # Choose initial conformation strategy
        if restarts_done % 2 == 1:
            chain = _random_saw_compact(n, rng)
        else:
            chain = _random_saw(n, rng)

        if chain is None:
            continue

        # SA optimization
        remaining = min(seq_deadline - time.time(), 3.0)
        if remaining < 0.5:
            break

        optimized_chain, energy = _sa_optimize(sequence, chain, time_budget=remaining)

        if best_chain is None or energy < best_energy:
            best_chain = optimized_chain
            best_energy = energy

    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
