"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = time.time() + 325  # safety margin before 330s kill


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Energy = -(# of H-H non-adjacent lattice contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
    """Randomized greedy chain growth with trap penalty."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            trap_penalty = -0.5 if free == 1 else 0.0
            options.append((gain + 0.1 * free + trap_penalty + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _is_valid_1move(chain, i, new_site, dimension):
    """Check if moving residue i to new_site keeps chain valid."""
    if new_site in chain:
        return False
    prev = chain[i - 1]
    nxt = chain[i + 1]
    d_prev = sum(abs(new_site[a] - prev[a]) for a in range(dimension))
    d_next = sum(abs(new_site[a] - nxt[a]) for a in range(dimension))
    if d_prev != 1 or d_next != 1:
        return False
    return True


def _is_valid_kink(chain, i, site1, site2, dimension):
    """Check if 2-residue kink (i->site1, i+1->site2) is valid."""
    if site1 in chain or site2 in chain:
        return False
    if site1 == site2:
        return False
    prev = chain[i - 1]
    nxt = chain[i + 2] if i + 2 < len(chain) else None
    d_prev = sum(abs(site1[a] - prev[a]) for a in range(dimension))
    d_mid = sum(abs(site2[a] - site1[a]) for a in range(dimension))
    if d_prev != 1 or d_mid != 1:
        return False
    if nxt is not None:
        d_next = sum(abs(site2[a] - nxt[a]) for a in range(dimension))
        if d_next != 1:
            return False
    return True


def _is_valid_jump(chain, j, new_pos, dimension):
    """Check if moving residue j to new_pos keeps chain valid.
    j is internal (1 <= j <= n-2), so we need unit steps to j-1 and j+1."""
    if new_pos in chain:
        return False
    prev = chain[j - 1]
    nxt = chain[j + 1]
    d_prev = sum(abs(new_pos[a] - prev[a]) for a in range(dimension))
    d_next = sum(abs(new_pos[a] - nxt[a]) for a in range(dimension))
    if d_prev != 1 or d_next != 1:
        return False
    return True


def _local_refine(sequence, chain, dimension, rng, max_iters=2000):
    """Hybrid local search: 1-moves, kink moves, and jump moves (move residue j
    to a neighbor of residue i, |i-j|>=3) with SA-style acceptance."""
    if len(chain) < 4:
        return chain

    n = len(chain)
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    current_chain = list(chain)
    current_energy = best_energy

    # Precompute H positions for faster energy delta
    h_positions = [i for i in range(n) if sequence[i] == 'H']

    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break

        improved = False
        move_type = rng.random()

        if move_type < 0.35:
            # 1-residue move
            j = rng.randint(1, n - 2)
            for site in _neighbours(current_chain[j]):
                if site in current_chain:
                    continue
                if not _is_valid_1move(current_chain, j, site, dimension):
                    continue
                old_pos = current_chain[j]
                current_chain[j] = site
                new_energy = _energy(sequence, current_chain)
                if new_energy < best_energy:
                    best_chain = current_chain[:]
                    best_energy = new_energy
                    current_energy = new_energy
                    improved = True
                    break
                elif new_energy == current_energy and rng.random() < 0.3:
                    current_energy = new_energy
                    improved = True
                    break
                else:
                    current_chain[j] = old_pos

        elif move_type < 0.6:
            # 2-residue kink move
            if n >= 4:
                i = rng.randint(1, n - 3)
                for site1 in _neighbours(current_chain[i]):
                    if site1 in current_chain:
                        continue
                    for site2 in _neighbours(site1):
                        if site2 in current_chain or site2 == site1:
                            continue
                        if not _is_valid_kink(current_chain, i, site1, site2, dimension):
                            continue
                        old1, old2 = current_chain[i], current_chain[i + 1]
                        current_chain[i] = site1
                        current_chain[i + 1] = site2
                        new_energy = _energy(sequence, current_chain)
                        if new_energy < best_energy:
                            best_chain = current_chain[:]
                            best_energy = new_energy
                            current_energy = new_energy
                            improved = True
                            break
                        elif new_energy == current_energy and rng.random() < 0.2:
                            current_energy = new_energy
                            improved = True
                            break
                        else:
                            current_chain[i] = old1
                            current_chain[i + 1] = old2
                    if improved:
                        break

        else:
            # Jump move: move residue j to a neighbor of residue i, |i-j|>=3
            j = rng.randint(1, n - 2)
            i = rng.randint(0, n - 1)
            if abs(i - j) < 3:
                continue
            anchor = current_chain[i]
            for new_pos in _neighbours(anchor):
                if new_pos in current_chain:
                    continue
                if not _is_valid_jump(current_chain, j, new_pos, dimension):
                    continue
                old_pos = current_chain[j]
                current_chain[j] = new_pos
                new_energy = _energy(sequence, current_chain)
                if new_energy < best_energy:
                    best_chain = current_chain[:]
                    best_energy = new_energy
                    current_energy = new_energy
                    improved = True
                    break
                elif new_energy <= current_energy + 1 and rng.random() < 0.2:
                    # Accept slightly worse to escape basins
                    current_energy = new_energy
                    improved = True
                    break
                else:
                    current_chain[j] = old_pos

        if not improved:
            # Try SA-style escape: accept a worse move
            escaped = False
            for _ in range(5):
                j = rng.randint(1, n - 2)
                for site in _neighbours(current_chain[j]):
                    if site in current_chain:
                        continue
                    if not _is_valid_1move(current_chain, j, site, dimension):
                        continue
                    old_pos = current_chain[j]
                    current_chain[j] = site
                    new_energy = _energy(sequence, current_chain)
                    if new_energy <= best_energy + 3:
                        current_energy = new_energy
                        escaped = True
                        break
                    else:
                        current_chain[j] = old_pos
                if escaped:
                    break
            if not escaped:
                # Stuck, break out
                if it > 50:
                    break
                else:
                    continue

    return best_chain


def fold(sequence, dimension):
    """Best conformation using many randomized greedy growths with hybrid local refinement."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))

    best_chain = None
    best_energy = 0  # energy is negative, 0 means no contacts

    # Scale number of attempts based on chain length
    if n <= 48:
        num_attempts = 800
    elif n <= 80:
        num_attempts = 400
    elif n <= 100:
        num_attempts = 200
    else:
        num_attempts = 100

    for attempt in range(num_attempts):
        if time.time() > _DEADLINE - 5:
            break

        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue

        # Refine all successful growths
        refined = _local_refine(sequence, chain, dimension, rng, max_iters=min(1500, 30 * n))
        value = _energy(sequence, refined)

        if value < best_energy:
            best_chain = refined
            best_energy = value

    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
