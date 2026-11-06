"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math

_DEADLINE = time.time() + 315  # safety margin before 330s kill


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng, trap_penalty_weight=0.3):
    """One randomized greedy chain growth with trap avoidance."""
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
            trap_penalty = -trap_penalty_weight if free == 1 else 0.0
            options.append((gain + 0.1 * free + trap_penalty + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _is_valid_move(chain, i, new_site, dimension):
    """Check that moving residue i to new_site keeps the chain self-avoiding and connected."""
    if new_site in chain:
        return False
    if i == 0 or i == len(chain) - 1:
        return False
    prev = chain[i - 1]
    nxt = chain[i + 1]
    d_prev = sum(abs(new_site[a] - prev[a]) for a in range(dimension))
    d_next = sum(abs(new_site[a] - nxt[a]) for a in range(dimension))
    if d_prev != 1 or d_next != 1:
        return False
    return True


def _is_valid_kink(chain, i, site1, site2, dimension):
    """Check that a 2-residue kink (i -> site1, i+1 -> site2) is valid."""
    if site1 in chain or site2 in chain:
        return False
    if site1 == site2:
        return False
    if i == 0 or i + 1 >= len(chain):
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


def _local_refine(sequence, chain, dimension, rng, max_steps=50):
    """Local search with 1-residue moves, 2-residue kink moves, and SA-style escape."""
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    n = len(chain)
    if n < 3:
        return best_chain

    stall_count = 0
    for step in range(max_steps):
        if time.time() > _DEADLINE - 5:
            break

        moved = False

        # --- 1-residue move at a random internal position ---
        i = rng.randint(1, n - 2)
        for site in _neighbours(chain[i]):
            if site in chain:
                continue
            if not _is_valid_move(chain, i, site, dimension):
                continue
            new_chain = chain[:]
            new_chain[i] = site
            new_energy = _energy(sequence, new_chain)
            if new_energy < best_energy:
                best_chain = new_chain
                best_energy = new_energy
                chain = new_chain
                moved = True
                stall_count = 0
                break
            elif new_energy == best_energy and rng.random() < 0.3:
                chain = new_chain
                moved = True
                stall_count = 0
                break

        # --- 2-residue kink move ---
        if not moved and n >= 4:
            i = rng.randint(1, n - 3)
            for site1 in _neighbours(chain[i]):
                if site1 in chain:
                    continue
                for site2 in _neighbours(site1):
                    if site2 in chain or site2 == site1:
                        continue
                    if not _is_valid_kink(chain, i, site1, site2, dimension):
                        continue
                    new_chain = chain[:]
                    new_chain[i] = site1
                    new_chain[i + 1] = site2
                    new_energy = _energy(sequence, new_chain)
                    if new_energy < best_energy:
                        best_chain = new_chain
                        best_energy = new_energy
                        chain = new_chain
                        moved = True
                        stall_count = 0
                        break
                    elif new_energy == best_energy and rng.random() < 0.2:
                        chain = new_chain
                        moved = True
                        stall_count = 0
                        break
                if moved:
                    break

        if not moved:
            # Simulated-annealing style: try a few random 1-moves even if they
            # worsen energy, to escape the local basin.
            escaped = False
            for _ in range(3):
                j = rng.randint(1, n - 2)
                for site in _neighbours(chain[j]):
                    if site in chain:
                        continue
                    if not _is_valid_move(chain, j, site, dimension):
                        continue
                    new_chain = chain[:]
                    new_chain[j] = site
                    new_energy = _energy(sequence, new_chain)
                    if new_energy <= best_energy + 2:
                        chain = new_chain
                        escaped = True
                        break
                if escaped:
                    break
            if escaped:
                stall_count += 1
                if stall_count >= 5:
                    break
            else:
                break

    return best_chain


def _deep_refine(sequence, chain, dimension, rng, time_limit=10.0):
    """Deep simulated annealing refinement for the best candidates."""
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    n = len(chain)
    if n < 3:
        return best_chain

    start_time = time.time()
    T = 2.0
    T_min = 0.1
    cooling = 0.995
    step = 0
    stall_count = 0

    while time.time() - start_time < time_limit:
        if time.time() > _DEADLINE - 5:
            break

        step += 1
        T = max(T_min, T * cooling)

        # Choose move type
        move_type = rng.random()
        moved = False

        if move_type < 0.6:
            # 1-residue move
            i = rng.randint(1, n - 2)
            for site in _neighbours(chain[i]):
                if site in chain:
                    continue
                if not _is_valid_move(chain, i, site, dimension):
                    continue
                new_chain = chain[:]
                new_chain[i] = site
                new_energy = _energy(sequence, new_chain)
                delta_e = new_energy - current_energy
                if delta_e < 0 or (T > T_min and rng.random() < math.exp(-delta_e / T)):
                    chain = new_chain
                    current_energy = new_energy
                    moved = True
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = new_chain[:]
                        stall_count = 0
                    else:
                        stall_count += 1
                    break
                else:
                    stall_count += 1
        else:
            # 2-residue kink move
            if n >= 4:
                i = rng.randint(1, n - 3)
                for site1 in _neighbours(chain[i]):
                    if site1 in chain:
                        continue
                    for site2 in _neighbours(site1):
                        if site2 in chain or site2 == site1:
                            continue
                        if not _is_valid_kink(chain, i, site1, site2, dimension):
                            continue
                        new_chain = chain[:]
                        new_chain[i] = site1
                        new_chain[i + 1] = site2
                        new_energy = _energy(sequence, new_chain)
                        delta_e = new_energy - current_energy
                        if delta_e < 0 or (T > T_min and rng.random() < math.exp(-delta_e / T)):
                            chain = new_chain
                            current_energy = new_energy
                            moved = True
                            if new_energy < best_energy:
                                best_energy = new_energy
                                best_chain = new_chain[:]
                                stall_count = 0
                            else:
                                stall_count += 1
                            break
                        else:
                            stall_count += 1
                    if moved:
                        break

        if stall_count >= 30:
            break

    return best_chain


def fold(sequence, dimension):
    """Best conformation using two-phase search: fast screening then deep refinement."""
    rng = random.Random(12345 + len(sequence))
    seq_len = len(sequence)

    # Phase 1: Fast screening with many diverse starts
    if seq_len <= 48:
        n_screen = 600
    elif seq_len <= 64:
        n_screen = 400
    elif seq_len <= 96:
        n_screen = 250
    else:
        n_screen = 150

    candidates = []
    seq_deadline = _DEADLINE - 15

    for r in range(n_screen):
        if time.time() > seq_deadline:
            break

        # Vary trap penalty to get diverse starts
        trap_w = 0.2 + 0.3 * rng.random()
        chain = _greedy_grow(sequence, dimension, rng, trap_penalty_weight=trap_w)
        if chain is None:
            continue

        # Quick local refinement
        refined = _local_refine(sequence, chain, dimension, rng, max_steps=min(30, 5 * seq_len // 10))
        value = _energy(sequence, refined)
        candidates.append((value, refined))

    if not candidates:
        # Fallback: straight line
        return [tuple([i] + [0] * (dimension - 1)) for i in range(seq_len)]

    # Sort by energy (most negative first)
    candidates.sort(key=lambda x: x[0])

    # Phase 2: Deep refinement on top candidates
    n_deep = min(10, len(candidates))
    best_chain = candidates[0][1]
    best_energy = candidates[0][0]

    for idx in range(n_deep):
        if time.time() > _DEADLINE - 5:
            break

        cand_chain = candidates[idx][1]
        # Allocate more time for better candidates
        time_alloc = 8.0 if idx == 0 else 5.0 if idx < 3 else 3.0
        if time.time() + time_alloc > _DEADLINE - 5:
            time_alloc = max(1.0, _DEADLINE - 5 - time.time())

        refined = _deep_refine(sequence, cand_chain, dimension, rng, time_limit=time_alloc)
        value = _energy(sequence, refined)

        if value < best_energy:
            best_energy = value
            best_chain = refined

    if best_chain is None:
        best_chain = candidates[0][1]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
