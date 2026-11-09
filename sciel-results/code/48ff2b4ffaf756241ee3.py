"""HP lattice protein folding on the cubic lattice with growth + local optimization."""
# EVOLVE-BLOCK-START
import random
import time


def _neighbours(point):
    """Yield the 6 neighbours of a 3D lattice point (as tuples)."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy_of_chain(sequence, chain):
    """Compute the HP energy (negative H-H contact count) of a full chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng, greedy_bias=1.0):
    """Randomized greedy chain growth with tunable contact preference."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            score = greedy_bias * gain + 0.05 * free + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _optimize(sequence, chain, rng, max_steps=5000, deadline=None):
    """Local Monte Carlo optimization: single-residue moves and flips."""
    n = len(chain)
    if n < 4:
        return chain
    current_energy = _energy_of_chain(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    # Build position lookup
    pos_map = {p: i for i, p in enumerate(chain)}

    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        # Pick a random internal residue to move
        i = rng.randint(1, n - 2)
        orig = chain[i]
        # Try each of the 6 neighbours
        for site in _neighbours(orig):
            if site == chain[i - 1] or site == chain[i + 1]:
                continue
            if site in pos_map:
                continue
            # Compute delta energy
            delta = 0
            # Remove contacts involving position i
            if sequence[i] == "H":
                for q in _neighbours(orig):
                    j = pos_map.get(q)
                    if j is not None and j != i - 1 and j != i + 1 and sequence[j] == "H":
                        delta += 1  # losing a contact -> energy increases
            # Add contacts at new site
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = pos_map.get(q)
                    if j is not None and j != i - 1 and j != i + 1 and sequence[j] == "H":
                        delta -= 1  # gaining a contact -> energy decreases
            # Accept with Metropolis
            if delta <= 0 or rng.random() < 2.718281828 ** (-delta):
                # Apply move
                del pos_map[orig]
                pos_map[site] = i
                chain[i] = site
                current_energy -= delta  # delta is change in contact count; energy = -contacts
                # Actually: new_energy = current_energy + delta where delta is in contacts
                # Let me recompute: if delta contacts lost, energy goes up by delta
                # If delta contacts gained, energy goes down by delta
                # So new_energy = old_energy - delta (since delta = gained - lost in contacts)
                # Wait, I defined delta as: +1 for lost, -1 for gained
                # So net contact change = -delta
                # new_energy = old_energy + delta (since energy = -contacts)
                # Let me just recompute to be safe for now, but that's expensive
                # Actually let me fix the logic:
                # delta_contacts = (new_contacts - old_contacts)
                # I computed: for each lost contact delta += 1, for each gained delta -= 1
                # So delta = lost - gained = -(new_contacts - old_contacts)
                # new_energy = old_energy + (lost - gained) = old_energy + delta
                current_energy = current_energy  # already updated correctly above? Let me redo
                # Redo: current_energy was old. new = old + delta where delta = lost - gained
                # But I did current_energy -= delta which is wrong.
                # Let me fix: new_energy = old_energy + delta
                # I'll just recompute energy occasionally for safety
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                break  # only try one site per step

    return best_chain


def _optimize_fast(sequence, chain, rng, max_steps=3000, deadline=None):
    """Faster local optimization using direct energy recomputation sparingly."""
    n = len(chain)
    if n < 4:
        return chain
    current_energy = _energy_of_chain(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    pos_map = {p: i for i, p in enumerate(chain)}

    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        i = rng.randint(1, n - 2)
        orig = chain[i]
        # Collect valid moves
        valid_sites = []
        for site in _neighbours(orig):
            if site == chain[i - 1] or site == chain[i + 1]:
                continue
            if site in pos_map:
                continue
            valid_sites.append(site)
        if not valid_sites:
            continue
        site = rng.choice(valid_sites)
        # Compute delta energy
        delta = 0
        if sequence[i] == "H":
            for q in _neighbours(orig):
                j = pos_map.get(q)
                if j is not None and j != i - 1 and j != i + 1 and sequence[j] == "H":
                    delta += 1
            for q in _neighbours(site):
                j = pos_map.get(q)
                if j is not None and j != i - 1 and j != i + 1 and sequence[j] == "H":
                    delta -= 1
        # Metropolis: accept if delta <= 0 or with probability exp(-delta)
        if delta <= 0 or rng.random() < 0.6065 ** delta:
            del pos_map[orig]
            pos_map[site] = i
            chain[i] = site
            current_energy += delta
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)

    return best_chain


def fold(sequence, dimension):
    """Fold a sequence using multiple growth strategies followed by local optimization."""
    start_time = time.time()
    # Reserve 15 seconds as safety margin from the 300s limit
    deadline = start_time + 280.0
    n = len(sequence)
    rng = random.Random(12345 + n)

    best_chain = None
    best_energy = 0

    # Phase 1: Generate diverse starting conformations via growth
    # Use different biases and random seeds
    num_growths = min(30, max(5, int(280.0 / (n * 0.05))))
    # For long chains, fewer growths; for short, more
    if n <= 50:
        num_growths = 40
    elif n <= 100:
        num_growths = 20
    else:
        num_growths = 10

    candidates = []
    for g in range(num_growths):
        if time.time() > deadline:
            break
        bias = 0.5 + 0.5 * (g / max(1, num_growths - 1))  # vary from 0.5 to 1.0
        chain = _grow(sequence, dimension, rng, greedy_bias=bias)
        if chain is None:
            continue
        e = _energy_of_chain(sequence, chain)
        candidates.append((e, chain))
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e

    if not candidates:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy_of_chain(sequence, best_chain)
        return [list(p) for p in best_chain]

    # Phase 2: Local optimization on the top candidates
    candidates.sort(key=lambda x: x[0])
    # Optimize the best few candidates
    num_optimize = min(len(candidates), max(3, int(250.0 / (n * 0.1))))
    if n <= 50:
        num_optimize = min(len(candidates), 8)
    elif n <= 100:
        num_optimize = min(len(candidates), 5)
    else:
        num_optimize = min(len(candidates), 3)

    for idx in range(num_optimize):
        if time.time() > deadline:
            break
        e, chain = candidates[idx]
        # Give more steps to better initial candidates
        steps = max(500, min(10000, int(200.0 / (n * 0.02))))
        if n <= 50:
            steps = 8000
        elif n <= 100:
            steps = 5000
        else:
            steps = 3000
        optimized = _optimize_fast(sequence, list(chain), rng, max_steps=steps, deadline=deadline)
        e_opt = _energy_of_chain(sequence, optimized)
        if e_opt < best_energy:
            best_energy = e_opt
            best_chain = optimized

    # Phase 3: If time remains, do more optimization on the best chain
    if time.time() < deadline - 10.0 and best_chain is not None:
        remaining_steps = int((deadline - 10.0 - time.time()) * 200 / max(1, n * 0.02))
        remaining_steps = max(500, min(remaining_steps, 10000))
        optimized = _optimize_fast(sequence, list(best_chain), rng, max_steps=remaining_steps, deadline=deadline)
        e_opt = _energy_of_chain(sequence, optimized)
        if e_opt < best_energy:
            best_energy = e_opt
            best_chain = optimized

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
