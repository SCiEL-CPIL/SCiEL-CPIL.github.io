"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = time.time() + 175.0


def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _greedy_grow(sequence, dimension, rng, temperature=0.0):
    """
    Randomized greedy chain growth.
    Prefer sites that create new H-H contacts, then sites with more free space.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)

    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = i
        chain.append(best_site)

    return chain


def _greedy_grow_from_fixed(suffix_seq, start_pos, prefix, full_sequence, rng, temperature=0.0):
    """
    Greedy growth for suffix, with proper access to full sequence for H checks.
    """
    if not suffix_seq:
        return []

    # Build occupied set from prefix with their global indices
    occupied = {p: i for i, p in enumerate(prefix)}
    chain = [start_pos]

    for i, res in enumerate(suffix_seq):
        global_idx = len(prefix) + i
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if res == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < global_idx - 1 and full_sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = global_idx
        chain.append(best_site)

    return chain[1:]


def _regrow_sa(sequence, chain, rng, max_iters=3000, start_temp=5.0, cool_rate=0.997):
    """
    Simulated annealing using regrow moves: pick a split point, regrow the suffix.
    This is the key mechanism for escaping local optima.
    """
    n = len(sequence)
    if n < 4:
        return chain, _compute_energy(sequence, chain)

    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    temp = start_temp
    no_improve = 0

    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        if temp < 0.05:
            break

        # Pick a random split point (keep at least 3 residues in prefix, 2 in suffix)
        min_split = 3
        max_split = max(min_split, n - 3)
        if max_split <= min_split:
            break
        split = rng.randint(min_split, max_split)

        prefix = chain[:split]
        suffix_seq = sequence[split:]

        # Regrow suffix from the end of prefix
        new_suffix = _greedy_grow_from_fixed(suffix_seq, prefix[-1], prefix, sequence, rng, temperature=temp)
        if new_suffix is None:
            temp *= cool_rate
            no_improve += 1
            continue

        new_chain = prefix + new_suffix
        new_energy = _compute_energy(sequence, new_chain)

        delta = new_energy - current_energy
        if delta < 0 or (temp > 0 and rng.random() < (10 ** (-delta / max(temp, 0.1)))):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        temp *= cool_rate

        if no_improve > 500:
            break

    return best_chain, best_energy


def _local_improve(sequence, chain, rng, max_moves=5000):
    """
    Improve conformation via local moves: kink flips and short regrows.
    """
    n = len(chain)
    if n < 3:
        return chain, _compute_energy(sequence, chain)

    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    no_improve_count = 0

    for _ in range(max_moves):
        if time.time() > _DEADLINE - 5:
            break

        move_type = rng.random()
        improved = False

        if move_type < 0.5:
            # Kink flip: try to move residue i to a neighbor that maintains connectivity
            i = rng.randint(1, n - 2)
            p = chain[i]
            prev, nxt = chain[i - 1], chain[i + 1]

            candidates = []
            for np in _neighbours(prev):
                if np == p:
                    continue
                if abs(np[0] - nxt[0]) + abs(np[1] - nxt[1]) != 1:
                    continue
                if np in occupied and occupied[np] != i:
                    continue
                candidates.append(np)

            if candidates:
                np = rng.choice(candidates)
                # Perform move
                del occupied[p]
                chain[i] = np
                occupied[np] = i
                new_energy = _compute_energy(sequence, chain)

                if new_energy < current_energy:
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                    no_improve_count = 0
                    improved = True
                else:
                    # Revert
                    del occupied[np]
                    chain[i] = p
                    occupied[p] = i
                    no_improve_count += 1
        else:
            # Short regrow: regrow last 3-5 residues
            length = rng.randint(3, min(5, n - 3))
            split = n - length
            prefix = chain[:split]
            suffix_seq = sequence[split:]

            new_suffix = _greedy_grow_from_fixed(suffix_seq, prefix[-1], prefix, sequence, rng, temperature=0.5)
            if new_suffix is not None:
                new_chain = prefix + new_suffix
                new_energy = _compute_energy(sequence, new_chain)

                if new_energy < current_energy:
                    # Update occupied set
                    occupied = {p: i for i, p in enumerate(new_chain)}
                    chain = new_chain
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                    no_improve_count = 0
                    improved = True
                else:
                    no_improve_count += 1

        if not improved:
            no_improve_count += 1
            if no_improve_count > 200:
                break

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold a sequence into a low-energy conformation.
    """
    if dimension != 2:
        raise ValueError("Only dimension 2 is supported")

    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]

    rng = random.Random(12345 + len(sequence))

    best_chain = None
    best_energy = float('inf')

    # Phase 1: Multiple greedy growths with different temperatures
    temperatures = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0]

    for temp in temperatures:
        for _ in range(15):
            if time.time() > _DEADLINE - 20:
                break
            chain = _greedy_grow(sequence, dimension, rng, temperature=temp)
            if chain is None:
                continue
            energy = _compute_energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain

        if time.time() > _DEADLINE - 20:
            break

    if best_chain is None:
        # Fallback: straight line
        best_chain = [[i, 0] for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)

    # Phase 2: Local improvement (kink flips + short regrows)
    if time.time() < _DEADLINE - 15:
        improved_chain, improved_energy = _local_improve(sequence, best_chain, rng, max_moves=5000)
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain

    # Phase 3: Simulated annealing with regrow moves (key for escaping local optima)
    if time.time() < _DEADLINE - 10:
        sa_chain, sa_energy = _regrow_sa(sequence, best_chain, rng, max_iters=3000, start_temp=5.0, cool_rate=0.997)
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = sa_chain

    # Phase 4: More greedy growths with local search + SA
    if time.time() < _DEADLINE - 5:
        for _ in range(30):
            if time.time() > _DEADLINE - 5:
                break
            chain = _greedy_grow(sequence, dimension, rng, temperature=0.5)
            if chain is None:
                continue
            energy = _compute_energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
            # Local improvement
            if time.time() < _DEADLINE - 5:
                improved = _local_improve(sequence, chain, rng, max_moves=2000)
                if improved[1] < best_energy:
                    best_energy = improved[1]
                    best_chain = improved[0]
            # SA on the improved chain
            if time.time() < _DEADLINE - 3:
                sa_c, sa_e = _regrow_sa(sequence, best_chain, rng, max_iters=1000, start_temp=3.0, cool_rate=0.998)
                if sa_e < best_energy:
                    best_energy = sa_e
                    best_chain = sa_c

    # Final local polish
    if time.time() < _DEADLINE - 2:
        final_chain, final_energy = _local_improve(sequence, best_chain, rng, max_moves=3000)
        if final_energy < best_energy:
            best_energy = final_energy
            best_chain = final_chain

    # Convert to list of lists
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
