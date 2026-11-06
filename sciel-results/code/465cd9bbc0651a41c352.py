"""HP lattice protein folding: fast randomized growth + local search with time guard."""
# EVOLVE-BLOCK-START
import time
import random

_T0 = time.time()
_DEADLINE = 200.0  # 210s wall-clock limit, 10s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(seq, chain):
    """Compute H-H contact energy (negative = better)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if seq[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and seq[j] == 'H':
                    contacts += 1
    return -contacts


def _grow(seq, dimension, rng):
    """Randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(seq)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if seq[i] == 'H':
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and seq[occupied[q]] == 'H')
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_search(seq, chain, rng, max_iters=500):
    """Simple local search: try moving each residue to a random neighbour, accept if energy improves."""
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(seq, chain)
    best_chain = list(chain)
    best_energy = current_energy

    for _ in range(max_iters):
        if _remaining() < 5.0:
            break
        # Pick a random interior residue
        i = rng.randint(1, len(chain) - 2)
        old_pos = chain[i]
        # Try moving to a random unoccupied neighbour
        candidates = [q for q in _neighbours(old_pos) if q not in occupied_set(chain, i)]
        if not candidates:
            continue
        new_pos = rng.choice(candidates)
        # Check self-avoidance
        if new_pos in where and where[new_pos] != i:
            continue
        # Tentatively move
        chain[i] = new_pos
        new_energy = _energy(seq, chain)
        if new_energy < current_energy:
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
        else:
            # Revert
            chain[i] = old_pos
    return best_chain, best_energy


def occupied_set(chain, exclude_idx):
    """Return set of occupied positions excluding the one at exclude_idx."""
    return {chain[j] for j in range(len(chain)) if j != exclude_idx}


def fold(sequence, dimension):
    """Best conformation found within time budget."""
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0  # 0 means no valid conformation yet

    # Phase 1: Randomized greedy growths
    while _remaining() > 30.0:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if best_chain is None or value < best_energy:
            best_chain = chain
            best_energy = value

    # Phase 2: Local search on best found
    if best_chain is not None and _remaining() > 10.0:
        best_chain, best_energy = _local_search(sequence, best_chain, rng, max_iters=2000)

    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(len(sequence))]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
