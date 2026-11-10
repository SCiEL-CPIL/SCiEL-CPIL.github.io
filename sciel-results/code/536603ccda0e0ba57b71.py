"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 180.0


def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


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


def _local_improve(sequence, chain, rng, max_moves=5000):
    """
    Improve conformation via single-residue moves that preserve connectivity.
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
        if time.time() > _DEADLINE - 10:
            break
        
        i = rng.randint(1, n - 2)
        p = chain[i]
        tried = False

        for np in _neighbours(p):
            if np == p:
                continue
            if np in occupied and occupied[np] != i:
                continue
            
            # Check if moving to np preserves connectivity
            prev, nxt = chain[i-1], chain[i+1]
            dist_prev = abs(np[0]-prev[0]) + abs(np[1]-prev[1])
            dist_next = abs(np[0]-nxt[0]) + abs(np[1]-nxt[1])
            
            if dist_prev != 1 or dist_next != 1:
                continue
            
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
            else:
                # Revert
                del occupied[np]
                chain[i] = p
                occupied[p] = i
                no_improve_count += 1
            
            tried = True
            break
        
        if not tried:
            no_improve_count += 1
        
        if no_improve_count > 500:
            break

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold using greedy restarts + local improvement.
    """
    rng = random.Random(42)
    best_chain = None
    best_energy = 0
    n = len(sequence)

    # Phase 1: Greedy restarts
    num_restarts = 3000
    for attempt in range(num_restarts):
        if time.time() > _DEADLINE - 20:
            break

        temp = rng.uniform(0.0, 3.0)
        chain = _greedy_grow(sequence, dimension, rng, temperature=temp)
        if chain is None:
            continue

        energy = _compute_energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

    # Phase 2: Local improvement on best
    if best_chain is not None:
        improved_chain, improved_energy = _local_improve(
            sequence, best_chain, rng, max_moves=10000)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy

    # Fallback
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
