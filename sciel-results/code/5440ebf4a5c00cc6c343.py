"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


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


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
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
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_search(sequence, chain, rng, deadline):
    """Local search: try moving individual residues to neighboring free sites."""
    n = len(chain)
    occupied = {p: i for i, p in enumerate(chain)}
    best_energy = _energy(sequence, chain)
    best_chain = list(chain)
    
    while time.time() < deadline:
        improved = False
        # Try multiple random moves per iteration
        for _ in range(n):
            if time.time() >= deadline:
                break
            i = rng.randint(1, n - 2)  # not endpoints
            current = chain[i]
            for new_site in _neighbours(current):
                # Check if new_site is free
                if new_site in occupied:
                    continue
                # Check unit step with neighbors
                prev = chain[i - 1]
                nxt = chain[i + 1]
                # Must be unit distance from prev and nxt
                if any(abs(new_site[a] - prev[a]) > 1 or abs(new_site[a] - prev[a]) == 0 for a in range(dimension)) and \
                   any(abs(new_site[a] - prev[a]) > 1 for a in range(dimension)):
                    continue
                # Proper check: Manhattan distance must be 1
                if sum(abs(new_site[a] - prev[a]) for a in range(dimension)) != 1:
                    continue
                if sum(abs(new_site[a] - nxt[a]) for a in range(dimension)) != 1:
                    continue
                # Also new_site should not equal prev or nxt (already covered by distance check mostly)
                
                # Try the move
                del occupied[current]
                occupied[new_site] = i
                chain[i] = new_site
                
                new_energy = _energy(sequence, chain)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                    improved = True
                else:
                    # Revert
                    occupied[current] = i
                    del occupied[new_site]
                    chain[i] = current
        
        if not improved:
            break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation with local search and dynamic time allocation."""
    start_time = time.time()
    deadline = start_time + 300  # safety margin before 330s kill
    
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1
    
    # First pass: greedy growths
    while time.time() < deadline:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
        
        # Run local search on this chain
        chain, value = _local_search(sequence, chain, rng, deadline)
        if value < best_energy:
            best, best_energy = chain, value
        
        # If we've found a good solution, use remaining time for more search
        # Check if we should continue
    
    if best is None:  # a straight line is always self-avoiding
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
