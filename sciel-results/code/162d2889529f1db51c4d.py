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


def _local_search(sequence, chain, rng, deadline):
    """Hill-climbing with valid moves (preserving connectivity)."""
    n = len(chain)
    best = list(chain)
    best_e = _energy(sequence, best)
    occupied = {p: i for i, p in enumerate(best)}
    
    # Precompute neighbors for all positions to speed up delta calculation? 
    # Or just calculate on the fly.
    
    while time.time() < deadline:
        improved = False
        # Try moving each residue
        for i in range(1, n - 1):
            old = best[i]
            prev = best[i-1]
            next_pos = best[i+1]
            
            # Find valid candidate sites:
            # Must be neighbor of prev AND neighbor of next_pos
            # AND not occupied by any other residue
            candidates = set()
            for site in _neighbours(prev):
                if site in _neighbours(next_pos):
                    if site not in occupied or occupied[site] == i:
                        if site != old:
                            candidates.add(site)
            
            if not candidates:
                continue
                
            for site in candidates:
                # Calculate energy delta
                # Remove old, add new
                # Contacts lost: neighbors of old that are H and occupied by j != i
                delta = 0
                if sequence[i] == 'H':
                    # Lost contacts
                    for q in _neighbours(old):
                        j = occupied.get(q)
                        if j is not None and j != i:
                            if sequence[j] == 'H':
                                delta -= 1
                    # Gained contacts
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j != i:
                            if sequence[j] == 'H':
                                delta += 1
                
                if delta > 0: # Energy decreases (more negative)
                    # Update state
                    occupied[old] = -1 # Mark as free
                    occupied[site] = i
                    best[i] = site
                    best_e += -delta # Wait, energy is -contacts. 
                    # If contacts increase by delta, energy decreases by delta.
                    # So new_energy = old_energy - delta.
                    best_e -= delta
                    improved = True
                    break # Move to next i
            
            if improved:
                continue # Restart loop? Or continue scanning? 
                # Restarting is safer to avoid interference, but slower. 
                # Let's just continue to next i for speed, but this might miss global improvements.
                # Given time constraints, simple hill climbing is fine.
        
        if not improved:
            break
            
    return best, best_e


def fold(sequence, dimension):
    """Best conformation via randomized growth + local search within a time budget."""
    start_time = time.time()
    # The evaluator kills at 330s for ALL sequences.
    # We don't know how many calls we get, but typically 26.
    # 330 / 26 ~ 12s per call.
    # Let's use a strict 10s budget per call to be safe.
    per_seq_deadline = start_time + 10.0

    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1

    # Phase 1: multiple greedy growths
    # Growth is fast O(N^2) roughly. N=100 -> 10000 ops. Very fast.
    # We can do many restarts.
    while time.time() < per_seq_deadline - 2.0:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value

    if best is None:
        # Fallback: straight line (might be invalid if N > dimension range, but self-avoiding)
        # Actually, a straight line of length N fits in 3D easily.
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = _energy(sequence, best)

    # Phase 2: local search polish on the best chain
    if time.time() < per_seq_deadline - 0.5:
        best, best_energy = _local_search(sequence, best, rng, per_seq_deadline - 0.5)

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
