"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = 310.0
_START_TIME = None

def _get_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME

def _time_left():
    return _DEADLINE - (time.time() - _get_start())

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

def _local_search(sequence, chain, dimension, rng, max_iters=2000):
    """Crank-move local search: translate a subchain by one unit in a random direction."""
    n = len(chain)
    if n < 5:
        return chain
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    no_improve = 0
    
    for _ in range(max_iters):
        if _time_left() < 2.0:
            break
        
        # Choose a subchain to crank
        start_idx = rng.randint(1, n - 3)
        end_idx = rng.randint(start_idx + 1, n - 2)
        subchain = chain[start_idx:end_idx + 1]
        
        # Pick a random direction
        ax = rng.randint(0, dimension - 1)
        s = rng.choice((-1, 1))
        
        # Compute new positions
        new_positions = []
        for p in subchain:
            np = list(p)
            np[ax] += s
            new_positions.append(tuple(np))
        
        # Remove old positions
        for i, p in enumerate(subchain):
            del occupied[p]
        
        # Check validity
        valid = True
        seen = set()
        for np in new_positions:
            if np in occupied or np in seen:
                valid = False
                break
            seen.add(np)
        
        if valid:
            # Check bond connectivity at boundaries
            if start_idx > 0:
                prev = chain[start_idx - 1]
                dist = sum(abs(a - b) for a, b in zip(prev, new_positions[0]))
                if dist != 1:
                    valid = False
            
            if valid and end_idx < n - 1:
                next_p = chain[end_idx + 1]
                dist = sum(abs(a - b) for a, b in zip(new_positions[-1], next_p))
                if dist != 1:
                    valid = False
        
        if not valid:
            # Restore
            for i, p in enumerate(subchain):
                occupied[p] = start_idx + i
            no_improve += 1
            if no_improve > 500:
                break
            continue
        
        # Apply move
        for i, np in enumerate(new_positions):
            occupied[np] = start_idx + i
        
        new_chain = chain[:start_idx] + new_positions + chain[end_idx + 1:]
        new_energy = _energy(sequence, new_chain)
        
        if new_energy < current_energy:
            chain = new_chain
            current_energy = new_energy
            no_improve = 0
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        else:
            # Revert
            for i, np in enumerate(new_positions):
                del occupied[np]
            for i, p in enumerate(subchain):
                occupied[p] = start_idx + i
            no_improve += 1
            if no_improve > 500:
                break
    
    return best_chain

def fold(sequence, dimension):
    """Best conformation using greedy growth with local search refinement."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    
    # Phase 1: Greedy growth with multiple restarts
    n_restarts = 100
    for _ in range(n_restarts):
        if _time_left() < 5.0:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if best is None or value < best_energy:
            best, best_energy = chain, value
    
    # Phase 2: Local search on the best greedy result
    if best is not None and _time_left() > 5.0:
        refined = _local_search(sequence, best, dimension, rng, max_iters=3000)
        if refined is not None:
            refined_energy = _energy(sequence, refined)
            if refined_energy < best_energy:
                best, best_energy = refined, refined_energy
    
    # Fallback: straight line is always self-avoiding
    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
