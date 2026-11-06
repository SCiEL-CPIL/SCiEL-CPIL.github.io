"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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


def _local_refine(sequence, chain, rng, max_steps=5000):
    """Local refinement: try random single-residue moves that improve energy."""
    n = len(chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    occupied = {tuple(p): i for i, p in enumerate(chain)}
    
    step = 0
    while step < max_steps and time.time() < _DEADLINE:
        step += 1
        # Pick a random residue to move (not endpoints)
        i = rng.randint(1, n - 2)
        orig_pos = tuple(chain[i])
        
        # Try all 6 neighbor positions
        candidates = list(_neighbours(orig_pos))
        rng.shuffle(candidates)
        
        moved = False
        for new_pos in candidates:
            if new_pos == orig_pos:
                continue
            if new_pos in occupied:
                continue
            # Check self-avoidance: new_pos must not collide with any other residue
            # (already checked via occupied, but need to make sure it's not adjacent constraint issue)
            # Actually, just need new_pos not in occupied (which includes all other residues)
            
            # Compute energy delta
            # Remove residue i from consideration, place at new_pos
            # Energy change: contacts involving residue i change
            # Let's just recompute locally
            
            # Temporarily move
            chain[i] = new_pos
            occupied[new_pos] = i
            del occupied[orig_pos]
            
            new_energy = _energy(sequence, chain)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
            else:
                # Revert
                chain[i] = orig_pos
                occupied[orig_pos] = i
                del occupied[new_pos]
                continue
            
            moved = True
            break
        
        if not moved:
            # No improvement found in this step; continue searching
            pass
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using multiple restarts with local refinement."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0  # 0 means no H-H contacts (unfolded)
    
    # Phase 1: Greedy growth with multiple restarts
    num_restarts = 100
    if n > 100:
        num_restarts = 30
    elif n > 60:
        num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > _DEADLINE - 30:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if best_chain is None or value < best_energy:
            best_chain = chain
            best_energy = value
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = 0
    
    # Phase 2: Local refinement on the best chain
    if time.time() < _DEADLINE - 10:
        refined_chain, refined_energy = _local_refine(sequence, best_chain, rng, max_steps=10000)
        if refined_energy < best_energy:
            best_chain = refined_chain
            best_energy = refined_energy
    
    # Phase 3: Try a few more refined restarts if time permits
    if time.time() < _DEADLINE - 20:
        for r in range(10):
            if time.time() > _DEADLINE - 15:
                break
            chain = _grow(sequence, dimension, rng)
            if chain is None:
                continue
            refined_chain, refined_energy = _local_refine(sequence, chain, rng, max_steps=5000)
            if refined_energy < best_energy:
                best_chain = refined_chain
                best_energy = refined_energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
