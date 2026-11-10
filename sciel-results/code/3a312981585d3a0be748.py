"""HP lattice protein folding using E2T Monte Carlo moves with simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = time.time() + 175.0


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    n = len(chain)
    if n < 2:
        return 0
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _is_valid_chain(chain, dimension):
    """Check self-avoidance and unit steps."""
    n = len(chain)
    if n < 2:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            prev = chain[i - 1]
            dist = sum(abs(a - b) for a, b in zip(p, prev))
            if dist != 1:
                return False
    return True


def _greedy_grow(sequence, dimension, rng):
    """Randomized greedy chain growth."""
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
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _e2t_move(chain, i, j, new_pos_i, new_pos_j):
    """Apply E2T move: move residues i and j to new positions."""
    chain[i] = new_pos_i
    chain[j] = new_pos_j
    return True


def _try_e2t_move(chain, i, j, occupied, rng):
    """Try an E2T move on residues i and j. Returns (success, new_i, new_j)."""
    n = len(chain)
    if i >= n or j >= n or i < 0 or j < 0:
        return False, None, None
    
    pi = chain[i]
    pj = chain[j]
    
    # Generate candidate positions for both residues
    candidates_i = []
    candidates_j = []
    
    for site in _neighbours(pi):
        if site in occupied and site != pi and site != pj:
            continue
        # Check if moving i to site keeps connectivity
        if i > 0 and site != chain[i-1] and abs(site[0]-chain[i-1][0]) + abs(site[1]-chain[i-1][1]) != 1:
            continue
        if i < n-1 and site != chain[i+1] and abs(site[0]-chain[i+1][0]) + abs(site[1]-chain[i+1][1]) != 1:
            continue
        candidates_i.append(site)
    
    for site in _neighbours(pj):
        if site in occupied and site != pi and site != pj:
            continue
        if j > 0 and site != chain[j-1] and abs(site[0]-chain[j-1][0]) + abs(site[1]-chain[j-1][1]) != 1:
            continue
        if j < n-1 and site != chain[j+1] and abs(site[0]-chain[j+1][0]) + abs(site[1]-chain[j+1][1]) != 1:
            continue
        candidates_j.append(site)
    
    if not candidates_i or not candidates_j:
        return False, None, None
    
    new_i = rng.choice(candidates_i)
    new_j = rng.choice(candidates_j)
    
    # Check for self-avoidance after move
    # Remove old positions temporarily
    occupied.pop(pi, None)
    occupied.pop(pj, None)
    
    if new_i in occupied or new_j in occupied:
        occupied[pi] = i
        occupied[pj] = j
        return False, None, None
    
    # Check all other positions for conflicts
    conflict = False
    for k in range(n):
        if k == i or k == j:
            continue
        pk = chain[k]
        if pk == new_i or pk == new_j:
            conflict = True
            break
    
    if conflict:
        occupied[pi] = i
        occupied[pj] = j
        return False, None, None
    
    # Update chain and occupied
    chain[i] = new_i
    chain[j] = new_j
    occupied[new_i] = i
    occupied[new_j] = j
    
    return True, new_i, new_j


def _local_improve_e2t(sequence, chain, temperature, max_steps=500, rng=None):
    """Local improvement using E2T moves with simulated annealing."""
    if rng is None:
        rng = random.Random()
    
    n = len(chain)
    if n < 3:
        return chain
    
    occupied = {chain[i]: i for i in range(n)}
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for step in range(max_steps):
        if time.time() > _DEADLINE - 5:
            break
        
        # Pick two adjacent residues to move
        i = rng.randint(0, n - 2)
        j = i + 1
        
        # Try E2T move
        success, new_i, new_j = _try_e2t_move(chain, i, j, occupied, rng)
        
        if not success:
            continue
        
        new_energy = _compute_energy(sequence, chain)
        delta = new_energy - current_energy
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or rng.random() < (2.718281828 ** (-delta / max(temperature, 0.01))):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
        else:
            # Revert
            chain[i], chain[j] = new_i, new_j
            occupied[new_i] = i
            occupied[new_j] = j
    
    return best_chain


def _exhaustive_local_search(sequence, chain, max_rounds=10):
    """Exhaustive local search: try all single-residue moves until no improvement."""
    n = len(chain)
    if n < 2:
        return chain
    
    occupied = {chain[i]: i for i in range(n)}
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    improved = True
    
    for _ in range(max_rounds):
        if not improved:
            break
        improved = False
        if time.time() > _DEADLINE - 5:
            break
        
        for i in range(1, n - 1):  # Don't move endpoints
            pi = chain[i]
            for site in _neighbours(pi):
                if site in occupied and site != pi:
                    continue
                
                # Check connectivity
                if abs(site[0] - chain[i-1][0]) + abs(site[1] - chain[i-1][1]) != 1:
                    continue
                if abs(site[0] - chain[i+1][0]) + abs(site[1] - chain[i+1][1]) != 1:
                    continue
                
                # Try the move
                old_pos = chain[i]
                chain[i] = site
                occupied.pop(old_pos, None)
                occupied[site] = i
                
                new_energy = _compute_energy(sequence, chain)
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                    improved = True
                    current_energy = new_energy
                else:
                    # Revert
                    chain[i] = old_pos
                    occupied.pop(site, None)
                    occupied[old_pos] = i
    
    return best_chain


def fold(sequence, dimension):
    """Fold using greedy growth followed by E2T Monte Carlo optimization."""
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    n = len(sequence)
    
    # Generate multiple initial conformations via greedy growth
    for attempt in range(50):
        if time.time() > _DEADLINE - 30:
            break
        
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Validate
        if not _is_valid_chain(chain, dimension):
            continue
        
        energy = _compute_energy(sequence, chain)
        
        if best_chain is None or energy < best_energy:
            best_chain = list(chain)
            best_energy = energy
        
        # If we have a good result, stop generating more
        if energy <= -n // 3:
            break
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    
    # Phase 1: E2T local improvement with simulated annealing
    if time.time() < _DEADLINE - 20:
        # Start with higher temperature, cool down
        for temp in [2.0, 1.0, 0.5, 0.2, 0.1]:
            if time.time() > _DEADLINE - 10:
                break
            improved_chain = _local_improve_e2t(sequence, best_chain, temp, max_steps=200, rng=rng)
            improved_energy = _compute_energy(sequence, improved_chain)
            if improved_energy < best_energy:
                best_chain = improved_chain
                best_energy = improved_energy
    
    # Phase 2: Exhaustive local search polish
    if time.time() < _DEADLINE - 5:
        polished = _exhaustive_local_search(sequence, best_chain, max_rounds=5)
        polished_energy = _compute_energy(sequence, polished)
        if polished_energy < best_energy:
            best_chain = polished
            best_energy = polished_energy
    
    # Final check
    if not _is_valid_chain(best_chain, dimension):
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
