"""HP lattice protein folding on the cubic lattice.
Uses a hybrid approach: fast greedy growth with multiple restarts,
then aggressive local search via tail re-growth with simulated annealing,
plus a final polish phase with short-tail refinement.
"""
import random
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 318.0  # Safety margin before 330s kill


def _neighbours(point):
    """Yield the 6 face-neighbours of point in 3D (fast path)."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def _energy(sequence, chain):
    """Compute total energy: negative count of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y, z = p
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng, temperature=1.0):
    """One randomized greedy chain growth with temperature control."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        x, y, z = chain[-1]
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                sx, sy, sz = site
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = 0
            sx, sy, sz = site
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            score = gain + 0.1 * free + rng.random() * temperature
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _regrow_tail(sequence, chain, start_idx, rng, max_iters=100, temperature=None):
    """Regrow the tail of the chain from start_idx to improve energy.
    Uses Simulated Annealing with adaptive initial temperature based on energy magnitude.
    Returns (best_chain, best_energy)."""
    if start_idx >= len(chain) - 1:
        return list(chain), _energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    n = len(sequence)
    
    # Adaptive initial temperature: scale with energy magnitude for larger exploration
    if temperature is None:
        temperature = max(1.0, min(abs(best_energy) * 0.3, 5.0))
    else:
        temperature = float(temperature)
    
    cooling_rate = 0.985
    min_temp = 0.03
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 3:
            break
        
        prefix = chain[:start_idx + 1]
        occupied = {p: i for i, p in enumerate(prefix)}
        
        current_tail = [tuple(p) for p in prefix]
        success = True
        
        for i in range(start_idx + 1, n):
            x, y, z = current_tail[-1]
            options = []
            for site in _neighbours(current_tail[-1]):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == "H":
                    sx, sy, sz = site
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == "H":
                            gain += 1
                free = 0
                sx, sy, sz = site
                for q in _neighbours(site):
                    if q not in occupied:
                        free += 1
                score = gain + 0.05 * free + rng.random() * temperature
                options.append((score, site))
            if not options:
                success = False
                break
            
            site = max(options)[1]
            occupied[site] = i
            current_tail.append(site)
        
        if not success:
            temperature = max(min_temp, temperature * 0.95)
            continue
        
        new_energy = _energy(sequence, current_tail)
        
        # Metropolis acceptance criterion
        if new_energy < best_energy:
            best_chain = current_tail
            best_energy = new_energy
            # Cool faster on improvement to lock in gains
            temperature = max(min_temp, temperature * 0.97)
        else:
            delta_e = new_energy - best_energy
            if delta_e > 0:
                prob = math.exp(-delta_e / temperature)
                if rng.random() < prob:
                    best_chain = current_tail
                    best_energy = new_energy
            temperature = max(min_temp, temperature * cooling_rate)
    
    return best_chain, best_energy


def _local_search(sequence, chain, rng, time_budget=30.0):
    """Iterative local search: repeatedly try regrowing from various positions
    with adaptive temperature scheduling."""
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    n = len(sequence)
    start_time = time.time()
    
    # Generate candidate split points: mix of short tails and strategic positions
    split_points = set()
    # Short tails (fast to evaluate, good for final polish)
    split_points.add(max(0, n - 8))
    split_points.add(max(0, n - 12))
    split_points.add(max(0, n - 18))
    split_points.add(max(0, n - 25))
    split_points.add(max(0, n - 35))
    # Strategic positions
    split_points.add(n // 4)
    split_points.add(n // 2)
    split_points.add(3 * n // 4)
    # Additional for longer chains
    if n > 60:
        split_points.add(max(0, n - 50))
        split_points.add(max(0, n - 65))
    
    # Sort by tail length (shorter first for quick wins)
    split_points = sorted(split_points, key=lambda sp: n - sp)
    
    # First pass: quick sweeps with moderate temperature
    for sp in split_points:
        if time.time() - start_time > time_budget * 0.6:
            break
        if sp >= n - 1:
            continue
        
        # Use adaptive temperature (None lets _regrow_tail decide)
        improved_chain, improved_energy = _regrow_tail(
            sequence, best_chain, sp, rng, max_iters=60, temperature=None
        )
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    # Second pass: deeper exploration on promising split points
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 5:
        # Focus on the 4 shortest-tail splits with more iterations
        for sp in split_points[:4]:
            if time.time() - start_time > time_budget:
                break
            if sp >= n - 1:
                continue
            improved_chain, improved_energy = _regrow_tail(
                sequence, best_chain, sp, rng, max_iters=120, temperature=None
            )
            if improved_energy < best_energy:
                best_chain = improved_chain
                best_energy = improved_energy
    
    # Third pass: if time remains, try longer tails with aggressive exploration
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 10:
        for sp in split_points[4:]:
            if time.time() - start_time > time_budget:
                break
            if sp >= n - 1:
                continue
            improved_chain, improved_energy = _regrow_tail(
                sequence, best_chain, sp, rng, max_iters=80, temperature=None
            )
            if improved_energy < best_energy:
                best_chain = improved_chain
                best_energy = improved_energy
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using fast growth + aggressive local search with deadline guard."""
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    n = len(sequence)
    remaining = _DEADLINE - time.time()
    
    # Phase 1: Multiple randomized greedy growths
    # Scale number of initial tries based on chain length and time budget
    if n <= 50:
        num_initial = min(70, int(remaining * 2))
    elif n <= 100:
        num_initial = min(35, int(remaining))
    else:
        num_initial = min(18, int(remaining * 0.5))
    
    num_initial = max(num_initial, 10)
    
    for trial in range(num_initial):
        if time.time() > _DEADLINE - 60:
            break
        
        # Vary temperature: early trials more exploratory, later more greedy
        temp = 1.5 if trial < num_initial // 3 else (0.8 if trial < 2 * num_initial // 3 else 0.4)
        chain = _grow(sequence, dimension, rng, temperature=temp)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best_chain = chain
            best_energy = value
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
        return [list(p) for p in best_chain]
    
    # Phase 2: Local search refinement
    remaining = _DEADLINE - time.time() - 5  # Leave 5s safety margin
    if remaining > 10:
        ls_budget = min(remaining, 200.0)
        best_chain, best_energy = _local_search(sequence, best_chain, rng, time_budget=ls_budget)
    
    # Phase 3: Final polish if time remains
    remaining = _DEADLINE - time.time() - 3
    if remaining > 10:
        n = len(sequence)
        for sp in [max(0, n - 8), max(0, n - 20), n // 2]:
            if time.time() > _DEADLINE - 3:
                break
            if sp >= n - 1:
                continue
            improved_chain, improved_energy = _regrow_tail(
                sequence, best_chain, sp, rng, max_iters=80, temperature=None
            )
            if improved_energy < best_energy:
                best_chain = improved_chain
                best_energy = improved_energy
    
    return [list(p) for p in best_chain]
