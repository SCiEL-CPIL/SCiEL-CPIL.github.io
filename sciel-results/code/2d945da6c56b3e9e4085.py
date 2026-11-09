"""HP lattice protein folding on the cubic lattice.
Uses simulated annealing with multiple initializations and local moves.
"""
import random
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 318.0  # Safety margin before 330s kill


def _neighbours(point):
    """Yield the 6 face-neighbours of point in 3D."""
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
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0, 0)]
    
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                options.append(site)
        if not options:
            return None
        site = rng.choice(options)
        occupied[site] = i
        chain.append(site)
    
    return chain


def _greedy_growth(sequence, dimension, rng):
    """One randomized greedy chain growth with contact-aware scoring."""
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 2.0 + free * 0.1 + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def _simulated_annealing(sequence, chain, rng, time_budget=10.0):
    """Improve chain using simulated annealing with pivot moves."""
    n = len(sequence)
    if n < 3:
        return chain
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    current_chain = list(chain)
    current_energy = best_energy
    
    start_time = time.time()
    t0 = 1.0
    t_final = 0.01
    cooling_rate = math.log(t_final / t0) / max(1, int(time_budget * 100))
    
    iteration = 0
    while time.time() - start_time < time_budget:
        if time.time() > _DEADLINE - 5:
            break
        
        # Choose a random pivot point (not endpoints)
        pivot = rng.randint(1, n - 2)
        
        # Try to move the segment from pivot to end
        # Extract the tail
        tail = current_chain[pivot:]
        prefix = current_chain[:pivot]
        
        # Remove tail positions from occupied set
        occupied = {p: i for i, p in enumerate(prefix)}
        
        # Try to regrow the tail from the pivot
        success = True
        new_tail = [current_chain[pivot]]
        occupied[current_chain[pivot]] = pivot
        
        for i in range(pivot + 1, n):
            options = []
            for site in _neighbours(new_tail[-1]):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == "H":
                    gain = sum(1 for q in _neighbours(site)
                               if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                score = gain * 2.0 + free * 0.1 + rng.random() * 0.5
                options.append((score, site))
            if not options:
                success = False
                break
            site = max(options)[1]
            occupied[site] = i
            new_tail.append(site)
        
        if not success:
            # Try a simpler move: rotate or flip a small segment
            if n >= 4:
                seg_len = rng.randint(2, min(5, n - 2))
                seg_start = rng.randint(1, n - 1 - seg_len)
                segment = current_chain[seg_start:seg_start + seg_len + 1]
                
                # Try reversing the segment
                reversed_segment = list(reversed(segment))
                new_chain = current_chain[:seg_start] + reversed_segment + current_chain[seg_start + seg_len + 1:]
                
                # Check self-avoidance
                valid = True
                pos_set = set(new_chain)
                if len(pos_set) != len(new_chain):
                    valid = False
                
                if valid:
                    new_energy = _energy(sequence, new_chain)
                    delta_e = new_energy - current_energy
                    if delta_e < 0 or rng.random() < math.exp(-delta_e / max(0.01, t0 * math.exp(cooling_rate * iteration))):
                        current_chain = new_chain
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(current_chain)
            iteration += 1
            continue
        
        new_chain = prefix + new_tail
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        temperature = t0 * math.exp(cooling_rate * iteration)
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(0.01, temperature)):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        iteration += 1
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using multiple initializations + simulated annealing."""
    rng = random.Random(54321 + len(sequence))
    n = len(sequence)
    
    if n <= 1:
        return [[0] * dimension]
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Generate initial candidates
    candidates = []
    
    # Try greedy growth multiple times
    num_greedy = min(50, max(10, n // 2))
    for _ in range(num_greedy):
        if time.time() > _DEADLINE - 30:
            break
        chain = _greedy_growth(sequence, dimension, rng)
        if chain is not None:
            candidates.append(chain)
    
    # Try random walks
    num_random = min(20, max(5, n // 4))
    for _ in range(num_random):
        if time.time() > _DEADLINE - 30:
            break
        chain = _random_walk(sequence, dimension, rng)
        if chain is not None:
            candidates.append(chain)
    
    if not candidates:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
        return [list(p) for p in best_chain]
    
    # Evaluate all candidates
    for chain in candidates:
        e = _energy(sequence, chain)
        if e < best_energy:
            best_energy = e
            best_chain = chain
    
    # Phase 2: Simulated annealing on top candidates
    if time.time() < _DEADLINE - 15:
        # Sort candidates by energy and take top few
        scored = [(_energy(sequence, c), c) for c in candidates]
        scored.sort()
        
        # Anneal the top candidates
        num_anneal = min(5, len(scored))
        for i in range(num_anneal):
            if time.time() > _DEADLINE - 5:
                break
            _, cand_chain = scored[i]
            remaining_time = _DEADLINE - 5 - time.time()
            if remaining_time < 1.0:
                break
            # Allocate time proportional to remaining
            time_per_anneal = min(remaining_time / (num_anneal - i), 15.0)
            improved = _simulated_annealing(sequence, cand_chain, rng, time_budget=time_per_anneal)
            e = _energy(sequence, improved)
            if e < best_energy:
                best_energy = e
                best_chain = improved
    
    # Phase 3: Additional refinement if time permits
    if time.time() < _DEADLINE - 3:
        remaining = _DEADLINE - 3 - time.time()
        if remaining > 2.0:
            improved = _simulated_annealing(sequence, best_chain, rng, time_budget=min(remaining, 20.0))
            e = _energy(sequence, improved)
            if e < best_energy:
                best_energy = e
                best_chain = improved
    
    return [list(p) for p in best_chain]
