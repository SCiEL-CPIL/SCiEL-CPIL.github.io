"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 185.0  # Safety margin before 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _energy_2d(sequence, chain):
    """Compute energy: -number of non-adjacent H-H contacts."""
    n = len(chain)
    where = {}
    for i, (x, y) in enumerate(chain):
        where[(x, y)] = i
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            for nx, ny in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
                j = where.get((nx, ny))
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _grow_greedy(sequence, rng, bias=1.0):
    """Randomized greedy chain growth with contact-biased move selection."""
    n = len(sequence)
    if n == 0:
        return []
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        x, y = chain[-1]
        options = []
        for nx, ny in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if (nx, ny) in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for mx, my in ((nx+1,ny), (nx-1,ny), (nx,ny+1), (nx,ny-1)):
                    j = occupied.get((mx, my))
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = 0
            for mx, my in ((nx+1,ny), (nx-1,ny), (nx,ny+1), (nx,ny-1)):
                if (mx, my) not in occupied:
                    free += 1
            score = bias * gain + 0.1 * free + rng.random() * 0.5
            options.append((score, (nx, ny)))
        
        if not options:
            return None
        
        best_site = max(options, key=lambda t: t[0])[1]
        occupied[best_site] = i
        chain.append(best_site)
    
    return chain


def _local_search(sequence, chain, max_iters=5000, temperature=1.0, rng=None):
    """Simulated annealing local search."""
    if rng is None:
        rng = random.Random()
    
    n = len(chain)
    if n < 4:
        return chain, _energy_2d(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy_2d(sequence, chain)
    current_chain = list(chain)
    current_energy = best_energy
    
    is_h = [c == 'H' for c in sequence]
    
    start_temp = temperature
    end_temp = 0.01
    cooling = (end_temp / start_temp) ** (1.0 / max(1, max_iters))
    
    for iteration in range(max_iters):
        if _time_left() < 5.0:
            break
        
        temp = start_temp * (cooling ** iteration)
        
        # Pick a random internal residue to move
        i = rng.randint(1, n - 2)
        x, y = current_chain[i]
        
        # Try moving to one of the 4 neighbours
        moves = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
        rng.shuffle(moves)
        
        for nx, ny in moves:
            # Check if new position is already occupied
            occupied = set(current_chain)
            if (nx, ny) in occupied:
                continue
            # Check not same as adjacent residues
            if i > 0 and (nx, ny) == current_chain[i-1]:
                continue
            if i < n-1 and (nx, ny) == current_chain[i+1]:
                continue
            
            new_chain = current_chain[:]
            new_chain[i] = (nx, ny)
            
            new_energy = _energy_2d(sequence, new_chain)
            delta_e = new_energy - current_energy
            if delta_e <= 0 or rng.random() < math.exp(-delta_e / max(temp, 1e-10)):
                current_chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation."""
    if dimension != 2:
        return [[i] + [0] * (dimension - 1) for i in range(len(sequence))]
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    if n == 2:
        return [[0, 0], [1, 0]]
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Multiple greedy growths
    biases = [0.5, 1.0, 1.5, 2.0, 3.0]
    for bias in biases:
        for _ in range(10):
            if _time_left() < 10.0:
                break
            chain = _grow_greedy(sequence, rng, bias=bias)
            if chain is None:
                continue
            energy = _energy_2d(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
    
    if best_chain is None:
        return [[i, 0] for i in range(n)]
    
    # Phase 2: Local search
    if _time_left() > 15.0:
        improved_chain, improved_energy = _local_search(sequence, best_chain, max_iters=3000, temperature=2.0, rng=rng)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    # Phase 3: More local search if time permits
    if _time_left() > 20.0:
        improved_chain, improved_energy = _local_search(sequence, best_chain, max_iters=5000, temperature=1.0, rng=rng)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    # Phase 4: More greedy + local search
    while _time_left() > 30.0:
        chain = _grow_greedy(sequence, rng, bias=rng.uniform(0.5, 3.0))
        if chain is None:
            continue
        energy = _energy_2d(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            if _time_left() > 25.0:
                improved_chain, improved_energy = _local_search(sequence, best_chain, max_iters=2000, temperature=1.5, rng=rng)
                if improved_energy < best_energy:
                    best_chain = improved_chain
                    best_energy = improved_energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
