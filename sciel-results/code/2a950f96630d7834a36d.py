"""HP lattice protein folding using multi-strategy search with simulated annealing."""
# EVOLVE-BLOCK-START
import time
import random

def _neighbours(point):
    """Yield the 4 neighbours of a point on the square lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    """Compute the HP contact energy of a chain conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _random_walk(n, rng, max_steps=10000):
    """Generate a random self-avoiding walk of length n."""
    if n <= 1:
        return [(0, 0)] if n == 1 else []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    direction = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    for i in range(1, n):
        x, y = chain[-1]
        shuffled_dirs = direction[:]
        rng.shuffle(shuffled_dirs)
        
        found = False
        for dx, dy in shuffled_dirs:
            new_pos = (x + dx, y + dy)
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied[new_pos] = i
                found = True
                break
        
        if not found:
            return None
    
    return chain

def _greedy_grow(sequence, rng):
    """Grow chain greedily, prioritizing H-H contact opportunities."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)] if n == 1 else []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            score = 0.0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        score += 10.0
            
            free_count = sum(1 for q in _neighbours(site) if q not in occupied)
            score += free_count * 0.1
            score += rng.random() * 0.5
            
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: -x[0])
        if rng.random() < 0.15 and len(candidates) > 1:
            idx = rng.randint(1, min(2, len(candidates) - 1))
            best_site = candidates[idx][1]
        else:
            best_site = candidates[0][1]
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain

def _spiral_template(n, rng):
    """Generate a compact spiral-shaped self-avoiding walk of length n."""
    if n <= 1:
        return [(0, 0)] if n == 1 else []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    direction = (1, 0)
    
    for i in range(1, n):
        found = False
        for step in range(4):
            x, y = chain[-1]
            dx, dy = direction
            new_pos = (x + dx, y + dy)
            
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied[new_pos] = i
                found = True
                break
            
            direction = (dy, -dx)
        
        if not found:
            return None
    
    return chain

def _local_search(sequence, chain, rng, max_iter=200):
    """Local search with single-residue moves."""
    n = len(sequence)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    for _ in range(max_iter):
        improved = False
        
        for i in range(1, n - 1):
            old_pos = best_chain[i]
            
            for new_pos in _neighbours(old_pos):
                if new_pos in occupied:
                    continue
                
                prev_pos = best_chain[i - 1]
                next_pos = best_chain[i + 1]
                
                dist_prev = abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1])
                dist_next = abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1])
                
                if dist_prev != 1 or dist_next != 1:
                    continue
                
                del occupied[old_pos]
                occupied[new_pos] = i
                best_chain[i] = new_pos
                
                new_energy = _energy(sequence, best_chain)
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    improved = True
                else:
                    del occupied[new_pos]
                    occupied[old_pos] = i
                    best_chain[i] = old_pos
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best_chain, best_energy

def _simulated_annealing(sequence, chain, rng, start_time, deadline):
    """Simulated annealing to escape local minima."""
    n = len(sequence)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    T = 5.0
    cooling = 0.995
    min_T = 0.1
    
    while T > min_T and time.time() < deadline - 5.0:
        improved_this_cycle = False
        
        for i in range(1, n - 1):
            if time.time() > deadline - 5.0:
                break
            
            old_pos = current_chain[i]
            valid_moves = []
            
            for new_pos in _neighbours(old_pos):
                if new_pos in occupied:
                    continue
                
                prev_pos = current_chain[i - 1]
                next_pos = current_chain[i + 1]
                
                dist_prev = abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1])
                dist_next = abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1])
                
                if dist_prev == 1 and dist_next == 1:
                    valid_moves.append(new_pos)
            
            if not valid_moves:
                continue
            
            new_pos = rng.choice(valid_moves)
            
            del occupied[old_pos]
            occupied[new_pos] = i
            current_chain[i] = new_pos
            
            new_energy = _energy(sequence, current_chain)
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(current_chain)
                    improved_this_cycle = True
            else:
                del occupied[new_pos]
                occupied[old_pos] = i
                current_chain[i] = old_pos
        
        T *= cooling
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation."""
    start_time = time.time()
    deadline = start_time + 195.0
    n = len(sequence)
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0
    
    strategies = [
        lambda: _random_walk(n, rng),
        lambda: _greedy_grow(sequence, rng),
        lambda: _spiral_template(n, rng),
    ]
    
    num_attempts = 0
    max_attempts = 3000
    
    while num_attempts < max_attempts and time.time() < deadline - 60.0:
        strategy_idx = rng.randint(0, len(strategies) - 1)
        chain = strategies[strategy_idx]()
        
        if chain is not None and len(chain) == n:
            num_attempts += 1
            energy = _energy(sequence, chain)
            
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    if time.time() < deadline - 30.0:
        improved_chain, improved_energy = _local_search(sequence, best_chain, rng, max_iter=300)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    if time.time() < deadline - 10.0:
        improved_chain, improved_energy = _simulated_annealing(sequence, best_chain, rng, start_time, deadline)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
