"""HP lattice protein folding on the square lattice using contact-driven greedy + fast SA."""
import random
import time
import math

_GLOBAL_START = time.time()
_TOTAL_BUDGET = 195.0

def _neighbours(point):
    x, y = point
    return ((x+1, y), (x-1, y), (x, y+1), (x, y-1))

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    h_indices = set(i for i, s in enumerate(sequence) if s == "H")
    contacts = 0
    for i in h_indices:
        x, y = chain[i]
        for q in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
            j = where.get(q)
            if j is not None and j > i + 1 and j in h_indices:
                contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    
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
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _try_pivot_move(sequence, chain, k, rot_choice):
    n = len(chain)
    if k >= n - 1:
        return None, None
    
    px, py = chain[k]
    
    if rot_choice == 0:
        def rot(x, y): return (-y, x)
    elif rot_choice == 1:
        def rot(x, y): return (-x, -y)
    else:
        def rot(x, y): return (y, -x)
        
    prefix_occupied = set(chain[:k+1])
    
    new_chain = list(chain[:k+1])
    
    for j in range(k + 1, n):
        ox = chain[j][0] - px
        oy = chain[j][1] - py
        nx, ny = rot(ox, oy)
        new_pos = (px + nx, py + ny)
        
        if new_pos in prefix_occupied:
            return None, None
            
        new_chain.append(new_pos)
        
    seen = set()
    for i, p in enumerate(new_chain):
        if p in seen:
            return None, None
        seen.add(p)
        if i > 0:
            dx = abs(new_chain[i][0] - new_chain[i-1][0])
            dy = abs(new_chain[i][1] - new_chain[i-1][1])
            if dx + dy != 1:
                return None, None
    
    new_energy = _energy(sequence, new_chain)
    return new_chain, new_energy

def _try_crankshaft(sequence, chain, k1, k2, rot_choice):
    n = len(chain)
    if k2 - k1 < 2:
        return None, None
    
    px, py = chain[k1]
    
    if rot_choice == 0:
        def rot(x, y): return (-y, x)
    elif rot_choice == 1:
        def rot(x, y): return (-x, -y)
    else:
        def rot(x, y): return (y, -x)
        
    prefix = chain[:k1+1]
    suffix = chain[k2+1:]
    seg = chain[k1+1:k2+1]
    
    fixed_occupied = set(prefix) | set(suffix)
    
    new_seg = []
    for j in range(len(seg)):
        ox = seg[j][0] - px
        oy = seg[j][1] - py
        nx, ny = rot(ox, oy)
        new_pos = (px + nx, py + ny)
        new_seg.append(new_pos)
    
    for pos in new_seg:
        if pos in fixed_occupied:
            return None, None
    if len(set(new_seg)) != len(new_seg):
        return None, None
    
    new_chain = prefix + new_seg + suffix
    
    seen = set()
    for i, p in enumerate(new_chain):
        if p in seen:
            return None, None
        seen.add(p)
        if i > 0:
            dx = abs(new_chain[i][0] - new_chain[i-1][0])
            dy = abs(new_chain[i][1] - new_chain[i-1][1])
            if dx + dy != 1:
                return None, None
    
    new_energy = _energy(sequence, new_chain)
    return new_chain, new_energy

def _pivot_search(sequence, chain, time_limit, rng):
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    T_start = 2.0
    T_end = 0.01
    start_time = time.time()
    
    max_iters = 300000
    cooling_factor = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        use_crankshaft = rng.random() < 0.3
        
        if use_crankshaft and n > 6:
            k1 = rng.randint(0, n - 3)
            k2 = rng.randint(k1 + 2, n - 1)
            rot_choice = rng.randint(0, 2)
            new_chain, new_energy = _try_crankshaft(sequence, current_chain, k1, k2, rot_choice)
        else:
            k = rng.randint(0, n - 2)
            rot_choice = rng.randint(0, 2)
            new_chain, new_energy = _try_pivot_move(sequence, current_chain, k, rot_choice)
        
        if new_chain is None:
            continue
            
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        T *= cooling_factor
        
    return best_chain, best_energy

def _greedy_polish(sequence, chain, max_rounds=100, deadline=None):
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    for _ in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        for k in range(n - 1):
            for rot_choice in range(3):
                new_chain, new_energy = _try_pivot_move(sequence, current_chain, k, rot_choice)
                if new_chain is not None and new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
                    break
            if improved:
                break
        
        if n > 6:
            for k1 in range(n - 2):
                for k2 in range(k1 + 2, n):
                    for rot_choice in range(3):
                        new_chain, new_energy = _try_crankshaft(sequence, current_chain, k1, k2, rot_choice)
                        if new_chain is not None and new_energy < current_energy:
                            current_chain = new_chain
                            current_energy = new_energy
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
            
    return current_chain, current_energy

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    
    elapsed = time.time() - _GLOBAL_START
    if elapsed > _TOTAL_BUDGET:
        return [[i, 0] for i in range(n)]
    
    remaining = _TOTAL_BUDGET - elapsed
    sa_time = remaining * 0.85
    polish_time = remaining * 0.05
    
    if sa_time < 1.0:
        rng = random.Random(12345 + n)
        initial = _greedy_grow(sequence, dimension, rng)
        if initial:
            return [list(p) for p in initial]
        return [[i, 0] for i in range(n)]
        
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0
    
    if n <= 30:
        num_restarts = 4
    elif n <= 50:
        num_restarts = 3
    else:
        num_restarts = 2
    time_per_restart = sa_time / num_restarts
    
    for k in range(num_restarts):
        if time.time() - _GLOBAL_START > _TOTAL_BUDGET - 10.0:
            break
            
        init_rng = random.Random(12345 + n + k * 100)
        initial = _greedy_grow(sequence, dimension, init_rng)
        
        if initial is None:
            initial = [(i, 0) for i in range(n)]
            
        refined_chain, refined_energy = _pivot_search(sequence, initial, time_per_restart, rng)
        
        if refined_chain is not None:
            if best_chain is None or refined_energy < best_energy:
                best_chain = refined_chain
                best_energy = refined_energy
                
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    remaining_for_polish = _TOTAL_BUDGET - 3.0 - (time.time() - _GLOBAL_START)
    if remaining_for_polish > 1.0:
        polish_rounds = 300 if n <= 40 else 200
        polished_chain, polished_energy = _greedy_polish(sequence, best_chain, max_rounds=polish_rounds, deadline=_TOTAL_BUDGET - 3.0)
        if polished_energy < best_energy:
            best_chain = polished_chain
            best_energy = polished_energy
            
    return [list(p) for p in best_chain]
