"""HP lattice protein folding - SA with local moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

_TOTAL_START = time.time()
_HARD_DEADLINE = _TOTAL_START + 320.0  # 10s safety margin before 330s kill

def _remaining():
    return _HARD_DEADLINE - time.time()

def _manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])

def _neighbours6(p):
    return (
        (p[0]+1, p[1], p[2]), (p[0]-1, p[1], p[2]),
        (p[0], p[1]+1, p[2]), (p[0], p[1]-1, p[2]),
        (p[0], p[1], p[2]+1), (p[0], p[1], p[2]-1),
    )

def _energy(sequence, chain):
    """Compute energy: -count of H-H non-bonded contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours6(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2

def _grow_simple(sequence, dimension, rng):
    """Simple random growth to produce a valid self-avoiding walk."""
    n = len(sequence)
    if n == 0:
        return []
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours6(last):
            if site not in occupied:
                options.append(site)
        if not options:
            return None
        site = rng.choice(options)
        chain.append(site)
        occupied[site] = i
    
    return chain

def _try_single_move(sequence, chain, occupied, is_h, i, rng):
    """Try a single-residue move at position i. Returns (new_chain, new_occupied, delta) or None."""
    n = len(sequence)
    if i < 1 or i >= n - 1:
        return None
    
    old = chain[i]
    prev = chain[i-1]
    nxt = chain[i+1]
    
    # Try all 6 directions
    for new in _neighbours6(old):
        if new == prev or new == nxt:
            continue
        if _manhattan(new, prev) != 1 or _manhattan(new, nxt) != 1:
            continue
        if new in occupied:
            continue
        
        # Compute delta
        delta = 0
        if is_h[i]:
            for q in _neighbours6(old):
                j = occupied.get(q)
                if j is not None and j != i and is_h[j] and abs(i-j) > 1:
                    delta += 1  # removing contact
            for q in _neighbours6(new):
                j = occupied.get(q)
                if j is not None and j != i and is_h[j] and abs(i-j) > 1:
                    delta -= 1  # adding contact
        
        new_chain = list(chain)
        new_chain[i] = new
        new_occupied = dict(occupied)
        del new_occupied[old]
        new_occupied[new] = i
        
        return (new_chain, new_occupied, delta)
    
    return None

def _try_2crank_move(sequence, chain, occupied, is_h, i, rng):
    """Try a 2-crank move at positions i, i+1. Returns (new_chain, new_occupied, delta) or None."""
    n = len(sequence)
    if i < 1 or i >= n - 2:
        return None
    
    j = i + 1
    old_i = chain[i]
    old_j = chain[j]
    prev = chain[i-1]
    nxt = chain[j+1]
    
    for new_i in _neighbours6(old_i):
        if new_i == prev:
            continue
        if _manhattan(new_i, prev) != 1:
            continue
        if new_i in occupied and occupied[new_i] not in (i, j):
            continue
        
        for new_j in _neighbours6(new_i):
            if new_j == old_i or new_j == prev:
                continue
            if _manhattan(new_i, new_j) != 1:
                continue
            if _manhattan(new_j, nxt) != 1:
                continue
            if new_j in occupied and occupied[new_j] not in (i, j):
                continue
            
            delta = 0
            for idx, old_p, new_p in [(i, old_i, new_i), (j, old_j, new_j)]:
                if is_h[idx]:
                    for q in _neighbours6(old_p):
                        k = occupied.get(q)
                        if k is not None and k not in (i, j) and is_h[k] and abs(idx-k) > 1:
                            delta += 1
                    for q in _neighbours6(new_p):
                        k = occupied.get(q)
                        if k is not None and k not in (i, j) and is_h[k] and abs(idx-k) > 1:
                            delta -= 1
            
            new_chain = list(chain)
            new_chain[i] = new_i
            new_chain[j] = new_j
            new_occupied = dict(occupied)
            del new_occupied[old_i]
            del new_occupied[old_j]
            new_occupied[new_i] = i
            new_occupied[new_j] = j
            
            return (new_chain, new_occupied, delta)
    
    return None

def _try_3crank_move(sequence, chain, occupied, is_h, i, rng):
    """Try a 3-crank move at positions i, i+1, i+2. Returns (new_chain, new_occupied, delta) or None."""
    n = len(sequence)
    if i < 1 or i >= n - 3:
        return None
    
    j = i + 1
    k = i + 2
    old_i = chain[i]
    old_j = chain[j]
    old_k = chain[k]
    prev = chain[i-1]
    nxt = chain[k+1]
    
    for new_i in _neighbours6(old_i):
        if new_i == prev:
            continue
        if _manhattan(new_i, prev) != 1:
            continue
        if new_i in occupied and occupied[new_i] not in (i, j, k):
            continue
        
        for new_j in _neighbours6(new_i):
            if new_j == old_i or new_j == prev:
                continue
            if _manhattan(new_i, new_j) != 1:
                continue
            if new_j in occupied and occupied[new_j] not in (i, j, k):
                continue
            
            for new_k in _neighbours6(new_j):
                if new_k == old_j or new_k == prev:
                    continue
                if _manhattan(new_j, new_k) != 1:
                    continue
                if _manhattan(new_k, nxt) != 1:
                    continue
                if new_k in occupied and occupied[new_k] not in (i, j, k):
                    continue
                
                delta = 0
                for idx, old_p, new_p in [(i, old_i, new_i), (j, old_j, new_j), (k, old_k, new_k)]:
                    if is_h[idx]:
                        for q in _neighbours6(old_p):
                            m = occupied.get(q)
                            if m is not None and m not in (i, j, k) and is_h[m] and abs(idx-m) > 1:
                                delta += 1
                        for q in _neighbours6(new_p):
                            m = occupied.get(q)
                            if m is not None and m not in (i, j, k) and is_h[m] and abs(idx-m) > 1:
                                delta -= 1
                
                new_chain = list(chain)
                new_chain[i] = new_i
                new_chain[j] = new_j
                new_chain[k] = new_k
                new_occupied = dict(occupied)
                del new_occupied[old_i]
                del new_occupied[old_j]
                del new_occupied[old_k]
                new_occupied[new_i] = i
                new_occupied[new_j] = j
                new_occupied[new_k] = k
                
                return (new_chain, new_occupied, delta)
    
    return None

def _sa_fold(sequence, time_limit, rng):
    """Simulated annealing with local moves."""
    n = len(sequence)
    if n < 4:
        chain = _grow_simple(sequence, 3, rng)
        if chain is None:
            chain = [[i, 0, 0] for i in range(n)]
        return chain
    
    is_h = [c == 'H' for c in sequence]
    
    # Initial growth
    chain = _grow_simple(sequence, 3, rng)
    if chain is None:
        chain = [[i, 0, 0] for i in range(n)]
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_e = _energy(sequence, chain)
    best_e = current_e
    best_chain = list(chain)
    
    # SA parameters
    T0 = 1.5
    Tend = 0.02
    start_time = time.time()
    no_improve = 0
    
    while _remaining() > 2.0 and time.time() < time_limit:
        # Temperature
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / (time_limit - start_time))
        T = T0 * math.exp(-4.0 * progress)
        
        # Choose move type
        r = rng.random()
        if r < 0.4:
            move_type = 0
        elif r < 0.75:
            move_type = 1
        else:
            move_type = 2
        
        result = None
        if move_type == 0 and n >= 3:
            i = rng.randrange(1, n - 1)
            result = _try_single_move(sequence, chain, occupied, is_h, i, rng)
        elif move_type == 1 and n >= 4:
            i = rng.randrange(1, n - 2)
            result = _try_2crank_move(sequence, chain, occupied, is_h, i, rng)
        elif move_type == 2 and n >= 5:
            i = rng.randrange(1, n - 3)
            result = _try_3crank_move(sequence, chain, occupied, is_h, i, rng)
        
        if result is None:
            continue
        
        new_chain, new_occupied, delta = result
        
        # SA acceptance
        if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            chain = new_chain
            occupied = new_occupied
            current_e += delta
            if current_e < best_e:
                best_e = current_e
                best_chain = list(chain)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # Early stop
        if no_improve > 30000 and T < 0.05:
            break
    
    return best_chain

def fold(sequence, dimension):
    """Fold HP chain on cubic lattice. Returns list of [x,y,z] points."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    if n == 2:
        return [[0, 0, 0], [1, 0, 0]]
    
    rng = random.Random(42 + len(sequence) * 7)
    
    remaining = _remaining()
    if remaining < 10.0:
        chain = _grow_simple(sequence, dimension, rng)
        if chain is None:
            chain = [[i, 0, 0] for i in range(n)]
        return [list(p) for p in chain]
    
    # Multiple SA restarts
    if n <= 48:
        num_restarts = max(3, int(remaining / 25.0))
    elif n <= 80:
        num_restarts = max(2, int(remaining / 35.0))
    else:
        num_restarts = max(1, int(remaining / 45.0))
    
    best_chain = None
    best_energy = 0
    
    for restart in range(num_restarts):
        if _remaining() < 5.0:
            break
        
        per_run = min(_remaining() - 3.0, 50.0)
        run_limit = time.time() + per_run
        
        chain = _sa_fold(sequence, run_limit, rng)
        e = _energy(sequence, chain)
        
        if best_chain is None or e < best_energy:
            best_energy = e
            best_chain = chain
    
    if best_chain is None:
        best_chain = [[i, 0, 0] for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
