"""HP lattice protein folding using simulated annealing with reversals."""
# EVOLVE-BLOCK-START
import random
import time

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _build_initial_chain(sequence, dimension, rng):
    """Create a snake-like initial conformation."""
    n = len(sequence)
    if n <= 1:
        return [[0, 0]]
    
    width = max(1, int(n ** 0.5))
    height = max(1, (n + width - 1) // width)
    
    chain = []
    occupied = set()
    for row in range(height):
        for col in range(width):
            if len(chain) >= n:
                break
            if row % 2 == 0:
                pos = (col, row)
            else:
                pos = (width - 1 - col, row)
            if pos not in occupied:
                chain.append([pos[0], pos[1]])
                occupied.add(pos)
        if len(chain) >= n:
            break
    
    while len(chain) < n:
        last = chain[-1]
        found = False
        for nb in _neighbours(last):
            if nb not in occupied:
                chain.append([nb[0], nb[1]])
                occupied.add(nb)
                found = True
                break
        if not found:
            break
    
    # Verify
    for i in range(1, len(chain)):
        dx = abs(chain[i][0] - chain[i-1][0])
        dy = abs(chain[i][1] - chain[i-1][1])
        if dx + dy != 1:
            return [[i, 0] for i in range(n)]
    
    return chain[:n]

def _random_growth(sequence, n, rng):
    """Randomized greedy chain growth."""
    chain = [[0, 0]]
    occupied = {(0, 0)}
    for i in range(1, n):
        last = chain[-1]
        options = []
        for nb in _neighbours(last):
            if nb not in occupied:
                score = rng.random()
                if sequence[i] == 'H':
                    for q in _neighbours(nb):
                        if q in occupied:
                            # Find index
                            for k, p in enumerate(chain):
                                if p[0] == q[0] and p[1] == q[1]:
                                    if k < i - 1 and sequence[k] == 'H':
                                        score += 1.0
                                    break
                options.append((score, nb))
        if not options:
            return None
        options.sort(reverse=True)
        top_k = min(3, len(options))
        chosen = options[rng.randint(0, top_k - 1)][1]
        chain.append([chosen[0], chosen[1]])
        occupied.add(chosen)
    return chain

def _compute_all_contacts(sequence, chain):
    """Return set of H-H contact pairs."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[(p[0], p[1])] = i
    
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] != 'H':
            continue
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts.add((i, j))
    return contacts

def _sa_fold(sequence, chain, rng, time_limit):
    """Simulated annealing with reversals and corner moves."""
    n = len(sequence)
    if n <= 2:
        return chain
    
    contacts = _compute_all_contacts(sequence, chain)
    best_chain = [list(p) for p in chain]
    best_energy = -len(contacts)
    
    start_time = time.time()
    T = 1.0
    T_end = 0.001
    cooling = 0.999
    
    e = 2.718281828
    
    while T > T_end:
        if time.time() - start_time > time_limit:
            break
        
        move_type = rng.random()
        
        if move_type < 0.6:
            # Reversal
            if n < 4:
                T *= cooling
                continue
            i = rng.randint(0, n - 3)
            j = rng.randint(i + 2, n - 1)
            
            # Check boundary adjacency
            if i > 0:
                dx = abs(chain[j][0] - chain[i-1][0])
                dy = abs(chain[j][1] - chain[i-1][1])
                if dx + dy != 1:
                    T *= cooling
                    continue
            if j < n - 1:
                dx = abs(chain[i][0] - chain[j+1][0])
                dy = abs(chain[i][1] - chain[j+1][1])
                if dx + dy != 1:
                    T *= cooling
                    continue
            
            # Apply reversal
            chain[i:j+1] = chain[i:j+1][::-1]
            
            # Recompute contacts involving residues in [i, j]
            where = {}
            for k, p in enumerate(chain):
                where[(p[0], p[1])] = k
            
            # Remove old contacts involving [i, j]
            new_contacts = set()
            for (a, b) in contacts:
                if not (i <= a <= j or i <= b <= j):
                    new_contacts.add((a, b))
            
            # Add new contacts involving [i, j]
            for k in range(i, j + 1):
                if sequence[k] != 'H':
                    continue
                p = chain[k]
                for q in _neighbours(p):
                    m = where.get(q)
                    if m is not None and m > k + 1 and sequence[m] == 'H':
                        new_contacts.add((k, m))
                    elif m is not None and k > m + 1 and sequence[m] == 'H':
                        new_contacts.add((m, k))
            
            delta = -len(new_contacts) - (-len(contacts))
            
            if delta < 0 or rng.random() < e ** (-delta / T):
                contacts = new_contacts
                energy = -len(contacts)
                if energy < best_energy:
                    best_energy = energy
                    best_chain = [list(p) for p in chain]
            else:
                chain[i:j+1] = chain[i:j+1][::-1]
        
        elif move_type < 0.85:
            # Corner move
            if n < 3:
                T *= cooling
                continue
            k = rng.randint(1, n - 2)
            last = chain[k]
            moved = False
            occupied = set((p[0], p[1]) for p in chain)
            
            for nb in _neighbours(last):
                if (nb[0], nb[1]) == (chain[k-1][0], chain[k-1][1]):
                    continue
                if (nb[0], nb[1]) == (chain[k+1][0], chain[k+1][1]):
                    continue
                if nb in occupied:
                    continue
                
                # Try move
                old_pos = (last[0], last[1])
                chain[k] = [nb[0], nb[1]]
                occupied.remove(old_pos)
                occupied.add(nb)
                
                # Check self-avoidance
                valid = True
                for m, p in enumerate(chain):
                    if m != k and (p[0], p[1]) == nb:
                        valid = False
                        break
                
                if not valid:
                    chain[k] = [old_pos[0], old_pos[1]]
                    occupied.remove(nb)
                    occupied.add(old_pos)
                    continue
                
                # Recompute contacts involving residue k
                where = {}
                for m, p in enumerate(chain):
                    where[(p[0], p[1])] = m
                
                new_contacts = set()
                for (a, b) in contacts:
                    if a != k and b != k:
                        new_contacts.add((a, b))
                
                if sequence[k] == 'H':
                    p = chain[k]
                    for q in _neighbours(p):
                        m = where.get(q)
                        if m is not None and m > k + 1 and sequence[m] == 'H':
                            new_contacts.add((k, m))
                        elif m is not None and k > m + 1 and sequence[m] == 'H':
                            new_contacts.add((m, k))
                
                delta = -len(new_contacts) - (-len(contacts))
                
                if delta < 0 or rng.random() < e ** (-delta / T):
                    contacts = new_contacts
                    energy = -len(contacts)
                    if energy < best_energy:
                        best_energy = energy
                        best_chain = [list(p) for p in chain]
                    moved = True
                    break
                else:
                    chain[k] = [old_pos[0], old_pos[1]]
                    occupied.remove(nb)
                    occupied.add(old_pos)
            
            if not moved:
                T *= cooling
        
        else:
            # Skip full moves for simplicity
            T *= cooling
        
        T *= cooling
    
    return best_chain

def fold(sequence, dimension):
    """Best conformation via multiple SA restarts with deadline guard."""
    n = len(sequence)
    if n <= 1:
        return [[0, 0]]
    
    start_time = time.time()
    total_budget = 200
    
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() - start_time > total_budget - 5:
            break
        
        if restart == 0:
            chain = [[i, 0] for i in range(n)]
        elif restart == 1:
            chain = _build_initial_chain(sequence, dimension, rng)
        else:
            chain = _random_growth(sequence, n, rng)
            if chain is None or len(chain) != n:
                chain = [[i, 0] for i in range(n)]
        
        # Verify
        pos_set = set()
        valid = True
        for p in chain:
            tp = (p[0], p[1])
            if tp in pos_set:
                valid = False
                break
            pos_set.add(tp)
        if valid:
            for i in range(1, n):
                dx = abs(chain[i][0] - chain[i-1][0])
                dy = abs(chain[i][1] - chain[i-1][1])
                if dx + dy != 1:
                    valid = False
                    break
        if not valid:
            chain = [[i, 0] for i in range(n)]
        
        remaining = total_budget - (time.time() - start_time) - 3
        if remaining < 1:
            break
        
        result = _sa_fold(sequence, chain, rng, remaining)
        
        # Compute energy
        where = {}
        for i, p in enumerate(result):
            where[(p[0], p[1])] = i
        contacts = 0
        for i, p in enumerate(result):
            if sequence[i] == 'H':
                for q in _neighbours(p):
                    j = where.get(q)
                    if j is not None and j > i + 1 and sequence[j] == 'H':
                        contacts += 1
        energy = -contacts
        
        if energy < best_energy:
            best_energy = energy
            best_chain = result
    
    if best_chain is None:
        best_chain = [[i, 0] for i in range(n)]
    
    return best_chain
# EVOLVE-BLOCK-END
