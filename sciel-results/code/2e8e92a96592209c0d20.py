"""Improved HP lattice protein folding with local search refinement."""
# EVOLVE-BLOCK-START
import random
import time

_TOTAL_BUDGET = 330.0
_DEADLINE = None
_START_TIME = None

def _init_time():
    global _START_TIME, _DEADLINE
    if _START_TIME is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + _TOTAL_BUDGET * 0.88

def _time_remaining():
    _init_time()
    return _DEADLINE - time.time()

def _is_deadline():
    return time.time() >= _DEADLINE

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
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng, start_pos=None):
    if start_pos is None:
        start_pos = (0, 0, 0)
    chain = [start_pos]
    occupied = {start_pos: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 2.0 + 0.05 * free + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        site = max(options, key=lambda x: x[0])[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _local_search(sequence, chain, rng, max_iter=5000):
    n = len(chain)
    if n < 5:
        return chain
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for iteration in range(max_iter):
        if _is_deadline() or _time_remaining() < 2.0:
            break
        
        move_type = rng.random()
        improved = False
        
        if move_type < 0.5:
            # Single residue relocation
            if n >= 4:
                i = rng.randint(1, n - 2)
                old_pos = chain[i]
                for new_pos in _neighbours(old_pos):
                    if new_pos == old_pos or new_pos in occupied:
                        continue
                    # Tentatively move
                    del occupied[old_pos]
                    occupied[new_pos] = i
                    chain[i] = new_pos
                    
                    new_energy = _energy(sequence, chain)
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                        current_energy = new_energy
                        improved = True
                        break
                    else:
                        # Revert
                        chain[i] = old_pos
                        del occupied[new_pos]
                        occupied[old_pos] = i
        
        elif move_type < 0.8:
            # Pivot: move residue i+1 to another neighbour of i
            if n >= 4:
                i = rng.randint(1, n - 2)
                old_next = chain[i + 1]
                for new_next in _neighbours(chain[i]):
                    if new_next == old_next or new_next in occupied:
                        continue
                    # Tentatively move
                    del occupied[old_next]
                    occupied[new_next] = i + 1
                    chain[i + 1] = new_next
                    
                    # Check self-avoidance for rest of chain
                    valid = True
                    for j in range(i + 2, n):
                        if chain[j] in occupied:
                            valid = False
                            break
                    if not valid:
                        chain[i + 1] = old_next
                        del occupied[new_next]
                        occupied[old_next] = i + 1
                        continue
                    
                    new_energy = _energy(sequence, chain)
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                        current_energy = new_energy
                        improved = True
                        break
                    else:
                        chain[i + 1] = old_next
                        del occupied[new_next]
                        occupied[old_next] = i + 1
        
        else:
            # Segment reversal
            if n >= 6:
                i = rng.randint(1, n - 4)
                j = rng.randint(i + 1, n - 2)
                new_chain = list(chain)
                new_chain[i:j+1] = list(reversed(chain[i:j+1]))
                # Check self-avoidance
                seen = set()
                valid = True
                for p in new_chain:
                    if p in seen:
                        valid = False
                        break
                    seen.add(p)
                if not valid:
                    continue
                new_energy = _energy(sequence, new_chain)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = new_chain
                    chain = new_chain
                    current_energy = new_energy
                    occupied = {p: k for k, p in enumerate(chain)}
                    improved = True
    
    return best_chain

def fold(sequence, dimension):
    _init_time()
    rng = random.Random(12345 + len(sequence))
    
    n = len(sequence)
    best_chain = None
    best_energy = 0
    
    max_restarts = 50
    if n > 100:
        max_restarts = 30
    elif n > 60:
        max_restarts = 40
    
    for restart in range(max_restarts):
        if _is_deadline() or _time_remaining() < 3.0:
            break
        
        start_pos = (rng.randint(-n, n), rng.randint(-n, n), rng.randint(-n, n))
        chain = _greedy_grow(sequence, dimension, rng, start_pos)
        if chain is None:
            continue
        
        chain = _local_search(sequence, chain, rng, max_iter=3000)
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
