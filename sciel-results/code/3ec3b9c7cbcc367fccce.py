"""
HP lattice protein folding on the cubic lattice.
Strategy: Monte Carlo growth with backtracking and local optimization.
"""
import random
import sys
import time

# EVOLVE-BLOCK-START

def _neighbours(point):
    """Yield the 6 neighbours of a 3D lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(seq, chain):
    """Compute energy of a chain."""
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i in range(n):
        if seq[i] != 'H':
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = pos_to_idx.get(q)
            if j is not None and abs(i - j) > 1 and seq[j] == 'H':
                if i < j:
                    contacts += 1
    return -contacts

def _beam_search_fold(seq, dimension, beam_width=200, rng=None):
    """
    Construct a chain using Beam Search.
    """
    if rng is None:
        rng = random.Random()
    
    n = len(seq)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    # Each beam entry is (score, chain_list, occupied_set)
    start = (0, 0, 0)
    beam = [(0, [start], {start: 0})]
    
    for i in range(1, n):
        new_beam = []
        is_h = seq[i] == 'H'
        
        for score, chain, occupied in beam:
            last_pos = chain[-1]
            for q in _neighbours(last_pos):
                if q in occupied:
                    continue
                
                new_score = score
                if is_h:
                    for q2 in _neighbours(q):
                        j = occupied.get(q2)
                        if j is not None and j < i - 1 and seq[j] == 'H':
                            new_score += 1
                
                new_chain = chain + [q]
                new_occupied = occupied.copy()
                new_occupied[q] = i
                
                new_beam.append((new_score, new_chain, new_occupied))
        
        if not new_beam:
            return None
        
        new_beam.sort(key=lambda x: -x[0])
        beam = new_beam[:beam_width]
        
    if not beam:
        return None
    
    return beam[0][1]

def _sa_optimize(seq, chain, iterations=5000, rng=None):
    """
    Simulated Annealing with Single Residue Moves.
    """
    if rng is None:
        rng = random.Random()
    
    n = len(chain)
    if n < 4:
        return chain
    
    current_chain = list(chain)
    current_energy = _energy(seq, current_chain)
    pos_to_idx = {p: i for i, p in enumerate(current_chain)}
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    T = 10.0
    cooling = 0.999
    
    for it in range(iterations):
        T *= cooling
        if T < 0.01:
            break
        
        i = rng.randint(1, n - 2)
        p = current_chain[i]
        axis = rng.randint(0, 2)
        step = rng.choice((-1, 1))
        q = list(p)
        q[axis] += step
        q = tuple(q)
        
        if q in pos_to_idx:
            continue
        
        prev_p = current_chain[i - 1]
        next_p = current_chain[i + 1]
        
        if abs(prev_p[0] - q[0]) + abs(prev_p[1] - q[1]) + abs(prev_p[2] - q[2]) != 1:
            continue
        if abs(next_p[0] - q[0]) + abs(next_p[1] - q[1]) + abs(next_p[2] - q[2]) != 1:
            continue
        
        pos_to_idx.pop(p)
        pos_to_idx[q] = i
        current_chain[i] = q
        
        new_energy = _energy(seq, current_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < pow(2.718281828, -delta / max(T, 0.01)):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(current_chain)
        else:
            pos_to_idx.pop(q)
            pos_to_idx[p] = i
            current_chain[i] = p
    
    return best_chain

def fold(sequence, dimension):
    """
    Fold a protein sequence on the cubic lattice.
    Returns a list of lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n < 4:
        return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    start_time = time.time()
    time_budget = 8.0
    
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = float('inf')
    
    beam_width = 100
    if n > 60:
        beam_width = 50
    
    try:
        chain = _beam_search_fold(sequence, dimension, beam_width=beam_width, rng=rng)
        if chain is not None and len(chain) == n:
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
    except Exception:
        pass
    
    if best_chain is not None:
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time > 0.5:
            iterations = int(remaining_time * 1000)
            iterations = max(1000, min(10000, iterations))
            opt_chain = _sa_optimize(sequence, best_chain, iterations=iterations, rng=rng)
            opt_energy = _energy(sequence, opt_chain)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_chain = opt_chain
    
    if best_chain is None:
        chain = [(0, 0, 0)]
        occupied = {chain[0]: 0}
        for i in range(1, n):
            valid = []
            for q in _neighbours(chain[-1]):
                if q not in occupied:
                    valid.append(q)
            if not valid:
                break
            q = rng.choice(valid)
            chain.append(q)
            occupied[q] = i
        
        if len(chain) == n:
            best_chain = chain
            best_energy = _energy(sequence, best_chain)
        else:
            best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
            best_energy = _energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
