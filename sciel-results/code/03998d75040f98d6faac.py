import time
import random
import math
import heapq

# EVOLVE-BLOCK-START

_DEADLINE = 315.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _compute_energy(sequence, chain):
    """Compute H-H contact energy (negative count)."""
    n = len(chain)
    if n == 0:
        return 0
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    contacts = 0
    is_h = [c == 'H' for c in sequence]
    
    for i, p in enumerate(chain):
        if is_h[i]:
            x, y, z = p
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                j = where.get((x+dx, y+dy, z+dz), -1)
                if j > i + 1 and is_h[j]:
                    contacts += 1
    return -contacts

def _beam_search(sequence, dimension, beam_width, rng, time_budget):
    """Beam search for folding. Returns list of (chain, energy) for top candidates."""
    n = len(sequence)
    is_h = [c == 'H' for c in sequence]
    
    start = (0, 0, 0)
    start_occ = {start: 0}
    start_chain = [start]
    start_contacts = 0
    
    beam = [(0, start_chain, start_occ, start_contacts)]
    
    for i in range(1, n):
        if _time_left() < 5.0:
            break
        
        new_beam = []
        for neg_score, chain, occ, contacts in beam:
            last = chain[-1]
            x, y, z = last
            
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                site = (x+dx, y+dy, z+dz)
                if site in occ:
                    continue
                
                new_occ = dict(occ)
                new_occ[site] = i
                
                new_contacts = contacts
                if is_h[i]:
                    sx, sy, sz = site
                    for dx2, dy2, dz2 in _NEIGHBOR_OFFSETS:
                        j = new_occ.get((sx+dx2, sy+dy2, sz+dz2), -1)
                        if 0 <= j < i and j < i - 1 and is_h[j]:
                            new_contacts += 1
                
                new_chain = chain + [site]
                new_neg = -new_contacts
                new_beam.append((new_neg, new_chain, new_occ, new_contacts))
        
        if not new_beam:
            break
        
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
    
    results = []
    for neg_score, chain, occ, contacts in beam:
        energy = -contacts
        results.append((chain, energy))
    return results

def _kink_move(chain, i):
    """
    Try to move residue i to an adjacent free site.
    Returns new_chain if valid, else None.
    """
    if i <= 0 or i >= len(chain) - 1:
        return None

    occupied = set(chain)
    occupied.discard(chain[i])

    prev_pos = chain[i-1]
    next_pos = chain[i+1]
    curr_pos = chain[i]
    
    cx, cy, cz = curr_pos
    
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        site = (cx+dx, cy+dy, cz+dz)
        if site in occupied:
            continue
        
        if abs(site[0] - prev_pos[0]) + abs(site[1] - prev_pos[1]) + abs(site[2] - prev_pos[2]) != 1:
            continue
        if abs(site[0] - next_pos[0]) + abs(site[1] - next_pos[1]) + abs(site[2] - next_pos[2]) != 1:
            continue
        
        new_chain = list(chain)
        new_chain[i] = site
        return new_chain
    return None

def _pivot_move(chain, i, axis, sign):
    """
    Perform a pivot move at index i.
    Rotates the tail (i+1 to end) around chain[i] by 90 degrees around 'axis' with 'sign'.
    Returns new_chain if valid, else None.
    """
    n = len(chain)
    if i <= 1 or i >= n - 1:
        return None
    
    head = chain[:i+1]
    tail = chain[i+1:]
    pivot = chain[i]
    
    head_set = set(head)
    new_tail = []
    seen_in_tail = set()
    
    px, py, pz = pivot
    
    for pos in tail:
        vx, vy, vz = pos[0] - px, pos[1] - py, pos[2] - pz
        
        if axis == 0:
            if sign == 1:
                nvx, nvy, nvz = vx, -vz, vy
            else:
                nvx, nvy, nvz = vx, vz, -vy
        elif axis == 1:
            if sign == 1:
                nvx, nvy, nvz = vz, vy, -vx
            else:
                nvx, nvy, nvz = -vz, vy, vx
        else:
            if sign == 1:
                nvx, nvy, nvz = -vy, vx, vz
            else:
                nvx, nvy, nvz = vy, -vx, vz
        
        new_pos = (px + nvx, py + nvy, pz + nvz)
        
        if new_pos in seen_in_tail or new_pos in head_set:
            return None
        
        seen_in_tail.add(new_pos)
        new_tail.append(new_pos)
        
    return head + new_tail

def _compute_energy_delta(sequence, chain, old_chain, i, is_h):
    """Compute energy difference when moving residue i from old_chain to chain.
    Returns new_energy - old_energy."""
    old_pos = old_chain[i]
    new_pos = chain[i]
    
    old_where = {p: idx for idx, p in enumerate(old_chain)}
    new_where = {p: idx for idx, p in enumerate(chain)}
    
    old_contacts = 0
    if is_h[i]:
        x, y, z = old_pos
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            j = old_where.get((x+dx, y+dy, z+dz), -1)
            if 0 <= j < len(chain) and j != i and abs(j - i) > 1 and is_h[j]:
                old_contacts += 1
    
    new_contacts = 0
    if is_h[i]:
        x, y, z = new_pos
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            j = new_where.get((x+dx, y+dy, z+dz), -1)
            if 0 <= j < len(chain) and j != i and abs(j - i) > 1 and is_h[j]:
                new_contacts += 1
    
    return new_contacts - old_contacts

def _local_optimize_sa(sequence, chain, time_budget, rng):
    """
    Simulated Annealing with Greedy initialization and mixed moves.
    Uses incremental energy computation for speed.
    """
    n = len(chain)
    is_h = [c == 'H' for c in sequence]
    
    current_chain = list(chain)
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    greedy_iter = 0
    max_greedy_iters = 3000 if n < 50 else 800
    while greedy_iter < max_greedy_iters and _time_left() > 5.0:
        improved = False
        greedy_iter += 1
        
        i = rng.randint(1, n - 2)
        new_chain = _kink_move(current_chain, i)
        if new_chain:
            delta = _compute_energy_delta(sequence, new_chain, current_chain, i, is_h)
            if delta < 0:
                current_chain = new_chain
                current_energy += delta
                improved = True
        
        if not improved and n > 10:
            i = rng.randint(2, n - 2)
            axis = rng.randint(0, 2)
            sign = rng.choice([-1, 1])
            new_chain = _pivot_move(current_chain, i, axis, sign)
            if new_chain:
                new_energy = _compute_energy(sequence, new_chain)
                if new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(current_chain)

    T = 2.5
    alpha = 0.995
    min_T = 0.01
    
    steps = 0
    while _time_left() > 3.0 and T > min_T:
        move_type = rng.random()
        new_chain = None
        
        if move_type < 0.55:
            i = rng.randint(1, n - 2)
            new_chain = _kink_move(current_chain, i)
            if new_chain:
                delta = _compute_energy_delta(sequence, new_chain, current_chain, i, is_h)
        elif move_type < 0.9 and n > 15:
            i = rng.randint(2, n - 2)
            axis = rng.randint(0, 2)
            sign = rng.choice([-1, 1])
            new_chain = _pivot_move(current_chain, i, axis, sign)
            if new_chain:
                new_energy = _compute_energy(sequence, new_chain)
                delta = new_energy - current_energy
        else:
            i = rng.randint(1, n - 3)
            max_len = min(6, n - i - 1)
            j = rng.randint(i + 1, i + max_len)
            
            new_chain = None
            if i > 0:
                d1 = abs(current_chain[i-1][0] - current_chain[j][0]) + \
                     abs(current_chain[i-1][1] - current_chain[j][1]) + \
                     abs(current_chain[i-1][2] - current_chain[j][2])
                if d1 == 1:
                    if j < n - 1:
                        d2 = abs(current_chain[j+1][0] - current_chain[i][0]) + \
                             abs(current_chain[j+1][1] - current_chain[i][1]) + \
                             abs(current_chain[j+1][2] - current_chain[i][2])
                        if d2 == 1:
                            new_chain = current_chain[:i] + list(reversed(current_chain[i:j+1])) + current_chain[j+1:]
                            if len(set(new_chain)) != len(new_chain):
                                new_chain = None
                    else:
                        new_chain = current_chain[:i] + list(reversed(current_chain[i:j+1])) + current_chain[j+1:]
                        if len(set(new_chain)) != len(new_chain):
                            new_chain = None
            else:
                if j < n - 1:
                    d2 = abs(current_chain[j+1][0] - current_chain[i][0]) + \
                         abs(current_chain[j+1][1] - current_chain[i][1]) + \
                         abs(current_chain[j+1][2] - current_chain[i][2])
                    if d2 == 1:
                        new_chain = current_chain[:i] + list(reversed(current_chain[i:j+1])) + current_chain[j+1:]
                        if len(set(new_chain)) != len(new_chain):
                            new_chain = None
                else:
                    new_chain = current_chain[:i] + list(reversed(current_chain[i:j+1])) + current_chain[j+1:]
                    if len(set(new_chain)) != len(new_chain):
                        new_chain = None
            
            if new_chain:
                new_energy = _compute_energy(sequence, new_chain)
                delta = new_energy - current_energy

        if new_chain is None:
            T *= alpha
            steps += 1
            continue
            
        if delta < 0 or (rng.random() < math.exp(-delta / T)):
            current_chain = new_chain
            current_energy += delta
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        T *= alpha
        steps += 1
        
    return best_chain, best_energy

def _multi_restart_fold(sequence, dimension):
    """
    Run beam search + local optimization multiple times with different seeds/beam widths.
    """
    n = len(sequence)
    best_chain = None
    best_energy = float('inf')
    
    if n <= 30:
        num_restarts = 6
        beam_widths = [2000, 1500, 1200, 1000, 800, 600]
    elif n <= 50:
        num_restarts = 5
        beam_widths = [1000, 800, 600, 500, 400]
    elif n <= 80:
        num_restarts = 4
        beam_widths = [500, 400, 300, 250]
    else:
        num_restarts = 3
        beam_widths = [300, 250, 200]
        
    if _time_left() < 50.0:
        num_restarts = min(num_restarts, 3)
    if _time_left() < 20.0:
        num_restarts = 1

    for restart_idx in range(num_restarts):
        if _time_left() < 8.0:
            break
        
        rng = random.Random(42 + restart_idx * 104729)
        bw = beam_widths[restart_idx % len(beam_widths)]
        
        bs_budget = min(12.0, _time_left() * 0.25)
        
        results = _beam_search(sequence, dimension, bw, rng, bs_budget)
        
        if not results:
            continue
        
        results.sort(key=lambda x: x[1])
        
        for cand_idx in range(min(3, len(results))):
            if _time_left() < 5.0:
                break
            
            chain = results[cand_idx][0]
            opt_chain, opt_energy = _local_optimize_sa(sequence, chain, _time_left(), rng)
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_chain = opt_chain
                
    if best_chain is None:
        return [[i, 0, 0] for i in range(n)]
    
    return [list(p) for p in best_chain]

def fold(sequence, dimension):
    """Fold a sequence on the cubic lattice. Returns list of lattice points."""
    n = len(sequence)
    
    return _multi_restart_fold(sequence, dimension)

# EVOLVE-BLOCK-END
