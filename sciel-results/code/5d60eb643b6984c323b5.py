# EVOLVE-BLOCK-START
import time
import random
import heapq
import math
from collections import defaultdict


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
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _beam_search_fold(sequence, dimension, beam_width=50, time_limit=30.0):
    """
    Bidirectional beam search: grow from N-terminus and C-terminus, then join.
    This explores different conformational basins.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    start_time = time.time()
    
    def _forward_beam(seq, width, steps):
        # Grow first 'steps' residues
        beam = [((0, 0, 0), {(0, 0, 0)}, [(0, 0, 0)], 0)]
        for i in range(1, steps):
            next_beam = []
            for last_pos, occupied, path, energy in beam:
                for nb in _neighbours(last_pos):
                    if nb in occupied:
                        continue
                    new_occ = occupied | {nb}
                    new_energy = energy
                    if seq[i] == 'H':
                        # Check contacts with existing H's
                        # We need to know which positions are H.
                        # For simplicity in beam search, we approximate or store H-positions.
                        # Storing full H-set is heavy. Let's just use energy heuristic based on local density?
                        # No, we need accurate energy.
                        # Let's store a set of H positions in the state.
                        pass 
            # This implementation is getting complex to maintain state efficiently in Python with sets.
            # Let's use a simpler heuristic-based beam search that tracks only position and energy,
            # assuming we can recompute energy at the end or use a cheap proxy.
            # Actually, for short prefixes (e.g., 20 residues), we can afford to store the path and compute energy.
            
            # Let's restart this function with a cleaner state: (pos, occupied_set, h_pos_set, path, energy)
            break
        
        # Simplified Beam Search for prefix
        beam = [((0, 0, 0), {(0, 0, 0)}, {(0, 0, 0)} if seq[0]=='H' else set(), [(0, 0, 0)], 0)]
        for i in range(1, steps):
            next_beam = []
            for last_pos, occupied, h_pos, path, energy in beam:
                for nb in _neighbours(last_pos):
                    if nb in occupied:
                        continue
                    new_occ = occupied | {nb}
                    new_h = h_pos
                    new_e = energy
                    if seq[i] == 'H':
                        new_h = h_pos | {nb}
                        # Check contacts
                        for q in _neighbours(nb):
                            if q in h_pos:
                                new_e -= 1
                    new_path = path + [nb]
                    next_beam.append((nb, new_occ, new_h, new_path, new_e))
            
            # Prune
            next_beam.sort(key=lambda x: x[4]) # Minimize energy (more negative is better)
            beam = next_beam[:width]
            
            if time.time() - start_time > time_limit:
                break
                
        return beam
    
    def _backward_beam(seq, width, steps):
        # Grow last 'steps' residues backwards
        # Reverse sequence
        rev_seq = seq[::-1]
        beam = [((0, 0, 0), {(0, 0, 0)}, {(0, 0, 0)} if rev_seq[0]=='H' else set(), [(0, 0, 0)], 0)]
        for i in range(1, steps):
            next_beam = []
            for last_pos, occupied, h_pos, path, energy in beam:
                for nb in _neighbours(last_pos):
                    if nb in occupied:
                        continue
                    new_occ = occupied | {nb}
                    new_h = h_pos
                    new_e = energy
                    if rev_seq[i] == 'H':
                        new_h = h_pos | {nb}
                        for q in _neighbours(nb):
                            if q in h_pos:
                                new_e -= 1
                    new_path = path + [nb]
                    next_beam.append((nb, new_occ, new_h, new_path, new_e))
            
            next_beam.sort(key=lambda x: x[4])
            beam = next_beam[:width]
            
            if time.time() - start_time > time_limit:
                break
                
        return beam
    
    # Strategy:
    # 1. Generate a set of "anchor" conformations for the middle or ends using beam search.
    # 2. Since joining two open chains is hard, let's just use the beam search to generate
    #    high-quality initial folds for the whole chain if n is small, or for a prefix if n is large.
    #    For n > 40, full beam search is too expensive.
    #    Let's use the beam search to generate a good "core" or just use it for the whole chain
    #    with a very narrow beam if n is small.
    
    if n <= 30:
        # Full beam search
        beam = [((0, 0, 0), {(0, 0, 0)}, {(0, 0, 0)} if sequence[0]=='H' else set(), [(0, 0, 0)], 0)]
        for i in range(1, n):
            next_beam = []
            for last_pos, occupied, h_pos, path, energy in beam:
                for nb in _neighbours(last_pos):
                    if nb in occupied:
                        continue
                    new_occ = occupied | {nb}
                    new_h = h_pos
                    new_e = energy
                    if sequence[i] == 'H':
                        new_h = h_pos | {nb}
                        for q in _neighbours(nb):
                            if q in h_pos:
                                new_e -= 1
                    new_path = path + [nb]
                    next_beam.append((nb, new_occ, new_h, new_path, new_e))
            
            next_beam.sort(key=lambda x: x[4])
            beam = next_beam[:beam_width]
            
            if time.time() - start_time > time_limit:
                break
        
        if beam:
            best_state = min(beam, key=lambda x: x[4])
            return best_state[3]
        return [(0,0,0)] + [(i,0,0) for i in range(1,n)] # Fallback

    else:
        # For longer chains, beam search on the whole thing is infeasible.
        # Return None to signal caller to use other methods.
        return None


def _pivot_optimize(chain, sequence, steps, time_limit):
    """
    Local search using pivot moves (rotations of the tail around a pivot point).
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    start_time = time.time()
    
    # Precompute neighbor lists for speed? No, on the fly is fine for N~100.
    
    for _ in range(steps):
        if time.time() - start_time > time_limit:
            break
        
        pivot_idx = random.randint(1, n - 2)
        p_prev = chain[pivot_idx - 1]
        p_pivot = chain[pivot_idx]
        
        # Determine bond vector
        v = [p_pivot[i] - p_prev[i] for i in range(3)]
        bond_axis = -1
        for i in range(3):
            if v[i] != 0:
                bond_axis = i
                break
        
        rot_axes = [i for i in range(3) if i != bond_axis]
        if not rot_axes:
            continue
        r_axis = random.choice(rot_axes)
        angle = random.choice([1, -1])
        
        def rotate_point(point, center, axis, angle):
            p = list(point)
            c = list(center)
            p[0] -= c[0]
            p[1] -= c[1]
            p[2] -= c[2]
            
            if axis == 0:
                if angle == 1:
                    p[1], p[2] = -p[2], p[1]
                else:
                    p[1], p[2] = p[2], -p[1]
            elif axis == 1:
                if angle == 1:
                    p[0], p[2] = p[2], -p[0]
                else:
                    p[0], p[2] = -p[2], p[0]
            elif axis == 2:
                if angle == 1:
                    p[0], p[1] = -p[1], p[0]
                else:
                    p[0], p[1] = p[1], -p[0]
            
            p[0] += c[0]
            p[1] += c[1]
            p[2] += c[2]
            return tuple(p)
        
        head_set = set(chain[:pivot_idx])
        valid = True
        new_tail_points = []
        occupied_tail = set()
        
        for i in range(pivot_idx + 1, n):
            orig_point = chain[i]
            new_point = rotate_point(orig_point, p_pivot, r_axis, angle)
            
            if new_point in head_set:
                valid = False
                break
            if new_point in occupied_tail:
                valid = False
                break
            if new_tail_points:
                prev_new = new_tail_points[-1]
                dist = sum(abs(new_point[k] - prev_new[k]) for k in range(3))
                if dist != 1:
                    valid = False
                    break
            occupied_tail.add(new_point)
            new_tail_points.append(new_point)
        
        if not valid:
            continue
        
        candidate_chain = chain[:pivot_idx + 1] + new_tail_points
        new_energy = _energy(sequence, candidate_chain)
        
        if new_energy < best_energy:
            best_chain = candidate_chain
            best_energy = new_energy
            # Accept move immediately (greedy hill climbing)
            chain = candidate_chain
            current_energy = new_energy
            
    return best_chain, best_energy


def _simulated_annealing(chain, sequence, time_limit):
    """
    Simulated Annealing with pivot moves.
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    T = 10.0
    cooling = 0.99
    
    while time.time() - start_time < time_limit:
        if T < 0.1:
            break
            
        pivot_idx = random.randint(1, n - 2)
        p_prev = current_chain[pivot_idx - 1]
        p_pivot = current_chain[pivot_idx]
        
        v = [p_pivot[i] - p_prev[i] for i in range(3)]
        bond_axis = -1
        for i in range(3):
            if v[i] != 0:
                bond_axis = i
                break
        
        rot_axes = [i for i in range(3) if i != bond_axis]
        if not rot_axes:
            continue
        r_axis = random.choice(rot_axes)
        angle = random.choice([1, -1])
        
        def rotate_point(point, center, axis, angle):
            p = list(point)
            c = list(center)
            p[0] -= c[0]
            p[1] -= c[1]
            p[2] -= c[2]
            
            if axis == 0:
                if angle == 1:
                    p[1], p[2] = -p[2], p[1]
                else:
                    p[1], p[2] = p[2], -p[1]
            elif axis == 1:
                if angle == 1:
                    p[0], p[2] = p[2], -p[0]
                else:
                    p[0], p[2] = -p[2], p[0]
            elif axis == 2:
                if angle == 1:
                    p[0], p[1] = -p[1], p[0]
                else:
                    p[0], p[1] = p[1], -p[0]
            
            p[0] += c[0]
            p[1] += c[1]
            p[2] += c[2]
            return tuple(p)
        
        head_set = set(current_chain[:pivot_idx])
        valid = True
        new_tail_points = []
        occupied_tail = set()
        
        for i in range(pivot_idx + 1, n):
            orig_point = current_chain[i]
            new_point = rotate_point(orig_point, p_pivot, r_axis, angle)
            
            if new_point in head_set:
                valid = False
                break
            if new_point in occupied_tail:
                valid = False
                break
            if new_tail_points:
                prev_new = new_tail_points[-1]
                dist = sum(abs(new_point[k] - prev_new[k]) for k in range(3))
                if dist != 1:
                    valid = False
                    break
            occupied_tail.add(new_point)
            new_tail_points.append(new_point)
        
        if not valid:
            continue
        
        candidate_chain = current_chain[:pivot_idx + 1] + new_tail_points
        new_energy = _energy(sequence, candidate_chain)
        
        delta = new_energy - current_energy
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_chain = candidate_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_chain = candidate_chain
                best_energy = new_energy
                
        T *= cooling
        
    return best_chain, best_energy


def _grow_random(sequence, dimension):
    """
    Generate a random self-avoiding walk.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            # Heuristic: prefer sites that create H-H contacts
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H":
                        gain += 1
            # Add some randomness
            score = gain + random.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None # Dead end
            
        # Choose best option
        options.sort(key=lambda x: x[0], reverse=True)
        # Sometimes pick randomly from top 3 to explore
        if len(options) > 3 and random.random() < 0.2:
            choice = random.choice(options[:3])
        else:
            choice = options[0]
            
        site = choice[1]
        occupied[site] = i
        chain.append(site)
        
    return chain


def fold(sequence, dimension):
    """
    Strategy: 
    1. Try beam search for short chains.
    2. For longer chains, use multi-start local search:
       - Generate initial folds via random greedy growth.
       - Optimize with pivot moves (hill climbing).
       - Occasionally use simulated annealing to escape local minima.
    3. Keep track of the best fold found within the time budget.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    start_time = time.time()
    deadline = start_time + 240 # Safety margin for 240s budget
    
    best_chain = None
    best_energy = 0
    
    # 1. Try Beam Search for short chains
    if n <= 30:
        bs_result = _beam_search_fold(sequence, dimension, beam_width=50, time_limit=10.0)
        if bs_result:
            e = _energy(sequence, bs_result)
            if e < best_energy:
                best_energy = e
                best_chain = bs_result
    
    # 2. Multi-start Local Search
    max_restarts = 100
    for i in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        # Generate initial fold
        chain = _grow_random(sequence, dimension)
        if chain is None:
            continue
        
        # Optimize
        # Alternate between hill climbing and simulated annealing
        if i % 3 == 0:
            # Simulated Annealing for exploration
            opt_chain, opt_energy = _simulated_annealing(chain, sequence, time_limit=2.0)
        else:
            # Hill Climbing for exploitation
            opt_chain, opt_energy = _pivot_optimize(chain, sequence, steps=50 + n, time_limit=2.0)
            
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_chain = opt_chain
            
    if best_chain is None:
        # Fallback: linear chain
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
