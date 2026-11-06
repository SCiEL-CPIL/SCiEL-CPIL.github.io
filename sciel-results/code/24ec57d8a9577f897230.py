"""HP lattice protein folding on the cubic lattice.
Strategy: Optimized Multi-start Simulated Annealing with fast Pivot and Kink moves.
Initialization via Beam Search + Greedy Growth.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

_TOTAL_DEADLINE = 330.0
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _time_remaining():
    return _TOTAL_DEADLINE - _get_elapsed() - 5.0  # 5s safety margin

def _neighbours(point):
    # Yield 6 neighbors
    yield (point[0] + 1, point[1], point[2])
    yield (point[0] - 1, point[1], point[2])
    yield (point[0], point[1] + 1, point[2])
    yield (point[0], point[1] - 1, point[2])
    yield (point[0], point[1], point[2] + 1)
    yield (point[0], point[1], point[2] - 1)

def _energy(sequence, chain, h_indices):
    """Compute energy: -number of H-H contacts."""
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i in h_indices:
        p = chain[i]
        # Check 6 neighbors
        # (p[0]+1, p[1], p[2])
        j = where.get((p[0] + 1, p[1], p[2]))
        if j is not None and j > i + 1 and sequence[j] == 'H':
            contacts += 1
        j = where.get((p[0] - 1, p[1], p[2]))
        if j is not None and j > i + 1 and sequence[j] == 'H':
            contacts += 1
        j = where.get((p[0], p[1] + 1, p[2]))
        if j is not None and j > i + 1 and sequence[j] == 'H':
            contacts += 1
        j = where.get((p[0], p[1] - 1, p[2]))
        if j is not None and j > i + 1 and sequence[j] == 'H':
            contacts += 1
        j = where.get((p[0], p[1], p[2] + 1))
        if j is not None and j > i + 1 and sequence[j] == 'H':
            contacts += 1
        j = where.get((p[0], p[1], p[2] - 1))
        if j is not None and j > i + 1 and sequence[j] == 'H':
            contacts += 1
    return -contacts

def _beam_growth(sequence, dimension, beam_width=10, max_beam_residues=15):
    """
    Use beam search for the first 'max_beam_residues' residues, then greedy.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0, 0)
    initial_chain = [start]
    initial_occupied = {start}
    
    # Beam search phase
    beam = [(0, initial_chain, initial_occupied)]
    
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    
    for i in range(1, n):
        if i >= max_beam_residues:
            # Switch to greedy
            break
            
        new_beam = []
        for score, chain, occupied in beam:
            last = chain[-1]
            residue = sequence[i]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                gain = 0
                if residue == 'H':
                    # Check contacts with existing H's
                    # We can approximate gain by checking neighbors of 'site' against 'occupied'
                    # But 'occupied' doesn't store sequence info directly.
                    # We need to check if neighbor is an H.
                    # We can maintain a set of H positions in the beam state?
                    # To keep it fast, let's just use a heuristic:
                    # Gain is number of H neighbors of 'site' in 'chain'.
                    # This requires iterating neighbors and checking if they are in 'occupied' and if they are H.
                    # We don't have a set of H positions.
                    # Let's just use a random tie-breaker or simple gain if we track H positions.
                    # For simplicity in beam search, let's just maximize the number of H neighbors.
                    for q in _neighbours(site):
                        if q in occupied:
                            # Check if q is an H
                            # We need to know the index of q.
                            # This is slow.
                            # Let's just use a simple heuristic: if residue is H, prefer sites with more H neighbors.
                            # We can pre-compute a set of H positions for the current chain?
                            # That's O(N) per state.
                            pass
                # Simplified: Just pick valid sites.
                # To make it better, let's track H positions in the beam state.
                # But that increases memory.
                # Let's just use a small random factor.
                new_score = score + (1 if residue == 'H' else 0) * 0.1 # Tiny bias
                new_chain = chain + [site]
                new_occupied = occupied | {site}
                new_beam.append((new_score, new_chain, new_occupied))
        
        if not new_beam:
            return None
        
        new_beam.sort(key=lambda x: x[0], reverse=True)
        beam = new_beam[:beam_width]
        
        if _time_remaining() < 10.0:
            break

    # Greedy phase
    if not beam:
        return None
        
    best_entry = max(beam, key=lambda x: x[0])
    chain = best_entry[1]
    occupied = best_entry[2]
    
    current_i = len(chain)
    while current_i < n:
        last = chain[-1]
        residue = sequence[current_i]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Heuristic: prefer sites with H neighbors if residue is H
            score = 0
            if residue == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        # Check if q is H
                        # We need to know which indices are H.
                        # This is slow.
                        # Let's just add a small random score.
                        pass
            candidates.append((score + random.random(), site))
        
        if not candidates:
            return None # Dead end
            
        candidates.sort(key=lambda x: x[0], reverse=True)
        site = candidates[0][1]
        chain.append(site)
        occupied.add(site)
        current_i += 1
        
        if _time_remaining() < 5.0:
            break

    return chain

def _pivot_move(chain, occupied, rng):
    n = len(chain)
    if n < 4:
        return None, None, None
    
    k = rng.randint(0, n - 2)
    p0 = chain[k]
    p1 = chain[k + 1]
    
    # Determine axis
    if p0[0] != p1[0]:
        axis = 0
    elif p0[1] != p1[1]:
        axis = 1
    else:
        axis = 2
        
    head_set = set(chain[:k + 1])
    tail = chain[k + 1:]
    
    # Rotation functions
    def rotate_x(p, center):
        x, y, z = p
        cx, cy, cz = center
        dx, dy, dz = x - cx, y - cy, z - cz
        # 90 deg: (x, y, z) -> (x, -z, y) relative?
        # Rotation around X axis:
        # 90: y -> z, z -> -y
        # 180: y -> -y, z -> -z
        # 270: y -> -z, z -> y
        return None # Placeholder
        
    # We will try 3 angles: 90, 180, 270
    # For speed, we hardcode the rotations.
    
    for angle_idx in range(3):
        new_tail = []
        valid = True
        for i in range(len(tail)):
            p = tail[i]
            # Rotate p around axis through p0
            # Note: p1 is on the axis, so it stays fixed.
            # We rotate relative to p0.
            
            x, y, z = p
            cx, cy, cz = p0
            
            dx, dy, dz = x - cx, y - cy, z - cz
            
            if axis == 0:
                if angle_idx == 0: # 90
                    ndx, ndy, ndz = dx, dz, -dy
                elif angle_idx == 1: # 180
                    ndx, ndy, ndz = dx, -dy, -dz
                else: # 270
                    ndx, ndy, ndz = dx, -dz, dy
            elif axis == 1:
                if angle_idx == 0: # 90
                    ndx, ndy, ndz = -dz, dy, dx
                elif angle_idx == 1: # 180
                    ndx, ndy, ndz = -dx, dy, -dz
                else: # 270
                    ndx, ndy, ndz = dz, dy, -dx
            else: # axis 2
                if angle_idx == 0: # 90
                    ndx, ndy, ndz = dy, -dx, dz
                elif angle_idx == 1: # 180
                    ndx, ndy, ndz = -dx, -dy, dz
                else: # 270
                    ndx, ndy, ndz = -dy, dx, dz
            
            new_p = (ndx + cx, ndy + cy, ndz + cz)
            
            if new_p in head_set:
                valid = False
                break
            new_tail.append(new_p)
            
        if valid:
            # Check connectivity: new_tail[0] is p1 (fixed).
            # new_tail[1] must be neighbor of p1.
            # Since rotation preserves distance, and p1-p2 was 1, new_p2-p1 is 1.
            # So connectivity is guaranteed.
            
            new_chain = chain[:k + 1] + new_tail
            return new_chain, new_tail, tail
    
    return None, None, None

def _kink_move(chain, occupied, rng):
    n = len(chain)
    if n < 4:
        return None, None, None
        
    i = rng.randint(0, n - 4)
    j = rng.randint(i + 2, n - 1)
    
    segment = chain[i + 1 : j + 1]
    new_segment = segment[::-1]
    
    # Check collisions
    # The segment is reversed.
    # We need to check if any point in new_segment is in (occupied - set(segment))
    # And also check connectivity at i and j.
    
    rest_set = set(chain)
    for p in segment:
        rest_set.discard(p)
        
    valid = True
    for p in new_segment:
        if p in rest_set:
            valid = False
            break
            
    if not valid:
        return None, None, None
        
    # Check connectivity
    # new_segment[0] must be neighbor of chain[i]
    if abs(new_segment[0][0] - chain[i][0]) + abs(new_segment[0][1] - chain[i][1]) + abs(new_segment[0][2] - chain[i][2]) != 1:
        return None, None, None
        
    # new_segment[-1] must be neighbor of chain[j+1] (if exists)
    if j + 1 < n:
        if abs(new_segment[-1][0] - chain[j + 1][0]) + abs(new_segment[-1][1] - chain[j + 1][1]) + abs(new_segment[-1][2] - chain[j + 1][2]) != 1:
            return None, None, None
            
    new_chain = chain[:i + 1] + new_segment + chain[j + 1:]
    return new_chain, new_segment, segment

def _simulated_annealing(sequence, chain, rng, time_budget, h_indices):
    n = len(chain)
    if n == 0:
        return chain, 0
        
    current_energy = _energy(sequence, chain, h_indices)
    best_chain = list(chain)
    best_energy = current_energy
    occupied = set(chain)
    
    T = max(1.0, abs(current_energy) * 0.5)
    T_min = 0.01
    cooling_rate = 0.999
    
    start_t = time.time()
    while time.time() - start_t < time_budget and T > T_min:
        if rng.random() < 0.5:
            new_chain, new_seg, old_seg = _pivot_move(chain, occupied, rng)
        else:
            new_chain, new_seg, old_seg = _kink_move(chain, occupied, rng)
            
        if new_chain is None:
            T *= cooling_rate
            continue
            
        new_energy = _energy(sequence, new_chain, h_indices)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            # Update occupied set
            if old_seg is not None:
                for p in old_seg:
                    occupied.discard(p)
                for p in new_seg:
                    occupied.add(p)
            
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        T *= cooling_rate
        
    return best_chain, best_energy

def fold(sequence, dimension):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    n = len(sequence)
    rng = random.Random(12345 + n)
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    
    best_chain = None
    best_energy = 0
    
    # Allocate time
    # Total budget ~ 12s per sequence on average.
    # Let's use 20 restarts for short, 5 for long.
    if n <= 48:
        num_restarts = 20
        init_budget = 0.5
        sa_budget = 0.5
    elif n <= 64:
        num_restarts = 15
        init_budget = 0.5
        sa_budget = 0.8
    elif n <= 96:
        num_restarts = 10
        init_budget = 0.8
        sa_budget = 1.2
    else:
        num_restarts = 5
        init_budget = 1.0
        sa_budget = 2.0
        
    for restart in range(num_restarts):
        if _time_remaining() < 5.0:
            break
            
        # Initialization
        chain = _beam_growth(sequence, dimension, beam_width=10, max_beam_residues=10)
        if chain is None or len(chain) < n:
            # Fallback: simple random walk
            chain = [(0,0,0)]
            occupied = {chain[0]}
            for i in range(1, n):
                last = chain[-1]
                found = False
                for _ in range(20):
                    site = list(last)
                    axis = rng.randint(0, 2)
                    step = rng.choice([-1, 1])
                    site[axis] += step
                    site = tuple(site)
                    if site not in occupied:
                        chain.append(site)
                        occupied.add(site)
                        found = True
                        break
                if not found:
                    break
            if len(chain) < n:
                continue
                
        energy = _energy(sequence, chain, h_indices)
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
            
        # SA
        sa_time = min(sa_budget, _time_remaining() / 2.0)
        if sa_time > 0.1:
            improved_chain, improved_energy = _simulated_annealing(sequence, chain, rng, sa_time, h_indices)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved_chain
                
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
