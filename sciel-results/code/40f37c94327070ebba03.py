# EVOLVE-BLOCK-START
import random
import time
import sys

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

def _grow(sequence, dimension, rng):
    """Randomized greedy growth with bias towards H-H contacts."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _local_optimize(sequence, chain, dimension, rng, steps, deadline):
    """
    Optimize chain using local moves: 
    1. Pivot (tail rotation)
    2. Crankshaft (3-residue rotation) - simplified as single residue move if possible? No, stick to Pivot for robustness.
    Actually, let's implement a robust Pivot and a simple "Crank" which is a rotation of a 3-residue segment.
    """
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    n = len(chain)
    
    start_time = time.time()
    
    for step in range(steps):
        if time.time() > deadline - 5:
            break
            
        if n < 4:
            break
            
        # Randomly choose move type
        # 50% Pivot, 50% Crank (if implemented)
        # For now, let's focus on Pivot as it is the most powerful local move.
        # We will add a "Crank" move: Rotate residues i, i+1, i+2 around the axis defined by bond (i-1, i) and (i+2, i+3)?
        # No, standard crankshaft: Rotate segment [i, i+1, i+2] around the axis connecting (i-1) and (i+3)?
        # This is complex. Let's use a simpler "Single Residue Move" if valid, or just more Pivots.
        
        # Let's implement a "Pivot" move.
        pivot_idx = rng.randint(1, n - 2)
        p_prev = chain[pivot_idx - 1]
        p_pivot = chain[pivot_idx]
        
        v = [p_pivot[i] - p_prev[i] for i in range(dimension)]
        bond_axis = -1
        for i in range(dimension):
            if v[i] != 0:
                bond_axis = i
                break
        if bond_axis == -1: continue
                
        rot_axes = [i for i in range(dimension) if i != bond_axis]
        if not rot_axes: continue
        
        r_axis = rng.choice(rot_axes)
        angle = rng.choice([1, -1])
        
        # Rotate tail [pivot_idx+1 .. N-1]
        head_set = set(chain[:pivot_idx])
        new_tail = []
        valid = True
        occupied_tail = set()
        
        for i in range(pivot_idx + 1, n):
            orig = chain[i]
            dx = orig[0] - p_pivot[0]
            dy = orig[1] - p_pivot[1]
            dz = orig[2] - p_pivot[2]
            
            if r_axis == 0: # Rotate around X
                if angle == 1:
                    nx, ny, nz = dx, -dz, dy
                else:
                    nx, ny, nz = dx, dz, -dy
            elif r_axis == 1: # Rotate around Y
                if angle == 1:
                    nx, ny, nz = dz, dy, -dx
                else:
                    nx, ny, nz = -dz, dy, dx
            else: # Rotate around Z
                if angle == 1:
                    nx, ny, nz = -dy, dx, dz
                else:
                    nx, ny, nz = dy, -dx, dz
            
            new_point = (p_pivot[0] + nx, p_pivot[1] + ny, p_pivot[2] + nz)
            
            if new_point in head_set:
                valid = False
                break
            if new_point in occupied_tail:
                valid = False
                break
            
            if new_tail:
                prev = new_tail[-1]
                if abs(new_point[0]-prev[0]) + abs(new_point[1]-prev[1]) + abs(new_point[2]-prev[2]) != 1:
                    valid = False
                    break
            
            occupied_tail.add(new_point)
            new_tail.append(new_point)
            
        if not valid:
            continue
            
        candidate_chain = chain[:pivot_idx + 1] + new_tail
        new_energy = _energy(sequence, candidate_chain)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_chain = candidate_chain
            chain = candidate_chain
            
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Uses randomized growth followed by local pivot optimization.
    Adaptive budget: spends time on optimization until deadline.
    """
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    start_time = time.time()
    
    # Global time management
    if not hasattr(fold, '_global_start'):
        fold._global_start = time.time()
        
    elapsed = time.time() - fold._global_start
    # Assume total budget 240s, kill at 330s.
    # We want to finish all 26 calls well before 330s.
    # Let's reserve 10s for the last call and overhead.
    # If we are on call k (1-indexed), we have used roughly (k-1) * avg_time.
    # Since we don't know k, let's just use a global deadline of 300s.
    # If time.time() > 300, return immediately.
    
    global_deadline = fold._global_start + 300.0
    # Also, limit each call to avoid one call eating too much time if others are slow?
    # No, let's just let it run until global deadline.
    
    # However, if one call takes 300s, we timeout.
    # Let's limit each call to a max of 10s if we are early, but allow more if we are late?
    # Actually, a fixed per-call limit of ~8s is safer for 26 calls (26*8=208s < 300s).
    # Let's use a dynamic per-call limit based on remaining time.
    # But we don't know how many calls are left.
    # Let's assume sequential and equal distribution.
    # Total calls = 26.
    # If we assume this is call i, we don't know i.
    # Let's just use a per-call limit of 8 seconds.
    
    call_deadline = time.time() + 8.0
    # But also respect global deadline
    if call_deadline > global_deadline:
        call_deadline = global_deadline

    n_restarts = 5
    # For long chains, fewer restarts, more optimization?
    # Let's do:
    # 1. Generate a few initial chains.
    # 2. Optimize each one.
    # 3. Keep the best.
    
    for i in range(n_restarts):
        if time.time() > call_deadline - 2:
            break
            
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Optimization budget for this restart
        remaining_restarts = n_restarts - i
        time_per_restart = (call_deadline - time.time()) / remaining_restarts
        # Convert to steps? 
        # 1 step ~ 10-50 us? 
        # 1 second ~ 20k-100k steps.
        # Let's use a fixed number of steps based on time.
        steps = int(time_per_restart * 5000) # Estimate 5k steps/sec for safety
        if steps < 100:
            steps = 100
            
        optimized_chain, opt_energy = _local_optimize(sequence, chain, dimension, rng, steps, call_deadline)
        
        if optimized_chain is not None:
            if best_chain is None or opt_energy < best_energy:
                best_chain = optimized_chain
                best_energy = opt_energy

    if best_chain is None:
        # Fallback
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
