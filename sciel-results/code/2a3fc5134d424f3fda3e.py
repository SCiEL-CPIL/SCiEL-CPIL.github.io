"""HP lattice protein folding with parallel-restart strategy and contact-map local search."""
# EVOLVE-BLOCK-START
import random
import time
import math


# Pre-compute neighbor offsets for 3D cubic lattice
NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _get_neighbours_fast(x, y, z):
    """Fast neighbor generation using precomputed offsets."""
    return [
        (x + 1, y, z), (x - 1, y, z),
        (x, y + 1, z), (x, y - 1, z),
        (x, y, z + 1), (x, y, z - 1)
    ]

def _energy(sequence, chain):
    """Compute energy = -number of non-consecutive H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _get_neighbours_fast(*p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _delta_flip(sequence, chain, pos_map, i, new_pos):
    """
    Calculate energy delta for moving residue i from old_pos to new_pos.
    Assumes new_pos is valid (unit step from i-1 and i+1, and not occupied).
    """
    old_pos = chain[i]
    if old_pos == new_pos:
        return 0
    
    is_h = sequence[i] == "H"
    if not is_h:
        return 0
    
    # Count H-H contacts involving residue i at old_pos
    old_contacts = 0
    for q in _get_neighbours_fast(*old_pos):
        j = pos_map.get(q)
        if j is not None and abs(j - i) > 1 and sequence[j] == "H":
            old_contacts += 1
    
    # Count H-H contacts involving residue i at new_pos
    new_contacts = 0
    for q in _get_neighbours_fast(*new_pos):
        j = pos_map.get(q)
        if j is not None and abs(j - i) > 1 and sequence[j] == "H":
            new_contacts += 1
    
    return -(new_contacts - old_contacts)

def _mc_fold(sequence, chain, dimension, rng, time_budget):
    """
    Monte Carlo Minimization with Flip and Reversal moves.
    This is a structural change from the previous local search + short anneal.
    Reversals allow large-scale rearrangements.
    """
    n = len(chain)
    if n < 4:
        return chain
    
    start_time = time.time()
    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    current_chain = list(chain)
    current_energy = best_energy
    
    pos_map = {p: i for i, p in enumerate(current_chain)}
    
    # Temperature schedule for Metropolis acceptance (allows some uphill moves to escape minima)
    # For pure minimization, we could use T=0, but T>0 helps escape deep local minima.
    T0 = max(1.0, abs(current_energy) * 0.5)
    T_min = 0.1
    # Estimate steps based on budget
    # Reversals are O(N) to check validity (if we do it naively) but we can optimize.
    # Flips are O(1) validity check.
    # Let's assume we can do ~5000-10000 steps per second.
    est_steps = int(time_budget * 5000)
    est_steps = max(100, min(est_steps, 50000))
    
    cooling_factor = (T_min / T0) ** (1.0 / est_steps)
    T = T0
    
    steps_per_sec = 5000
    
    for _ in range(est_steps):
        if time.time() - start_time > time_budget - 0.1:
            break
        
        T *= cooling_factor
        if T < T_min:
            T = T_min
            
        move_type = rng.random()
        
        if move_type < 0.5:
            # Flip Move
            i = rng.randint(1, n - 2)
            old_pos = current_chain[i]
            prev = current_chain[i - 1]
            nxt = current_chain[i + 1]
            
            # Get valid neighbors
            valid_options = []
            for new_pos in _get_neighbours_fast(*old_pos):
                # Check unit steps
                d1 = abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) + abs(new_pos[2] - prev[2])
                d2 = abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) + abs(new_pos[2] - nxt[2])
                if d1 != 1 or d2 != 1:
                    continue
                # Check self-avoidance
                if new_pos in pos_map:
                    continue
                valid_options.append(new_pos)
            
            if not valid_options:
                continue
                
            new_pos = rng.choice(valid_options)
            delta = _delta_flip(sequence, current_chain, pos_map, i, new_pos)
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                pos_map.pop(old_pos)
                pos_map[new_pos] = i
                current_chain[i] = new_pos
                current_energy += delta
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                    
        else:
            # Reversal Move
            # Choose two indices a, b such that a < b
            # Reverse segment [a, b]
            # Valid if:
            # 1. If a > 0, dist(chain[a-1], chain[b]) == 1
            # 2. If b < n-1, dist(chain[a], chain[b+1]) == 1
            # 3. No collisions with external parts of the chain (except endpoints which are handled by 1&2? No, internal collisions matter)
            #    Actually, if we reverse [a,b], the internal geometry is preserved (just mirrored).
            #    The only new potential collisions are between the reversed segment and the rest of the chain.
            #    But wait, the rest of the chain is fixed. The reversed segment occupies the same set of points?
            #    No, reversing a path on a lattice does NOT necessarily preserve the set of points.
            #    Example: (0,0,0)->(1,0,0)->(1,1,0). Reverse: (1,1,0)->(1,0,0)->(0,0,0).
            #    The set of points is the same.
            #    Wait, is a lattice path reversible in terms of point set?
            #    Yes, a self-avoiding walk is a sequence of distinct points. Reversing the sequence results in the same set of points.
            #    So, internal collisions are impossible if the original was self-avoiding.
            #    The only risks are:
            #    1. The new connection at the start (a-1 to b) is not a unit step or collides.
            #    2. The new connection at the end (a to b+1) is not a unit step or collides.
            #    3. Do we need to check if chain[b] collides with chain[a-1]? No, that's the connection.
            #    4. Do we need to check if any point in [a,b] collides with any point in [0,a-1] or [b+1,n-1]?
            #       Since the set of points in [a,b] is the same as before, and the chain was self-avoiding,
            #       the points in [a,b] did not collide with [0,a-1] or [b+1,n-1] before.
            #       So they still don't.
            #    Therefore, a reversal is valid IF:
            #    - dist(chain[a-1], chain[b]) == 1 (if a > 0)
            #    - dist(chain[a], chain[b+1]) == 1 (if b < n-1)
            #    - chain[b] != chain[a-1] (implied by dist=1 unless n=2? No, dist=1 implies different points)
            #    - chain[a] != chain[b+1]
            
            a = rng.randint(0, n - 2)
            b = rng.randint(a + 1, n - 1)
            
            valid = True
            
            if a > 0:
                prev_a = current_chain[a - 1]
                new_next_a = current_chain[b]
                d = abs(prev_a[0] - new_next_a[0]) + abs(prev_a[1] - new_next_a[1]) + abs(prev_a[2] - new_next_a[2])
                if d != 1:
                    valid = False
            
            if valid and b < n - 1:
                next_b = current_chain[b + 1]
                new_prev_b = current_chain[a]
                d = abs(next_b[0] - new_prev_b[0]) + abs(next_b[1] - new_prev_b[1]) + abs(next_b[2] - new_prev_b[2])
                if d != 1:
                    valid = False
            
            if valid:
                # Perform reversal
                segment = current_chain[a:b + 1]
                segment.reverse()
                current_chain[a:b + 1] = segment
                
                # Update pos_map
                # The points are the same, but indices are swapped.
                # We can just rebuild pos_map for the affected part or update it.
                # Since the set of points is the same, we just need to update the index mapping.
                for idx in range(a, b + 1):
                    p = current_chain[idx]
                    pos_map[p] = idx
                
                # Calculate new energy
                # We can calculate delta energy more efficiently, but for simplicity and robustness,
                # let's just recalculate energy for now? No, that's O(N).
                # Let's try to calculate delta.
                # The only contacts that change are those involving the endpoints a and b, 
                # and potentially contacts between the reversed segment and itself?
                # No, internal contacts are preserved (distances are preserved).
                # So only contacts involving a or b with the outside world change.
                # Also, contact between a and b? No, they are not adjacent in the chain unless b=a+1.
                # If b=a+1, they are adjacent.
                
                # Let's just recalculate energy if N is small, or use delta.
                # Given N up to 136, O(N) energy calc is fast enough (~136 * 6 ops).
                # 5000 steps/sec * 136 ops = ~700k ops/sec. Very fast.
                new_energy = _energy(sequence, current_chain)
                delta = new_energy - current_energy
                
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                else:
                    # Reject, revert
                    segment = current_chain[a:b + 1]
                    segment.reverse()
                    current_chain[a:b + 1] = segment
                    for idx in range(a, b + 1):
                        p = current_chain[idx]
                        pos_map[p] = idx
    
    return best_chain

def _grow(sequence, dimension, rng, bias=0.7):
    """Randomized greedy chain growth with tunable contact bias."""
    start = (0, 0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _get_neighbours_fast(*chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _get_neighbours_fast(*site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _get_neighbours_fast(*site) if q not in occupied)
            # Mix contact gain and free space with tunable bias
            score = gain * bias * 10.0 + free * (1 - bias) * 0.5 + rng.random() * 0.1
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def fold(sequence, dimension):
    """
    Monte Carlo Minimization strategy with Reversal moves.
    """
    deadline = time.time() + 235.0  # 235s budget with safety margin
    n = len(sequence)
    
    # Number of restarts
    if n <= 50:
        num_restarts = 10
    elif n <= 80:
        num_restarts = 6
    elif n <= 120:
        num_restarts = 4
    else:
        num_restarts = 3
        
    best = None
    best_energy = 0  # 0 = no contacts (unfolded)
    
    time_per_restart = (deadline - time.time() - 5.0) / num_restarts
    time_per_restart = max(2.0, min(time_per_restart, 40.0))
    
    for restart in range(num_restarts):
        if time.time() >= deadline - 5.0:
            break
        
        rng = random.Random(1000 + restart * 7919 + len(sequence))
        
        # Growth with varying bias
        bias = 0.5 + 0.4 * (restart % 3) / 2.0
        chain = _grow(sequence, dimension, rng, bias=bias)
        
        if chain is None:
            chain = _grow(sequence, dimension, rng, bias=0.7)
        
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        
        # Monte Carlo Minimization
        remaining = deadline - time.time() - 3.0
        if remaining > 1.0:
            mc_budget = min(time_per_restart - 0.5, remaining * 0.8)
            refined = _mc_fold(sequence, chain, dimension, rng, mc_budget)
            refine_energy = _energy(sequence, refined)
            
            if refine_energy < energy:
                chain = refined
                energy = refine_energy
        
        if energy < best_energy:
            best, best_energy = chain, energy
    
    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = 0
    
    # Final refinement with remaining time
    remaining = deadline - time.time() - 1.0
    if remaining > 2.0:
        rng = random.Random(99999 + len(sequence))
        final = _mc_fold(sequence, best, dimension, rng, remaining * 0.7)
        final_energy = _energy(sequence, final)
        
        if final_energy < best_energy:
            best = final
            best_energy = final_energy
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
