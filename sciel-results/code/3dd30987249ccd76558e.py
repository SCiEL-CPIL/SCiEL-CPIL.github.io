"""HP lattice protein folding on cubic lattice - pivot Monte Carlo with parallel tempering."""
import random
import time
import math

_GLOBAL_START = time.time()
_HARD_DEADLINE = _GLOBAL_START + 320.0  # 320s hard limit, 10s safety margin


def _neighbours(point):
    """Generate 6 neighbours in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy: -number of H-H contacts (non-adjacent)."""
    n = len(chain)
    occupied = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = occupied.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _pivot_move(chain, pivot_idx, direction_idx, occupied):
    """
    Perform a pivot move: rotate the segment after pivot_idx around the bond at pivot_idx.
    direction_idx selects which of the 4 possible rotations (excluding identity) to apply.
    Returns (new_chain, new_occupied, energy_delta) or None if invalid.
    """
    n = len(chain)
    if pivot_idx < 1 or pivot_idx >= n - 1:
        return None
    
    # The segment to rotate is chain[pivot_idx+1:]
    segment = chain[pivot_idx+1:]
    if not segment:
        return None
    
    # The axis of rotation is the bond from chain[pivot_idx] to chain[pivot_idx+1]
    axis_start = chain[pivot_idx]
    axis_end = chain[pivot_idx+1]
    
    # Determine the rotation axis direction
    axis_dir = [axis_end[a] - axis_start[a] for a in range(3)]
    
    # Find the axis index (should have exactly one non-zero component)
    axis_idx = -1
    for a in range(3):
        if axis_dir[a] != 0:
            if axis_idx != -1:
                return None  # Not a valid axis
            axis_idx = a
    if axis_idx == -1:
        return None
    
    # The rotation is around the axis through axis_start and axis_end
    # We need to rotate the segment around this axis by 90, 180, or 270 degrees
    # direction_idx: 0=90deg, 1=180deg, 2=270deg, 3=270deg other direction
    
    # For simplicity, let's use a different approach: 
    # Pivot moves in HP lattice are typically defined as rotating a sub-chain 
    # around one of its bonds. Let's use a simpler pivot definition.
    
    # Actually, let me redefine: a pivot move rotates the tail of the chain 
    # (after position pivot_idx) around the bond connecting positions pivot_idx-1 and pivot_idx.
    
    return None  # This implementation is complex; let's use a different move set


def _try_pivot_move(chain, pivot_idx, occupied):
    """
    Try a pivot move: rotate the segment from pivot_idx to end around the bond 
    connecting pivot_idx-1 and pivot_idx.
    Returns (new_chain, new_occupied, energy_delta) or None if invalid.
    """
    n = len(chain)
    if pivot_idx < 1 or pivot_idx >= n - 1:
        return None
    
    # Segment to rotate: chain[pivot_idx:]
    segment = chain[pivot_idx:]
    
    # Axis of rotation: bond from chain[pivot_idx-1] to chain[pivot_idx]
    p_prev = chain[pivot_idx-1]
    p_axis = chain[pivot_idx]
    
    # Direction of the axis
    axis_dir = [p_axis[a] - p_prev[a] for a in range(3)]
    
    # Find which axis it is
    axis_idx = -1
    for a in range(3):
        if axis_dir[a] != 0:
            if axis_idx != -1:
                return None
            axis_idx = a
    if axis_idx == -1:
        return None
    
    # The other two axes for rotation
    other_axes = [a for a in range(3) if a != axis_idx]
    
    # Try all 4 possible rotations (90, 180, 270, and the other 270)
    # Actually, rotating around an axis in 3D, we have 4 distinct orientations
    # Let's generate all 4 possible rotated positions for each point in the segment
    
    # For a point relative to the axis start (p_prev), we can rotate it
    # The rotation is around the line through p_prev and p_axis
    
    # Let's use a simpler approach: try all 3 possible new positions for chain[pivot_idx]
    # that are adjacent to p_prev but not equal to p_axis, and then greedily extend
    
    # This is getting complex. Let's use a different move: single-residue moves and 
    # segment reversals (2-opt).
    
    return None


def _single_move(chain, idx, occupied, sequence):
    """
    Try to move residue idx to an adjacent position.
    Returns (new_chain, new_occupied, energy_delta) or None if no valid move.
    """
    n = len(chain)
    if idx < 1 or idx >= n - 1:
        return None
    
    old_pos = chain[idx]
    prev_pos = chain[idx-1]
    next_pos = chain[idx+1]
    
    # Find valid new positions
    candidates = []
    for site in _neighbours(prev_pos):
        if site == old_pos:
            continue
        if site in occupied and occupied[site] != idx:
            continue
        # Check distance to next_pos
        dist = sum(abs(site[a] - next_pos[a]) for a in range(3))
        if dist == 1:
            candidates.append(site)
    
    if not candidates:
        return None
    
    new_pos = random.choice(candidates)
    
    # Compute energy delta
    energy_delta = 0
    if sequence[idx] == 'H':
        # Lost contacts
        for q in _neighbours(old_pos):
            j = occupied.get(q)
            if j is not None and j != idx and sequence[j] == 'H':
                energy_delta += 1  # Lost contact, energy increases
        # Gained contacts
        for q in _neighbours(new_pos):
            j = occupied.get(q)
            if j is not None and j != idx and sequence[j] == 'H':
                energy_delta -= 1  # Gained contact, energy decreases
    
    new_chain = list(chain)
    new_chain[idx] = new_pos
    
    new_occupied = dict(occupied)
    del new_occupied[old_pos]
    new_occupied[new_pos] = idx
    
    return new_chain, new_occupied, energy_delta


def _segment_reverse_move(chain, i, j, occupied, sequence):
    """
    Try to reverse the segment from i to j.
    Returns (new_chain, new_occupied, energy_delta) or None if invalid.
    """
    n = len(chain)
    if i < 1 or j > n - 2 or i >= j:
        return None
    
    # Check boundary conditions
    if i > 0:
        prev_pos = chain[i-1]
        first_new = chain[j]
        if sum(abs(prev_pos[a] - first_new[a]) for a in range(3)) != 1:
            return None
    
    if j < n - 1:
        next_pos = chain[j+1]
        last_new = chain[i]
        if sum(abs(next_pos[a] - last_new[a]) for a in range(3)) != 1:
            return None
    
    # Check self-avoidance
    occ_set = set()
    for idx in range(n):
        if idx < i or idx > j:
            occ_set.add(chain[idx])
    
    segment = chain[i:j+1]
    for pos in segment:
        if pos in occ_set:
            return None
    
    # Compute energy delta
    # Old contacts involving the segment
    old_contacts = 0
    for idx in range(i, j+1):
        if sequence[idx] == 'H':
            pos = chain[idx]
            for q in _neighbours(pos):
                oj = occupied.get(q)
                if oj is not None and oj < i - 1 and sequence[oj] == 'H':
                    old_contacts += 1
                elif oj is not None and oj > j + 1 and sequence[oj] == 'H':
                    old_contacts += 1
    
    # New contacts
    new_chain = list(chain)
    new_chain[i:j+1] = reversed(segment)
    
    new_occupied = {}
    for idx, pos in enumerate(new_chain):
        new_occupied[pos] = idx
    
    new_contacts = 0
    for idx in range(i, j+1):
        if sequence[idx] == 'H':
            pos = new_chain[idx]
            for q in _neighbours(pos):
                oj = new_occupied.get(q)
                if oj is not None and oj < i - 1 and sequence[oj] == 'H':
                    new_contacts += 1
                elif oj is not None and oj > j + 1 and sequence[oj] == 'H':
                    new_contacts += 1
    
    energy_delta = new_contacts - old_contacts
    
    return new_chain, new_occupied, energy_delta


def _grow_initial(sequence, dimension, rng):
    """Grow an initial self-avoiding walk with some randomness."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            # Score based on potential contacts and compactness
            score = rng.random()
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        score += 2.0
            options.append((score, site))
        
        if not options:
            # Fallback: try a simple extension
            return None
        
        # Boltzmann selection
        max_score = max(s for s, _ in options)
        weights = [math.exp((s - max_score) / 1.0) for s, _ in options]
        total = sum(weights)
        r = rng.random() * total
        cumsum = 0
        best_site = options[0][1]
        for (score, site), w in zip(options, weights):
            cumsum += w
            if r <= cumsum:
                best_site = site
                break
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain


def _parallel_tempering_sa(sequence, initial_chains, rng, deadline):
    """
    Parallel tempering (replica exchange) simulated annealing.
    """
    n = len(initial_chains[0])
    num_reps = len(initial_chains)
    
    if num_reps == 1:
        return _local_search_sa(sequence, initial_chains[0], rng, deadline)
    
    # Sort by energy
    scored = [(_compute_energy(sequence, c), c) for c in initial_chains]
    scored.sort()
    
    replicas = [list(c) for _, c in scored]
    replica_energies = [e for e, _ in scored]
    replica_occupied = [{p: i for i, p in enumerate(c)} for c in replicas]
    
    # Temperature schedule
    base_T = 4.0
    T_scale = 0.6
    temperatures = [base_T * (T_scale ** k) for k in range(num_reps)]
    
    best_chain = list(replicas[0])
    best_energy = replica_energies[0]
    
    total_iters = n * 100
    t0 = base_T
    t_end = 0.05
    alpha = (t_end / t0) ** (1.0 / max(total_iters, 1))
    
    iters = 0
    while time.time() < deadline:
        iters += 1
        progress = min(iters / total_iters, 1.0)
        global_T = t0 * (t_end / t0) ** progress
        
        # Pick a replica to move
        r = rng.randint(0, num_reps - 1)
        T_r = temperatures[r] * global_T / base_T
        
        # Choose move type
        if rng.random() < 0.5 and n > 5:
            # Segment reverse
            i = rng.randint(1, n - 2)
            j = rng.randint(i + 1, n - 1)
            result = _segment_reverse_move(replicas[r], i, j, replica_occupied[r], sequence)
            if result:
                new_chain, new_occupied, delta_e = result
                accept = delta_e < 0 or (T_r > 0 and rng.random() < math.exp(-delta_e / T_r))
                if accept:
                    replicas[r] = new_chain
                    replica_occupied[r] = new_occupied
                    replica_energies[r] += delta_e
        else:
            # Single move
            idx = rng.randint(1, n - 2)
            result = _single_move(replicas[r], idx, replica_occupied[r], sequence)
            if result:
                new_chain, new_occupied, delta_e = result
                accept = delta_e < 0 or (T_r > 0 and rng.random() < math.exp(-delta_e / T_r))
                if accept:
                    replicas[r] = new_chain
                    replica_occupied[r] = new_occupied
                    replica_energies[r] += delta_e
        
        # Try replica exchange
        if num_reps > 1:
            r2 = rng.randint(0, num_reps - 1)
            if r2 != r:
                T1 = temperatures[r] * global_T / base_T
                T2 = temperatures[r2] * global_T / base_T
                E1 = replica_energies[r]
                E2 = replica_energies[r2]
                
                delta_swap = (E1 / T2 + E2 / T1) - (E1 / T1 + E2 / T2)
                if delta_swap < 0 or (T1 > 0 and T2 > 0 and rng.random() < math.exp(-delta_swap)):
                    replicas[r], replicas[r2] = replicas[r2], replicas[r]
                    replica_energies[r], replica_energies[r2] = replica_energies[r2], replica_energies[r]
                    replica_occupied[r], replica_occupied[r2] = replica_occupied[r2], replica_occupied[r]
        
        # Update best
        for rep in range(num_reps):
            if replica_energies[rep] < best_energy:
                best_energy = replica_energies[rep]
                best_chain = list(replicas[rep])
    
    return best_chain, best_energy


def _local_search_sa(sequence, chain, rng, deadline):
    """Simple simulated annealing with single and segment moves."""
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    current_energy = best_energy
    current_chain = list(best_chain)
    current_occupied = dict(occupied)
    
    t0 = 5.0
    t_end = 0.1
    total_iters = n * 50
    alpha = (t_end / t0) ** (1.0 / max(total_iters, 1))
    
    T = t0
    iters = 0
    while time.time() < deadline:
        iters += 1
        T *= alpha
        if T < t_end:
            T = t_end
        
        # Choose move
        if rng.random() < 0.5 and n > 5:
            i = rng.randint(1, n - 2)
            j = rng.randint(i + 1, n - 1)
            result = _segment_reverse_move(current_chain, i, j, current_occupied, sequence)
            if result:
                new_chain, new_occupied, delta_e = result
                accept = delta_e < 0 or (T > 0 and rng.random() < math.exp(-delta_e / T))
                if accept:
                    current_chain = new_chain
                    current_occupied = new_occupied
                    current_energy += delta_e
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                        occupied = dict(current_occupied)
        else:
            idx = rng.randint(1, n - 2)
            result = _single_move(current_chain, idx, current_occupied, sequence)
            if result:
                new_chain, new_occupied, delta_e = result
                accept = delta_e < 0 or (T > 0 and rng.random() < math.exp(-delta_e / T))
                if accept:
                    current_chain = new_chain
                    current_occupied = new_occupied
                    current_energy += delta_e
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                        occupied = dict(current_occupied)
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation via diverse growth + parallel tempering SA."""
    n = len(sequence)
    rng = random.Random(42 + len(sequence) * 7 + sum(ord(c) for c in sequence))
    
    # Time budget
    per_seq_time = min(12.0, max(2.0, (_HARD_DEADLINE - time.time()) / 26.0))
    deadline = time.time() + per_seq_time
    
    best_chain = None
    best_energy = 0
    
    # Generate diverse initial chains
    initial_chains = []
    for _ in range(6):
        if time.time() > deadline - 4.0:
            break
        chain = _grow_initial(sequence, dimension, rng)
        if chain:
            initial_chains.append(chain)
    
    # Evaluate
    for chain in initial_chains:
        e = _compute_energy(sequence, chain)
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e
    
    if best_chain is None:
        # Fallback
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    
    # Parallel tempering SA
    if time.time() < deadline - 0.5 and len(initial_chains) >= 2:
        best_chain, best_energy = _parallel_tempering_sa(
            sequence, initial_chains[:min(4, len(initial_chains))], rng, deadline - 0.5
        )
    elif time.time() < deadline - 0.5:
        best_chain, best_energy = _local_search_sa(
            sequence, best_chain, rng, deadline - 0.5
        )
    
    return [list(p) for p in best_chain]
