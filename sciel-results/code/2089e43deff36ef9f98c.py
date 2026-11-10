"""HP lattice protein folding on cubic lattice - multi-replica simulated annealing."""
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


def _delta_energy_move(sequence, chain, i, old_pos, new_pos, occupied):
    """Compute energy change when moving residue i from old_pos to new_pos.
    Returns delta such that new_energy = old_energy - delta (delta > 0 means improvement)."""
    if sequence[i] != 'H':
        return 0
    delta = 0
    # Lost contacts at old_pos
    for q in _neighbours(old_pos):
        j = occupied.get(q)
        if j is not None and j != i and sequence[j] == 'H':
            delta -= 1
    # Gained contacts at new_pos
    for q in _neighbours(new_pos):
        j = occupied.get(q)
        if j is not None and j != i and sequence[j] == 'H':
            delta += 1
    return delta


def _initial_chain(sequence, dimension, rng):
    """Generate an initial self-avoiding walk using greedy growth with compactness bias."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0, 0)]
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    for i in range(1, n):
        best_site = None
        best_score = -10000
        
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            candidates.append(site)
        
        if not candidates:
            # Fallback: extend in a straight line
            if len(chain) >= 2:
                dx = chain[-1][0] - chain[-2][0]
                dy = chain[-1][1] - chain[-2][1]
                dz = chain[-1][2] - chain[-2][2]
                site = (chain[-1][0] + dx, chain[-1][1] + dy, chain[-1][2] + dz)
                if site not in occupied:
                    chain.append(site)
                    occupied[site] = i
                    continue
            # Last resort
            for site in _neighbours(last):
                if site not in occupied:
                    chain.append(site)
                    occupied[site] = i
                    break
            continue
        
        for site in candidates:
            score = 0
            # Prefer compactness: closer to centroid
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)
            cz = sum(p[2] for p in chain) / len(chain)
            dist = abs(site[0] - cx) + abs(site[1] - cy) + abs(site[2] - cz)
            score -= dist * 0.5
            
            # Prefer H-H contacts
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        score += 5
            
            # Add small random perturbation
            score += rng.random() * 0.1
            
            if score > best_score:
                best_score = score
                best_site = site
        
        if best_site is not None:
            chain.append(best_site)
            occupied[best_site] = i
    
    return chain


def _try_single_move(sequence, chain, occupied, i, rng):
    """Try a single-residue pivot move at position i. Returns (new_chain, new_occupied, delta_e) or None."""
    old = chain[i]
    prev = chain[i-1]
    next_pos = chain[i+1]
    
    candidates = []
    for site in _neighbours(prev):
        if site == old:
            continue
        if site in occupied and occupied[site] != i:
            continue
        # Must be adjacent to next_pos
        dist = sum(abs(site[a] - next_pos[a]) for a in range(3))
        if dist == 1:
            candidates.append(site)
    
    if not candidates:
        return None
    
    site = rng.choice(candidates)
    delta = _delta_energy_move(sequence, chain, i, old, site, occupied)
    delta_e = -delta  # Energy change (negative is improvement)
    
    # Apply move
    new_chain = list(chain)
    new_occupied = dict(occupied)
    new_occupied[old] = -1
    new_occupied[site] = i
    new_chain[i] = site
    
    return new_chain, new_occupied, delta_e


def _try_2opt_move(sequence, chain, occupied, i, j, rng):
    """Try a 2-opt move reversing segment [i, j]. Returns (new_chain, new_occupied, delta_e) or None."""
    if i >= j:
        return None
    
    segment = chain[i:j+1]
    new_segment = list(reversed(segment))
    
    # Validate connectivity
    valid = True
    if i > 0:
        prev_pos = chain[i-1]
        first_new = new_segment[0]
        if sum(abs(prev_pos[a] - first_new[a]) for a in range(3)) != 1:
            valid = False
    if valid and j < len(chain) - 1:
        next_pos = chain[j+1]
        last_new = new_segment[-1]
        if sum(abs(next_pos[a] - last_new[a]) for a in range(3)) != 1:
            valid = False
    
    if not valid:
        return None
    
    # Check self-avoidance
    occ_set = set()
    for idx, pos in enumerate(chain):
        if idx < i or idx > j:
            occ_set.add(pos)
    for pos in new_segment:
        if pos in occ_set:
            return None
    
    # Compute energy delta
    occ_map = {}
    for idx, pos in enumerate(chain):
        if idx < i or idx > j:
            occ_map[pos] = idx
    
    old_cc = 0
    for idx in range(i, j+1):
        if sequence[idx] == 'H':
            pos = chain[idx]
            for q in _neighbours(pos):
                oj = occ_map.get(q)
                if oj is not None and sequence[oj] == 'H':
                    old_cc += 1
    
    new_cc = 0
    for idx in range(i, j+1):
        if sequence[idx] == 'H':
            new_pos = chain[i+j-idx]
            for q in _neighbours(new_pos):
                oj = occ_map.get(q)
                if oj is not None and sequence[oj] == 'H':
                    new_cc += 1
    
    delta_e = -(new_cc - old_cc)
    
    # Apply move
    new_chain = list(chain)
    for k in range(j - i + 1):
        new_chain[i + k] = new_segment[k]
    
    new_occupied = {}
    for idx, pos in enumerate(new_chain):
        new_occupied[pos] = idx
    
    return new_chain, new_occupied, delta_e


def _simulated_annealing(sequence, chain, rng, deadline, num_reps=4):
    """Multi-replica simulated annealing."""
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    # Initialize replicas
    replicas = [list(chain) for _ in range(num_reps)]
    replica_occupied = [{p: i for i, p in enumerate(c)} for c in replicas]
    replica_energies = [_compute_energy(sequence, c) for c in replicas]
    
    # Temperature schedule
    T_start = 3.0
    T_end = 0.05
    total_iters = 50000
    
    # Track best
    best_energy = min(replica_energies)
    best_chain = replicas[replica_energies.index(best_energy)]
    
    iters = 0
    while time.time() < deadline:
        iters += 1
        
        # Pick a random replica
        r = rng.randint(0, num_reps - 1)
        
        # Current temperature
        progress = min(iters / total_iters, 1.0)
        T = T_start * (T_end / T_start) ** progress
        
        # Pick a move type
        move_type = rng.random()
        if move_type < 0.7:
            # Single residue move
            i = rng.randint(1, n - 2)
            result = _try_single_move(sequence, replicas[r], replica_occupied[r], i, rng)
            if result is None:
                continue
            new_chain, new_occupied, delta_e = result
        else:
            # 2-opt move
            i = rng.randint(1, n - 2)
            j = rng.randint(i + 1, n - 1)
            result = _try_2opt_move(sequence, replicas[r], replica_occupied[r], i, j, rng)
            if result is None:
                continue
            new_chain, new_occupied, delta_e = result
        
        # Metropolis acceptance
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-9)):
            replicas[r] = new_chain
            replica_occupied[r] = new_occupied
            replica_energies[r] += delta_e
            
            if replica_energies[r] < best_energy:
                best_energy = replica_energies[r]
                best_chain = list(replicas[r])
        
        # Replica exchange: try to swap adjacent replicas
        if num_reps > 1 and rng.random() < 0.05:
            r1 = rng.randint(0, num_reps - 2)
            r2 = r1 + 1
            # Compute delta for swapping (using same T for simplicity)
            E1, E2 = replica_energies[r1], replica_energies[r2]
            if E1 != E2:
                # Swap if it improves the lower-energy replica
                if rng.random() < 0.5:
                    replicas[r1], replicas[r2] = replicas[r2], replicas[r1]
                    replica_energies[r1], replica_energies[r2] = replica_energies[r2], replica_energies[r1]
                    replica_occupied[r1], replica_occupied[r2] = replica_occupied[r2], replica_occupied[r1]
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation via multi-replica simulated annealing."""
    n = len(sequence)
    rng = random.Random(42 + len(sequence) * 7 + sum(ord(c) for c in sequence))
    
    # Time budget per sequence
    per_seq_time = 11.0
    deadline = time.time() + per_seq_time
    
    # Generate initial chain
    chain = _initial_chain(sequence, dimension, rng)
    if chain is None or len(chain) != n:
        # Fallback: straight line
        chain = [(i, 0, 0) for i in range(n)]
    
    # Run simulated annealing
    chain, energy = _simulated_annealing(sequence, chain, rng, deadline, num_reps=4)
    
    # Verify self-avoidance
    if len(set(chain)) != n:
        # Should not happen, but fallback
        chain = [(i, 0, 0) for i in range(n)]
        energy = _compute_energy(sequence, chain)
    
    return [list(p) for p in chain]
