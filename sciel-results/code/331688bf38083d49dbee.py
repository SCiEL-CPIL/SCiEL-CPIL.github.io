"""HP lattice protein folding on cubic lattice - multi-replica simulated annealing with 2D projection initialization."""
import random
import time
import math

_GLOBAL_START = time.time()
_HARD_DEADLINE = _GLOBAL_START + 315.0  # 315s hard limit, 15s safety margin


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
    Returns delta_e such that new_energy = old_energy + delta_e."""
    if sequence[i] != 'H':
        return 0
    delta_e = 0
    # Lost contacts at old_pos
    for q in _neighbours(old_pos):
        j = occupied.get(q)
        if j is not None and j != i and sequence[j] == 'H':
            delta_e += 1
    # Gained contacts at new_pos
    for q in _neighbours(new_pos):
        j = occupied.get(q)
        if j is not None and j != i and sequence[j] == 'H':
            delta_e -= 1
    return delta_e


def _grow_greedy(sequence, dimension, rng):
    """Generate initial chain using greedy growth with compactness and H-H contact bias."""
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    for i in range(1, n):
        best_site = None
        best_score = -1e9
        
        last = chain[-1]
        prev = chain[-2] if len(chain) >= 2 else None
        
        # Candidate sites: neighbours of last that are unoccupied
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Check it doesn't create a loop (must be adjacent to next position if exists)
            # For now, just check unoccupied
            
            score = 0.0
            
            # Compactness: prefer sites closer to center of mass
            com_x = sum(p[0] for p in chain) / len(chain)
            com_y = sum(p[1] for p in chain) / len(chain)
            com_z = sum(p[2] for p in chain) / len(chain)
            dist = abs(site[0] - com_x) + abs(site[1] - com_y) + abs(site[2] - com_z)
            score -= dist * 0.3
            
            # H-H contact gain
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        score += 5.0
            
            # Avoid backtracking to previous position
            if prev is not None and site == prev:
                score -= 10.0
            
            score += rng.random() * 0.5
            
            if score > best_score:
                best_score = score
                best_site = site
        
        if best_site is None:
            # Fallback: try any unoccupied neighbour
            for site in _neighbours(last):
                if site not in occupied:
                    best_site = site
                    break
            if best_site is None:
                # Last resort: extend in straight line
                dx = last[0] - chain[-2][0] if len(chain) >= 2 else 1
                dy = last[1] - chain[-2][1] if len(chain) >= 2 else 0
                dz = last[2] - chain[-2][2] if len(chain) >= 2 else 0
                best_site = (last[0] + dx, last[1] + dy, last[2] + dz)
                if best_site in occupied:
                    for axis in range(3):
                        for step in (-1, 1):
                            cand = list(last)
                            cand[axis] += step
                            if tuple(cand) not in occupied:
                                best_site = tuple(cand)
                                break
                        if best_site not in occupied:
                            break
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain


def _grow_compact_2d_projection(sequence, dimension, rng):
    """Generate initial chain by first solving in 2D (xy-plane), then lifting to 3D."""
    n = len(sequence)
    if n == 0:
        return []
    
    # Phase 1: Greedy growth in 2D (z=0)
    chain2d = [(0, 0)]
    occupied2d = {(0, 0): 0}
    
    for i in range(1, n):
        best_site = None
        best_score = -1e9
        
        last = chain2d[-1]
        prev = chain2d[-2] if len(chain2d) >= 2 else None
        
        for site in [(last[0]+1, last[1]), (last[0]-1, last[1]), 
                     (last[0], last[1]+1), (last[0], last[1]-1)]:
            if site in occupied2d:
                continue
            
            score = 0.0
            
            # Compactness
            com_x = sum(p[0] for p in chain2d) / len(chain2d)
            com_y = sum(p[1] for p in chain2d) / len(chain2d)
            dist = abs(site[0] - com_x) + abs(site[1] - com_y)
            score -= dist * 0.3
            
            # H-H contact gain
            if sequence[i] == 'H':
                for q in [(site[0]+1, site[1]), (site[0]-1, site[1]), 
                          (site[0], site[1]+1), (site[0], site[1]-1)]:
                    j = occupied2d.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        score += 5.0
            
            if prev is not None and site == prev:
                score -= 10.0
            
            score += rng.random() * 0.5
            
            if score > best_score:
                best_score = score
                best_site = site
        
        if best_site is None:
            for site in [(last[0]+1, last[1]), (last[0]-1, last[1]), 
                         (last[0], last[1]+1), (last[0], last[1]-1)]:
                if site not in occupied2d:
                    best_site = site
                    break
            if best_site is None:
                best_site = (last[0] + 1, last[1])
                # This shouldn't happen often
        
        chain2d.append(best_site)
        occupied2d[best_site] = i
    
    # Phase 2: Lift to 3D by adding z-coordinates to resolve conflicts
    # Start with all z=0, then fix any overlaps (shouldn't be any since 2D is self-avoiding)
    chain3d = [(x, y, 0) for x, y in chain2d]
    
    # Try to improve by moving some residues to z=1 or z=-1 to gain H-H contacts
    occupied3d = {p: i for i, p in enumerate(chain3d)}
    
    for i in range(n):
        if sequence[i] != 'H':
            continue
        best_z = 0
        best_gain = 0
        for z in (-1, 0, 1):
            if z == 0:
                continue
            new_pos = (chain3d[i][0], chain3d[i][1], z)
            if new_pos in occupied3d and occupied3d[new_pos] != i:
                continue
            gain = 0
            for q in _neighbours(new_pos):
                j = occupied3d.get(q)
                if j is not None and j != i and sequence[j] == 'H':
                    gain += 1
            # Subtract lost contacts at old position
            old_pos = chain3d[i]
            for q in _neighbours(old_pos):
                j = occupied3d.get(q)
                if j is not None and j != i and sequence[j] == 'H':
                    gain -= 1
            if gain > best_gain:
                best_gain = gain
                best_z = z
        
        if best_z != 0:
            old_pos = chain3d[i]
            new_pos = (chain3d[i][0], chain3d[i][1], best_z)
            occupied3d[old_pos] = -1
            occupied3d[new_pos] = i
            chain3d[i] = new_pos
    
    return chain3d


def _initial_chain(sequence, dimension, rng):
    """Generate initial chain using the better of greedy 3D and 2D projection."""
    n = len(sequence)
    if n == 0:
        return []
    
    chain1 = _grow_greedy(sequence, dimension, rng)
    chain2 = _grow_compact_2d_projection(sequence, dimension, rng)
    
    e1 = _compute_energy(sequence, chain1)
    e2 = _compute_energy(sequence, chain2)
    
    if e2 < e1:
        return chain2
    return chain1


def _try_single_move(replica, sequence, occupied, rng, T):
    """Attempt a single-residue move. Returns (success, delta_e, new_occupied)."""
    n = len(replica)
    if n < 3:
        return False, 0, occupied
    
    i = rng.randint(1, n - 2)
    old = replica[i]
    prev = replica[i-1]
    next_pos = replica[i+1]
    
    candidates = []
    for site in _neighbours(prev):
        if site == old:
            continue
        if site in occupied and occupied[site] != i:
            continue
        # Check adjacency to next_pos
        dist = sum(abs(site[a] - next_pos[a]) for a in range(len(site)))
        if dist == 1:
            candidates.append(site)
    
    if not candidates:
        return False, 0, occupied
    
    site = rng.choice(candidates)
    delta_e = _delta_energy_move(sequence, replica, i, old, site, occupied)
    
    # Metropolis acceptance
    if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-9)):
        new_occupied = dict(occupied)
        new_occupied[old] = -1
        new_occupied[site] = i
        replica[i] = site
        return True, delta_e, new_occupied
    
    return False, 0, occupied


def _try_2opt_move(replica, sequence, occupied, rng, T):
    """Attempt a 2-opt move (reverse segment). Returns (success, delta_e, new_occupied)."""
    n = len(replica)
    if n < 4:
        return False, 0, occupied
    
    i = rng.randint(1, n - 2)
    j = rng.randint(i + 1, n - 1)
    
    segment = replica[i:j+1]
    new_segment = list(reversed(segment))
    
    # Validate connectivity
    valid = True
    if i > 0:
        prev_pos = replica[i-1]
        first_new = new_segment[0]
        if sum(abs(prev_pos[a] - first_new[a]) for a in range(len(prev_pos))) != 1:
            valid = False
    if valid and j < n - 1:
        next_pos = replica[j+1]
        last_new = new_segment[-1]
        if sum(abs(next_pos[a] - last_new[a]) for a in range(len(next_pos))) != 1:
            valid = False
    
    if not valid:
        return False, 0, occupied
    
    # Check self-avoidance
    occ_set = set()
    for idx, pos in enumerate(replica):
        if idx < i or idx > j:
            occ_set.add(pos)
    for pos in new_segment:
        if pos in occ_set:
            return False, 0, occupied
    
    # Compute energy delta
    occ_map = {}
    for idx, pos in enumerate(replica):
        if idx < i or idx > j:
            occ_map[pos] = idx
    
    old_cc = 0
    for idx in range(i, j+1):
        if sequence[idx] == 'H':
            pos = replica[idx]
            for q in _neighbours(pos):
                oj = occ_map.get(q)
                if oj is not None and sequence[oj] == 'H':
                    old_cc += 1
    
    new_cc = 0
    for idx in range(i, j+1):
        if sequence[idx] == 'H':
            new_pos = replica[i+j-idx]
            for q in _neighbours(new_pos):
                oj = occ_map.get(q)
                if oj is not None and sequence[oj] == 'H':
                    new_cc += 1
    
    delta_e = -(new_cc - old_cc)
    
    # Metropolis acceptance
    if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-9)):
        for k in range(j - i + 1):
            replica[i + k] = new_segment[k]
        
        new_occupied = {}
        for idx, pos in enumerate(replica):
            new_occupied[pos] = idx
        return True, delta_e, new_occupied
    
    return False, 0, occupied


def _simulated_annealing(sequence, chain, rng, deadline, num_reps=6):
    """Multi-replica simulated annealing."""
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    # Temperature schedule: start high, cool down
    T_start = 8.0
    T_end = 0.05
    
    # Create replicas
    replicas = [list(chain) for _ in range(num_reps)]
    replica_energies = [_compute_energy(sequence, c) for c in replicas]
    replica_occupied = [{p: i for i, p in enumerate(c)} for c in replicas]
    
    best_energy = min(replica_energies)
    best_chain = replicas[replica_energies.index(best_energy)]
    
    total_time = deadline - time.time()
    if total_time <= 0:
        return best_chain, best_energy
    
    # Estimate total iterations
    target_iters = int(total_time * 5000)  # ~5000 iters/sec
    target_iters = max(target_iters, 1000)
    
    for it in range(target_iters):
        if time.time() > deadline:
            break
        
        # Linear cooling
        frac = it / max(target_iters - 1, 1)
        T = T_start * (T_end / T_start) ** frac
        
        # Pick random replica
        r = rng.randint(0, num_reps - 1)
        
        # Choose move type
        if rng.random() < 0.65:
            success, delta_e, new_occ = _try_single_move(
                replicas[r], sequence, replica_occupied[r], rng, T)
            if success:
                replica_occupied[r] = new_occ
                replica_energies[r] += delta_e
                if replica_energies[r] < best_energy:
                    best_energy = replica_energies[r]
                    best_chain = list(replicas[r])
        else:
            success, delta_e, new_occ = _try_2opt_move(
                replicas[r], sequence, replica_occupied[r], rng, T)
            if success:
                replica_occupied[r] = new_occ
                replica_energies[r] += delta_e
                if replica_energies[r] < best_energy:
                    best_energy = replica_energies[r]
                    best_chain = list(replicas[r])
        
        # Replica exchange (swap adjacent replicas with probability)
        if rng.random() < 0.15 and num_reps > 1:
            r1 = rng.randint(0, num_reps - 2)
            r2 = r1 + 1
            E1, E2 = replica_energies[r1], replica_energies[r2]
            
            # For SA, we swap if it helps the lower-T replica
            # Simple heuristic: swap if the lower-energy chain goes to higher-T replica
            if E1 < E2 and r1 > r2:
                # Swap to put better chain at lower temperature
                replicas[r1], replicas[r2] = replicas[r2], replicas[r1]
                replica_energies[r1], replica_energies[r2] = replica_energies[r2], replica_energies[r1]
                replica_occupied[r1], replica_occupied[r2] = replica_occupied[r2], replica_occupied[r1]
            elif E2 < E1 and r2 > r1:
                replicas[r1], replicas[r2] = replicas[r2], replicas[r1]
                replica_energies[r1], replica_energies[r2] = replica_energies[r2], replica_energies[r1]
                replica_occupied[r1], replica_occupied[r2] = replica_occupied[r2], replica_occupied[r1]
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation via multi-replica simulated annealing."""
    n = len(sequence)
    rng = random.Random(42 + len(sequence) * 7 + sum(ord(c) for c in sequence))
    
    # Time budget: leave margin for all sequences
    per_seq_time = 10.0
    deadline = time.time() + per_seq_time
    
    # Generate initial chain
    chain = _initial_chain(sequence, dimension, rng)
    if chain is None or len(chain) != n:
        chain = [(i, 0, 0) for i in range(n)]
    
    # Run simulated annealing
    chain, energy = _simulated_annealing(sequence, chain, rng, deadline, num_reps=6)
    
    # Verify self-avoidance
    if len(set(chain)) != n:
        chain = [(i, 0, 0) for i in range(n)]
        energy = _compute_energy(sequence, chain)
    
    return [list(p) for p in chain]
