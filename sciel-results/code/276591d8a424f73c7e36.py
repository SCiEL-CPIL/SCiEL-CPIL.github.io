"""HP lattice protein folding on cubic lattice - replica exchange Monte Carlo."""
import random
import time
import math

_GLOBAL_START = time.time()
_HARD_DEADLINE = _GLOBAL_START + 310.0  # 310s hard limit, 20s safety margin


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


def _initial_chain_snake(sequence, dimension, rng):
    """Generate an initial self-avoiding walk using a space-filling curve approach."""
    n = len(sequence)
    # Use a simple space-filling pattern
    chain = []
    x, y, z = 0, 0, 0
    chain.append((x, y, z))
    occupied = {(x, y, z): 0}
    
    # Try to fill a roughly cubic box
    side = int(math.ceil(n ** (1/3))) + 2
    
    for i in range(1, n):
        best_site = None
        best_score = -1000
        
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            # Score based on how "central" the site is (prefer compact)
            cx = site[0] / max(side, 1)
            cy = site[1] / max(side, 1)
            cz = site[2] / max(side, 1)
            dist_from_center = abs(cx - 0.5) + abs(cy - 0.5) + abs(cz - 0.5)
            
            # Prefer H-H contacts
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            
            score = gain * 10 - dist_from_center + rng.random() * 0.5
            if score > best_score:
                best_score = score
                best_site = site
        
        if best_site is None:
            # Fallback: try to continue in a straight line
            dx = chain[-1][0] - chain[-2][0]
            dy = chain[-1][1] - chain[-2][1]
            dz = chain[-1][2] - chain[-2][2]
            best_site = (chain[-1][0] + dx, chain[-1][1] + dy, chain[-1][2] + dz)
            if best_site in occupied:
                # Last resort: pick any unoccupied neighbour
                for site in _neighbours(chain[-1]):
                    if site not in occupied:
                        best_site = site
                        break
                if best_site in occupied:
                    return None
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain


def _monte_carlo_fold(sequence, chain, rng, deadline, num_reps=8):
    """Replica exchange Monte Carlo with multiple temperature replicas."""
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    # Initialize replicas at different temperatures
    temperatures = []
    T_max = 5.0
    T_min = 0.1
    for i in range(num_reps):
        T = T_max * (T_min / T_max) ** (i / max(num_reps - 1, 1))
        temperatures.append(T)
    
    # Each replica starts as a copy of the initial chain
    replicas = [list(chain) for _ in range(num_reps)]
    replica_energies = [_compute_energy(sequence, c) for c in replicas]
    replica_occupied = [{p: i for i, p in enumerate(c)} for c in replicas]
    
    # Track best across all replicas
    best_energy = min(replica_energies)
    best_chain = replicas[replica_energies.index(best_energy)]
    
    # Precompute H positions for faster energy delta
    h_positions = [i for i in range(n) if sequence[i] == 'H']
    
    iters = 0
    while time.time() < deadline:
        iters += 1
        
        # Pick a random replica
        r = rng.randint(0, num_reps - 1)
        T = temperatures[r]
        
        # Pick a move type
        if rng.random() < 0.6:
            # Single residue move
            i = rng.randint(1, n - 2)
            old = replicas[r][i]
            prev = replicas[r][i-1]
            next_pos = replicas[r][i+1]
            
            candidates = []
            for site in _neighbours(prev):
                if site == old:
                    continue
                if site in replica_occupied[r] and replica_occupied[r][site] != i:
                    continue
                dist = sum(abs(site[a] - next_pos[a]) for a in range(len(site)))
                if dist == 1:
                    candidates.append(site)
            
            if not candidates:
                continue
            
            site = rng.choice(candidates)
            delta = _delta_energy_move(sequence, replicas[r], i, old, site, replica_occupied[r])
            delta_e = -delta  # Energy change
            
            # Metropolis acceptance
            if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-9)):
                # Apply move
                replica_occupied[r][old] = -1
                replica_occupied[r][site] = i
                replicas[r][i] = site
                replica_energies[r] += delta_e
                
                if replica_energies[r] < best_energy:
                    best_energy = replica_energies[r]
                    best_chain = list(replicas[r])
        else:
            # 2-opt move (reverse segment)
            i = rng.randint(1, n - 2)
            j = rng.randint(i + 1, n - 1)
            
            segment = replicas[r][i:j+1]
            new_segment = list(reversed(segment))
            
            # Validate
            valid = True
            if i > 0:
                prev_pos = replicas[r][i-1]
                first_new = new_segment[0]
                if sum(abs(prev_pos[a] - first_new[a]) for a in range(len(prev_pos))) != 1:
                    valid = False
            if valid and j < n - 1:
                next_pos = replicas[r][j+1]
                last_new = new_segment[-1]
                if sum(abs(next_pos[a] - last_new[a]) for a in range(len(next_pos))) != 1:
                    valid = False
            
            if valid:
                # Check self-avoidance
                occ_set = set()
                for idx, pos in enumerate(replicas[r]):
                    if idx < i or idx > j:
                        occ_set.add(pos)
                for pos in new_segment:
                    if pos in occ_set:
                        valid = False
                        break
            
            if valid:
                # Compute energy delta
                occ_map = {}
                for idx, pos in enumerate(replicas[r]):
                    if idx < i or idx > j:
                        occ_map[pos] = idx
                
                old_cc = 0
                for idx in range(i, j+1):
                    if sequence[idx] == 'H':
                        pos = replicas[r][idx]
                        for q in _neighbours(pos):
                            oj = occ_map.get(q)
                            if oj is not None and sequence[oj] == 'H':
                                old_cc += 1
                
                new_cc = 0
                for idx in range(i, j+1):
                    if sequence[idx] == 'H':
                        new_pos = replicas[r][i+j-idx]
                        for q in _neighbours(new_pos):
                            oj = occ_map.get(q)
                            if oj is not None and sequence[oj] == 'H':
                                new_cc += 1
                
                delta_e = -(new_cc - old_cc)
                
                if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-9)):
                    # Apply move
                    for k in range(j - i + 1):
                        replicas[r][i + k] = new_segment[k]
                    
                    # Rebuild occupied
                    new_occ = {}
                    for idx, pos in enumerate(replicas[r]):
                        new_occ[pos] = idx
                    replica_occupied[r] = new_occ
                    replica_energies[r] += delta_e
                    
                    if replica_energies[r] < best_energy:
                        best_energy = replica_energies[r]
                        best_chain = list(replicas[r])
        
        # Replica exchange: try to swap adjacent replicas
        if rng.random() < 0.1 and num_reps > 1:
            r1 = rng.randint(0, num_reps - 2)
            r2 = r1 + 1
            T1, T2 = temperatures[r1], temperatures[r2]
            E1, E2 = replica_energies[r1], replica_energies[r2]
            
            # Delta for swapping
            delta_swap = (E1 / T2 + E2 / T1) - (E1 / T1 + E2 / T2)
            
            if delta_swap < 0 or rng.random() < math.exp(-delta_swap):
                # Swap replicas
                replicas[r1], replicas[r2] = replicas[r2], replicas[r1]
                replica_energies[r1], replica_energies[r2] = replica_energies[r2], replica_energies[r1]
                replica_occupied[r1], replica_occupied[r2] = replica_occupied[r2], replica_occupied[r1]
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation via replica exchange Monte Carlo."""
    n = len(sequence)
    rng = random.Random(42 + len(sequence) * 7 + sum(ord(c) for c in sequence))
    
    # Time budget
    per_seq_time = 11.0
    deadline = time.time() + per_seq_time
    
    # Generate initial chain
    chain = _initial_chain_snake(sequence, dimension, rng)
    if chain is None or len(chain) != n:
        # Fallback: straight line
        chain = [(i, 0, 0) for i in range(n)]
    
    # Run Monte Carlo
    chain, energy = _monte_carlo_fold(sequence, chain, rng, deadline, num_reps=8)
    
    # Verify self-avoidance
    if len(set(chain)) != n:
        # Should not happen, but fallback
        chain = [(i, 0, 0) for i in range(n)]
        energy = _compute_energy(sequence, chain)
    
    return [list(p) for p in chain]
