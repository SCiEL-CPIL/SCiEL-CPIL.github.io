"""HP lattice protein folding on cubic lattice - parallel tempering Monte Carlo on SAW scaffold."""
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


def _compute_energy(sequence, chain, occupied=None):
    """Compute energy: -number of H-H contacts (non-adjacent)."""
    n = len(chain)
    if occupied is None:
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


def _grow_scaffold(sequence, dimension, rng, temperature=0.0):
    """Grow a self-avoiding walk scaffold with Metropolis-like acceptance.
    Uses a fixed set of candidate moves and accepts/rejects based on energy change."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        # Generate all valid candidate sites
        candidates = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                candidates.append(site)
        
        if not candidates:
            return None
        
        if len(candidates) == 1:
            best_site = candidates[0]
        elif temperature > 0:
            # Metropolis-like selection: score each candidate
            scores = []
            for site in candidates:
                gain = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == "H":
                            gain += 1
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                score = gain * 2.0 + free * 0.1
                scores.append(score)
            
            # Boltzmann selection
            max_score = max(scores)
            weights = [math.exp((s - max_score) / max(temperature, 1e-6)) for s in scores]
            total = sum(weights)
            r = rng.random() * total
            cumsum = 0
            best_site = candidates[0]
            for (site, w) in zip(candidates, weights):
                cumsum += w
                if r <= cumsum:
                    best_site = site
                    break
        else:
            # Greedy: pick best scoring candidate
            best_score = -1e10
            best_site = candidates[0]
            for site in candidates:
                gain = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == "H":
                            gain += 1
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                score = gain * 2.0 + free * 0.1
                if score > best_score:
                    best_score = score
                    best_site = site
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain


def _grow_compact_scaffold(sequence, dimension, rng):
    """Growth that packs tightly, preferring sites with many occupied neighbours."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                candidates.append(site)
        
        if not candidates:
            return None
        
        best_score = -1e10
        best_site = candidates[0]
        for site in candidates:
            pack_score = 0
            for q in _neighbours(site):
                if q in occupied:
                    pack_score += 1
            contact_score = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        contact_score += 1
            score = contact_score * 2 + pack_score * 0.1 + rng.random() * 0.3
            if score > best_score:
                best_score = score
                best_site = site
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain


def _local_search_sa(sequence, chain, rng, deadline, max_iters_per_residue=3):
    """Simulated annealing with single moves and 2-opt moves."""
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    current_energy = best_energy
    current_chain = list(best_chain)
    current_occupied = dict(occupied)
    
    # Cooling schedule
    t0 = 5.0
    t_end = 0.1
    total_iters = n * max_iters_per_residue * 10
    alpha = (t_end / t0) ** (1.0 / max(total_iters, 1))
    
    T = t0
    improved = False
    
    while time.time() < deadline:
        improved = False
        T *= alpha
        if T < t_end:
            T = t_end
        
        for _ in range(n):
            if time.time() > deadline:
                break
            
            # Choose move type
            if rng.random() < 0.7 and n > 4:
                # 2-opt move
                i = rng.randint(1, n - 2)
                j = rng.randint(i + 1, n - 1)
                
                new_chain = list(current_chain)
                segment = new_chain[i:j+1]
                new_segment = list(reversed(segment))
                
                valid = True
                if i > 0:
                    prev_pos = new_chain[i-1]
                    first_new = new_segment[0]
                    if sum(abs(prev_pos[a] - first_new[a]) for a in range(len(prev_pos))) != 1:
                        valid = False
                if valid and j < n - 1:
                    next_pos = new_chain[j+1]
                    last_new = new_segment[-1]
                    if sum(abs(next_pos[a] - last_new[a]) for a in range(len(next_pos))) != 1:
                        valid = False
                
                if valid:
                    occ_set = set()
                    for idx, pos in enumerate(new_chain):
                        if idx < i or idx > j:
                            occ_set.add(pos)
                    for pos in new_segment:
                        if pos in occ_set:
                            valid = False
                            break
                
                if valid:
                    occ_map = {}
                    for idx, pos in enumerate(new_chain):
                        if idx < i or idx > j:
                            occ_map[pos] = idx
                    
                    old_cc = 0
                    for idx in range(i, j+1):
                        if sequence[idx] == 'H':
                            pos = current_chain[idx]
                            for q in _neighbours(pos):
                                oj = occ_map.get(q)
                                if oj is not None and sequence[oj] == 'H':
                                    old_cc += 1
                    
                    new_cc = 0
                    for idx in range(i, j+1):
                        if sequence[idx] == 'H':
                            new_pos = new_chain[i+j-idx]
                            for q in _neighbours(new_pos):
                                oj = occ_map.get(q)
                                if oj is not None and sequence[oj] == 'H':
                                    new_cc += 1
                    
                    delta_e = -(new_cc - old_cc)
                    accept = delta_e < 0 or (T > 0 and rng.random() < math.exp(-delta_e / T))
                    
                    if accept:
                        for k in range(j - i + 1):
                            new_chain[i + k] = new_segment[k]
                        for idx, pos in enumerate(new_chain):
                            current_occupied[pos] = idx
                        for pos in list(current_occupied.keys()):
                            if pos not in new_chain:
                                del current_occupied[pos]
                        
                        current_chain = new_chain
                        current_energy += delta_e
                        
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(current_chain)
                            occupied = dict(current_occupied)
                        improved = True
            else:
                # Single move
                idx = rng.randint(1, n - 2)
                old = current_chain[idx]
                prev = current_chain[idx-1]
                next_pos = current_chain[idx+1]
                
                candidates = []
                for site in _neighbours(prev):
                    if site == old:
                        continue
                    if site in current_occupied and current_occupied[site] != idx:
                        continue
                    dist = sum(abs(site[a] - next_pos[a]) for a in range(len(site)))
                    if dist == 1:
                        candidates.append(site)
                
                if candidates:
                    site = rng.choice(candidates)
                    delta = _delta_energy_move(sequence, current_chain, idx, old, site, current_occupied)
                    delta_e = -delta
                    
                    accept = delta_e < 0 or (T > 0 and rng.random() < math.exp(-delta_e / T))
                    
                    if accept:
                        current_occupied[old] = -1
                        current_occupied[site] = idx
                        current_chain[idx] = site
                        current_energy += delta_e
                        
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(current_chain)
                            occupied = dict(current_occupied)
                        improved = True
        
        if not improved and T < t_end * 1.1:
            break
    
    return best_chain, best_energy


def _parallel_tempering_sa(sequence, initial_chains, rng, deadline, num_temps=5):
    """Parallel tempering Monte Carlo on a fixed SAW scaffold.
    Multiple replicas at different temperatures explore conformational space,
    with Metropolis acceptance and periodic swap attempts between replicas."""
    n = len(initial_chains[0])
    if n < 4:
        return initial_chains[0], _compute_energy(sequence, initial_chains[0])
    
    # Use top chains as initial replicas
    scored = [(_compute_energy(sequence, c), c) for c in initial_chains]
    scored.sort()
    
    # Create replicas from top chains, pad with perturbations if needed
    replicas = []
    for e, c in scored[:num_temps]:
        replicas.append(list(c))
    while len(replicas) < num_temps:
        # Create perturbed version of best chain
        base = list(scored[0][1])
        idx = rng.randint(1, n - 2)
        old = base[idx]
        prev = base[idx-1]
        next_pos = base[idx+1]
        occ = {p: i for i, p in enumerate(base)}
        candidates = []
        for site in _neighbours(prev):
            if site == old:
                continue
            if site in occ and occ[site] != idx:
                continue
            dist = sum(abs(site[a] - next_pos[a]) for a in range(len(site)))
            if dist == 1:
                candidates.append(site)
        if candidates:
            site = rng.choice(candidates)
            base[idx] = site
            replicas.append(base)
        else:
            replicas.append(list(scored[0][1]))
    
    replica_energies = [_compute_energy(sequence, c) for c in replicas]
    replica_occupied = [{p: i for i, p in enumerate(c)} for c in replicas]
    
    # Temperature ladder: geometric spacing
    T_high = 4.0
    T_low = 0.05
    temps = []
    for k in range(num_temps):
        ratio = (T_low / T_high) ** (1.0 / max(num_temps - 1, 1))
        temps.append(T_high * (ratio ** k))
    
    best_chain = list(replicas[0])
    best_energy = replica_energies[0]
    
    # Main loop
    while time.time() < deadline:
        # Pick a random replica to update
        r = rng.randint(0, num_temps - 1)
        T_r = temps[r]
        
        # Choose move type
        if rng.random() < 0.6 and n > 4:
            # 2-opt move
            i = rng.randint(1, n - 2)
            j = rng.randint(i + 1, n - 1)
            
            new_chain = list(replicas[r])
            segment = new_chain[i:j+1]
            new_segment = list(reversed(segment))
            
            valid = True
            if i > 0:
                prev_pos = new_chain[i-1]
                first_new = new_segment[0]
                if sum(abs(prev_pos[a] - first_new[a]) for a in range(len(prev_pos))) != 1:
                    valid = False
            if valid and j < n - 1:
                next_pos = new_chain[j+1]
                last_new = new_segment[-1]
                if sum(abs(next_pos[a] - last_new[a]) for a in range(len(next_pos))) != 1:
                    valid = False
            
            if valid:
                occ_set = set()
                for idx, pos in enumerate(new_chain):
                    if idx < i or idx > j:
                        occ_set.add(pos)
                for pos in new_segment:
                    if pos in occ_set:
                        valid = False
                        break
            
            if valid:
                occ_map = {}
                for idx, pos in enumerate(new_chain):
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
                        new_pos = new_chain[i+j-idx]
                        for q in _neighbours(new_pos):
                            oj = occ_map.get(q)
                            if oj is not None and sequence[oj] == 'H':
                                new_cc += 1
                
                delta_e = -(new_cc - old_cc)
                # Metropolis acceptance
                if delta_e <= 0:
                    accept = True
                else:
                    accept = rng.random() < math.exp(-delta_e / max(T_r, 1e-10))
                
                if accept:
                    for k in range(j - i + 1):
                        new_chain[i + k] = new_segment[k]
                    for idx, pos in enumerate(new_chain):
                        replica_occupied[r][pos] = idx
                    for pos in list(replica_occupied[r].keys()):
                        if pos not in new_chain:
                            del replica_occupied[r][pos]
                    
                    replicas[r] = new_chain
                    replica_energies[r] += delta_e
        else:
            # Single move
            idx = rng.randint(1, n - 2)
            old = replicas[r][idx]
            prev = replicas[r][idx-1]
            next_pos = replicas[r][idx+1]
            
            candidates = []
            for site in _neighbours(prev):
                if site == old:
                    continue
                if site in replica_occupied[r] and replica_occupied[r][site] != idx:
                    continue
                dist = sum(abs(site[a] - next_pos[a]) for a in range(len(site)))
                if dist == 1:
                    candidates.append(site)
            
            if candidates:
                site = rng.choice(candidates)
                delta = _delta_energy_move(sequence, replicas[r], idx, old, site, replica_occupied[r])
                delta_e = -delta
                
                # Metropolis acceptance
                if delta_e <= 0:
                    accept = True
                else:
                    accept = rng.random() < math.exp(-delta_e / max(T_r, 1e-10))
                
                if accept:
                    replica_occupied[r][old] = -1
                    replica_occupied[r][site] = idx
                    replicas[r][idx] = site
                    replica_energies[r] += delta_e
        
        # Periodic swap attempt between two random replicas
        if rng.random() < 0.3:
            r1 = rng.randint(0, num_temps - 1)
            r2 = rng.randint(0, num_temps - 1)
            if r1 != r2:
                T1 = temps[r1]
                T2 = temps[r2]
                E1 = replica_energies[r1]
                E2 = replica_energies[r2]
                
                # Swap acceptance: detailed balance
                delta_swap = (E1 / T2 + E2 / T1) - (E1 / T1 + E2 / T2)
                if delta_swap <= 0:
                    do_swap = True
                else:
                    do_swap = rng.random() < math.exp(-delta_swap)
                
                if do_swap:
                    replicas[r1], replicas[r2] = replicas[r2], replicas[r1]
                    replica_energies[r1], replica_energies[r2] = replica_energies[r2], replica_energies[r1]
                    replica_occupied[r1], replica_occupied[r2] = replica_occupied[r2], replica_occupied[r1]
        
        # Update global best
        for r in range(num_temps):
            if replica_energies[r] < best_energy:
                best_energy = replica_energies[r]
                best_chain = list(replicas[r])
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation via scaffold growth + parallel tempering Monte Carlo."""
    n = len(sequence)
    rng = random.Random(42 + len(sequence) * 7 + sum(ord(c) for c in sequence))
    
    # Time budget: adapt dynamically based on number of sequences
    per_seq_time = min(12.0, max(2.0, (_HARD_DEADLINE - time.time()) / 26.0))
    deadline = time.time() + per_seq_time
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Generate diverse initial chains using scaffold growth
    initial_chains = []
    
    # Strategy 1: Scaffold growth with various temperatures
    for temp in [0.0, 0.15, 0.3, 0.5]:
        if time.time() > deadline - 5.0:
            break
        chain = _grow_scaffold(sequence, dimension, rng, temperature=temp)
        if chain:
            initial_chains.append(chain)
    
    # Strategy 2: Compact scaffold growth
    for _ in range(2):
        if time.time() > deadline - 5.0:
            break
        chain = _grow_compact_scaffold(sequence, dimension, rng)
        if chain:
            initial_chains.append(chain)
    
    # Evaluate all initial chains
    for chain in initial_chains:
        e = _compute_energy(sequence, chain)
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e
    
    if best_chain is None:
        # Fallback: simple snake pattern
        best_chain = []
        for i in range(n):
            x = i % 8
            y = (i // 8) % 8
            z = (i // 64) % 8
            best_chain.append((x, y, z))
        if len(set(best_chain)) < n:
            best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    
    # Phase 2: Parallel tempering Monte Carlo on the scaffold
    if time.time() < deadline - 0.5:
        # Use all initial chains as seeds for parallel tempering
        num_temps = min(5, max(2, len(initial_chains)))
        best_chain, best_energy = _parallel_tempering_sa(
            sequence, initial_chains, rng, deadline - 0.5, num_temps=num_temps
        )
    
    # Phase 3: If time remains, do a final local search polish
    remaining = deadline - time.time()
    if remaining > 1.5:
        best_chain, best_energy = _local_search_sa(
            sequence, best_chain, rng, time.time() + remaining - 0.3, max_iters_per_residue=2
        )
    
    return [list(p) for p in best_chain]
