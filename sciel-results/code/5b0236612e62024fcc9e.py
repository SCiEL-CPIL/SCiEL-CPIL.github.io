"""HP lattice protein folding on cubic lattice - Replica Exchange SA with diverse greedy initializations."""
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

def _delta_energy_single(sequence, chain, i, old_pos, new_pos, occupied):
    """Compute energy change for a single residue move."""
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

def _grow_greedy(sequence, dimension, rng, bias=0.0):
    """
    Fast greedy growth. 
    bias: higher bias prefers more compact (higher occupancy neighbor count) sites.
    """
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
        
        best_site = None
        best_score = -1e10
        
        for site in candidates:
            # Score 1: H-H contacts
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            
            # Score 2: Compactness (number of occupied neighbors)
            compact = 0
            for q in _neighbours(site):
                if q in occupied:
                    compact += 1
            
            # Score 3: Randomness
            noise = rng.random()
            
            # Weighted score
            score = gain * 2.0 + compact * bias + noise
            
            if score > best_score:
                best_score = score
                best_site = site
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain

def _replica_exchange_sa(sequence, initial_chains, rng, deadline, num_temps=4):
    """
    Replica Exchange Simulated Annealing.
    Uses single moves and 2-opt moves.
    """
    n = len(initial_chains[0])
    if n < 4:
        return initial_chains[0], _compute_energy(sequence, initial_chains[0])
    
    # Prepare replicas
    # We assume initial_chains are already diverse and sorted by energy or just a list
    replicas = [list(c) for c in initial_chains[:num_temps]]
    while len(replicas) < num_temps:
        # Duplicate best with perturbation
        best_idx = min(range(len(replicas)), key=lambda i: _compute_energy(sequence, replicas[i]))
        base = list(replicas[best_idx])
        # Perturb
        idx = rng.randint(1, n - 2)
        old = base[idx]
        prev = base[idx-1]
        next_pos = base[idx+1]
        occ = {p: i for i, p in enumerate(base)}
        candidates = []
        for site in _neighbours(prev):
            if site == old: continue
            if site in occ and occ[site] != idx: continue
            if sum(abs(site[a] - next_pos[a]) for a in range(len(site))) == 1:
                candidates.append(site)
        if candidates:
            base[idx] = rng.choice(candidates)
        replicas.append(base)
        
    replica_energies = [_compute_energy(sequence, c) for c in replicas]
    replica_occupied = [{p: i for i, p in enumerate(c)} for c in replicas]
    
    # Temperature ladder
    T_high = 3.0
    T_low = 0.1
    temps = []
    for k in range(num_temps):
        ratio = (T_low / T_high) ** (1.0 / max(num_temps - 1, 1))
        temps.append(T_high * (ratio ** k))
    
    best_chain = list(replicas[0])
    best_energy = replica_energies[0]
    
    # Main loop
    while time.time() < deadline:
        r = rng.randint(0, num_temps - 1)
        T_r = temps[r]
        
        # Choose move type: 0.7 Single, 0.3 2-opt
        if rng.random() < 0.7:
            # Single move
            idx = rng.randint(1, n - 2)
            old = replicas[r][idx]
            prev = replicas[r][idx-1]
            next_pos = replicas[r][idx+1]
            
            candidates = []
            for site in _neighbours(prev):
                if site == old: continue
                if site in replica_occupied[r] and replica_occupied[r][site] != idx: continue
                if sum(abs(site[a] - next_pos[a]) for a in range(len(site))) == 1:
                    candidates.append(site)
            
            if candidates:
                site = rng.choice(candidates)
                delta = _delta_energy_single(sequence, replicas[r], idx, old, site, replica_occupied[r])
                delta_e = -delta # Energy change
                
                if delta_e <= 0 or (T_r > 0 and rng.random() < math.exp(-delta_e / T_r)):
                    # Accept
                    replica_occupied[r][old] = -1
                    replica_occupied[r][site] = idx
                    replicas[r][idx] = site
                    replica_energies[r] += delta_e
        else:
            # 2-opt move (Reversal)
            i = rng.randint(1, n - 2)
            j = rng.randint(i + 1, n - 1)
            
            # Check connectivity at boundaries
            valid = True
            if i > 0:
                prev_pos = replicas[r][i-1]
                first_new = replicas[r][j]
                if sum(abs(prev_pos[a] - first_new[a]) for a in range(len(prev_pos))) != 1:
                    valid = False
            if valid and j < n - 1:
                next_pos = replicas[r][j+1]
                last_new = replicas[r][i]
                if sum(abs(next_pos[a] - last_new[a]) for a in range(len(next_pos))) != 1:
                    valid = False
            
            if valid:
                # Check self-avoidance
                # The segment i..j is reversed.
                # We need to ensure that the new positions (which are the old positions of the segment, just in reverse order)
                # do not collide with the outside part.
                # Actually, the positions are the same set, just the order is reversed.
                # Wait, in a lattice SAW, the *positions* of the residues in the segment remain the same, 
                # but the *indices* are swapped? No, the physical chain is reversed.
                # The set of occupied sites is identical. 
                # So self-avoidance is automatically preserved IF the chain was valid before.
                # The only constraints are the connectivity at the cut points (i-1, i) and (j, j+1).
                
                # Calculate energy change
                # The contacts within the segment are preserved (just reversed order).
                # The contacts between the segment and the outside change.
                
                # Outside set
                outside_indices = set(range(n)) - set(range(i, j+1))
                outside_occ = {replicas[r][k]: k for k in outside_indices}
                
                # Old contacts involving the segment
                old_cc = 0
                for k in range(i, j+1):
                    if sequence[k] == 'H':
                        pos = replicas[r][k]
                        for q in _neighbours(pos):
                            oj = outside_occ.get(q)
                            if oj is not None and sequence[oj] == 'H':
                                old_cc += 1
                
                # New contacts involving the segment
                # The segment is reversed: index k in old corresponds to index i+j-k in new.
                # The position at new index k is the old position at index i+j-k.
                new_cc = 0
                for k in range(i, j+1):
                    if sequence[k] == 'H':
                        new_pos = replicas[r][i+j-k]
                        for q in _neighbours(new_pos):
                            oj = outside_occ.get(q)
                            if oj is not None and sequence[oj] == 'H':
                                new_cc += 1
                                
                delta_e = -(new_cc - old_cc)
                
                if delta_e <= 0 or (T_r > 0 and rng.random() < math.exp(-delta_e / T_r)):
                    # Accept: reverse the segment
                    for k in range(i, j+1):
                        replicas[r][k] = replicas[r][i+j-k]
                    # Update occupied map
                    # Since positions are the same, we just need to update indices
                    for k in range(i, j+1):
                        replica_occupied[r][replicas[r][k]] = k
                    replica_energies[r] += delta_e
                    
        # Swap attempt
        if rng.random() < 0.5:
            r1 = rng.randint(0, num_temps - 1)
            r2 = rng.randint(0, num_temps - 1)
            if r1 != r2:
                T1, T2 = temps[r1], temps[r2]
                E1, E2 = replica_energies[r1], replica_energies[r2]
                delta_swap = (E1 / T2 + E2 / T1) - (E1 / T1 + E2 / T2)
                if delta_swap <= 0 or rng.random() < math.exp(-delta_swap):
                    replicas[r1], replicas[r2] = replicas[r2], replicas[r1]
                    replica_energies[r1], replica_energies[r2] = replica_energies[r2], replica_energies[r1]
                    replica_occupied[r1], replica_occupied[r2] = replica_occupied[r2], replica_occupied[r1]
        
        # Update best
        for r_idx in range(num_temps):
            if replica_energies[r_idx] < best_energy:
                best_energy = replica_energies[r_idx]
                best_chain = list(replicas[r_idx])
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Best conformation via diverse greedy growth + Replica Exchange SA."""
    n = len(sequence)
    rng = random.Random(42 + len(sequence) * 7 + sum(ord(c) for c in sequence))
    
    # Time budget
    # 26 sequences, 300s total. ~11s per sequence.
    # Use 10s to be safe.
    per_seq_deadline = time.time() + 10.0
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Generate diverse initial chains
    initial_chains = []
    
    # Generate chains with different biases and random seeds
    biases = [0.0, 0.5, 1.0, 2.0, 0.0, 0.5, 1.0, 2.0]
    for i in range(min(8, max(2, n // 5))): # Scale number of starts with length
        if time.time() > per_seq_deadline - 5.0:
            break
        bias = biases[i % len(biases)]
        chain = _grow_greedy(sequence, dimension, rng, bias=bias)
        if chain:
            initial_chains.append(chain)
            
    if not initial_chains:
        # Fallback
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
        return [list(p) for p in best_chain]
        
    # Evaluate and sort
    for chain in initial_chains:
        e = _compute_energy(sequence, chain)
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e
            
    # Phase 2: Replica Exchange SA
    if time.time() < per_seq_deadline - 1.0:
        # Use top chains for replicas
        scored = [(_compute_energy(sequence, c), c) for c in initial_chains]
        scored.sort()
        top_chains = [c for e, c in scored[:4]]
        
        best_chain, best_energy = _replica_exchange_sa(
            sequence, top_chains, rng, per_seq_deadline - 1.0, num_temps=4
        )
        
    return [list(p) for p in best_chain]
