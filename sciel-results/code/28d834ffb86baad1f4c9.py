"""HP lattice protein folding on cubic lattice - diverse growth + intensive local search."""
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


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0, 0)]
    
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Compactness bias: prefer sites closer to centroid
            if len(chain) > 5:
                cx = sum(p[0] for p in chain) / len(chain)
                cy = sum(p[1] for p in chain) / len(chain)
                cz = sum(p[2] for p in chain) / len(chain)
                dist = abs(site[0] - cx) + abs(site[1] - cy) + abs(site[2] - cz)
                gain -= dist * 0.3
            score = gain + 0.1 * free + rng.random() * 0.5
            candidates.append((score, site))
        
        if not candidates:
            # Fallback: try to extend in a direction that doesn't collide
            for site in _neighbours(last):
                if site not in occupied:
                    chain.append(site)
                    occupied[site] = i
                    break
            else:
                return None
        
        best_site = max(candidates, key=lambda x: x[0])[1]
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain


def _local_search(sequence, chain, occupied, deadline, rng):
    """Hill-climbing with single-residue moves and 2-opt moves."""
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_occupied = dict(occupied)
    best_e = _compute_energy(sequence, best_chain)
    
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        
        # Try single-residue moves
        for i in range(1, n - 1):
            old = best_chain[i]
            prev = best_chain[i - 1]
            next_pos = best_chain[i + 1]
            
            # Find valid candidate sites: neighbor of prev AND neighbor of next_pos
            candidates = []
            for site in _neighbours(prev):
                if site == old:
                    continue
                if site in best_occupied and best_occupied[site] != i:
                    continue
                dist = sum(abs(site[a] - next_pos[a]) for a in range(3))
                if dist == 1:
                    candidates.append(site)
            
            if not candidates:
                continue
            
            # Try each candidate, take the best improving move
            best_delta = 0
            best_site = None
            for site in candidates:
                delta = _delta_energy_move(sequence, best_chain, i, old, site, best_occupied)
                if delta > best_delta:
                    best_delta = delta
                    best_site = site
            
            if best_site is not None and best_delta > 0:
                # Apply move
                best_occupied[old] = -1
                best_occupied[best_site] = i
                best_chain[i] = best_site
                best_e -= best_delta
                improved = True
                no_improve_count = 0
                # Restart scanning from beginning for safety
                break
        
        if improved:
            continue
        
        # Try 2-opt moves (segment reversal)
        if n > 6:
            # Randomly try a few 2-opt moves
            for _ in range(3):
                i = rng.randint(1, n - 3)
                j = rng.randint(i + 1, n - 2)
                
                segment = best_chain[i:j+1]
                new_segment = list(reversed(segment))
                
                # Validate connectivity
                valid = True
                if i > 0:
                    prev_pos = best_chain[i-1]
                    first_new = new_segment[0]
                    if sum(abs(prev_pos[a] - first_new[a]) for a in range(3)) != 1:
                        valid = False
                if valid and j < n - 1:
                    next_pos = best_chain[j+1]
                    last_new = new_segment[-1]
                    if sum(abs(next_pos[a] - last_new[a]) for a in range(3)) != 1:
                        valid = False
                
                if not valid:
                    continue
                
                # Check self-avoidance
                occ_set = set()
                for idx, pos in enumerate(best_chain):
                    if idx < i or idx > j:
                        occ_set.add(pos)
                for pos in new_segment:
                    if pos in occ_set:
                        break
                else:
                    # Compute energy delta
                    occ_map = {}
                    for idx, pos in enumerate(best_chain):
                        if idx < i or idx > j:
                            occ_map[pos] = idx
                    
                    old_cc = 0
                    for idx in range(i, j+1):
                        if sequence[idx] == 'H':
                            pos = best_chain[idx]
                            for q in _neighbours(pos):
                                oj = occ_map.get(q)
                                if oj is not None and sequence[oj] == 'H':
                                    old_cc += 1
                    
                    new_cc = 0
                    for idx in range(i, j+1):
                        if sequence[idx] == 'H':
                            new_pos = best_chain[i+j-idx]
                            for q in _neighbours(new_pos):
                                oj = occ_map.get(q)
                                if oj is not None and sequence[oj] == 'H':
                                    new_cc += 1
                    
                    delta = new_cc - old_cc
                    if delta > 0:
                        # Apply 2-opt
                        for k in range(j - i + 1):
                            best_chain[i + k] = new_segment[k]
                        best_occupied = {p: idx for idx, p in enumerate(best_chain)}
                        best_e -= delta
                        improved = True
                        no_improve_count = 0
                        break
        
        if not improved:
            no_improve_count += 1
    
    return best_chain, best_e


def fold(sequence, dimension):
    """Best conformation via diverse growth + intensive local search."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    # Seed RNG for reproducibility
    seed = 42 + len(sequence) * 7 + sum(ord(c) for c in sequence)
    rng = random.Random(seed)
    
    # Time budget: use up to 11s per sequence, but track global deadline
    per_seq_deadline = min(time.time() + 11.0, _HARD_DEADLINE - 5.0)
    
    # Phase 1: Generate many diverse initial chains via randomized growth
    # Growth is O(N^2) and fast, so we can do many restarts
    initial_chains = []
    growth_deadline = per_seq_deadline - 4.0  # Leave 4s for local search
    
    # Do at least 5 growths, or as many as fit in growth_deadline
    min_growths = 5
    max_growths = 50
    
    count = 0
    while count < max_growths and time.time() < growth_deadline:
        chain = _grow(sequence, dimension, rng)
        if chain is not None and len(chain) == n:
            initial_chains.append(chain)
        count += 1
    
    # Ensure we have at least one chain
    if not initial_chains:
        initial_chains.append([(i, 0, 0) for i in range(n)])
    
    # Phase 2: Local search on top-k diverse chains
    # Rank by energy, take top-k for intensive polishing
    # k depends on remaining time and chain length
    remaining_for_search = per_seq_deadline - time.time()
    
    # Sort by energy
    for chain in initial_chains:
        chain_e = _compute_energy(sequence, chain)
        chain.append  # placeholder
        initial_chains[initial_chains.index(chain)] = (chain, chain_e)
    
    # Actually, let's do this more cleanly
    scored_chains = []
    for chain in initial_chains:
        e = _compute_energy(sequence, chain)
        scored_chains.append((e, chain))
    
    scored_chains.sort(key=lambda x: x[0])  # Sort by energy (most negative first)
    
    # Take top-k for local search
    # More time -> more chains to polish
    if remaining_for_search > 6.0:
        k = min(5, len(scored_chains))
    elif remaining_for_search > 3.0:
        k = min(3, len(scored_chains))
    else:
        k = min(2, len(scored_chains))
    
    # Local search on each top chain, with time budget per chain
    best_chain = None
    best_energy = float('inf')
    
    search_deadline = per_seq_deadline - 1.0  # Leave 1s safety margin
    per_chain_time = max(0.5, (search_deadline - time.time()) / k)
    
    for rank in range(k):
        e, chain = scored_chains[rank]
        chain_occupied = {p: i for i, p in enumerate(chain)}
        
        chain_deadline = min(time.time() + per_chain_time, search_deadline)
        if time.time() >= chain_deadline:
            break
        
        polished_chain, polished_e = _local_search(sequence, chain, chain_occupied, chain_deadline, rng)
        
        if polished_e < best_energy:
            best_energy = polished_e
            best_chain = polished_chain
    
    # If no chain was polished (shouldn't happen), use the best initial
    if best_chain is None:
        best_chain = scored_chains[0][1]
        best_energy = scored_chains[0][0]
    
    # Verify self-avoidance
    if len(set(best_chain)) != n:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]
