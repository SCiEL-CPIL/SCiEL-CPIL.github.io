"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

# Global deadline guard: must finish well before 330s wall-clock.
_TOTAL_START = time.time()
_SAFETY_MARGIN = 15.0  # seconds to leave before hard kill

def _remaining_time():
    """Return remaining seconds before our safety deadline."""
    return _TOTAL_START + 300.0 - _SAFETY_MARGIN - time.time()

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

def _grow(sequence, dimension, rng, temp=0.0):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            score = gain * 5.0 + free * 0.3 + rng.random() * (0.5 + temp)
            options.append((score, site))
        if not options:
            return None
        if temp > 0 and rng.random() < 0.2:
            options.sort(key=lambda x: -x[0])
            k = min(len(options), 1 + int(rng.expovariate(1.0 / max(0.5, temp))))
            site = options[rng.randint(0, k - 1)][1]
        else:
            site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _local_search(sequence, chain, rng, deadline):
    """
    Hill-climbing with residue moves. Each move relocates one internal residue
    to a site that is a neighbor of both its predecessor and successor.
    This preserves connectivity by construction, making validity O(1).
    """
    n = len(chain)
    if n < 3:
        return list(chain), _energy(sequence, chain)
    
    best = list(chain)
    best_e = _energy(sequence, best)
    occupied = {p: i for i, p in enumerate(best)}
    
    # Precompute neighbor sets for all positions for fast intersection
    # Actually, we'll compute on the fly since chains are short
    
    while time.time() < deadline:
        improved = False
        # Try moving each residue in random order for diversity
        indices = list(range(1, n - 1))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() >= deadline:
                break
            
            old = best[i]
            prev = best[i-1]
            next_pos = best[i+1]
            
            # Find valid candidate sites: must be neighbor of prev AND next_pos
            # AND not occupied by any other residue (or is the current position)
            candidates = set()
            for site in _neighbours(prev):
                for site2 in _neighbours(next_pos):
                    if site == site2:
                        if site != old:
                            # Check if site is free
                            occ_idx = occupied.get(site)
                            if occ_idx is None or occ_idx == i:
                                candidates.add(site)
                        break
            
            if not candidates:
                continue
            
            for site in candidates:
                # Calculate energy delta
                delta = 0
                if sequence[i] == 'H':
                    # Lost contacts: neighbors of old that are H and not i
                    for q in _neighbours(old):
                        j = occupied.get(q)
                        if j is not None and j != i:
                            if sequence[j] == 'H':
                                delta -= 1
                    # Gained contacts: neighbors of site that are H and not i
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j != i:
                            if sequence[j] == 'H':
                                delta += 1
                
                if delta > 0:  # Energy decreases (more negative)
                    # Update state
                    del occupied[old]
                    occupied[site] = i
                    best[i] = site
                    best_e -= delta
                    improved = True
                    break  # Move to next residue
        
        if not improved:
            break
            
    return best, best_e

def _divide_and_conquer_fold(sequence, dimension, rng, deadline):
    """
    Divide-and-conquer: split sequence into blocks, fold each independently,
    then assemble with local search on the full chain.
    """
    n = len(sequence)
    
    # For short chains, just do direct folding
    if n <= 30:
        best_chain = None
        best_energy = 0
        while time.time() < deadline:
            chain = _grow(sequence, dimension, rng, rng.uniform(0.0, 0.5))
            if chain is None:
                continue
            value = _energy(sequence, chain)
            if value < best_energy:
                best_chain = chain
                best_energy = value
        if best_chain is None:
            best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
            best_energy = _energy(sequence, best_chain)
        if time.time() < deadline - 1.0:
            best_chain, best_energy = _local_search(sequence, best_chain, rng, deadline)
        return best_chain, best_energy
    
    # For longer chains, divide into blocks
    # Block size: aim for 12-20 residues per block
    block_size = min(16, max(10, n // 4))
    blocks = []
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        blocks.append((start, end))
    
    # Fold each block independently
    block_chains = []
    for (start, end) in blocks:
        sub_seq = sequence[start:end]
        best_block = None
        best_block_e = 0
        # Time budget per block: divide remaining time
        block_deadline = time.time() + max(0.5, (deadline - time.time()) / len(blocks) * 0.6)
        
        while time.time() < block_deadline:
            chain = _grow(sub_seq, dimension, rng, rng.uniform(0.0, 0.3))
            if chain is None:
                continue
            value = _energy(sub_seq, chain)
            if value < best_block_e:
                best_block = chain
                best_block_e = value
        
        if best_block is None:
            # Fallback: straight line for this block
            best_block = [tuple([i] + [0] * (dimension - 1)) for i in range(end - start)]
        
        block_chains.append(best_block)
    
    # Assemble blocks: place first block at origin, then position subsequent blocks
    # to minimize clashes and maximize inter-block H-H contacts
    full_chain = []
    occupied = set()
    
    for b_idx, block in enumerate(block_chains):
        if b_idx == 0:
            # Place first block at origin
            for p in block:
                full_chain.append(p)
                occupied.add(p)
        else:
            # Try to place this block near the end of the previous block
            # The last point of the previous block is the connection point
            prev_end = full_chain[-1]
            
            # Try several translations to find a non-clashing position
            # that maximizes H-H contacts with the existing chain
            best_offset = None
            best_score = -999999
            best_placed = None
            
            # Search offsets in a neighborhood of the previous end
            search_radius = 3
            for dx in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
                    for dz in range(-search_radius, search_radius + 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        offset = (prev_end[0] + dx, prev_end[1] + dy, prev_end[2] + dz)
                        
                        # Check if this placement is valid (no clashes)
                        valid = True
                        placed = []
                        for p in block:
                            new_p = (p[0] + offset[0], p[1] + offset[1], p[2] + offset[2])
                            if new_p in occupied:
                                valid = False
                                break
                            placed.append(new_p)
                        
                        if not valid:
                            continue
                        
                        # Check connectivity: first point of block must be adjacent to last point of prev
                        if placed[0] != prev_end:
                            diff = sum(abs(placed[0][d] - prev_end[d]) for d in range(dimension))
                            if diff != 1:
                                continue
                        
                        # Score: count H-H contacts between this block and existing chain
                        score = 0
                        for p in placed:
                            if sequence[full_chain.__len__() + block_chains[b_idx-1].__len__() + block.index(p)] == 'H':
                                for q in _neighbours(p):
                                    if q in occupied:
                                        # Find which residue is at q
                                        for idx, op in enumerate(full_chain):
                                            if op == q:
                                                if sequence[idx] == 'H' and idx != len(full_chain) + block.index(p) - 1:
                                                    score += 1
                                                break
                        
                        if score > best_score:
                            best_score = score
                            best_offset = offset
                            best_placed = placed
            
            if best_placed is None:
                # Fallback: place in a spiral outward
                placed = []
                for p in block:
                    new_p = (prev_end[0] + p[0], prev_end[1] + p[1], prev_end[2] + p[2])
                    # If clash, try shifting
                    while new_p in occupied:
                        new_p = (new_p[0] + 1, new_p[1], new_p[2])
                    placed.append(new_p)
            else:
                placed = best_placed
            
            for p in placed:
                full_chain.append(p)
                occupied.add(p)
    
    # Now do local search on the assembled chain
    if time.time() < deadline - 0.5:
        full_chain, best_e = _local_search(sequence, full_chain, rng, deadline)
    else:
        best_e = _energy(sequence, full_chain)
    
    return full_chain, best_e

def fold(sequence, dimension):
    """
    Best conformation using:
    1. Multiple greedy growths with varying temperatures
    2. Divide-and-conquer for longer chains
    3. Residue-move local search for refinement
    """
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    deadline = _TOTAL_START + 300.0 - _SAFETY_MARGIN
    time_left = deadline - time.time()
    
    if time_left <= 0:
        # Out of time, return straight line
        return [list(p) for p in [tuple([i] + [0] * (dimension - 1)) for i in range(n)]]
    
    best_chain = None
    best_energy = 0
    
    # Strategy selection based on chain length
    if n <= 48:
        # Direct approach: many greedy growths + local search
        # Allocate 70% of time to growth, 30% to local search
        growth_deadline = time.time() + time_left * 0.7
        search_deadline = deadline
        
        while time.time() < growth_deadline:
            temp = rng.uniform(0.0, 0.5)
            chain = _grow(sequence, dimension, rng, temp)
            if chain is None:
                continue
            value = _energy(sequence, chain)
            if value < best_energy:
                best_chain = chain
                best_energy = value
        
        if best_chain is None:
            best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
            best_energy = _energy(sequence, best_chain)
        
        if time.time() < search_deadline - 0.5:
            best_chain, best_energy = _local_search(sequence, best_chain, rng, search_deadline)
    
    elif n <= 80:
        # Moderate: mix of direct growth and divide-and-conquer
        # Try divide-and-conquer first, then refine with direct growth
        dc_deadline = time.time() + time_left * 0.4
        
        dc_chain, dc_energy = _divide_and_conquer_fold(sequence, dimension, rng, dc_deadline)
        if dc_chain is not None:
            best_chain = dc_chain
            best_energy = dc_energy
        
        # Then do direct growths on remaining time
        growth_deadline = time.time() + (deadline - time.time()) * 0.5
        while time.time() < growth_deadline:
            temp = rng.uniform(0.0, 0.4)
            chain = _grow(sequence, dimension, rng, temp)
            if chain is None:
                continue
            value = _energy(sequence, chain)
            if value < best_energy:
                best_chain = chain
                best_energy = value
        
        if best_chain is None:
            best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
            best_energy = _energy(sequence, best_chain)
        
        if time.time() < deadline - 0.5:
            best_chain, best_energy = _local_search(sequence, best_chain, rng, deadline)
    
    else:
        # Long chains: primarily divide-and-conquer
        dc_deadline = time.time() + time_left * 0.6
        
        dc_chain, dc_energy = _divide_and_conquer_fold(sequence, dimension, rng, dc_deadline)
        if dc_chain is not None:
            best_chain = dc_chain
            best_energy = dc_energy
        
        # Add some direct growths
        growth_deadline = time.time() + (deadline - time.time()) * 0.3
        while time.time() < growth_deadline:
            temp = rng.uniform(0.0, 0.3)
            chain = _grow(sequence, dimension, rng, temp)
            if chain is None:
                continue
            value = _energy(sequence, chain)
            if value < best_energy:
                best_chain = chain
                best_energy = value
        
        if best_chain is None:
            best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
            best_energy = _energy(sequence, best_chain)
        
        if time.time() < deadline - 0.5:
            best_chain, best_energy = _local_search(sequence, best_chain, rng, deadline)
    
    # Final fallback
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
