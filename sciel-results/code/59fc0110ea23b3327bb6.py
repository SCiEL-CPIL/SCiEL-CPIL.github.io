"""HP lattice protein folding - divide-and-conquer with block assembly."""
import random
import time
import sys

_DEADLINE = 320.0
_start_time = None


def _get_remaining_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _DEADLINE - (time.time() - _start_time)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: -number of H-H contacts on non-adjacent lattice neighbours."""
    grid = {}
    for i, p in enumerate(chain):
        grid[tuple(p)] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = grid.get(tuple(q))
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _energy_fast(sequence, chain):
    """Fast energy computation using a dict."""
    grid = {tuple(p): i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for axis in range(3):
                for step in (-1, 1):
                    q = list(p)
                    q[axis] += step
                    j = grid.get(tuple(q))
                    if j is not None and j > i + 1 and sequence[j] == "H":
                        contacts += 1
    return -contacts


def _greedy_grow_block(sequence, start_idx, end_idx, dimension, rng, fixed_neighbors=None):
    """
    Greedily grow a block from start_idx to end_idx-1.
    fixed_neighbors: dict mapping residue index to its fixed position (for boundary constraints)
    Returns list of positions for residues start_idx..end_idx-1, or None on failure.
    """
    n_block = end_idx - start_idx
    if n_block <= 0:
        return []
    
    # Determine starting position
    if fixed_neighbors and start_idx in fixed_neighbors:
        start_pos = fixed_neighbors[start_idx]
    else:
        # Try to start from a position that allows growth
        start_pos = (0, 0, 0)
    
    chain = [start_pos]
    occupied = {start_pos: start_idx}
    
    for i in range(start_idx + 1, end_idx):
        last = chain[-1]
        options = []
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            # Check if this site is a fixed neighbor that doesn't match
            if fixed_neighbors and i in fixed_neighbors:
                if site != fixed_neighbors[i]:
                    continue
            
            # Count new H-H contacts
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            
            # Count free neighbours
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            
            score = gain * 10.0 + free * 0.5 + rng.random() * 1.0
            options.append((score, site))
        
        if not options:
            return None
        
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def _local_search_block(sequence, chain, dimension, rng, max_iters=500):
    """Local search to improve a block's conformation."""
    best_chain = list(chain)
    n = len(best_chain)
    if n < 3:
        return best_chain
    
    best_energy = _energy_fast(sequence, best_chain)
    grid = {tuple(p): i for i, p in enumerate(best_chain)}
    
    for _ in range(max_iters):
        if _get_remaining_time() < 5.0:
            break
        
        improved = False
        i = rng.randint(1, n - 2)
        current = best_chain[i]
        
        for site in _neighbours(current):
            if tuple(site) in grid:
                continue
            
            prev = best_chain[i-1]
            next_ = best_chain[i+1]
            
            d1 = sum(abs(site[d] - prev[d]) for d in range(dimension))
            d2 = sum(abs(site[d] - next_[d]) for d in range(dimension))
            
            if d1 != 1 or d2 != 1:
                continue
            
            # Try the move
            old_pos = current
            best_chain[i] = site
            del grid[tuple(old_pos)]
            grid[tuple(site)] = i
            
            new_energy = _energy_fast(sequence, best_chain)
            if new_energy < best_energy:
                best_energy = new_energy
                improved = True
            else:
                # Revert
                best_chain[i] = old_pos
                del grid[tuple(site)]
                grid[tuple(old_pos)] = i
        
        if not improved:
            # Try a few more
            for _ in range(5):
                i = rng.randint(1, n - 2)
                current = best_chain[i]
                for site in _neighbours(current):
                    if tuple(site) in grid:
                        continue
                    prev = best_chain[i-1]
                    next_ = best_chain[i+1]
                    d1 = sum(abs(site[d] - prev[d]) for d in range(dimension))
                    d2 = sum(abs(site[d] - next_[d]) for d in range(dimension))
                    if d1 != 1 or d2 != 1:
                        continue
                    old_pos = current
                    best_chain[i] = site
                    del grid[tuple(old_pos)]
                    grid[tuple(site)] = i
                    new_energy = _energy_fast(sequence, best_chain)
                    if new_energy < best_energy:
                        best_energy = new_energy
                        improved = True
                    else:
                        best_chain[i] = old_pos
                        del grid[tuple(site)]
                        grid[tuple(old_pos)] = i
                    if improved:
                        break
                if improved:
                    break
    
    return best_chain


def _assemble_blocks(block_chains, sequence, dimension, rng):
    """
    Assemble pre-folded blocks into a single chain.
    block_chains: list of lists, each containing positions for a block.
    We need to connect them with unit steps.
    """
    if not block_chains:
        return []
    
    # Start with first block
    full_chain = list(block_chains[0])
    
    for block_idx in range(1, len(block_chains)):
        block = list(block_chains[block_idx])
        if not block:
            continue
        
        # We need to connect the last position of full_chain to the first position of block
        # with a path of unit steps. Try to find a valid connection.
        
        # First, check if they're already adjacent
        last_pos = full_chain[-1]
        first_pos = block[0]
        
        dist = sum(abs(last_pos[d] - first_pos[d]) for d in range(dimension))
        
        if dist == 1:
            full_chain.extend(block)
            continue
        
        # Need to find a path. Try translating the block to be adjacent.
        # Try all possible translations that make block[0] adjacent to last_pos
        found = False
        for axis in range(dimension):
            for step in (-1, 1):
                # Translate block so that block[0] is at last_pos + step in axis
                target_first = list(last_pos)
                target_first[axis] += step
                
                # Translation vector
                trans = [target_first[d] - block[0][d] for d in range(dimension)]
                
                # Apply translation
                translated_block = [[p[d] + trans[d] for d in range(dimension)] for p in block]
                
                # Check self-avoidance with existing chain
                occupied = set(tuple(p) for p in full_chain)
                valid = True
                for p in translated_block:
                    if tuple(p) in occupied:
                        valid = False
                        break
                    # Also check it's not adjacent to itself in a bad way
                if not valid:
                    continue
                
                # Check unit steps within translated block
                for k in range(len(translated_block) - 1):
                    d = sum(abs(translated_block[k][d2] - translated_block[k+1][d2]) for d2 in range(dimension))
                    if d != 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check connection: last_pos to translated_block[0]
                d_conn = sum(abs(last_pos[d] - translated_block[0][d]) for d in range(dimension))
                if d_conn != 1:
                    continue
                
                # Valid! Use this translation
                full_chain.extend(translated_block)
                found = True
                break
            
            if found:
                break
        
        if not found:
            # Fallback: just append with a straight line extension
            # This may not be valid, but we'll fix it later
            # Try to extend with a simple path
            for axis in range(dimension):
                for step in (-1, 1):
                    target = list(last_pos)
                    target[axis] += step
                    if tuple(target) not in set(tuple(p) for p in full_chain):
                        full_chain.append(target)
                        break
                if full_chain[-1] != last_pos:
                    break
            
            # Now try to connect to block
            # For simplicity, just try to place block starting from current end
            # This is a simplified fallback
            for p in block:
                # Try to place p relative to current end
                placed = False
                for axis in range(dimension):
                    for step in (-1, 1):
                        new_pos = list(full_chain[-1])
                        new_pos[axis] += step
                        if tuple(new_pos) not in set(tuple(q) for q in full_chain):
                            full_chain.append(new_pos)
                            placed = True
                            break
                    if placed:
                        break
                if not placed:
                    break
    
    return full_chain


def _full_local_search(sequence, chain, dimension, rng, max_iters=2000):
    """Full chain local search with single-residue moves."""
    best_chain = list(chain)
    n = len(best_chain)
    best_energy = _energy_fast(sequence, best_chain)
    grid = {tuple(p): i for i, p in enumerate(best_chain)}
    
    for _ in range(max_iters):
        if _get_remaining_time() < 10.0:
            break
        
        improved = False
        i = rng.randint(1, n - 2)
        current = best_chain[i]
        
        for site in _neighbours(current):
            if tuple(site) in grid:
                continue
            
            prev = best_chain[i-1]
            next_ = best_chain[i+1]
            
            d1 = sum(abs(site[d] - prev[d]) for d in range(dimension))
            d2 = sum(abs(site[d] - next_[d]) for d in range(dimension))
            
            if d1 != 1 or d2 != 1:
                continue
            
            old_pos = current
            best_chain[i] = site
            del grid[tuple(old_pos)]
            grid[tuple(site)] = i
            
            new_energy = _energy_fast(sequence, best_chain)
            if new_energy < best_energy:
                best_energy = new_energy
                improved = True
            else:
                best_chain[i] = old_pos
                del grid[tuple(site)]
                grid[tuple(old_pos)] = i
        
        if not improved:
            for _ in range(3):
                i = rng.randint(1, n - 2)
                current = best_chain[i]
                for site in _neighbours(current):
                    if tuple(site) in grid:
                        continue
                    prev = best_chain[i-1]
                    next_ = best_chain[i+1]
                    d1 = sum(abs(site[d] - prev[d]) for d in range(dimension))
                    d2 = sum(abs(site[d] - next_[d]) for d in range(dimension))
                    if d1 != 1 or d2 != 1:
                        continue
                    old_pos = current
                    best_chain[i] = site
                    del grid[tuple(old_pos)]
                    grid[tuple(site)] = i
                    new_energy = _energy_fast(sequence, best_chain)
                    if new_energy < best_energy:
                        best_energy = new_energy
                        improved = True
                    else:
                        best_chain[i] = old_pos
                        del grid[tuple(site)]
                        grid[tuple(old_pos)] = i
                    if improved:
                        break
                if improved:
                    break
    
    return best_chain, best_energy


def _sweep_polish(sequence, chain, dimension, rng):
    """Sweep through all residues trying to improve each one."""
    best_chain = list(chain)
    n = len(best_chain)
    best_energy = _energy_fast(sequence, best_chain)
    grid = {tuple(p): i for i, p in enumerate(best_chain)}
    
    for sweep in range(50):
        if _get_remaining_time() < 10.0:
            break
        
        improved = False
        for i in range(1, n - 1):
            current = best_chain[i]
            prev = best_chain[i-1]
            next_ = best_chain[i+1]
            
            for site in _neighbours(current):
                if tuple(site) in grid:
                    continue
                
                d1 = sum(abs(site[d] - prev[d]) for d in range(dimension))
                d2 = sum(abs(site[d] - next_[d]) for d in range(dimension))
                
                if d1 != 1 or d2 != 1:
                    continue
                
                old_pos = current
                best_chain[i] = site
                del grid[tuple(old_pos)]
                grid[tuple(site)] = i
                
                new_energy = _energy_fast(sequence, best_chain)
                if new_energy < best_energy:
                    best_energy = new_energy
                    improved = True
                else:
                    best_chain[i] = old_pos
                    del grid[tuple(site)]
                    grid[tuple(old_pos)] = i
                
                if improved:
                    break
        
        if not improved:
            break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold using divide-and-conquer: split into blocks, fold each independently,
    then assemble and polish.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = len(sequence)
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0
    
    # Strategy 1: Full greedy growth (baseline)
    for restart in range(30):
        if _get_remaining_time() < 30.0:
            break
        chain = _greedy_grow_block(sequence, 0, n, dimension, rng)
        if chain is not None:
            chain = _local_search_block(sequence, chain, dimension, rng, max_iters=200)
            e = _energy_fast(sequence, chain)
            if e < best_energy:
                best_chain = chain
                best_energy = e
    
    # Strategy 2: Divide and conquer
    # Split into blocks of size ~12-20
    block_sizes = []
    remaining = n
    while remaining > 0:
        if remaining <= 15:
            block_sizes.append(remaining)
            remaining = 0
        else:
            bs = 12 + rng.randint(0, 8)
            block_sizes.append(bs)
            remaining -= bs
    
    n_attempts = 5 if n <= 50 else 3
    
    for attempt in range(n_attempts):
        if _get_remaining_time() < 20.0:
            break
        
        # Fold each block independently
        block_chains = []
        success = True
        for bs in block_sizes:
            if not block_chains:
                block = _greedy_grow_block(sequence, 0, bs, dimension, rng)
            else:
                start = sum(block_sizes[:len(block_chains)])
                end = start + bs
                block = _greedy_grow_block(sequence, start, end, dimension, rng)
            
            if block is None:
                success = False
                break
            
            # Local search on block
            block = _local_search_block(sequence, start, end, dimension, rng, max_iters=100) if len(block) > 2 else block
            block_chains.append(block)
        
        if not success or not block_chains:
            continue
        
        # Assemble
        assembled = _assemble_blocks(block_chains, sequence, dimension, rng)
        if len(assembled) != n:
            continue
        
        # Verify validity
        valid = True
        seen = set()
        for p in assembled:
            if tuple(p) in seen:
                valid = False
                break
            seen.add(tuple(p))
        if valid:
            for k in range(n - 1):
                d = sum(abs(assembled[k][d2] - assembled[k+1][d2]) for d2 in range(dimension))
                if d != 1:
                    valid = False
                    break
        
        if not valid:
            continue
        
        e = _energy_fast(sequence, assembled)
        if e < best_energy:
            best_chain = assembled
            best_energy = e
        
        # Polish the assembled chain
        if _get_remaining_time() > 15.0:
            polished, pe = _full_local_search(sequence, assembled, dimension, rng, max_iters=500)
            if pe < best_energy:
                best_chain = polished
                best_energy = pe
    
    # Final polish on best chain
    if best_chain is not None and _get_remaining_time() > 10.0:
        best_chain, best_energy = _sweep_polish(sequence, best_chain, dimension, rng)
    
    # Fallback
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
