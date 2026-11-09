"""HP lattice protein folding on the cubic lattice.
Combines randomized greedy growth with tail re-growth and pivot-move local search.
"""
import random
import time

# Global deadline: evaluator kills at 330s wall-clock. Use 320s safety margin.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0


def _neighbours(point):
    """Yield the 6 face-neighbours of point in 3D."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def _energy(sequence, chain):
    """Compute the total energy of the chain. Energy = -num_HH_contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
    """
    One randomized greedy chain growth.
    Heuristic: prefer sites with new H-H contacts, then free space, with small noise.
    """
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    for i in range(1, n):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            # Free space to avoid dead ends
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Score: contacts weighted heavily, free space secondary, noise for diversity
            score = 2.0 * gain + 0.3 * free + rng.random() * 0.8
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best option with some randomness among top candidates
        options.sort(key=lambda x: -x[0])
        if len(options) > 1 and rng.random() < 0.3:
            # Sometimes pick second-best for diversity
            best_site = options[1][1]
        else:
            best_site = options[0][1]
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain


def _regrow_tail(sequence, chain, i, rng):
    """
    Re-grow the tail from index i onwards, given the prefix up to i-1.
    Returns (new_chain, new_energy) or (None, None) if stuck.
    """
    prefix = chain[:i]
    occupied = {p: idx for idx, p in enumerate(prefix)}
    new_chain = list(prefix)
    n = len(sequence)
    
    for k in range(i, n):
        last = new_chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[k] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < k - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = 2.0 * gain + 0.3 * free + rng.random() * 0.8
            options.append((score, site))
        
        if not options:
            return None, None
        
        options.sort(key=lambda x: -x[0])
        if len(options) > 1 and rng.random() < 0.3:
            best_site = options[1][1]
        else:
            best_site = options[0][1]
        new_chain.append(best_site)
        occupied[best_site] = k
    
    energy = _energy(sequence, new_chain)
    return new_chain, energy


def _pivot_move(sequence, chain, rng):
    """
    Perform a pivot move: rotate a sub-chain around a pivot point.
    Returns (new_chain, new_energy) or (None, None) if invalid.
    """
    n = len(sequence)
    if n < 4:
        return None, None
    
    # Choose pivot position (not at ends)
    pivot = rng.randint(1, n - 2)
    
    # Choose rotation axis (one of the 6 directions)
    axes = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    axis = axes[rng.randint(0, 5)]
    
    # Get the sub-chain to rotate (from pivot to end)
    subchain = chain[pivot:]
    
    # Build occupied set from prefix
    prefix = chain[:pivot]
    occupied = {p: i for i, p in enumerate(prefix)}
    
    # Rotate subchain around pivot along axis
    # The pivot stays fixed; subsequent points are rotated
    new_subchain = [subchain[0]]  # pivot stays
    valid = True
    for k in range(1, len(subchain)):
        old_pt = subchain[k]
        # Translate to pivot-relative coordinates
        rel = (old_pt[0] - subchain[0][0], old_pt[1] - subchain[0][1], old_pt[2] - subchain[0][2])
        # Apply rotation: this is a simple reflection/translation
        # For a pivot move, we rotate the entire tail by 180 degrees around an axis through the pivot
        # Actually, let's do a proper pivot: move the tail to a different direction
        # Simpler: try moving the next residue in a different direction
        pass
    
    # Alternative simpler pivot: just try to re-route from pivot
    # This is essentially a tail re-growth, so let's do something different
    # Try a "crankshaft" style: pick two non-adjacent residues and try to swap their local configurations
    
    # Let's implement a proper pivot: choose a pivot, and for the tail after pivot,
    # try rotating it 180 degrees around an axis defined by two bonds
    if n < 5:
        return None, None
    
    # Pick pivot at position p, and look at bond from p-1 to p and p to p+1
    p = rng.randint(1, n - 2)
    if p >= n - 1:
        return None, None
    
    # The tail starts at p+1. We want to rotate the tail around the bond (p-1, p)
    # Actually, standard pivot: rotate everything from position p onwards around position p
    # Let's try: for each possible new direction from position p, check if it's valid
    
    # Get current direction from p-1 to p
    d_in = (chain[p][0] - chain[p-1][0], chain[p][1] - chain[p-1][1], chain[p][2] - chain[p-1][2])
    
    # Possible new directions (not the reverse of d_in)
    possible_dirs = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                if dx + dy + dz != 1 and dx + dy + dz != -1:
                    continue
                # Not the reverse of incoming direction
                if dx == -d_in[0] and dy == -d_in[1] and dz == -d_in[2]:
                    continue
                possible_dirs.append((dx, dy, dz))
    
    if not possible_dirs:
        return None, None
    
    # Pick a random new direction
    new_dir = possible_dirs[rng.randint(0, len(possible_dirs) - 1)]
    
    # Build new chain: prefix up to p, then extend from p in new_dir
    # This is essentially a tail re-growth from p+1 with a bias toward new_dir
    prefix = chain[:p+1]
    occupied = {pt: i for i, pt in enumerate(prefix)}
    new_chain = list(prefix)
    
    # Grow from p+1 onward, with initial bias toward new_dir
    for k in range(p+1, n):
        last = new_chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[k] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < k - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Bonus if this site is in the preferred direction from the pivot
            if k == p+1:
                expected = (last[0] + new_dir[0], last[1] + new_dir[1], last[2] + new_dir[2])
                if site == expected:
                    gain += 1.0  # Bonus for following the pivot direction
            
            score = 2.0 * gain + 0.3 * free + rng.random() * 0.8
            options.append((score, site))
        
        if not options:
            return None, None
        
        options.sort(key=lambda x: -x[0])
        if len(options) > 1 and rng.random() < 0.2:
            best_site = options[1][1]
        else:
            best_site = options[0][1]
        new_chain.append(best_site)
        occupied[best_site] = k
    
    energy = _energy(sequence, new_chain)
    return new_chain, energy


def fold(sequence, dimension):
    """
    Fold the sequence on the cubic lattice.
    Phase 1: Multiple randomized greedy growths.
    Phase 2: Local optimization via tail re-growth and pivot moves.
    Phase 3: Final polish with aggressive local search.
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    # Check remaining time
    time_remaining = _DEADLINE - time.time()
    if time_remaining < 1.0:
        return [[i, 0, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    # Phase 1: Multiple greedy growths (30% of time)
    phase1_budget = time_remaining * 0.3
    est_growth_time = n * 0.0015
    num_growths = max(5, int(phase1_budget / est_growth_time))
    num_growths = min(num_growths, 3000)
    
    for _ in range(num_growths):
        if time.time() > _DEADLINE - 15:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = 0
    
    # Phase 2: Local optimization (50% of remaining time)
    current_chain = list(best_chain)
    current_energy = best_energy
    
    time_remaining = _DEADLINE - time.time()
    if time_remaining < 1.0:
        return [list(p) for p in best_chain]
    
    phase2_budget = time_remaining * 0.5
    # Estimate time per local move
    est_move_time = (n * 0.001)  # rough estimate
    num_moves = max(10, int(phase2_budget / est_move_time))
    num_moves = min(num_moves, 100000)
    
    no_improve_count = 0
    for step in range(num_moves):
        if time.time() > _DEADLINE - 10:
            break
        
        # Choose move type: 50% tail regrowth, 50% pivot
        if rng.random() < 0.5:
            # Tail re-growth
            if rng.random() < 0.4:
                i = rng.randint(1, max(1, n // 3))
            elif rng.random() < 0.7:
                i = rng.randint(n // 3, max(n // 3 + 1, n * 2 // 3))
            else:
                i = rng.randint(n * 2 // 3, n - 1)
            i = min(i, n - 1)
            new_chain, new_energy = _regrow_tail(sequence, current_chain, i, rng)
        else:
            # Pivot move
            new_chain, new_energy = _pivot_move(sequence, current_chain, rng)
        
        if new_chain is None:
            no_improve_count += 1
            continue
        
        delta = new_energy - current_energy
        if delta < 0:
            # Improvement
            current_chain = new_chain
            current_energy = new_energy
            no_improve_count = 0
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        else:
            # Worse or same: accept with probability based on "temperature"
            # Simple geometric cooling
            frac = step / max(1, num_moves)
            temp = 3.0 * (1.0 - frac)
            if rng.random() < min(1.0, temp / max(1.0, abs(delta))):
                current_chain = new_chain
                current_energy = new_energy
                no_improve_count += 1
            else:
                no_improve_count += 1
        
        # If stuck for too long, try a bigger restart
        if no_improve_count > 200:
            # Do a fresh growth and use it as new starting point if better
            fresh_chain = _grow(sequence, dimension, rng)
            if fresh_chain is not None:
                fresh_energy = _energy(sequence, fresh_chain)
                if fresh_energy < current_energy:
                    current_chain = fresh_chain
                    current_energy = fresh_energy
            no_improve_count = 0
    
    # Phase 3: Final polish (remaining time, 10% safety margin)
    time_remaining = _DEADLINE - time.time()
    if time_remaining > 2.0:
        # Do a few more aggressive tail re-growths on the best chain
        polish_chain = list(best_chain)
        polish_energy = best_energy
        est_polish_time = n * 0.001
        num_polish = max(5, int((time_remaining - 2.0) / est_polish_time))
        num_polish = min(num_polish, 5000)
        
        for _ in range(num_polish):
            if time.time() > _DEADLINE - 3:
                break
            i = rng.randint(1, n - 1)
            new_chain, new_energy = _regrow_tail(sequence, polish_chain, i, rng)
            if new_chain is not None and new_energy < polish_energy:
                polish_chain = new_chain
                polish_energy = new_energy
        
        if polish_energy < best_energy:
            best_energy = polish_energy
            best_chain = polish_chain
    
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
