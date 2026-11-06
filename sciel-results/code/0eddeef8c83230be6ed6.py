"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = time.time() + 318  # safety margin before 330s kill


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _initial_energy(sequence, chain):
    """Calculate energy for a chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
    """Randomized greedy chain growth."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            trap_penalty = -0.5 if free == 1 else 0.0
            options.append((gain + 0.1 * free + trap_penalty + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_refine(sequence, chain, dimension, rng, max_iters=2000):
    """
    Local search with incremental energy updates.
    Moves: 1-move, Kink, Jump.
    """
    if len(chain) < 4:
        return chain

    n = len(chain)
    best_chain = list(chain)
    current_chain = list(chain)
    
    occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = _initial_energy(sequence, current_chain)
    best_energy = current_energy
    
    # Precompute H positions for faster lookup
    h_positions = set(i for i in range(n) if sequence[i] == 'H')
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 3:
            break
            
        improved = False
        move_type = rng.random()
        
        if move_type < 0.45:
            # 1-residue move
            j = rng.randint(1, n - 2)
            old_pos = current_chain[j]
            for new_pos in _neighbours(old_pos):
                if new_pos in occupied:
                    continue
                prev_pos = current_chain[j-1]
                next_pos = current_chain[j+1]
                
                if abs(new_pos[0]-prev_pos[0]) + abs(new_pos[1]-prev_pos[1]) + abs(new_pos[2]-prev_pos[2]) != 1:
                    continue
                if abs(new_pos[0]-next_pos[0]) + abs(new_pos[1]-next_pos[1]) + abs(new_pos[2]-next_pos[2]) != 1:
                    continue
                
                delta_e = 0
                if sequence[j] == 'H':
                    for q in _neighbours(old_pos):
                        idx_q = occupied.get(q)
                        if idx_q is not None and idx_q != j and abs(idx_q - j) > 1 and sequence[idx_q] == 'H':
                            delta_e -= 1
                    for q in _neighbours(new_pos):
                        idx_q = occupied.get(q)
                        if idx_q is not None and idx_q != j and abs(idx_q - j) > 1 and sequence[idx_q] == 'H':
                            delta_e += 1
                
                new_energy = current_energy + delta_e
                
                if new_energy < best_energy:
                    current_chain[j] = new_pos
                    del occupied[old_pos]
                    occupied[new_pos] = j
                    current_energy = new_energy
                    best_energy = new_energy
                    best_chain = current_chain[:]
                    improved = True
                    break
                elif new_energy == current_energy and rng.random() < 0.3:
                    current_chain[j] = new_pos
                    del occupied[old_pos]
                    occupied[new_pos] = j
                    current_energy = new_energy
                    improved = True
                    break

        elif move_type < 0.70:
            # Kink move: move residues i, i+1 together
            if n >= 4:
                i = rng.randint(1, n - 3)
                p1_old = current_chain[i]
                p2_old = current_chain[i+1]
                
                # Try moving the pair to a new configuration
                # p1_new must be neighbor of chain[i-1]
                # p2_new must be neighbor of p1_new
                # chain[i+2] must be neighbor of p2_new
                
                prev_pos = current_chain[i-1]
                next_pos = current_chain[i+2]
                
                for p1_new in _neighbours(prev_pos):
                    if p1_new in occupied:
                        continue
                    # p1_new cannot be same as p2_old or chain[i-1]
                    if p1_new == p2_old:
                        continue
                    for p2_new in _neighbours(p1_new):
                        if p2_new in occupied:
                            continue
                        if p2_new == p1_old:
                            continue
                        # Check if next_pos is neighbor of p2_new
                        if abs(p2_new[0]-next_pos[0]) + abs(p2_new[1]-next_pos[1]) + abs(p2_new[2]-next_pos[2]) != 1:
                            continue
                        
                        # Calculate energy delta for both residues
                        delta_e = 0
                        
                        # Residue i
                        if sequence[i] == 'H':
                            for q in _neighbours(p1_old):
                                idx_q = occupied.get(q)
                                if idx_q is not None and idx_q != i and idx_q != i+1 and abs(idx_q - i) > 1 and sequence[idx_q] == 'H':
                                    delta_e -= 1
                            for q in _neighbours(p1_new):
                                idx_q = occupied.get(q)
                                if idx_q is not None and idx_q != i and idx_q != i+1 and abs(idx_q - i) > 1 and sequence[idx_q] == 'H':
                                    delta_e += 1
                        
                        # Residue i+1
                        if sequence[i+1] == 'H':
                            for q in _neighbours(p2_old):
                                idx_q = occupied.get(q)
                                if idx_q is not None and idx_q != i and idx_q != i+1 and abs(idx_q - (i+1)) > 1 and sequence[idx_q] == 'H':
                                    delta_e -= 1
                            for q in _neighbours(p2_new):
                                idx_q = occupied.get(q)
                                if idx_q is not None and idx_q != i and idx_q != i+1 and abs(idx_q - (i+1)) > 1 and sequence[idx_q] == 'H':
                                    delta_e += 1
                        
                        # Also check if p1_new and p2_new form a new H-H contact between i and i+1
                        # (they are adjacent in chain, so this doesn't count)
                        
                        new_energy = current_energy + delta_e
                        
                        if new_energy < best_energy:
                            # Apply move
                            current_chain[i] = p1_new
                            current_chain[i+1] = p2_new
                            del occupied[p1_old]
                            del occupied[p2_old]
                            occupied[p1_new] = i
                            occupied[p2_new] = i+1
                            current_energy = new_energy
                            best_energy = new_energy
                            best_chain = current_chain[:]
                            improved = True
                            break
                    if improved:
                        break

        else:
            # Jump move: move residue j to a neighbor of residue i, |i-j| >= 3
            j = rng.randint(1, n - 2)
            i = rng.randint(0, n - 1)
            if abs(i - j) < 3:
                continue
            
            old_pos = current_chain[j]
            anchor = current_chain[i]
            
            for new_pos in _neighbours(anchor):
                if new_pos in occupied:
                    continue
                
                prev_pos = current_chain[j-1]
                next_pos = current_chain[j+1]
                
                if abs(new_pos[0]-prev_pos[0]) + abs(new_pos[1]-prev_pos[1]) + abs(new_pos[2]-prev_pos[2]) != 1:
                    continue
                if abs(new_pos[0]-next_pos[0]) + abs(new_pos[1]-next_pos[1]) + abs(new_pos[2]-next_pos[2]) != 1:
                    continue
                
                delta_e = 0
                if sequence[j] == 'H':
                    for q in _neighbours(old_pos):
                        idx_q = occupied.get(q)
                        if idx_q is not None and idx_q != j and abs(idx_q - j) > 1 and sequence[idx_q] == 'H':
                            delta_e -= 1
                    for q in _neighbours(new_pos):
                        idx_q = occupied.get(q)
                        if idx_q is not None and idx_q != j and abs(idx_q - j) > 1 and sequence[idx_q] == 'H':
                            delta_e += 1
                
                new_energy = current_energy + delta_e
                
                if new_energy < best_energy:
                    current_chain[j] = new_pos
                    del occupied[old_pos]
                    occupied[new_pos] = j
                    current_energy = new_energy
                    best_energy = new_energy
                    best_chain = current_chain[:]
                    improved = True
                    break
                elif new_energy <= current_energy + 1 and rng.random() < 0.2:
                    current_chain[j] = new_pos
                    del occupied[old_pos]
                    occupied[new_pos] = j
                    current_energy = new_energy
                    improved = True
                    break

        if not improved:
            # Try SA-style escape: accept a worse 1-move
            escaped = False
            for _ in range(5):
                j = rng.randint(1, n - 2)
                old_pos = current_chain[j]
                for new_pos in _neighbours(old_pos):
                    if new_pos in occupied:
                        continue
                    prev_pos = current_chain[j-1]
                    next_pos = current_chain[j+1]
                    
                    if abs(new_pos[0]-prev_pos[0]) + abs(new_pos[1]-prev_pos[1]) + abs(new_pos[2]-prev_pos[2]) != 1:
                        continue
                    if abs(new_pos[0]-next_pos[0]) + abs(new_pos[1]-next_pos[1]) + abs(new_pos[2]-next_pos[2]) != 1:
                        continue
                    
                    delta_e = 0
                    if sequence[j] == 'H':
                        for q in _neighbours(old_pos):
                            idx_q = occupied.get(q)
                            if idx_q is not None and idx_q != j and abs(idx_q - j) > 1 and sequence[idx_q] == 'H':
                                delta_e -= 1
                        for q in _neighbours(new_pos):
                            idx_q = occupied.get(q)
                            if idx_q is not None and idx_q != j and abs(idx_q - j) > 1 and sequence[idx_q] == 'H':
                                delta_e += 1
                    
                    new_energy = current_energy + delta_e
                    if new_energy <= best_energy + 3:
                        current_chain[j] = new_pos
                        del occupied[old_pos]
                        occupied[new_pos] = j
                        current_energy = new_energy
                        escaped = True
                        break
                    else:
                        continue
                if escaped:
                    break
            
            if not escaped:
                if it > 100:
                    break
                else:
                    continue

    return best_chain


def fold(sequence, dimension):
    """Best conformation using many randomized greedy growths with hybrid local refinement."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))

    best_chain = None
    best_energy = 0  # energy is negative, 0 means no contacts

    # Scale number of attempts based on chain length
    if n <= 48:
        num_attempts = 1200
    elif n <= 80:
        num_attempts = 600
    elif n <= 100:
        num_attempts = 300
    else:
        num_attempts = 120

    for attempt in range(num_attempts):
        if time.time() > _DEADLINE - 3:
            break

        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue

        max_iter = min(2000, 40 * n)
        refined = _local_refine(sequence, chain, dimension, rng, max_iters=max_iter)
        value = _initial_energy(sequence, refined)

        if value < best_energy:
            best_chain = refined
            best_energy = value

    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
