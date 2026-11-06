"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = time.time() + 300  # safety margin before 330s kill


def _check_deadline():
    return time.time() > _DEADLINE


def _neighbours3(point):
    """Yield the 6 neighbors of a point in 3D."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def _is_valid_move(chain, pos, new_pos, occupied):
    """Check if moving chain[pos] to new_pos is valid."""
    if new_pos in occupied and occupied[new_pos] != pos:
        return False
    # Check that the move doesn't break connectivity with neighbors
    if pos > 0:
        if new_pos == chain[pos - 1]:
            return False
    if pos < len(chain) - 1:
        if new_pos == chain[pos + 1]:
            return False
    # Check unit step from previous
    if pos > 0:
        prev = chain[pos - 1]
        if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) + abs(new_pos[2] - prev[2]) != 1:
            return False
    # Check unit step from next
    if pos < len(chain) - 1:
        nxt = chain[pos + 1]
        if abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) + abs(new_pos[2] - nxt[2]) != 1:
            return False
    return True


def _compute_contacts(sequence, chain, occupied):
    """Compute number of H-H contacts."""
    contacts = 0
    for i in range(len(chain)):
        if sequence[i] == 'H':
            for nb in _neighbours3(chain[i]):
                j = occupied.get(nb, -1)
                if j != -1 and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return contacts


def _energy(sequence, chain):
    """Compute energy = -contacts."""
    occupied = {p: i for i, p in enumerate(chain)}
    return -_compute_contacts(sequence, chain, occupied)


def _build_occupied(chain):
    return {p: i for i, p in enumerate(chain)}


def _initial_chain_spiral(sequence, dimension):
    """Generate an initial conformation using a spiral pattern."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0, 0)]
    
    # Create a compact spiral/helix
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    # Try to grow in a helical pattern
    directions = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    dir_idx = 0
    
    for i in range(1, n):
        placed = False
        # Try to continue in a helical pattern
        if i % 4 == 1:
            dir_idx = 0
        elif i % 4 == 2:
            dir_idx = 1
        elif i % 4 == 3:
            dir_idx = 2
        elif i % 4 == 0:
            dir_idx = 4 if (i // 4) % 2 == 0 else 5
        
        d = directions[dir_idx]
        last = chain[-1]
        new_pos = (last[0] + d[0], last[1] + d[1], last[2] + d[2])
        
        if new_pos not in occupied:
            chain.append(new_pos)
            occupied[new_pos] = i
            placed = True
        else:
            # Try other directions
            for d in directions:
                new_pos = (last[0] + d[0], last[1] + d[1], last[2] + d[2])
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied[new_pos] = i
                    placed = True
                    break
        
        if not placed:
            # Fallback: extend in a straight line
            for d in directions:
                new_pos = (last[0] + d[0], last[1] + d[1], last[2] + d[2])
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied[new_pos] = i
                    placed = True
                    break
            if not placed:
                # Last resort: place far away
                for offset in range(1, 50):
                    new_pos = (last[0] + offset, 0, 0)
                    if new_pos not in occupied:
                        chain.append(new_pos)
                        occupied[new_pos] = i
                        break
    
    return chain


def _initial_chain_random_grow(sequence, dimension, rng):
    """Randomized greedy growth with contact preference."""
    start = (0, 0, 0)
    chain, occupied = [start], {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours3(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours3(site):
                    j = occupied.get(q, -1)
                    if j != -1 and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours3(site) if q not in occupied)
            score = gain * 10 + free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def _simulated_annealing(sequence, chain, rng, time_budget):
    """Simulated annealing with multiple move types."""
    n = len(sequence)
    if n <= 1:
        return chain
    
    occupied = _build_occupied(chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    
    start_time = time.time()
    
    while T > T_min:
        if _check_deadline() or (time.time() - start_time) > time_budget:
            break
        
        move_type = rng.random()
        accepted = False
        
        if move_type < 0.4:
            # Single residue move: try to move one residue to a nearby position
            pos = rng.randint(1, n - 2)  # avoid endpoints for simplicity
            candidates = []
            for d in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                # Try moving to adjacent positions from current
                cur = chain[pos]
                for dd in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                    new_pos = (cur[0] + dd[0], cur[1] + dd[1], cur[2] + dd[2])
                    if _is_valid_move(chain, pos, new_pos, occupied):
                        candidates.append(new_pos)
            
            if candidates:
                new_pos = rng.choice(candidates)
                # Compute energy change
                old_pos = chain[pos]
                # Remove old position
                del occupied[old_pos]
                chain[pos] = new_pos
                occupied[new_pos] = pos
                
                new_energy = _energy(sequence, chain)
                delta = new_energy - current_energy
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                    accepted = True
                else:
                    # Revert
                    chain[pos] = old_pos
                    occupied[old_pos] = pos
                    del occupied[new_pos]
        
        elif move_type < 0.7:
            # Segment reversal: reverse a subsegment
            if n > 3:
                length = rng.randint(2, min(10, n - 2))
                start_pos = rng.randint(1, n - 1 - length)
                end_pos = start_pos + length - 1
                
                # Check if reversal is valid
                # The segment [start_pos, end_pos] gets reversed
                # Need to check connectivity at boundaries
                if start_pos > 0 and end_pos < n - 1:
                    # Check that chain[start_pos-1] is adjacent to chain[end_pos]
                    # and chain[end_pos+1] is adjacent to chain[start_pos]
                    left = chain[start_pos - 1]
                    right = chain[end_pos + 1]
                    new_left = chain[end_pos]
                    new_right = chain[start_pos]
                    
                    if (abs(left[0] - new_left[0]) + abs(left[1] - new_left[1]) + abs(left[2] - new_left[2]) == 1 and
                        abs(right[0] - new_right[0]) + abs(right[1] - new_right[1]) + abs(right[2] - new_right[2]) == 1):
                        # Check no self-intersection in reversed segment
                        valid = True
                        segment = chain[start_pos:end_pos + 1][::-1]
                        for k, p in enumerate(segment):
                            idx = start_pos + k
                            if p in occupied and occupied[p] != idx:
                                valid = False
                                break
                        
                        if valid:
                            # Apply reversal
                            old_segment = chain[start_pos:end_pos + 1][:]
                            new_segment = old_segment[::-1]
                            
                            # Update occupied
                            for i, p in enumerate(old_segment):
                                del occupied[p]
                            for i, p in enumerate(new_segment):
                                chain[start_pos + i] = p
                                occupied[p] = start_pos + i
                            
                            new_energy = _energy(sequence, chain)
                            delta = new_energy - current_energy
                            
                            if delta < 0 or rng.random() < math.exp(-delta / T):
                                current_energy = new_energy
                                if current_energy < best_energy:
                                    best_energy = current_energy
                                    best_chain = list(chain)
                                accepted = True
                            else:
                                # Revert
                                for i, p in enumerate(old_segment):
                                    chain[start_pos + i] = p
                                    occupied[p] = start_pos + i
                        else:
                            # Not valid, skip
                            pass
                    else:
                        pass
                elif start_pos == 0 and end_pos < n - 1:
                    # Reversal starting at 0
                    right = chain[end_pos + 1]
                    new_right = chain[end_pos]  # after reversal, end_pos becomes start
                    if (abs(right[0] - new_right[0]) + abs(right[1] - new_right[1]) + abs(right[2] - new_right[2]) == 1):
                        valid = True
                        segment = chain[start_pos:end_pos + 1][::-1]
                        for k, p in enumerate(segment):
                            idx = start_pos + k
                            if p in occupied and occupied[p] != idx:
                                valid = False
                                break
                        if valid:
                            old_segment = chain[start_pos:end_pos + 1][:]
                            new_segment = old_segment[::-1]
                            for i, p in enumerate(old_segment):
                                del occupied[p]
                            for i, p in enumerate(new_segment):
                                chain[start_pos + i] = p
                                occupied[p] = start_pos + i
                            new_energy = _energy(sequence, chain)
                            delta = new_energy - current_energy
                            if delta < 0 or rng.random() < math.exp(-delta / T):
                                current_energy = new_energy
                                if current_energy < best_energy:
                                    best_energy = current_energy
                                    best_chain = list(chain)
                                accepted = True
                            else:
                                for i, p in enumerate(old_segment):
                                    chain[start_pos + i] = p
                                    occupied[p] = start_pos + i
                elif start_pos > 0 and end_pos == n - 1:
                    left = chain[start_pos - 1]
                    new_left = chain[start_pos]  # after reversal, start_pos becomes end
                    if (abs(left[0] - new_left[0]) + abs(left[1] - new_left[1]) + abs(left[2] - new_left[2]) == 1):
                        valid = True
                        segment = chain[start_pos:end_pos + 1][::-1]
                        for k, p in enumerate(segment):
                            idx = start_pos + k
                            if p in occupied and occupied[p] != idx:
                                valid = False
                                break
                        if valid:
                            old_segment = chain[start_pos:end_pos + 1][:]
                            new_segment = old_segment[::-1]
                            for i, p in enumerate(old_segment):
                                del occupied[p]
                            for i, p in enumerate(new_segment):
                                chain[start_pos + i] = p
                                occupied[p] = start_pos + i
                            new_energy = _energy(sequence, chain)
                            delta = new_energy - current_energy
                            if delta < 0 or rng.random() < math.exp(-delta / T):
                                current_energy = new_energy
                                if current_energy < best_energy:
                                    best_energy = current_energy
                                    best_chain = list(chain)
                                accepted = True
                            else:
                                for i, p in enumerate(old_segment):
                                    chain[start_pos + i] = p
                                    occupied[p] = start_pos + i
        
        else:
            # Try a random local perturbation: move a few residues
            if n > 3:
                num_moves = rng.randint(1, 3)
                for _ in range(num_moves):
                    pos = rng.randint(1, n - 2)
                    cur = chain[pos]
                    for dd in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                        new_pos = (cur[0] + dd[0], cur[1] + dd[1], cur[2] + dd[2])
                        if _is_valid_move(chain, pos, new_pos, occupied):
                            old_pos = chain[pos]
                            del occupied[old_pos]
                            chain[pos] = new_pos
                            occupied[new_pos] = pos
                            break
                    else:
                        continue
        
        T *= cooling
    
    return best_chain


def _local_search(sequence, chain, rng, time_budget):
    """Iterative local search: accept only improving moves."""
    n = len(sequence)
    if n <= 1:
        return chain
    
    occupied = _build_occupied(chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    no_improve_count = 0
    
    while True:
        if _check_deadline() or (time.time() - start_time) > time_budget:
            break
        
        improved = False
        move_type = rng.random()
        
        if move_type < 0.5:
            # Single residue move
            pos = rng.randint(1, n - 2)
            candidates = []
            cur = chain[pos]
            for dd in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                new_pos = (cur[0] + dd[0], cur[1] + dd[1], cur[2] + dd[2])
                if _is_valid_move(chain, pos, new_pos, occupied):
                    candidates.append(new_pos)
            
            if candidates:
                new_pos = rng.choice(candidates)
                old_pos = chain[pos]
                del occupied[old_pos]
                chain[pos] = new_pos
                occupied[new_pos] = pos
                
                new_energy = _energy(sequence, chain)
                if new_energy < current_energy:
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                    improved = True
                else:
                    chain[pos] = old_pos
                    occupied[old_pos] = pos
                    del occupied[new_pos]
        
        elif move_type < 0.8:
            # Segment reversal
            if n > 3:
                length = rng.randint(2, min(8, n - 2))
                start_pos = rng.randint(1, n - 1 - length)
                end_pos = start_pos + length - 1
                
                if start_pos > 0 and end_pos < n - 1:
                    left = chain[start_pos - 1]
                    right = chain[end_pos + 1]
                    new_left = chain[end_pos]
                    new_right = chain[start_pos]
                    
                    if (abs(left[0] - new_left[0]) + abs(left[1] - new_left[1]) + abs(left[2] - new_left[2]) == 1 and
                        abs(right[0] - new_right[0]) + abs(right[1] - new_right[1]) + abs(right[2] - new_right[2]) == 1):
                        valid = True
                        segment = chain[start_pos:end_pos + 1][::-1]
                        for k, p in enumerate(segment):
                            idx = start_pos + k
                            if p in occupied and occupied[p] != idx:
                                valid = False
                                break
                        
                        if valid:
                            old_segment = chain[start_pos:end_pos + 1][:]
                            new_segment = old_segment[::-1]
                            for i, p in enumerate(old_segment):
                                del occupied[p]
                            for i, p in enumerate(new_segment):
                                chain[start_pos + i] = p
                                occupied[p] = start_pos + i
                            
                            new_energy = _energy(sequence, chain)
                            if new_energy < current_energy:
                                current_energy = new_energy
                                if current_energy < best_energy:
                                    best_energy = current_energy
                                    best_chain = list(chain)
                                improved = True
                            else:
                                for i, p in enumerate(old_segment):
                                    chain[start_pos + i] = p
                                    occupied[p] = start_pos + i
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 50:
                break
        else:
            no_improve_count = 0
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation found within time budget."""
    global _DEADLINE
    _DEADLINE = time.time() + 300  # reset deadline for each call
    
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(12345 + len(sequence))
    
    # Try multiple initial conformations and optimization strategies
    best_chain = None
    best_energy = 0  # 0 means no contacts (worst case)
    
    # Strategy 1: Spiral initial + SA
    chain = _initial_chain_spiral(sequence, dimension)
    if chain and len(chain) == n:
        chain = _simulated_annealing(sequence, chain, rng, 30)
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if _check_deadline():
        if best_chain is None:
            best_chain = [(i, 0, 0) for i in range(n)]
        return [list(p) for p in best_chain]
    
    # Strategy 2: Random growth + SA
    for attempt in range(3):
        if _check_deadline():
            break
        chain = _initial_chain_random_grow(sequence, dimension, rng)
        if chain and len(chain) == n:
            chain = _simulated_annealing(sequence, chain, rng, 20)
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
    
    if _check_deadline():
        if best_chain is None:
            best_chain = [(i, 0, 0) for i in range(n)]
        return [list(p) for p in best_chain]
    
    # Strategy 3: Local search from best
    if best_chain is not None:
        chain = _local_search(sequence, best_chain, rng, 30)
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if _check_deadline():
        if best_chain is None:
            best_chain = [(i, 0, 0) for i in range(n)]
        return [list(p) for p in best_chain]
    
    # Strategy 4: More SA restarts if time allows
    for attempt in range(5):
        if _check_deadline():
            break
        chain = _initial_chain_random_grow(sequence, dimension, rng)
        if chain and len(chain) == n:
            chain = _simulated_annealing(sequence, chain, rng, 15)
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
    
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
