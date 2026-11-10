"""HP lattice protein folding using simulated annealing with optimized moves."""
import time
import random
import math


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _initial_spiral(n, dimension=2):
    """Generate a compact spiral initial conformation."""
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    direction = (1, 0)  # start going right
    
    # Spiral pattern: right, up, left, left, down, down, right, right, ...
    # Step lengths: 1,1,2,2,3,3,4,4,...
    step_len = 1
    steps_in_run = 2
    run_count = 0
    total_steps_done = 0
    
    while len(chain) < n:
        # Determine direction for this step
        # Directions cycle: R, U, L, D
        dir_idx = (run_count) % 4
        if dir_idx == 0:
            direction = (1, 0)
        elif dir_idx == 1:
            direction = (0, 1)
        elif dir_idx == 2:
            direction = (-1, 0)
        else:
            direction = (0, -1)
        
        # Take steps in this direction
        for _ in range(steps_in_run):
            if len(chain) >= n:
                break
            new_pos = (chain[-1][0] + direction[0], chain[-1][1] + direction[1])
            if new_pos in occupied:
                break
            chain.append(new_pos)
            occupied[new_pos] = len(chain) - 1
            total_steps_done += 1
        
        if len(chain) >= n:
            break
        
        # After finishing a run, increment step length every 2 runs
        run_count += 1
        if run_count % 2 == 0:
            step_len += 1
            steps_in_run = step_len * 2
        # Actually let's use a simpler spiral
        # Reset and use a proper square spiral
        break
    
    # Fallback: simple square spiral
    if len(chain) < n:
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        # Square spiral: go right 1, up 1, left 2, down 2, right 3, up 3, ...
        directions = [(1,0), (0,1), (-1,0), (0,-1)]
        dir_idx = 0
        step_count = 1
        steps_in_this_dir = 1
        steps_done_in_dir = 0
        
        while len(chain) < n:
            dx, dy = directions[dir_idx]
            for _ in range(steps_in_this_dir):
                if len(chain) >= n:
                    break
                new_pos = (chain[-1][0] + dx, chain[-1][1] + dy)
                if new_pos in occupied:
                    break
                chain.append(new_pos)
                occupied[new_pos] = len(chain) - 1
                steps_done_in_dir += 1
            
            if len(chain) >= n:
                break
            
            dir_idx = (dir_idx + 1) % 4
            if dir_idx == 0:
                step_count += 1
                steps_in_this_dir = step_count
            elif dir_idx == 2:
                step_count += 1
                steps_in_this_dir = step_count
            steps_done_in_dir = 0
    
    return chain


def _compute_energy(sequence, chain):
    """Compute HP energy: negative count of non-adjacent H-H contacts."""
    n = len(chain)
    if n < 3:
        return 0
    
    # Build position to index map
    pos_to_idx = {}
    for i, pos in enumerate(chain):
        pos_to_idx[tuple(pos)] = i
    
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        # Check 4 neighbors
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            neighbor = (x + dx, y + dy)
            j = pos_to_idx.get(neighbor)
            if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                contacts += 1
    
    return -contacts // 2  # Each contact counted twice


def _is_valid_walk(chain, sequence):
    """Check if chain is a valid self-avoiding walk."""
    n = len(chain)
    if n != len(sequence):
        return False
    seen = set()
    for i, pos in enumerate(chain):
        p = tuple(pos)
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            prev = tuple(chain[i-1])
            dx = abs(p[0] - prev[0])
            dy = abs(p[1] - prev[1])
            if dx + dy != 1:
                return False
    return True


def _pivot_move(chain, rng):
    """Try a pivot move: rotate a subchain around a bond."""
    n = len(chain)
    if n < 3:
        return None
    
    # Choose pivot point
    pivot = rng.randint(1, n - 2)
    
    # Get the tail
    tail = chain[pivot + 1:]
    head = chain[:pivot + 1]
    
    # The bond direction from pivot to pivot+1
    dx = tail[0][0] - head[-1][0]
    dy = tail[0][1] - head[-1][1]
    
    # Try rotating the tail 90 or 180 degrees
    rotations = []
    if dx != 0:  # horizontal bond
        rotations.append((dy, dx))   # 90 deg
        rotations.append((-dx, -dy)) # 180 deg
    else:  # vertical bond
        rotations.append((-dy, dx))  # 90 deg
        rotations.append((-dx, -dy)) # 180 deg
    
    rng.shuffle(rotations)
    
    for rx, ry in rotations:
        # Apply rotation to all tail points
        new_tail = []
        for i, (x, y) in enumerate(tail):
            if i == 0:
                new_tail.append((head[-1][0] + rx, head[-1][1] + ry))
            else:
                prev_x, prev_y = new_tail[-1]
                new_tail.append((prev_x + (x - tail[i-1][0]), prev_y + (y - tail[i-1][1])))
        
        new_chain = head + new_tail
        
        # Check self-avoidance
        seen = set()
        valid = True
        for pos in new_chain:
            p = tuple(pos)
            if p in seen:
                valid = False
                break
            seen.add(p)
        
        if valid:
            return new_chain
    
    return None


def _crankshaft_move(chain, rng):
    """Try a crankshaft move: rotate a short segment (2-3 residues)."""
    n = len(chain)
    if n < 4:
        return None
    
    # Choose segment length and start
    seg_len = rng.randint(2, min(3, n - 2))
    start = rng.randint(1, n - seg_len - 1)
    
    segment = chain[start:start + seg_len]
    before = chain[:start]
    after = chain[start + seg_len:]
    
    # The bond before the segment
    dx_before = segment[0][0] - before[-1][0]
    dy_before = segment[0][1] - before[-1][1]
    
    # Try rotating the segment
    if dx_before != 0:
        rotations = [(dy_before, dx_before), (-dx_before, -dy_before)]
    else:
        rotations = [(-dy_before, dx_before), (-dx_before, -dy_before)]
    
    rng.shuffle(rotations)
    
    for rx, ry in rotations:
        new_segment = []
        for i, (x, y) in enumerate(segment):
            if i == 0:
                new_segment.append((before[-1][0] + rx, before[-1][1] + ry))
            else:
                prev_x, prev_y = new_segment[-1]
                new_segment.append((prev_x + (x - segment[i-1][0]), prev_y + (y - segment[i-1][1])))
        
        new_chain = before + new_segment + after
        
        # Check self-avoidance
        seen = set()
        valid = True
        for pos in new_chain:
            p = tuple(pos)
            if p in seen:
                valid = False
                break
            seen.add(p)
        
        if valid:
            return new_chain
    
    return None


def _reversal_move(chain, rng):
    """Try a chain reversal move: reverse a subchain."""
    n = len(chain)
    if n < 4:
        return None
    
    start = rng.randint(1, n - 2)
    end = rng.randint(start + 1, n - 1)
    
    new_chain = chain[:start] + chain[start:end+1][::-1] + chain[end+1:]
    
    # Check self-avoidance
    seen = set()
    valid = True
    for pos in new_chain:
        p = tuple(pos)
        if p in seen:
            valid = False
            break
        seen.add(p)
    
    if valid:
        return new_chain
    
    return None


def _simulated_annealing(sequence, initial_chain, time_budget, rng):
    """Run simulated annealing to optimize the conformation."""
    n = len(sequence)
    if n < 3:
        return initial_chain
    
    best_chain = initial_chain[:]
    best_energy = _compute_energy(sequence, initial_chain)
    current_chain = initial_chain[:]
    current_energy = best_energy
    
    # Temperature schedule
    t0 = 1.0
    t_min = 0.01
    cooling_rate = 0.995
    
    temperature = t0
    start_time = time.time()
    
    while temperature > t_min:
        # Check time budget
        if time.time() - start_time > time_budget:
            break
        
        # Choose a move
        move_type = rng.random()
        if move_type < 0.5:
            new_chain = _pivot_move(current_chain, rng)
        elif move_type < 0.8:
            new_chain = _crankshaft_move(current_chain, rng)
        else:
            new_chain = _reversal_move(current_chain, rng)
        
        if new_chain is None:
            # Try another move
            new_chain = _pivot_move(current_chain, rng)
            if new_chain is None:
                new_chain = _crankshaft_move(current_chain, rng)
        
        if new_chain is None:
            temperature *= cooling_rate
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Metropolis criterion
        if delta_e < 0 or rng.random() < math.exp(-delta_e / temperature):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
        
        temperature *= cooling_rate
    
    return best_chain


def fold(sequence, dimension):
    """Fold a sequence into a low-energy conformation using simulated annealing."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    if n == 2:
        return [[0, 0], [1, 0]]
    
    # Track overall time
    start_time = time.time()
    overall_budget = 120.0  # Leave margin for 180s kill
    
    rng = random.Random(42 + len(sequence))
    
    # Generate initial spiral
    initial_chain = _initial_spiral(n, dimension)
    
    # If spiral failed, use a simple path
    if not _is_valid_walk(initial_chain, sequence):
        initial_chain = [[i, 0] for i in range(n)]
    
    # Run multiple SA rounds with different random seeds
    best_chain = initial_chain[:]
    best_energy = _compute_energy(sequence, initial_chain)
    
    # Allocate time per round
    num_rounds = 5
    time_per_round = (overall_budget - 5.0) / num_rounds  # 5s safety margin
    
    for round_idx in range(num_rounds):
        # Check time
        elapsed = time.time() - start_time
        if elapsed > overall_budget - 10.0:  # 10s safety margin
            break
        
        # Generate a new initial conformation with some randomness
        if round_idx == 0:
            init_chain = initial_chain[:]
        else:
            # Perturb the best chain found so far
            init_chain = best_chain[:]
            # Apply some random moves to perturb
            for _ in range(n // 2):
                move_type = rng.random()
                if move_type < 0.5:
                    new_c = _pivot_move(init_chain, rng)
                elif move_type < 0.8:
                    new_c = _crankshaft_move(init_chain, rng)
                else:
                    new_c = _reversal_move(init_chain, rng)
                if new_c is not None:
                    init_chain = new_c
        
        # Run SA
        result_chain = _simulated_annealing(sequence, init_chain, time_per_round, rng)
        result_energy = _compute_energy(sequence, result_chain)
        
        if result_energy < best_energy:
            best_energy = result_energy
            best_chain = result_chain[:]
    
    # Final validation
    if not _is_valid_walk(best_chain, sequence):
        # Fallback to simple path
        best_chain = [[i, 0] for i in range(n)]
    
    return [list(p) for p in best_chain]
