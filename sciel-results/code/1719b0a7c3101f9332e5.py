# EVOLVE-BLOCK-START
import random
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute HP contact energy: -number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _is_valid_chain(chain):
    """Check if the chain is self-avoiding and has unit steps."""
    if not chain:
        return True
    occupied = set()
    for i, p in enumerate(chain):
        if p in occupied:
            return False
        occupied.add(p)
        if i > 0:
            prev = chain[i - 1]
            if abs(p[0] - prev[0]) + abs(p[1] - prev[1]) != 1:
                return False
    return True


def _local_search(sequence, chain, time_budget, rng):
    """Simulated annealing local search with pivot and reversal moves."""
    n = len(chain)
    if n <= 2:
        return chain
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, chain)
    current_chain = list(chain)
    current_energy = best_energy
    
    # Precompute H positions for faster contact detection
    h_positions = set(i for i, c in enumerate(sequence) if c == 'H')
    
    start_time = time.time()
    max_time = time_budget
    initial_temp = 2.0
    cooling_rate = 0.999
    
    temp = initial_temp
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() - start_time < max_time:
        # Choose move type: 60% pivot, 40% reversal
        if rng.random() < 0.6:
            # Pivot Move
            if n > 2:
                # Bias pivot selection toward H-rich regions
                if h_positions and rng.random() < 0.7:
                    # Pick a random H position as pivot
                    h_list = list(h_positions)
                    pivot_idx = h_list[rng.randint(0, len(h_list) - 1)]
                    if pivot_idx == 0:
                        pivot_idx = 1
                else:
                    pivot_idx = rng.randint(1, n - 1)
                
                pivot = current_chain[pivot_idx - 1]
                tail = current_chain[pivot_idx:]
                
                # Choose a random rotation: 1, 2, or 3 (90, 180, 270 deg)
                rot = rng.choice([1, 2, 3])
                rotated = []
                valid = True
                occupied_set = set(current_chain[:pivot_idx])
                
                for pos in tail:
                    dx = pos[0] - pivot[0]
                    dy = pos[1] - pivot[1]
                    
                    if rot == 1:
                        ndx, ndy = dy, -dx
                    elif rot == 2:
                        ndx, ndy = -dx, -dy
                    else:  # rot == 3
                        ndx, ndy = -dy, dx
                    
                    new_pos = (pivot[0] + ndx, pivot[1] + ndy)
                    
                    if new_pos in occupied_set:
                        valid = False
                        break
                    occupied_set.add(new_pos)
                    
                    if rotated:
                        prev_rot = rotated[-1]
                        if abs(new_pos[0] - prev_rot[0]) + abs(new_pos[1] - prev_rot[1]) != 1:
                            valid = False
                            break
                    else:
                        if abs(new_pos[0] - pivot[0]) + abs(new_pos[1] - pivot[1]) != 1:
                            valid = False
                            break
                            
                    rotated.append(new_pos)
                
                if valid:
                    new_chain = current_chain[:pivot_idx] + rotated
                    new_energy = _compute_energy(sequence, new_chain)
                else:
                    continue
            else:
                continue
        else:
            # Reversal Move
            if n > 3:
                # Bias reversal toward H-rich segments
                if h_positions and rng.random() < 0.7:
                    h_list = sorted(h_positions)
                    start_idx = h_list[rng.randint(0, len(h_list) - 1)]
                    # Find a segment containing this H
                    min_start = max(1, start_idx - 5)
                    max_end = min(n - 1, start_idx + 5)
                    start_idx = rng.randint(min_start, max(1, start_idx))
                    end_idx = rng.randint(start_idx + 1, max_end)
                else:
                    start_idx = rng.randint(1, n - 2)
                    end_idx = rng.randint(start_idx + 1, n - 1)
                
                new_chain = current_chain[:start_idx] + list(reversed(current_chain[start_idx:end_idx + 1])) + current_chain[end_idx + 1:]
                
                if _is_valid_chain(new_chain):
                    new_energy = _compute_energy(sequence, new_chain)
                else:
                    continue
            else:
                continue

        delta = new_energy - current_energy
        if delta < 0 or (temp > 0.01 and rng.random() < pow(2.718281828, -delta / temp)):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_chain = list(current_chain)
                best_energy = current_energy
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        temp *= cooling_rate
        
        if no_improve_count > max_no_improve:
            current_chain = list(best_chain)
            current_energy = best_energy
            temp = initial_temp * 0.5
            no_improve_count = 0
    
    return best_chain


def _generate_initial_chain(sequence, rng):
    """Generate a random self-avoiding walk as initial conformation, biased toward compactness."""
    n = len(sequence)
    if n <= 2:
        return [(0, 0), (1, 0)][:n]
    
    # Start at origin, grow randomly with bias toward center
    chain = [(0, 0)]
    occupied = {(0, 0)}
    direction = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
    
    for i in range(1, n):
        candidates = []
        last = chain[-1]
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_pos = (last[0] + dx, last[1] + dy)
            if new_pos not in occupied:
                # Score candidate by distance to origin (prefer closer)
                dist = abs(new_pos[0]) + abs(new_pos[1])
                candidates.append((dist, new_pos))
        
        if not candidates:
            return None
        
        # Prefer candidates closer to origin (compact)
        candidates.sort(key=lambda x: x[0])
        # With some probability, pick the best; otherwise random from top half
        if rng.random() < 0.6:
            new_pos = candidates[0][1]
        else:
            top_half = candidates[:max(1, len(candidates) // 2)]
            new_pos = top_half[rng.randint(0, len(top_half) - 1)][1]
        
        direction = (new_pos[0] - last[0], new_pos[1] - last[1])
        chain.append(new_pos)
        occupied.add(new_pos)
    
    return chain


def fold(sequence, dimension):
    """Fold HP sequence using multi-start simulated annealing local search."""
    n = len(sequence)
    if n <= 2:
        return [[0, 0], [1, 0]][:n]
    
    # Time budget: distribute based on sequence length
    # Total budget is 120s. 10 sequences.
    # Give more time to longer sequences
    time_per_seq = min(2.0 + n * 0.12, 15.0)
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = float('inf')
    
    # Multi-start: try different initial conformations
    num_starts = max(3, min(20, int(time_per_seq / 0.6)))
    time_per_start = time_per_seq / num_starts
    
    for start in range(num_starts):
        # Generate initial chain
        initial = _generate_initial_chain(sequence, rng)
        if initial is None:
            # Fallback: straight line
            initial = [(i, 0) for i in range(n)]
        
        # Local search
        improved = _local_search(sequence, initial, time_per_start, rng)
        
        energy = _compute_energy(sequence, improved)
        if energy < best_energy:
            best_energy = energy
            best_chain = improved
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
