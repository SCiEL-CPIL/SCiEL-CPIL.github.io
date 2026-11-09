"""HP lattice protein folding on the cubic lattice.
Uses a combination of randomized greedy growth and simulated annealing
with local moves (pivot, crankshaft, diagonal) and a deadline guard.
"""
import random
import time

# Deadline: evaluator kills at 330s wall-clock. Use 320s safety margin.
_START_TIME = None
_DEADLINE = 320.0  # seconds from start of fold calls


def _neighbours(p):
    """Yield the 6 face-neighbours of point p in 3D."""
    for axis in range(len(p)):
        for step in (-1, 1):
            q = list(p)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy as minus the number of H-H non-adjacent contacts."""
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
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
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
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move(chain, sequence, rng, occupied):
    """Try a pivot move: rotate a segment of the chain."""
    n = len(chain)
    if n < 4:
        return False
    # Pick a pivot index
    i = rng.randint(1, n - 3)
    # The segment from i to end is rotated
    pivot = chain[i]
    segment = chain[i:]
    
    # Try rotating the segment by trying all 6 possible orientations
    # We need to check if the new positions are valid
    orig_segment = list(segment)
    
    # Generate all 24 rotations of the segment relative to pivot
    # For simplicity, try a few random rotations
    for _ in range(5):
        # Pick a random axis and angle
        axis = rng.randint(0, 2)
        angle = rng.choice([90, 180, 270])
        
        # Apply rotation to each point in segment relative to pivot
        new_segment = [list(pivot)]
        for j in range(1, len(segment)):
            p = segment[j]
            new_p = list(pivot)
            # Compute offset from pivot
            offset = [p[k] - pivot[k] for k in range(3)]
            # Apply rotation
            if angle == 90:
                if axis == 0:
                    new_p[0] = pivot[0]
                    new_p[1] = pivot[1] - offset[2]
                    new_p[2] = pivot[2] + offset[1]
                elif axis == 1:
                    new_p[0] = pivot[0] + offset[2]
                    new_p[1] = pivot[1]
                    new_p[2] = pivot[2] - offset[0]
                else:
                    new_p[0] = pivot[0] - offset[1]
                    new_p[1] = pivot[1] + offset[0]
                    new_p[2] = pivot[2]
            elif angle == 180:
                new_p[0] = pivot[0] - offset[0]
                new_p[1] = pivot[1] - offset[1]
                new_p[2] = pivot[2] - offset[2]
            else:  # 270
                if axis == 0:
                    new_p[0] = pivot[0]
                    new_p[1] = pivot[1] + offset[2]
                    new_p[2] = pivot[2] - offset[1]
                elif axis == 1:
                    new_p[0] = pivot[0] - offset[2]
                    new_p[1] = pivot[1]
                    new_p[2] = pivot[2] + offset[0]
                else:
                    new_p[0] = pivot[0] + offset[1]
                    new_p[1] = pivot[1] - offset[0]
                    new_p[2] = pivot[2]
            new_segment.append(tuple(new_p))
        
        # Check validity
        valid = True
        # Check that new_segment[0] == pivot (should be)
        if new_segment[0] != pivot:
            valid = False
        # Check self-avoidance for the new segment
        new_occ = set()
        for k, p in enumerate(new_segment):
            if p in new_occ:
                valid = False
                break
            new_occ.add(p)
        # Check against occupied (excluding the original segment)
        orig_occ = set(chain[i:])
        for p in new_segment:
            if p in occupied and p not in orig_occ:
                valid = False
                break
            if p in orig_occ and p not in new_occ:
                # This shouldn't happen if we're replacing
                pass
        
        if valid:
            # Check that consecutive points are unit steps
            for k in range(len(new_segment) - 1):
                diff = sum(abs(new_segment[k][a] - new_segment[k+1][a]) for a in range(3))
                if diff != 1:
                    valid = False
                    break
        
        if valid:
            # Apply the move
            chain[i:] = new_segment
            # Update occupied
            for p in orig_occ:
                del occupied[p]
            for p in new_segment:
                occupied[p] = chain.index(p)  # This is O(n), but okay for small n
            # Actually, let's update occupied more carefully
            # Rebuild occupied for the moved part
            for idx in range(i, n):
                occupied[chain[idx]] = idx
            return True
    
    return False


def _crankshaft_move(chain, sequence, rng, occupied):
    """Try a crankshaft move: rotate a middle segment."""
    n = len(chain)
    if n < 6:
        return False
    # Pick start and end of the middle segment
    i = rng.randint(1, n - 5)
    j = rng.randint(i + 2, n - 2)
    segment = chain[i:j+1]
    
    # Try rotating the segment
    for _ in range(3):
        axis = rng.randint(0, 2)
        angle = rng.choice([90, 180, 270])
        
        new_segment = []
        for p in segment:
            new_p = list(p)
            offset = [p[k] - segment[0][k] for k in range(3)]
            if angle == 90:
                if axis == 0:
                    new_p[0] = segment[0][0]
                    new_p[1] = segment[0][1] - offset[2]
                    new_p[2] = segment[0][2] + offset[1]
                elif axis == 1:
                    new_p[0] = segment[0][0] + offset[2]
                    new_p[1] = segment[0][1]
                    new_p[2] = segment[0][2] - offset[0]
                else:
                    new_p[0] = segment[0][0] - offset[1]
                    new_p[1] = segment[0][1] + offset[0]
                    new_p[2] = segment[0][2]
            elif angle == 180:
                new_p[0] = segment[0][0] - offset[0]
                new_p[1] = segment[0][1] - offset[1]
                new_p[2] = segment[0][2] - offset[2]
            else:
                if axis == 0:
                    new_p[0] = segment[0][0]
                    new_p[1] = segment[0][1] + offset[2]
                    new_p[2] = segment[0][2] - offset[1]
                elif axis == 1:
                    new_p[0] = segment[0][0] - offset[2]
                    new_p[1] = segment[0][1]
                    new_p[2] = segment[0][2] + offset[0]
                else:
                    new_p[0] = segment[0][0] + offset[1]
                    new_p[1] = segment[0][1] - offset[0]
                    new_p[2] = segment[0][2]
            new_segment.append(tuple(new_p))
        
        # Check validity
        valid = True
        new_occ = set(new_segment)
        for p in new_occ:
            if p in occupied and p not in set(segment):
                valid = False
                break
        
        if valid:
            # Check unit steps within new segment
            for k in range(len(new_segment) - 1):
                diff = sum(abs(new_segment[k][a] - new_segment[k+1][a]) for a in range(3))
                if diff != 1:
                    valid = False
                    break
            # Check unit steps at boundaries
            if i > 0:
                diff = sum(abs(chain[i-1][a] - new_segment[0][a]) for a in range(3))
                if diff != 1:
                    valid = False
            if j < n - 1:
                diff = sum(abs(new_segment[-1][a] - chain[j+1][a]) for a in range(3))
                if diff != 1:
                    valid = False
        
        if valid:
            chain[i:j+1] = new_segment
            for p in segment:
                del occupied[p]
            for idx in range(i, j+1):
                occupied[chain[idx]] = idx
            return True
    
    return False


def _sa_fold(sequence, dimension, rng, time_budget):
    """Simulated annealing fold starting from a greedy growth."""
    chain = _grow(sequence, dimension, rng)
    if chain is None:
        # Try a simpler initialization
        chain = [(i, 0, 0) for i in range(len(sequence))]
        if not all(sum(abs(chain[i][a] - chain[i+1][a]) for a in range(3)) == 1 for i in range(len(chain)-1)):
            return None
        # Check self-avoidance
        if len(set(chain)) != len(chain):
            return None
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)
    
    n = len(chain)
    start_temp = 2.0
    end_temp = 0.01
    steps = int(time_budget * 1000)  # Approximate steps
    
    for step in range(steps):
        if time.time() - _START_TIME > _DEADLINE - 5:
            break
        
        temp = start_temp * (end_temp / start_temp) ** (step / max(1, steps))
        
        # Choose a move
        move_type = rng.random()
        if move_type < 0.4:
            success = _pivot_move(chain, sequence, rng, occupied)
        elif move_type < 0.7:
            success = _crankshaft_move(chain, sequence, rng, occupied)
        else:
            # Local move: try to move one residue
            i = rng.randint(1, n - 2)
            p = chain[i]
            for q in _neighbours(p):
                if q in occupied and occupied[q] not in (i - 1, i + 1):
                    continue
                if q in occupied and occupied[q] in (i - 1, i + 1):
                    # Would collide with neighbor
                    continue
                # Check if moving i to q is valid
                # Remove i from occupied temporarily
                old_pos = p
                del occupied[old_pos]
                if q not in occupied:
                    # Check unit steps
                    if i > 0 and sum(abs(chain[i-1][a] - q[a]) for a in range(3)) != 1:
                        occupied[old_pos] = i
                        continue
                    if i < n - 1 and sum(abs(q[a] - chain[i+1][a]) for a in range(3)) != 1:
                        occupied[old_pos] = i
                        continue
                    # Valid move
                    chain[i] = q
                    occupied[q] = i
                    new_energy = _energy(sequence, chain)
                    delta = new_energy - current_energy
                    if delta < 0 or rng.random() < min(1.0, (-delta / max(temp, 1e-6))):
                        current_energy = new_energy
                        if new_energy < best_energy:
                            best_energy = new_energy
                            best_chain = list(chain)
                    else:
                        # Revert
                        chain[i] = old_pos
                        occupied[old_pos] = i
                    success = True
                    break
                else:
                    occupied[old_pos] = i
            if not success:
                occupied[old_pos] = i
        
        if not success:
            continue
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using greedy growth and simulated annealing."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    # Time budget for this sequence
    elapsed = time.time() - _START_TIME
    remaining = _DEADLINE - elapsed
    if remaining < 5:
        # Not enough time, return a simple chain
        return [list(p) for p in [(i, 0, 0) for i in range(n)]]
    
    # Allocate time: some for greedy, some for SA
    greedy_time = min(remaining * 0.3, 30)
    sa_time = remaining - greedy_time - 5
    
    # Try multiple greedy growths
    best_chain = None
    best_energy = 0
    
    greedy_count = int(greedy_time / 0.01)  # ~100 greedy trials per second
    greedy_count = max(50, min(greedy_count, 500))
    
    for _ in range(greedy_count):
        if time.time() - _START_TIME > _DEADLINE - 10:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if best_chain is None or value < best_energy:
            best_chain = chain
            best_energy = value
    
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Simulated annealing
    if sa_time > 5:
        sa_chain = _sa_fold(sequence, dimension, rng, sa_time)
        if sa_chain is not None:
            sa_energy = _energy(sequence, sa_chain)
            if sa_energy < best_energy:
                best_chain = sa_chain
                best_energy = sa_energy
    
    return [list(p) for p in best_chain]
