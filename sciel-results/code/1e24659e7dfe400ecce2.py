"""HP lattice protein folding: simulated annealing with local search moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

_T0 = time.time()
_DEADLINE = 195.0  # wall-clock limit with safety margin (210s kill, leave 15s buffer)


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _neighbours(point):
    """Yield 4-neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _compute_energy(sequence, chain):
    """Compute energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    if n < 3:
        return 0
    # Map position -> index
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
    
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        # Check 4 neighbors
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (x + dx, y + dy)
            j = pos_to_idx.get(nb)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _initial_conformation(sequence, rng):
    """Generate a random self-avoiding walk as initial conformation."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0)}
    pos_to_idx = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for nb in _neighbours(last):
            if nb not in occupied:
                options.append(nb)
        
        if not options:
            # Backtrack and try different path - for simplicity, restart
            return _initial_conformation(sequence, rng)
        
        # Prefer moves that create H-H contacts
        if sequence[i] == 'H':
            scored = []
            for nb in options:
                gain = 0
                x, y = nb
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    pp = (x + dx, y + dy)
                    if pp in pos_to_idx:
                        idx = pos_to_idx[pp]
                        if sequence[idx] == 'H':
                            gain += 1
                scored.append((gain + rng.random(), nb))
            scored.sort(reverse=True)
            next_pos = scored[0][1]
        else:
            next_pos = rng.choice(options)
        
        chain.append(next_pos)
        occupied.add(next_pos)
        pos_to_idx[next_pos] = i
    
    return chain


def _pivot_move(chain, rng):
    """Perform a pivot move: rotate a suffix of the chain around a pivot point."""
    n = len(chain)
    if n < 4:
        return None
    
    # Choose pivot index (not at ends)
    pivot = rng.randint(1, n - 2)
    pivot_pos = chain[pivot]
    suffix = chain[pivot + 1:]
    
    # Generate all 4 rotations of the suffix relative to pivot
    rotations = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        rotated = []
        valid = True
        for j, (x, y) in enumerate(suffix):
            # Rotate relative to pivot
            rx, ry = x - pivot_pos[0], y - pivot_pos[1]
            # Apply 90-degree rotation
            if dx == 1 and dy == 0:  # This is just translation, not rotation
                pass
            # Let's do proper rotation: 90 deg CCW: (x,y) -> (-y,x)
            # Actually, let's rotate the vector
            if j == 0:
                # First element must be adjacent to pivot
                new_x, new_y = -ry, rx  # 90 deg CCW
                if (pivot_pos[0] + new_x, pivot_pos[1] + new_y) in set(chain[:pivot+1]):
                    valid = False
                    break
                rotated.append((pivot_pos[0] + new_x, pivot_pos[1] + new_y))
            else:
                prev = rotated[-1]
                # The new position should be adjacent to prev
                # We rotate the entire suffix
                new_x, new_y = -ry, rx
                pos = (pivot_pos[0] + new_x, pivot_pos[1] + new_y)
                if pos in set(chain[:pivot+1]):
                    valid = False
                    break
                rotated.append(pos)
        
        if valid and len(rotated) == len(suffix):
            new_chain = chain[:pivot + 1] + rotated
            # Check self-avoidance for the new suffix
            if len(set(new_chain)) == n:
                rotations.append(new_chain)
    
    if not rotations:
        return None
    return rng.choice(rotations)


def _reversal_move(chain, rng):
    """Perform a reversal move: reverse a segment of the chain."""
    n = len(chain)
    if n < 4:
        return None
    
    # Choose segment length
    # Longer reversals are more disruptive, shorter are more local
    # Pick a random length between 2 and min(10, n-2)
    max_len = min(10, n - 2)
    if max_len < 2:
        return None
    seg_len = rng.randint(2, max_len)
    
    # Choose start index
    start = rng.randint(0, n - seg_len - 1)
    end = start + seg_len - 1
    
    # The segment is chain[start:end+1]
    # Reversing it means the new segment is chain[end:start-1:-1]
    # But we must check connectivity and self-avoidance
    
    # New chain structure:
    # prefix = chain[:start]
    # segment = chain[start:end+1]
    # suffix = chain[end+1:]
    
    # Reversed segment
    new_segment = list(reversed(chain[start:end+1]))
    
    # Check connectivity at boundaries
    # 1. Prefix to New Segment
    if start > 0:
        last_prefix = chain[start - 1]
        first_new_seg = new_segment[0]
        if abs(last_prefix[0] - first_new_seg[0]) + abs(last_prefix[1] - first_new_seg[1]) != 1:
            return None
    else:
        # If start is 0, no prefix check needed, but we must ensure the chain doesn't hit itself later
        pass
        
    # 2. New Segment to Suffix
    if end < n - 1:
        last_new_seg = new_segment[-1]
        first_suffix = chain[end + 1]
        if abs(last_new_seg[0] - first_suffix[0]) + abs(last_new_seg[1] - first_suffix[1]) != 1:
            return None
    else:
        # If end is n-1, no suffix check needed
        pass
        
    # Construct new chain
    new_chain = chain[:start] + new_segment + chain[end+1:]
    
    # Check self-avoidance
    if len(set(new_chain)) != n:
        return None
        
    return new_chain


def _segment_relocation_move(chain, rng):
    """Perform a segment relocation: cut a segment and insert it elsewhere."""
    n = len(chain)
    if n < 6:
        return None
    
    # Choose segment length
    seg_len = rng.randint(2, min(5, n - 4))
    
    # Choose start index for segment to be cut
    # Must leave at least 1 residue before and 1 after to maintain connectivity options
    start = rng.randint(1, n - seg_len - 1)
    end = start + seg_len - 1
    
    segment = chain[start:end+1]
    prefix = chain[:start]
    suffix = chain[end+1:]
    
    # Try to insert the segment at a different location
    # The insertion point is between index k and k+1 in the remaining chain (prefix + suffix)
    # Remaining chain length is n - seg_len
    # Possible insertion indices: 0 to n - seg_len
    # But we must ensure the segment connects properly.
    
    # To simplify, we pick a random insertion index in the *new* sequence
    # Let's iterate over possible insertion points in the "gap" sequence
    # Gap sequence is prefix + suffix
    gap_len = n - seg_len
    
    # Pick a random insertion point
    # We avoid inserting immediately back where it was (start) to force a change
    # Insertion index k means insert before element k of the gap
    # k ranges from 0 to gap_len
    
    # We need to check:
    # 1. Connection from gap[k-1] to segment[0]
    # 2. Connection from segment[-1] to gap[k]
    # 3. Self-avoidance
    
    # To be efficient, we don't try all k. We pick a few random candidates.
    candidates = set()
    for _ in range(5):
        k = rng.randint(0, gap_len)
        if k == start: # Same position, skip
            continue
        candidates.add(k)
        
    for k in candidates:
        # Construct new chain
        # gap[:k] + segment + gap[k:]
        
        # Check connectivity
        # Left neighbor: gap[k-1] if k > 0
        # Right neighbor: gap[k] if k < gap_len
        
        # We need to map gap indices to original chain indices to check positions
        # But we can just build the new chain and check.
        
        new_chain = chain[:k] if k < start else chain[:k] # This logic is tricky with indices
        
        # Let's do it more carefully.
        # Original chain: C[0..n-1]
        # Segment: C[start..end]
        # Gap: C[0..start-1] + C[end+1..n-1]
        
        # If we insert at index k in the gap:
        # If k < start:
        #   New Chain: C[0..k-1] + Segment + C[k..start-1] + C[end+1..n-1]
        # If k >= start:
        #   New Chain: C[0..start-1] + C[end+1..end+k-start] + Segment + C[end+k-start+1..n-1]
        
        # This index shifting is error-prone. Let's just construct the list.
        gap_list = chain[:start] + chain[end+1:]
        
        # Insert segment into gap_list at index k
        new_chain = gap_list[:k] + segment + gap_list[k:]
        
        # Check self-avoidance
        if len(set(new_chain)) != n:
            continue
            
        # Check connectivity
        valid = True
        for i in range(n - 1):
            if abs(new_chain[i][0] - new_chain[i+1][0]) + abs(new_chain[i][1] - new_chain[i+1][1]) != 1:
                valid = False
                break
        if valid:
            return new_chain
            
    return None


def _try_local_search(sequence, chain, rng, max_iters=500):
    """Perform local search with simulated annealing."""
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    T = 5.0
    cooling = 0.999
    
    for it in range(max_iters):
        if _remaining() < 5.0:
            break
        
        # Choose a move
        r = rng.random()
        if r < 0.4:
            new_chain = _pivot_move(chain, rng)
        elif r < 0.7:
            new_chain = _reversal_move(chain, rng)
        else:
            new_chain = _segment_relocation_move(chain, rng)
        
        if new_chain is None:
            continue
        
        # Verify self-avoidance
        if len(set(new_chain)) != n:
            continue
        
        # Verify connectivity
        valid = True
        for i in range(n - 1):
            if abs(new_chain[i][0] - new_chain[i+1][0]) + abs(new_chain[i][1] - new_chain[i+1][1]) != 1:
                valid = False
                break
        if not valid:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Simulated annealing acceptance
        if delta < 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(new_chain)
        
        T *= cooling
        if T < 0.1:
            T = 0.1
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold sequence into low-energy conformation using simulated annealing."""
    if dimension != 2:
        # Fallback for non-2D (shouldn't happen per task)
        return [[i] + [0] * (dimension - 1) for i in range(len(sequence))]
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0  # Start with unfolded (energy 0)
    
    # Time budget: use most of the available time
    start_time = time.time()
    time_budget = min(_remaining() - 5.0, 180.0)  # Leave 5s buffer
    
    # Number of restarts based on sequence length
    if n <= 30:
        num_restarts = 50
    elif n <= 50:
        num_restarts = 30
    elif n <= 70:
        num_restarts = 20
    else:
        num_restarts = 15
    
    iters_per_restart = max(100, int(500000 / n))
    
    for restart in range(num_restarts):
        if _remaining() < 10.0:
            break
        
        # Generate initial conformation
        chain = _initial_conformation(sequence, rng)
        if chain is None or len(chain) != n:
            continue
        
        # Verify it's valid
        if len(set(chain)) != n:
            continue
        valid = True
        for i in range(n - 1):
            if abs(chain[i][0] - chain[i+1][0]) + abs(chain[i][1] - chain[i+1][1]) != 1:
                valid = False
                break
        if not valid:
            continue
        
        # Run local search
        chain, energy = _try_local_search(sequence, chain, rng, max_iters=iters_per_restart)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
        
        # Check time
        if time.time() - start_time > time_budget:
            break
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [[x, y] for x, y in best_chain]
# EVOLVE-BLOCK-END
