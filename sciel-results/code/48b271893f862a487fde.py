"""HP lattice protein folding - using simulated annealing with segment moves."""
# EVOLVE-BLOCK-START
import time
import random
import math

# Global deadline
_DEADLINE = time.time() + 185.0


def _remaining():
    return _DEADLINE - time.time()


def _neighbours(point):
    """Return 4 neighbours of a 2D lattice point."""
    x, y = point
    return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]


def _energy(sequence, chain):
    """Compute HP energy: -1 for each non-consecutive H-H contact."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _is_valid(chain):
    """Check if chain is self-avoiding with unit steps."""
    if not chain:
        return True
    seen = set()
    for p in chain:
        if p in seen:
            return False
        seen.add(p)
    for i in range(len(chain) - 1):
        dist = abs(chain[i][0] - chain[i+1][0]) + abs(chain[i][1] - chain[i+1][1])
        if dist != 1:
            return False
    return True


def _initial_chain(sequence):
    """Create a valid initial chain using a snake pattern."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        # Try to continue in a snake pattern
        # Determine direction based on position
        x, y = last
        
        # Simple strategy: go right, then down, then left, then up, repeat
        # But we need to avoid occupied cells
        candidates = []
        for q in _neighbours(last):
            if q not in occupied:
                # Prefer continuing in the same general direction
                candidates.append(q)
        
        if not candidates:
            # If stuck, just extend in a new direction (may cause issues)
            # For safety, extend right or down
            q = (x + 1, y)
            if q in occupied:
                q = (x, y + 1)
            if q in occupied:
                q = (x - 1, y)
            if q in occupied:
                q = (x, y - 1)
            if q in occupied:
                # This shouldn't happen for reasonable n
                return [(i, 0) for i in range(n)]
            candidates = [q]
        
        # Choose the candidate that maximizes potential contacts
        best_q = candidates[0]
        best_score = -1
        for q in candidates:
            score = 0
            if sequence[i] == 'H':
                for r in _neighbours(q):
                    j = occupied.get(r)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        score += 1
            # Add small preference for free space
            free = sum(1 for r in _neighbours(q) if r not in occupied)
            score += free * 0.1
            if score > best_score:
                best_score = score
                best_q = q
        
        chain.append(best_q)
        occupied[best_q] = i
    
    return chain


def _sa_fold(sequence, chain, time_limit):
    """
    Simulated annealing with single-residue moves and segment moves.
    Returns the best chain found.
    """
    n = len(chain)
    if n <= 1:
        return chain
    
    rng = random.Random(123 + len(sequence))
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Temperature schedule
    T_start = 5.0
    T_end = 0.1
    total_time = time_limit
    start_time = time.time()
    
    # Build position map for fast lookup
    pos_map = {p: i for i, p in enumerate(current_chain)}
    
    iteration = 0
    max_iterations = int(total_time * 10000)  # Estimate
    
    while time.time() - start_time < total_time:
        if _remaining() < 2.0:
            break
        
        # Choose move type
        move_type = rng.random()
        new_chain = None
        new_energy = None
        
        if move_type < 0.7:
            # Single residue move
            i = rng.randint(1, n - 2)
            p = current_chain[i]
            for q in _neighbours(p):
                if q == current_chain[i-1] or q == current_chain[i+1]:
                    continue
                if q in pos_map and pos_map[q] != i:
                    continue
                
                # Check unit steps
                prev = current_chain[i-1]
                nxt = current_chain[i+1]
                if abs(q[0] - prev[0]) + abs(q[1] - prev[1]) != 1:
                    continue
                if abs(q[0] - nxt[0]) + abs(q[1] - nxt[1]) != 1:
                    continue
                
                # Compute energy delta
                delta = 0
                if sequence[i] == 'H':
                    # Remove old contacts
                    for r in _neighbours(p):
                        j = pos_map.get(r)
                        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                            delta += 1
                    # Add new contacts
                    for r in _neighbours(q):
                        j = pos_map.get(r)
                        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                            delta -= 1
                
                new_energy = current_energy + delta
                
                # Accept?
                if new_energy < current_energy or rng.random() < math.exp(-delta / max(T_start, 0.01)):
                    # Apply move
                    del pos_map[p]
                    current_chain[i] = q
                    pos_map[q] = i
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    break
                
        elif move_type < 0.9:
            # Segment move: rotate a short segment
            seg_len = rng.choice([2, 3, 4, 5])
            if n > seg_len + 2:
                i = rng.randint(1, n - seg_len - 1)
                # Segment from i to i+seg_len-1
                # Pivot at i-1, connection at i+seg_len
                
                # Get direction from i-1 to i
                d1 = (current_chain[i][0] - current_chain[i-1][0],
                      current_chain[i][1] - current_chain[i-1][1])
                
                # Perpendicular direction
                perp = (-d1[1], d1[0])
                
                # Try shifting segment by perp
                new_seg = []
                valid = True
                for k in range(seg_len):
                    old_p = current_chain[i + k]
                    new_p = (old_p[0] + perp[0], old_p[1] + perp[1])
                    new_seg.append(new_p)
                
                # Check validity
                for new_p in new_seg:
                    j = pos_map.get(new_p)
                    if j is not None and not (i <= j < i + seg_len):
                        valid = False
                        break
                
                if valid:
                    # Check connections
                    if abs(new_seg[0][0] - current_chain[i-1][0]) + abs(new_seg[0][1] - current_chain[i-1][1]) != 1:
                        valid = False
                    if valid:
                        if abs(new_seg[-1][0] - current_chain[i+seg_len][0]) + abs(new_seg[-1][1] - current_chain[i+seg_len][1]) != 1:
                            valid = False
                    # Check internal steps
                    if valid:
                        for k in range(seg_len - 1):
                            if abs(new_seg[k+1][0] - new_seg[k][0]) + abs(new_seg[k+1][1] - new_seg[k][1]) != 1:
                                valid = False
                                break
                
                if valid:
                    # Compute energy delta
                    delta = 0
                    for k in range(seg_len):
                        old_p = current_chain[i + k]
                        new_p = new_seg[k]
                        if sequence[i + k] == 'H':
                            # Remove old contacts with outside
                            for r in _neighbours(old_p):
                                j = pos_map.get(r)
                                if j is not None and j != i + k and sequence[j] == 'H' and abs(j - (i + k)) > 1:
                                    if not (i <= j < i + seg_len):
                                        delta += 1
                            # Add new contacts with outside
                            for r in _neighbours(new_p):
                                j = pos_map.get(r)
                                if j is not None and j != i + k and sequence[j] == 'H' and abs(j - (i + k)) > 1:
                                    if not (i <= j < i + seg_len):
                                        delta -= 1
                    
                    new_energy = current_energy + delta
                    
                    # Accept?
                    if new_energy < current_energy or rng.random() < math.exp(-delta / max(T_start, 0.01)):
                        # Apply move
                        for k in range(seg_len):
                            old_p = current_chain[i + k]
                            del pos_map[old_p]
                        for k in range(seg_len):
                            current_chain[i + k] = new_seg[k]
                            pos_map[new_seg[k]] = i + k
                        current_energy = new_energy
                        
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(current_chain)
        
        else:
            # Reversal move: reverse a segment
            seg_len = rng.randint(2, min(6, n - 2))
            if n > seg_len + 1:
                i = rng.randint(1, n - seg_len - 1)
                # Reverse segment from i to i+seg_len-1
                
                # Check if reversal is valid (no overlaps)
                new_seg = [current_chain[i + k] for k in range(seg_len - 1, -1, -1)]
                valid = True
                for new_p in new_seg:
                    j = pos_map.get(new_p)
                    if j is not None and not (i <= j < i + seg_len):
                        valid = False
                        break
                
                if valid:
                    # Check connections
                    if abs(new_seg[0][0] - current_chain[i-1][0]) + abs(new_seg[0][1] - current_chain[i-1][1]) != 1:
                        valid = False
                    if valid:
                        if abs(new_seg[-1][0] - current_chain[i+seg_len][0]) + abs(new_seg[-1][1] - current_chain[i+seg_len][1]) != 1:
                            valid = False
                
                if valid:
                    # Compute energy delta
                    delta = 0
                    for k in range(seg_len):
                        old_p = current_chain[i + k]
                        new_p = new_seg[k]
                        if old_p != new_p:  # Only if position changed
                            if sequence[i + k] == 'H':
                                # Remove old contacts with outside
                                for r in _neighbours(old_p):
                                    j = pos_map.get(r)
                                    if j is not None and j != i + k and sequence[j] == 'H' and abs(j - (i + k)) > 1:
                                        if not (i <= j < i + seg_len):
                                            delta += 1
                                # Add new contacts with outside
                                for r in _neighbours(new_p):
                                    j = pos_map.get(r)
                                    if j is not None and j != i + k and sequence[j] == 'H' and abs(j - (i + k)) > 1:
                                        if not (i <= j < i + seg_len):
                                            delta -= 1
                    
                    new_energy = current_energy + delta
                    
                    # Accept?
                    if new_energy < current_energy or rng.random() < math.exp(-delta / max(T_start, 0.01)):
                        # Apply move
                        for k in range(seg_len):
                            old_p = current_chain[i + k]
                            del pos_map[old_p]
                        for k in range(seg_len):
                            current_chain[i + k] = new_seg[k]
                            pos_map[new_seg[k]] = i + k
                        current_energy = new_energy
                        
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(current_chain)
        
        # Cool down
        progress = (time.time() - start_time) / total_time
        T_start = T_start * math.exp(-0.5 * progress)
        T_start = max(T_start, T_end)
    
    return best_chain


def fold(sequence, dimension):
    """
    Find a low-energy conformation for the given HP sequence on a square lattice.
    Uses simulated annealing with multiple restarts.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    # Determine time budget per sequence
    # Total budget is ~185s for all 10 sequences, so ~18s per sequence
    # But we need to be conservative
    time_per_seq = 15.0
    
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = 0  # Unfolded chain has energy 0
    
    # Multiple restarts
    num_restarts = min(5, max(2, int(time_per_seq / 3)))
    
    for restart in range(num_restarts):
        if _remaining() < 5.0:
            break
        
        # Create initial chain
        chain = _initial_chain(sequence)
        
        if not _is_valid(chain):
            # Fallback to straight line
            chain = [(i, 0) for i in range(n)]
        
        # Run SA
        time_for_sa = time_per_seq / num_restarts - 0.5
        if time_for_sa < 1.0:
            time_for_sa = 1.0
        
        improved_chain = _sa_fold(sequence, chain, time_for_sa)
        
        energy = _energy(sequence, improved_chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = improved_chain
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
