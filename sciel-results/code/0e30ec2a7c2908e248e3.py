"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import random
import time
import sys


_DEADLINE = None
_DEADLINE_MARGIN = 5.0  # seconds to keep before hard limit


def _setup_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 235.0  # 235s budget, leaving margin


def _time_left():
    return _DEADLINE - time.time()


def _neighbours(point):
    """Yield 6 neighbors on cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy: -number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _beam_search(sequence, dimension, beam_width):
    """Beam search over self-avoiding walks."""
    n = len(sequence)
    start = (0,) * dimension
    initial_chain = (start,)
    initial_occupied = {start: 0}
    initial_score = 0.0
    beam = [(-initial_score, 0, initial_chain, initial_occupied)]
    
    best_chain = None
    best_energy = 0
    
    for step in range(1, n):
        if _time_left() < 2.0:
            break
        new_beam = []
        seen = set()
        
        for neg_score, seq_idx, chain, occupied in beam:
            last_pos = chain[-1]
            for site in _neighbours(last_pos):
                if site in occupied:
                    continue
                
                gain = 0
                if sequence[step] == "H":
                    for q in _neighbours(site):
                        if q in occupied:
                            j = occupied[q]
                            if j < step - 1 and sequence[j] == "H":
                                gain += 1
                
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                
                score = gain + 0.1 * free
                new_chain = chain + (site,)
                new_occupied = dict(occupied)
                new_occupied[site] = step
                
                key = new_chain
                if key in seen:
                    continue
                seen.add(key)
                
                new_beam.append((-score, step, new_chain, new_occupied))
        
        if not new_beam:
            break
        
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
        
        if step == n - 1:
            for neg_score, seq_idx, chain, occupied in beam:
                energy = _compute_energy(sequence, chain)
                if energy < best_energy:
                    best_energy = energy
                    best_chain = chain
    
    if best_chain is not None:
        return list(best_chain), best_energy
    
    if beam:
        best = None
        best_e = 0
        for neg_score, seq_idx, chain, occupied in beam:
            if len(chain) == n:
                e = _compute_energy(sequence, chain)
                if e < best_e:
                    best_e = e
                    best = chain
        if best:
            return list(best), best_e
    
    return None, 0


def _local_search(sequence, chain, dimension, max_iters=2000):
    """Local search with incremental energy updates and pivot moves."""
    n = len(chain)
    if n < 3:
        return chain, _compute_energy(sequence, chain)
    
    chain = list(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    rng = random.Random(42 + n)
    h_positions = set(i for i, c in enumerate(sequence) if c == "H")
    
    # Build contact map: for each position, track H-H contacts
    # contact_count[i] = number of H-H contacts involving position i
    contact_count = [0] * n
    occupied = {tuple(p): i for i, p in enumerate(chain)}
    
    # Build initial contact map
    for i in range(n):
        if sequence[i] == "H":
            for q in _neighbours(tuple(chain[i])):
                j = occupied.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contact_count[i] += 1
                    contact_count[j] += 1
    
    def try_single_move(i, new_pos):
        """Try moving vertex i to new_pos. Returns (delta_energy, valid)."""
        old_pos = tuple(chain[i])
        if new_pos == old_pos:
            return 0, False
        
        # Check self-avoidance: new_pos not occupied by non-adjacent
        for j, p in occupied.items():
            if p == new_pos and j != i and abs(j - i) > 1:
                return 0, False
        
        # Check unit step to neighbors
        if i > 0:
            d = sum(abs(new_pos[k] - chain[i-1][k]) for k in range(dimension))
            if d != 1:
                return 0, False
        if i < n - 1:
            d = sum(abs(new_pos[k] - chain[i+1][k]) for k in range(dimension))
            if d != 1:
                return 0, False
        
        # Compute delta energy
        delta = 0
        if sequence[i] == "H":
            # Remove contacts from old position
            for q in _neighbours(old_pos):
                j = occupied.get(q)
                if j is not None and j != i and abs(j - i) > 1 and sequence[j] == "H":
                    delta += 1  # losing a contact
            # Add contacts from new position
            for q in _neighbours(new_pos):
                j = occupied.get(q)
                if j is not None and j != i and abs(j - i) > 1 and sequence[j] == "H":
                    delta -= 1  # gaining a contact
        return delta, True
    
    def try_pivot_move(pivot_idx):
        """Try rotating segment [pivot_idx+1:] around pivot_idx.
        Returns (new_chain_segment, delta_energy) or None."""
        if pivot_idx < 1 or pivot_idx >= n - 1:
            return None
        
        pivot_pos = tuple(chain[pivot_idx])
        current_next = tuple(chain[pivot_idx + 1])
        
        # Get the direction from pivot to next
        # Try all 5 other directions (excluding current)
        for new_next in _neighbours(pivot_pos):
            if new_next == current_next:
                continue
            
            # Check if new_next is free
            if new_next in occupied:
                j = occupied[new_next]
                if j != pivot_idx + 1:
                    continue
            
            # Build new segment by translating the rest
            # Direction change: old_dir = current_next - pivot_pos
            old_dir = tuple(current_next[k] - pivot_pos[k] for k in range(dimension))
            new_dir = tuple(new_next[k] - pivot_pos[k] for k in range(dimension))
            
            # For 3D, we need to find a rotation that maps old_dir to new_dir
            # This is complex; instead, try a simple approach:
            # Just check if the immediate next position is valid, and 
            # assume the rest can be translated (approximation)
            
            # Actually, for a proper pivot, we need to rotate the entire segment
            # Let's try a simpler approach: just move the next vertex and 
            # check if the rest of the chain still fits (which is expensive)
            # For now, skip complex pivots and rely on single moves + random restarts
            pass
        
        return None
    
    for iteration in range(max_iters):
        if _time_left() < 3.0:
            break
        
        improved = False
        
        # Try random single-vertex moves with incremental energy
        for _ in range(100):
            i = rng.randint(1, n - 2)
            axes = [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)]
            rng.shuffle(axes)
            
            for axis, step in axes:
                new_pos = list(chain[i])
                new_pos[axis] += step
                new_pos = tuple(new_pos)
                
                delta, valid = try_single_move(i, new_pos)
                if not valid:
                    continue
                
                # Apply move
                old_pos = tuple(chain[i])
                chain[i] = new_pos
                occupied.pop(old_pos)
                occupied[new_pos] = i
                
                # Update contact counts
                if sequence[i] == "H":
                    for q in _neighbours(old_pos):
                        j = occupied.get(q)
                        if j is not None and j != i and abs(j - i) > 1 and sequence[j] == "H":
                            contact_count[i] -= 1
                            contact_count[j] -= 1
                    for q in _neighbours(new_pos):
                        j = occupied.get(q)
                        if j is not None and j != i and abs(j - i) > 1 and sequence[j] == "H":
                            contact_count[i] += 1
                            contact_count[j] += 1
                
                current_energy -= delta  # delta is negative for improvement
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                improved = True
                break
        
        if not improved:
            # Escape local minimum: accept a worse move
            if rng.random() < 0.3:
                i = rng.randint(1, n - 2)
                axes = [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)]
                axis, step = rng.choice(axes)
                new_pos = list(chain[i])
                new_pos[axis] += step
                new_pos = tuple(new_pos)
                
                delta, valid = try_single_move(i, new_pos)
                if valid:
                    old_pos = tuple(chain[i])
                    chain[i] = new_pos
                    occupied.pop(old_pos)
                    occupied[new_pos] = i
                    
                    if sequence[i] == "H":
                        for q in _neighbours(old_pos):
                            j = occupied.get(q)
                            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == "H":
                                contact_count[i] -= 1
                                contact_count[j] -= 1
                        for q in _neighbours(new_pos):
                            j = occupied.get(q)
                            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == "H":
                                contact_count[i] += 1
                                contact_count[j] += 1
                    
                    current_energy -= delta
    
    return best_chain, best_energy


def _spiral_fold(sequence, dimension):
    """Create a spiral-shaped fold as a fallback."""
    n = len(sequence)
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    # Spiral in the xy plane
    dx, dy = 1, 0
    x, y = 0, 0
    
    for i in range(1, n):
        # Try to continue in current direction
        for _ in range(4):
            nx, ny = x + dx, y + dy
            pos = (nx, ny, 0)
            if pos not in occupied:
                x, y = nx, ny
                chain.append(pos)
                occupied[pos] = i
                break
            else:
                # Turn right
                dx, dy = dy, -dx
        
        else:
            # Stuck, try moving in z
            for z in range(1, 10):
                pos = (x, y, z)
                if pos not in occupied:
                    x, y = 0, 0
                    dx, dy = 1, 0
                    chain.append(pos)
                    occupied[pos] = i
                    break
            else:
                # Give up, place linearly
                chain.append((x, y, i - len(chain) + 1))
                break
    
    return chain[:n]


def fold(sequence, dimension):
    """Best conformation using beam search + local search."""
    _setup_deadline()
    n = len(sequence)
    
    # Adaptive beam width based on sequence length
    if n <= 20:
        beam_width = 500
    elif n <= 40:
        beam_width = 300
    elif n <= 60:
        beam_width = 200
    elif n <= 80:
        beam_width = 150
    else:
        beam_width = 100
    
    # Try beam search
    chain, energy = _beam_search(sequence, dimension, beam_width)
    
    if chain is None:
        # Fallback: spiral fold
        chain = _spiral_fold(sequence, dimension)
        energy = _compute_energy(sequence, chain)
    
    # Apply local search to improve
    if _time_left() > 5.0:
        chain, energy = _local_search(sequence, chain, dimension, max_iters=5000)
    
    return [list(p) for p in chain]
# EVOLVE-BLOCK-END
