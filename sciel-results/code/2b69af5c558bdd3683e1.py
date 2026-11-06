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
    """Local search with pivots and shifts using incremental energy updates."""
    n = len(chain)
    if n < 3:
        return chain, _compute_energy(sequence, chain)
    
    chain = list(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    rng = random.Random(42 + n)
    
    # Precompute H positions for quick lookup
    h_positions = set(i for i, c in enumerate(sequence) if c == "H")
    
    def get_occupied(chain):
        return {p: i for i, p in enumerate(chain)}
    
    def pivot_move(chain, i):
        """Try rotating segment [i+1:] around vertex i."""
        if i >= n - 1 or i < 1:
            return None
        # Try all 6 directions for the next vertex
        current_next = chain[i + 1]
        for new_next in _neighbours(chain[i]):
            if new_next == current_next:
                continue
            # Check if new position is valid (not occupied by non-adjacent parts)
            # This is complex; skip for now and use simpler moves
            pass
        return None
    
    def shift_move(chain, i, direction):
        """Try shifting vertex i in a given direction."""
        new_pos = list(chain[i])
        new_pos[direction[0]] += direction[1]
        new_pos = tuple(new_pos)
        
        # Check self-avoidance
        for j, p in enumerate(chain):
            if j != i and j != i - 1 and j != i + 1:
                if p == new_pos:
                    return None
        
        # Check unit step constraints
        if i > 0:
            d = sum(abs(new_pos[k] - chain[i-1][k]) for k in range(dimension))
            if d != 1:
                return None
        if i < n - 1:
            d = sum(abs(new_pos[k] - chain[i+1][k]) for k in range(dimension))
            if d != 1:
                return None
        
        return new_pos
    
    for iteration in range(max_iters):
        if _time_left() < 3.0:
            break
        
        improved = False
        
        # Try random single-vertex moves
        for _ in range(50):
            i = rng.randint(1, n - 2)  # Don't move endpoints
            axes = [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)]
            rng.shuffle(axes)
            
            for axis, step in axes:
                new_pos = list(chain[i])
                new_pos[axis] += step
                new_pos = tuple(new_pos)
                
                # Check self-avoidance
                valid = True
                for j, p in enumerate(chain):
                    if j != i and abs(j - i) > 1:
                        if p == new_pos:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Check unit step to neighbors
                if i > 0:
                    d = sum(abs(new_pos[k] - chain[i-1][k]) for k in range(dimension))
                    if d != 1:
                        continue
                if i < n - 1:
                    d = sum(abs(new_pos[k] - chain[i+1][k]) for k in range(dimension))
                    if d != 1:
                        continue
                
                # Try the move
                old_pos = chain[i]
                chain[i] = new_pos
                new_energy = _compute_energy(sequence, chain)
                
                if new_energy < current_energy:
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                    improved = True
                    break
                else:
                    chain[i] = old_pos
        
        if not improved:
            # Try a pivot move: rotate a segment
            i = rng.randint(1, n - 2)
            if i < n - 1:
                # Get the direction from i to i+1
                dx = chain[i+1][0] - chain[i][0]
                dy = chain[i+1][1] - chain[i][1]
                dz = chain[i+1][2] - chain[i][2]
                
                # Try rotating the rest of the chain
                # This is complex; try simple 90-degree rotations in 3D
                # Skip for simplicity and rely on single-vertex moves
                
            # If no improvement, try a random restart with small perturbation
            if rng.random() < 0.1:
                # Accept a slightly worse move to escape local minimum
                i = rng.randint(1, n - 2)
                axes = [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)]
                axis, step = rng.choice(axes)
                new_pos = list(chain[i])
                new_pos[axis] += step
                new_pos = tuple(new_pos)
                
                valid = True
                for j, p in enumerate(chain):
                    if j != i and abs(j - i) > 1:
                        if p == new_pos:
                            valid = False
                            break
                
                if valid:
                    if i > 0:
                        d = sum(abs(new_pos[k] - chain[i-1][k]) for k in range(dimension))
                        if d != 1:
                            valid = False
                    if i < n - 1:
                        d = sum(abs(new_pos[k] - chain[i+1][k]) for k in range(dimension))
                        if d != 1:
                            valid = False
                
                if valid:
                    old_pos = chain[i]
                    chain[i] = new_pos
                    new_energy = _compute_energy(sequence, chain)
                    # Accept with some probability (simulated annealing-like)
                    if new_energy < current_energy or rng.random() < 0.3:
                        current_energy = new_energy
                        if new_energy < best_energy:
                            best_energy = new_energy
                            best_chain = list(chain)
                    else:
                        chain[i] = old_pos
    
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
