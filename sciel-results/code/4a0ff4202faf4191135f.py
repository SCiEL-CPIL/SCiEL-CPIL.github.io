"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _build_h_set(sequence):
    return set(i for i, c in enumerate(sequence) if c == 'H')


def _initial_grow(sequence, dimension, rng):
    """Greedy growth to get a valid self-avoiding walk."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain * 5.0 + free * 0.1 + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _compute_energy(sequence, chain, h_set):
    """Compute energy from scratch."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i in h_set:
        p = chain[i]
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and j > i + 1 and j in h_set:
                contacts += 1
    return -contacts


def _delta_single_move(sequence, chain, pos, new_pos, h_set):
    """Compute delta energy for moving residue at index pos to new_pos.
    Returns None if move is invalid (self-avoiding violation or adjacency violation)."""
    i = pos
    # Check self-avoiding: new_pos must not be occupied by any other residue
    # and must not be adjacent to chain[i-1] or chain[i+1] (except the one we're moving from)
    old_pos = chain[i]
    
    # new_pos must not be in chain except at position i
    # Build set of occupied positions excluding i
    # For efficiency, we check against neighbors of old_pos and the path
    # Actually we need to check if new_pos is occupied by any j != i
    # And new_pos must be adjacent to old_pos? No, single move means we pick a new position
    # adjacent to old_pos that is not occupied.
    
    # Check if new_pos is occupied
    # We need a quick lookup. Let's use the fact that we can check neighbors.
    # Actually for a single move, new_pos must be a neighbor of old_pos.
    # Check if new_pos is in the chain at some other position.
    # This is O(n) in worst case. For better performance, maintain a position->index map.
    # But for delta computation during SA, we can afford O(n) if n is moderate.
    # For n up to 136, O(n) per move is acceptable.
    
    # Check self-avoiding
    for j, p in enumerate(chain):
        if j != i and p == new_pos:
            return None
    
    # Check adjacency: new_pos must not be adjacent to chain[i-1] or chain[i+1]
    # because those would create non-consecutive adjacency (which is allowed for contacts)
    # Actually, the constraint is: consecutive residues must be adjacent.
    # Moving residue i to new_pos: we need chain[i-1] adjacent to new_pos AND new_pos adjacent to chain[i+1].
    # Wait, no. The chain is a path. If we move residue i, we break the path.
    # A single-residue move in SA for self-avoiding walks is tricky because it breaks the path.
    # Better approach: use segment moves or use a different representation.
    
    # Actually, for HP folding, a common approach is to use "moves" that preserve the self-avoiding walk property.
    # Simplest valid moves: 
    # 1. Move a terminal residue (first or last) to a new position adjacent to its neighbor.
    # 2. Reverse a segment.
    # 3. Move a segment.
    
    # Let's use a different approach: generate moves that are guaranteed valid.
    # For a single residue move, only terminals can be moved easily.
    # Or we can use "pivot" moves: pick a point and rotate the tail.
    
    # This is getting complex. Let me use a simpler but effective approach:
    # Use only moves that are known to preserve self-avoiding walk:
    # 1. Terminal move: move first or last residue to an adjacent free site.
    # 2. Segment reversal: reverse a contiguous segment (preserves adjacency).
    # 3. Segment move: move a contiguous segment to a new location (more complex).
    
    # For simplicity and correctness, let's use terminal moves and segment reversals.
    return None  # We'll handle valid moves in the SA loop


def _try_terminal_move(sequence, chain, h_set, which_end):
    """Try to move the terminal residue (index 0 or len-1) to a new adjacent position.
    Returns (new_chain, delta_energy) or None if no valid move."""
    i = which_end
    old_pos = chain[i]
    neighbor_idx = 1 if i == 0 else len(chain) - 2
    neighbor_pos = chain[neighbor_idx]
    
    # Find all free neighbors of old_pos that are adjacent to neighbor_pos
    # Actually, the new position must be adjacent to neighbor_pos (to maintain chain connectivity)
    # and must not be occupied by any other residue.
    
    for new_pos in _neighbours(neighbor_pos):
        # Check if new_pos is occupied by any residue other than i
        valid = True
        for j, p in enumerate(chain):
            if j != i and p == new_pos:
                valid = False
                break
        if not valid:
            continue
        # Also new_pos must not be the same as old_pos (that's a no-op)
        if new_pos == old_pos:
            continue
        # Check that new_pos is adjacent to neighbor_pos (already guaranteed by generation)
        # Compute delta energy
        # Remove old_pos from consideration, add new_pos
        # We need to count H-H contacts involving position i
        
        # Build where map for current chain
        where = {p: idx for idx, p in enumerate(chain)}
        
        # Count contacts lost (involving old_pos and H at i)
        contacts_lost = 0
        if i in h_set:
            for q in _neighbours(old_pos):
                j = where.get(q)
                if j is not None and j != i and j != neighbor_idx and j in h_set:
                    # Check if this contact is counted (j > i or i > j, but we count each pair once)
                    # In _compute_energy, we count pair (i,j) if i < j and both H and adjacent
                    # So if i < j, this contact is counted; if j < i, it's counted when processing j
                    # Let's just count all H-H adjacencies involving i
                    contacts_lost += 1
        
        # Count contacts gained (involving new_pos and H at i)
        contacts_gained = 0
        if i in h_set:
            for q in _neighbours(new_pos):
                j = where.get(q)
                if j is not None and j != i and j != neighbor_idx and j in h_set:
                    contacts_gained += 1
        
        delta = -(contacts_gained - contacts_lost)
        
        # Apply the move
        new_chain = list(chain)
        new_chain[i] = new_pos
        
        return new_chain, delta
    
    return None


def _try_segment_reversal(sequence, chain, h_set, start, end):
    """Reverse the segment from start to end (inclusive). Returns (new_chain, delta_energy) or None."""
    if start >= end:
        return None
    
    # Check that the reversal is valid: after reversal, chain[start-1] must be adjacent to chain[end]
    # and chain[end+1] must be adjacent to chain[start]
    # (assuming start > 0 and end < len-1)
    
    # For simplicity, only do reversals where the endpoints maintain adjacency
    # This is complex to check. Let's skip this for now and use a simpler approach.
    return None


def fold(sequence, dimension):
    """Best conformation using simulated annealing with time budget."""
    n = len(sequence)
    h_set = _build_h_set(sequence)
    
    # Time budget: aim to finish well before 330s
    # Reserve time for this call; the evaluator calls fold once per sequence
    # but we don't know how many sequences. Let's use a per-call budget.
    # Actually, the problem says 240s CPU budget for ALL sequences together.
    # So we need to be efficient. Let's use a fixed number of iterations or a time limit.
    
    # Use a time-based approach: spend up to 10 seconds per sequence (adjustable)
    # But we don't know total time. Let's use a heuristic: limit iterations based on n.
    
    # For robustness, use a combination: max iterations and a time check.
    start_time = time.time()
    time_limit = 8.0  # seconds per sequence, should be safe for 26 sequences
    
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0  # energy is negative, so 0 means no contacts
    
    # Generate initial conformation
    init_chain = _initial_grow(sequence, dimension, rng)
    if init_chain is None:
        # Fallback: straight line
        init_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    best_chain = init_chain
    best_energy = _compute_energy(sequence, init_chain, h_set)
    
    # Simulated annealing
    current_chain = list(init_chain)
    current_energy = best_energy
    
    # SA parameters
    T0 = 2.0
    Tend = 0.01
    # Number of iterations: scale with n
    n_iter = max(5000, min(50000, n * 200))
    
    for iteration in range(n_iter):
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        # Cooling schedule
        T = T0 * (Tend / T0) ** (iteration / n_iter)
        
        # Generate a move: terminal move
        which_end = rng.choice([0, n - 1])
        result = _try_terminal_move(sequence, current_chain, h_set, which_end)
        if result is None:
            # No valid terminal move, try the other end
            which_end = 1 if which_end == 0 else 0
            result = _try_terminal_move(sequence, current_chain, h_set, which_end)
            if result is None:
                # Stuck, break or restart
                break
        
        new_chain, delta = result
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
            current_chain = new_chain
            current_energy += delta
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
    
    # If we got stuck, try a few more random restarts
    restarts = 3
    for r in range(restarts):
        if time.time() - start_time > time_limit:
            break
        
        init_chain = _initial_grow(sequence, dimension, rng)
        if init_chain is None:
            continue
        
        e = _compute_energy(sequence, init_chain, h_set)
        if e < best_energy:
            best_energy = e
            best_chain = list(init_chain)
    
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
