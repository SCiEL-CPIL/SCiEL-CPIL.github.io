"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

_DEADLINE = time.time() + 310  # safety margin before 330s kill


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
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
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _mc_fold(sequence, dimension, rng, max_steps, start_temp, init_chain=None):
    """Monte Carlo folding with cooling. Optimized with NumPy and local contact deltas."""
    n = len(sequence)
    if init_chain is None:
        chain = _grow(sequence, dimension, rng)
    else:
        chain = list(init_chain)
        
    if chain is None or len(chain) != n:
        chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    # Precompute H positions and initial energy
    is_H = np.array([1 if c == 'H' else 0 for c in sequence], dtype=np.int8)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    # For fast delta calculation, we can maintain a set of occupied positions
    occupied = set(chain)
    
    # We will use a simplified move set: pivots are good but expensive to check.
    # Let's use a "slide" move (move one residue) and "pivot" (rotate suffix).
    # To keep it fast, we'll implement a fast local move and a standard pivot.
    
    # Precompute neighbors for speed? No, dimension is 3, it's small.
    
    # Optimization: Use a list of H indices for contact counting
    h_indices = [i for i in range(n) if sequence[i] == 'H']
    
    # We will track energy incrementally.
    # State: chain (list of tuples), occupied (set of tuples), energy (int)
    
    # To speed up energy delta calculation for local moves:
    # If we move residue i from pos A to pos B:
    # 1. Remove contacts involving i at A.
    # 2. Add contacts involving i at B.
    # Contacts are H-H pairs at distance 1 (lattice neighbors) that are not consecutive in sequence.
    
    def get_contacts(pos, idx):
        """Return set of indices j such that (i, j) is a contact if i is at pos."""
        contacts = set()
        if sequence[idx] != 'H':
            return contacts
        for q in _neighbours(pos):
            if q in occupied:
                j = chain.index(q) # This is O(N), bad.
                # We need a position-to-index map.
                pass
        return contacts

    # Let's stick to a simpler but robust approach:
    # Recalculate energy only when a move is accepted, but use a fast rejection for invalid moves.
    # However, for 136 residues, recalculating energy is O(N^2) or O(N) with a map.
    # With a map, O(N) is fine.
    
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    def calc_energy_delta(idx, old_pos, new_pos):
        """Calculate change in energy if residue idx moves from old_pos to new_pos."""
        delta = 0
        if sequence[idx] == 'H':
            # Remove old contacts
            for q in _neighbours(old_pos):
                if q in pos_to_idx:
                    j = pos_to_idx[q]
                    if j != idx and abs(i - j) > 1 and sequence[j] == 'H':
                        delta += 1 # We lose this contact (energy goes up by 1, i.e., less negative)
            
            # Add new contacts
            for q in _neighbours(new_pos):
                if q in pos_to_idx:
                    j = pos_to_idx[q]
                    if j != idx and abs(i - j) > 1 and sequence[j] == 'H':
                        delta -= 1 # We gain this contact (energy goes down by 1)
        return delta

    # Note: The above delta calculation assumes the rest of the chain stays fixed.
    # This is only valid for local moves (single residue move).
    # For pivots, the delta is complex.
    
    # Strategy:
    # 1. Use local moves (single residue) with fast delta.
    # 2. Use pivots with full energy recalculation (less frequent).
    
    for step in range(max_steps):
        if time.time() > _DEADLINE - 5:
            break
        
        move_type = rng.random()
        
        if move_type < 0.6:
            # Local move: move one interior residue to a neighbor
            if n < 3:
                continue
            i = rng.randint(1, n - 2)
            old_pos = chain[i]
            neighbors = list(_neighbours(old_pos))
            rng.shuffle(neighbors)
            moved = False
            for new_pos in neighbors:
                if new_pos in occupied:
                    continue
                # Check validity: new_pos is empty.
                # We don't need to check if new_pos is adjacent to old_pos's neighbors in a way that breaks connectivity?
                # Moving a single residue breaks the chain unless we treat it as a "move" that keeps connectivity?
                # No, a single residue move breaks the chain topology (i-1 -> i -> i+1).
                # If we move i, the step i-1 -> i and i -> i+1 might not be unit steps anymore.
                # So single residue moves are NOT valid for self-avoiding walks unless we allow "jumps" which we can't.
                # We must use moves that preserve the walk property.
                # Valid moves:
                # 1. Pivot: Rotate a suffix.
                # 2. Bridge move: Move a segment.
                # 3. "Flip" move? No.
                #
                # Actually, the standard "local move" in SAW optimization is often a "pivot" or a "slide" where a segment is moved.
                # A single residue move is only valid if the new position is adjacent to BOTH i-1 and i+1.
                # This is very restrictive.
                #
                # Let's stick to Pivots as the primary move, and maybe "segment moves" for long chains.
                # For now, let's just do Pivots but optimize the check.
                moved = True # Break inner loop, fall through to pivot logic below
            if not moved:
                continue
            
        # Perform Pivot Move
        if n < 3:
            continue
        pivot = rng.randint(1, n - 2)
        axis = rng.randint(0, dimension - 1)
        sign = rng.choice((-1, 1))
        
        pivot_point = chain[pivot]
        suffix = chain[pivot + 1:]
        
        # Quick check: will the rotated suffix collide?
        # We can pre-calculate the new suffix and check against occupied (excluding the suffix itself).
        occupied_prefix = set(chain[:pivot + 1])
        
        new_suffix = []
        valid = True
        for p in suffix:
            rel = (p[0] - pivot_point[0], p[1] - pivot_point[1], p[2] - pivot_point[2])
            # Rotate
            if axis == 0:
                if sign == 1: new_rel = (rel[0], -rel[2], rel[1])
                else: new_rel = (rel[0], rel[2], -rel[1])
            elif axis == 1:
                if sign == 1: new_rel = (rel[2], rel[1], -rel[0])
                else: new_rel = (-rel[2], rel[1], rel[0])
            else:
                if sign == 1: new_rel = (rel[1], -rel[0], rel[2])
                else: new_rel = (-rel[1], rel[0], rel[2])
            
            new_point = (pivot_point[0] + new_rel[0], pivot_point[1] + new_rel[1], pivot_point[2] + new_rel[2])
            if new_point in occupied_prefix:
                valid = False
                break
            occupied_prefix.add(new_point)
            new_suffix.append(new_point)
        
        if not valid:
            continue
        
        # Move is valid. Calculate new energy.
        # Instead of full recalc, we can try to estimate, but full recalc is safer and O(N).
        # For N=136, O(N) is ~1000 ops, very fast.
        new_chain = chain[:pivot + 1] + new_suffix
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta <= 0 or (start_temp > 0 and rng.random() < min(1.0, max(1e-10, 2.718281828 ** (-delta / start_temp)))):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        # Cool down
        if step % max(1, max_steps // 10) == 0 and start_temp > 0:
            start_temp = max(0.1, start_temp * 0.7)
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation via Multi-start Simulated Annealing with time budget."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Generate diverse initial states
    # More restarts for shorter chains to find global optima quickly.
    num_starts = 50 if n < 50 else 20
    
    candidates = []
    for _ in range(num_starts):
        chain = _grow(sequence, dimension, rng)
        if chain is not None and len(chain) == n:
            e = _energy(sequence, chain)
            candidates.append((e, chain))
            if best_chain is None or e < best_energy:
                best_chain = chain
                best_energy = e
    
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
        candidates = [(best_energy, best_chain)]
    
    # Sort candidates by energy
    candidates.sort(key=lambda x: x[0])
    
    # Phase 2: Deep Annealing on top survivors
    # Allocate time based on remaining budget
    start_time = time.time()
    time_elapsed = start_time - (time.time() - 310) # Roughly 0 at start of fold if called early
    # Actually, let's just use _DEADLINE
    remaining_time = _DEADLINE - time.time() - 5
    
    if remaining_time <= 0:
        return [list(p) for p in best_chain]
    
    # How many survivors to process?
    # For short chains, we can process many. For long chains, fewer but deeper.
    num_survivors = min(len(candidates), 5 if n > 100 else 10)
    
    # Time per survivor
    time_per_survivor = remaining_time / num_survivors
    
    # Estimate steps per second
    # Pivot move is the bottleneck.
    # Let's do a quick calibration on the best chain
    t0 = time.time()
    test_count = 100
    for _ in range(test_count):
        _mc_fold(sequence, dimension, rng, 1, 1.0, init_chain=best_chain) # Dummy run to warm up? No, just measure pivot cost
    # Actually, let's just measure pivot generation cost
    t1 = time.time()
    for _ in range(test_count):
        # Just generate a pivot move logic without full MC
        pivot = rng.randint(1, n - 2)
        axis = rng.randint(0, dimension - 1)
        sign = rng.choice((-1, 1))
        pivot_point = best_chain[pivot]
        suffix = best_chain[pivot + 1:]
        occupied_prefix = set(best_chain[:pivot + 1])
        for p in suffix:
            rel = (p[0] - pivot_point[0], p[1] - pivot_point[1], p[2] - pivot_point[2])
            if axis == 0:
                if sign == 1: new_rel = (rel[0], -rel[2], rel[1])
                else: new_rel = (rel[0], rel[2], -rel[1])
            elif axis == 1:
                if sign == 1: new_rel = (rel[2], rel[1], -rel[0])
                else: new_rel = (-rel[2], rel[1], rel[0])
            else:
                if sign == 1: new_rel = (rel[1], -rel[0], rel[2])
                else: new_rel = (-rel[1], rel[0], rel[2])
            new_point = (pivot_point[0] + new_rel[0], pivot_point[1] + new_rel[1], pivot_point[2] + new_rel[2])
            if new_point in occupied_prefix:
                break
            occupied_prefix.add(new_point)
    t2 = time.time()
    elapsed = t2 - t0
    if elapsed > 0:
        steps_per_sec = test_count / elapsed
    else:
        steps_per_sec = 10000
    
    # Adjust steps per survivor
    max_steps_per_survivor = int(time_per_survivor * steps_per_sec * 0.8)
    max_steps_per_survivor = max(100, min(max_steps_per_survivor, 50000))
    
    # Run Annealing on survivors
    for i in range(num_survivors):
        if time.time() > _DEADLINE - 5:
            break
        
        e, chain = candidates[i]
        # Start temperature higher for better exploration
        start_temp = 2.0 if n < 50 else 1.5
        
        new_chain, new_energy = _mc_fold(sequence, dimension, rng, max_steps_per_survivor, start_temp, init_chain=chain)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_chain = new_chain
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
