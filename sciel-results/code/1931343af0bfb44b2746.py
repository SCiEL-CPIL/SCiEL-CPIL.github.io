"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 170.0

# Directions: 0:R, 1:D, 2:L, 3:U
DIRS = [(1, 0), (0, -1), (-1, 0), (0, 1)]
OPPOSITE = {0: 2, 2: 0, 1: 3, 3: 1}

def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    n = len(chain)
    if n < 2:
        return 0
    # Map position to index
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y = p
            for dx, dy in DIRS:
                q = (x + dx, y + dy)
                j = pos_to_idx.get(q)
                if j is not None:
                    # Check if j is adjacent in sequence (i-1 or i+1)
                    if abs(i - j) > 1 and (i + j) % (2 * n) != 0: # Handle circular adjacency if needed, but chain is linear here
                        if sequence[j] == 'H':
                            contacts += 1
    return -contacts // 2

def _generate_hamiltonian_cycle(n, rng):
    """
    Generate a random Hamiltonian cycle on a square grid large enough to fit n residues.
    For simplicity, we use a 'snake' pattern on a bounding box or a random SAW extended to a cycle.
    However, generating a random Hamiltonian cycle for arbitrary n is hard.
    Alternative: Use a fixed 'snake' Hamiltonian path on a grid that fits n, and treat it as the base.
    Then we map the sequence to positions in this cycle.
    
    Actually, a better approach for HP folding is to fix the *lattice sites* in a cycle (or path) 
    and permute the *sequence*? No, the sequence order is fixed.
    
    Let's stick to the standard approach but change the search space.
    Instead of growing a chain, we start with a random Self-Avoiding Walk (SAW) of length n.
    Then we use 2-opt moves (reversing a segment of the walk) to improve energy.
    This requires the walk to remain self-avoiding.
    
    A 2-opt move on a SAW:
    Take indices i and j. Reverse the segment chain[i:j+1].
    Check if the new segment is self-avoiding and doesn't clash with the rest.
    This is expensive to check naively O(n^2).
    
    Let's use a different structural change: **Simulated Annealing with Regrow Moves on a Randomized Initial SAW**.
    The current program uses Greedy Growth. 
    New Strategy: **Random SAW Generation + Pivot/Regrow SA**.
    
    1. Generate a random SAW of length n.
    2. Use SA with moves:
       a. Pivot: Rotate a random suffix.
       b. Regrow: Pick a split point, regrow the suffix greedily with noise.
    """
    pass

def _generate_random_saw(n, rng):
    """Generate a random Self-Avoiding Walk of length n on 2D lattice."""
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0)}
    last_dir = -1 # No previous direction
    
    for _ in range(1, n):
        last = chain[-1]
        x, y = last
        options = []
        for d, (dx, dy) in enumerate(DIRS):
            if d == OPPOSITE.get(last_dir, -1):
                continue
            new_pos = (x + dx, y + dy)
            if new_pos not in occupied:
                options.append((d, new_pos))
        
        if not options:
            return None # Dead end
        
        # Randomly choose next step
        d, new_pos = rng.choice(options)
        chain.append(new_pos)
        occupied.add(new_pos)
        last_dir = d
        
    return chain

def _regrow_suffix(chain, split, sequence, rng, temperature=0.0):
    """
    Regrow the suffix of the chain starting from index split.
    chain[split:] is removed and regrown from chain[split-1].
    """
    if split <= 0:
        return None
    if split >= len(chain):
        return chain
    
    prefix = chain[:split]
    suffix_seq = sequence[split:]
    
    # Build occupied set from prefix
    occupied = {p: i for i, p in enumerate(prefix)}
    current_energy_prefix = 0 # We only care about incremental energy
    
    new_chain = list(prefix)
    last = prefix[-1]
    last_dir = -1
    if len(prefix) > 1:
        # Determine last_dir from prefix[-2] to prefix[-1]
        dx = last[0] - prefix[-2][0]
        dy = last[1] - prefix[-2][1]
        for d, (ddx, ddy) in enumerate(DIRS):
            if ddx == dx and ddy == dy:
                last_dir = d
                break
    
    for i, res in enumerate(suffix_seq):
        global_idx = split + i
        options = []
        x, y = last
        for d, (dx, dy) in enumerate(DIRS):
            if d == OPPOSITE.get(last_dir, -1):
                continue
            new_pos = (x + dx, y + dy)
            if new_pos in occupied:
                continue
            
            # Calculate gain in energy
            gain = 0
            if res == 'H':
                # Check neighbors of new_pos in occupied
                nx, ny = new_pos
                for ndx, ndy in DIRS:
                    npos = (nx + ndx, ny + ndy)
                    j = occupied.get(npos)
                    if j is not None:
                        # Check if j is adjacent in sequence to global_idx
                        # j is in prefix, so j < global_idx
                        # Adjacent if j == global_idx - 1 (which is last, already checked by no-backtrack? No, last is chain[-1], j could be last)
                        # If j == global_idx - 1, it's a bond, not a contact.
                        if j != global_idx - 1:
                            if sequence[j] == 'H':
                                gain += 1
            
            # Free space heuristic
            free = 0
            for ndx, ndy in DIRS:
                npos = (new_pos[0] + ndx, new_pos[1] + ndy)
                if npos not in occupied:
                    free += 1
            
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, d, new_pos))
        
        if not options:
            return None
        
        # Choose best option
        best_score, best_d, best_pos = max(options, key=lambda x: x[0])
        new_chain.append(best_pos)
        occupied[best_pos] = global_idx
        last = best_pos
        last_dir = best_d
        
    return new_chain

def _pivot_move(chain, pivot_idx, rng):
    """
    Rotate the suffix of the chain starting at pivot_idx around the pivot point.
    This is a 2D rotation.
    """
    if pivot_idx <= 0 or pivot_idx >= len(chain):
        return chain
    
    pivot = chain[pivot_idx]
    prefix = chain[:pivot_idx]
    suffix = chain[pivot_idx:]
    
    # Translate suffix so pivot is at origin
    tx, ty = pivot
    translated = [(x - tx, y - ty) for x, y in suffix]
    
    # Rotate 90, 180, or 270 degrees
    angle = rng.choice([1, 2, 3]) # 1:90, 2:180, 3:270
    rotated = []
    for x, y in translated:
        if angle == 1: # 90 deg CCW: (x, y) -> (-y, x)
            rotated.append((-y, x))
        elif angle == 2: # 180 deg: (x, y) -> (-x, -y)
            rotated.append((-x, -y))
        else: # 270 deg CCW: (x, y) -> (y, -x)
            rotated.append((y, -x))
    
    # Translate back
    new_suffix = [(x + tx, y + ty) for x, y in rotated]
    
    new_chain = prefix + new_suffix
    
    # Check self-avoidance
    seen = set()
    for p in new_chain:
        if p in seen:
            return None
        seen.add(p)
        
    # Check connectivity (should be preserved by rotation if pivot is fixed)
    # But we need to ensure no clashes with prefix
    # The check above `if p in seen` handles self-clash.
    # We also need to ensure the first step of suffix doesn't clash with prefix[-1]
    # Actually, the rotation keeps pivot fixed, so chain[pivot_idx-1] is connected to pivot.
    # The rest of the suffix is rotated.
    # We must check if any point in new_suffix clashes with prefix[:-1]
    prefix_set = set(prefix[:-1])
    for p in new_suffix:
        if p in prefix_set:
            return None
            
    return new_chain

def fold(sequence, dimension):
    """
    Fold the sequence using Random SAW initialization and Simulated Annealing
    with Pivot and Regrow moves.
    """
    rng = random.Random(42)
    n = len(sequence)
    
    if n == 1:
        return [[0, 0]]
    
    best_chain = None
    best_energy = 0
    
    # Time allocation
    # For short sequences, random SAW + SA is very fast.
    # For long sequences, we need more time for SA.
    
    # Phase 1: Generate initial random SAWs
    # Try a few random SAWs to get a diverse starting point
    start_time = time.time()
    
    # Attempt to generate a valid SAW
    # If n is large, random SAW might fail or be very slow.
    # We limit the number of attempts.
    
    candidate_chains = []
    for _ in range(10):
        if time.time() > _DEADLINE - 10:
            break
        chain = _generate_random_saw(n, rng)
        if chain:
            energy = _compute_energy(sequence, chain)
            candidate_chains.append((energy, chain))
            
    if not candidate_chains:
        # Fallback: straight line
        return [[i, 0] for i in range(n)]
        
    # Sort by energy
    candidate_chains.sort(key=lambda x: x[0])
    best_energy, best_chain = candidate_chains[0]
    
    # Phase 2: Simulated Annealing
    # We run SA on the best candidate and maybe a few others if time permits.
    
    current_chain = best_chain
    current_energy = best_energy
    
    # SA Parameters
    # Temperature schedule
    start_temp = 5.0
    end_temp = 0.1
    cool_rate = 0.995
    temp = start_temp
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < _DEADLINE - 10:
        if temp < end_temp:
            break
            
        # Choose move
        move_type = rng.random()
        
        new_chain = None
        
        if move_type < 0.5:
            # Regrow move
            # Pick split point
            # Bias towards later splits for larger moves? Or random?
            # Let's try random split between 2 and n-2
            if n > 4:
                split = rng.randint(2, n - 2)
                new_chain = _regrow_suffix(current_chain, split, sequence, rng, temperature=temp)
            else:
                new_chain = _regrow_suffix(current_chain, 1, sequence, rng, temperature=temp)
        else:
            # Pivot move
            if n > 3:
                pivot_idx = rng.randint(1, n - 2)
                new_chain = _pivot_move(current_chain, pivot_idx, rng)
            else:
                new_chain = None
                
        if new_chain is None:
            temp *= cool_rate
            no_improve += 1
            continue
            
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or (temp > 0 and rng.random() < math.exp(-delta / temp)):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        temp *= cool_rate
        
        if no_improve > max_no_improve:
            # Restart SA with a new random SAW if possible, or just break
            # Let's try to restart with a fresh random SAW if we have time
            if time.time() < _DEADLINE - 20:
                fresh_chain = _generate_random_saw(n, rng)
                if fresh_chain:
                    fresh_energy = _compute_energy(sequence, fresh_chain)
                    if fresh_energy < best_energy:
                        best_energy = fresh_energy
                        best_chain = fresh_chain
                        current_chain = fresh_chain
                        current_energy = fresh_energy
                        temp = start_temp
                        no_improve = 0
                        continue
            break
            
    # Phase 3: Local improvement (single residue moves) to polish
    if time.time() < _DEADLINE - 5:
        # Simple local search
        improved = True
        while improved and time.time() < _DEADLINE - 5:
            improved = False
            for i in range(1, n - 1):
                if time.time() > _DEADLINE - 5:
                    break
                p = current_chain[i]
                x, y = p
                for dx, dy in DIRS:
                    np = (x + dx, y + dy)
                    # Check if np is occupied
                    # We need an occupied set for this
                    # This is O(n) per check, so O(n^2) per iteration.
                    # Might be too slow for n=100 if done many times.
                    # Let's skip full local polish if n is large.
                    if n > 50:
                        continue
                    
                    # Check occupancy
                    occupied = set(current_chain)
                    if np in occupied:
                        continue
                    
                    # Check connectivity
                    prev = current_chain[i-1]
                    nxt = current_chain[i+1]
                    if abs(np[0]-prev[0]) + abs(np[1]-prev[1]) != 1:
                        continue
                    if abs(np[0]-nxt[0]) + abs(np[1]-nxt[1]) != 1:
                        continue
                        
                    # Try move
                    current_chain[i] = np
                    new_energy = _compute_energy(sequence, current_chain)
                    if new_energy < current_energy:
                        current_energy = new_energy
                        improved = True
                        if new_energy < best_energy:
                            best_energy = new_energy
                            best_chain = list(current_chain)
                    else:
                        current_chain[i] = p
                        
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
