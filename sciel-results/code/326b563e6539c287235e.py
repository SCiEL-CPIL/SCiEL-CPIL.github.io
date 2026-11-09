"""Improved HP lattice folding using Simulated Annealing with kink moves."""
# EVOLVE-BLOCK-START
import random
import math
import time

def _compute_initial_energy(sequence, chain):
    """Compute initial energy and build contact map."""
    n = len(chain)
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    energy = 0
    # Count H-H contacts
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            # Check 4 neighbors
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                j = pos_to_idx.get((x+dx, y+dy))
                if j is not None and abs(i-j) > 1 and sequence[j] == 'H':
                    # Avoid double counting: only count if i < j
                    if i < j:
                        energy -= 1
    return energy

def _generate_random_walk(n, rng):
    """Generate a random self-avoiding walk."""
    if n == 1:
        return [(0,0)]
    
    # Start with a random direction
    dirs = [(1,0), (-1,0), (0,1), (0,-1)]
    rng.shuffle(dirs)
    start_dir = dirs[0]
    
    chain = [(0,0)]
    occupied = {(0,0)}
    
    # Try to build a walk
    # For short chains, random walk is easy. For longer, it might fail.
    # We use a simple random walk with backtracking if stuck? No, just restart.
    # But for n>20, random walk is hard to keep self-avoiding.
    # Better: Use a "greedy" random walk that prefers empty spots, but it's still prone to trapping.
    # Let's try a simple random walk. If it fails, we return None.
    
    for i in range(1, n):
        current = chain[-1]
        options = []
        for dx, dy in dirs:
            new_pos = (current[0]+dx, current[1]+dy)
            if new_pos not in occupied:
                options.append(new_pos)
        
        if not options:
            return None
        
        # Random choice
        new_pos = rng.choice(options)
        chain.append(new_pos)
        occupied.add(new_pos)
        
    return chain

def fold(sequence, dimension):
    if dimension != 2:
        raise ValueError("Only 2D is supported")
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer for 120s budget
    
    # Parameters for Simulated Annealing
    # Kink moves: move a segment of length 2 (i, i+1) by 180 degrees?
    # No, standard kink move: pick i, move (i, i+1) such that i-1 -> i -> i+1 becomes i-1 -> i' -> i+1?
    # Actually, the most common efficient move is the "180-degree turn" or "kink".
    # Let's use the "kink" move: pick i (1 <= i <= n-2).
    # The segment (i, i+1) is moved.
    # Wait, standard kink move:
    # Original: ... A - B - C ...
    # Move B to B' such that A-B' is a unit step and B'-C is a unit step?
    # This implies B' is one of the neighbors of A or C?
    # Actually, if we move B, we break the chain.
    # A better move: "Pivot" or "Kink".
    # Let's use the "Kink" move where we rotate the segment (i, i+1) around i?
    # No, that's not self-avoiding easily.
    
    # Let's use the standard "180-degree turn" (also known as kink):
    # Pick i. The residues i and i+1 are moved.
    # Actually, the most effective simple move is moving a single residue i (1..n-2) to a new position
    # such that it remains connected to i-1 and i+1.
    # This is what the old code did, but it was slow.
    
    # Let's optimize the single-residue move with incremental energy.
    
    best_chain = None
    best_energy = 0 # Maximize negative energy (minimize value)
    
    # We will run multiple restarts if time permits.
    
    def run_sa(chain, max_iter):
        """Run Simulated Annealing on the given chain."""
        n = len(chain)
        if n <= 2:
            return chain, _compute_initial_energy(sequence, chain)
        
        current_chain = [tuple(p) for p in chain]
        current_energy = _compute_initial_energy(sequence, current_chain)
        
        # Build pos_to_idx for incremental updates
        pos_to_idx = {p: i for i, p in enumerate(current_chain)}
        
        # Precompute H residues
        is_h = [c == 'H' for c in sequence]
        
        # Helper to update energy when moving i to new_pos
        def calc_delta_energy(i, old_pos, new_pos):
            """Calculate change in energy when moving residue i from old_pos to new_pos."""
            delta = 0
            # Remove contacts involving i at old_pos
            if is_h[i]:
                x, y = old_pos
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    j = pos_to_idx.get((x+dx, y+dy))
                    if j is not None and abs(i-j) > 1 and is_h[j]:
                        if i < j:
                            delta += 1 # Energy was -1, removing it means +1 change
                        else:
                            delta += 1 # Energy was -1 (counted as j<i), removing it means +1 change
                            # Wait, if we counted i<j, we subtracted 1. If we remove contact, energy increases by 1.
                            # So delta += 1.
            
            # Add contacts involving i at new_pos
            if is_h[i]:
                x, y = new_pos
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    j = pos_to_idx.get((x+dx, y+dy))
                    if j is not None and abs(i-j) > 1 and is_h[j]:
                        # Check if this contact is valid (i.e., j is not moving, which it isn't)
                        # We must ensure we don't double count if we already counted it in removal?
                        # No, old_pos and new_pos are different.
                        # But we must ensure that the neighbor j is not i itself (impossible) or i-1/i+1 (checked).
                        # We count if i < j to avoid double counting in the total sum?
                        # In _compute_initial_energy, we counted if i < j.
                        # So here, we should only add if i < j.
                        if i < j:
                            delta -= 1 # New contact formed, energy decreases by 1
            
            return delta

        T = 1.0
        T_min = 0.01
        cooling = 0.9999
        rng = random.Random(12345 + n)
        
        best_chain_local = list(current_chain)
        best_energy_local = current_energy
        
        for _ in range(max_iter):
            if T < T_min:
                break
            
            # Pick a random residue to move (1 to n-2)
            i = rng.randint(1, n - 2)
            
            # Find valid moves for i
            old_pos = current_chain[i]
            x, y = old_pos
            
            # Neighbors of old_pos
            candidates = []
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                new_pos = (x+dx, y+dy)
                
                # Check if new_pos is occupied (excluding i itself)
                if new_pos in pos_to_idx:
                    idx = pos_to_idx[new_pos]
                    if idx != i:
                        continue # Occupied by another residue
                
                # Check connectivity to i-1
                prev_pos = current_chain[i-1]
                if abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1]) != 1:
                    continue
                
                # Check connectivity to i+1
                next_pos = current_chain[i+1]
                if abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1]) != 1:
                    continue
                
                candidates.append(new_pos)
            
            if not candidates:
                continue
            
            new_pos = rng.choice(candidates)
            
            # Calculate delta energy
            delta_e = calc_delta_energy(i, old_pos, new_pos)
            
            # Acceptance criterion
            if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
                # Accept move
                # Update pos_to_idx
                del pos_to_idx[old_pos]
                pos_to_idx[new_pos] = i
                current_chain[i] = new_pos
                current_energy += delta_e
                
                if current_energy < best_energy_local:
                    best_energy_local = current_energy
                    best_chain_local = list(current_chain)
            
            T *= cooling
            
            # Time check
            if time.time() - start_time > time_limit:
                break
        
        return best_chain_local, best_energy_local

    # Try a few different initial chains
    # 1. Random walk
    # 2. Straight line (unfolding)
    
    # Strategy: Run SA on a random walk. If it fails, try another.
    
    attempts = 0
    max_attempts = 10 # Limit attempts to avoid overhead
    
    while time.time() - start_time < time_limit and attempts < max_attempts:
        # Generate random walk
        chain = _generate_random_walk(n, random.Random(attempts))
        if chain is None:
            # If random walk fails (trapped), try a straight line?
            # Straight line is always valid.
            chain = [(i, 0) for i in range(n)]
        
        # Run SA
        # Adjust max_iter based on time remaining
        remaining = time_limit - (time.time() - start_time)
        # Estimate iterations per second? Hard to know.
        # Let's just run a fixed number of iterations per attempt, say 50000.
        # Or run until time runs out.
        # Better: Run SA for a chunk of time.
        
        # Let's run SA with a high max_iter, but break on time.
        res_chain, res_energy = run_sa(chain, max_iter=100000)
        
        if best_chain is None or res_energy < best_energy:
            best_chain = res_chain
            best_energy = res_energy
            
        attempts += 1
        
        # If we found a very good energy, we might stop early?
        # Hard to know "very good" without best known.
        # Let's just keep trying until time is up.

    if best_chain is None:
        # Fallback
        best_chain = [(i, 0) for i in range(n)]
        
    return [[p[0], p[1]] for p in best_chain]

# EVOLVE-BLOCK-END
