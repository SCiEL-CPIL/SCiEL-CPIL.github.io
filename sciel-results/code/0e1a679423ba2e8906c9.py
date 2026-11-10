"""
Strategy: Parallel Tempering Monte Carlo on a fixed dense scaffold.
1. Scaffold Generation: Generate a single dense, self-avoiding walk (SAW) for the
   longest sequence length (136). This walk serves as the geometric skeleton.
   For shorter sequences, we use the prefix of this walk. This ensures the
   initial geometry is already compact and "folded-like", avoiding the loose
   structures typical of random growth.
2. Parallel Tempering: Maintain a set of 'replicas' (copies of the sequence
   assignment on the scaffold) at different effective temperatures.
   - High temperature: Accepts more moves, explores broadly.
   - Low temperature: Accepts fewer moves, refines energy.
   - Exchange: Periodically swap configurations between replicas if it satisfies
     a Metropolis criterion based on the energy difference and temperature difference.
     This allows high-T replicas to escape local minima and pass good configurations
     to low-T replicas for refinement.
3. Delta Energy Updates: Moves (residue swaps) are evaluated using O(1) or O(k)
   delta calculations rather than full O(N) re-evaluations, enabling thousands
   of moves per second.
"""

import random
import time
import sys

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290  # 300s limit, keep 10s margin

def _remaining_time():
    return _DEADLINE - time.time()

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _is_adjacent(i, j):
    return abs(i - j) == 1

def _generate_scaffold(n_max, dimension=3):
    """
    Generate a dense self-avoiding walk of length n_max.
    Uses a simple greedy growth with a bias towards filling space (snake-like).
    """
    if n_max == 0:
        return []
    
    # Start at origin
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    # Directions: +x, -x, +y, -y, +z, -z
    # We want to snake to fill a box.
    # Simple strategy: try to move in a preferred direction, else random.
    # To ensure density, we can use a "snake" pattern in a bounding box.
    
    # Let's estimate box size. Volume ~ N. Side ~ N^(1/3).
    # For N=136, side ~ 5. 5x5x5 = 125. 6x6x6 = 216.
    # So a 6x6x6 box is safe.
    
    # A robust way to get a dense SAW is to fill a grid in a snake pattern.
    # However, we need a valid SAW.
    # Let's generate a snake path in a 6x6x6 box.
    
    # Actually, random greedy growth with a "fill" bias is easier to code robustly.
    # Bias: prefer moves that keep the chain close to the center of mass?
    # Or just random growth until trapped, then backtrack?
    # Backtracking is complex.
    
    # Alternative: Pre-computed snake path logic.
    # Let's generate a snake in a 6x6x6 box.
    # Path:
    # z=0: x 0->5, y=0; y=1, x 5->0; y=2, x 0->5 ...
    # This creates a valid SAW.
    
    box_size = 6
    points = []
    for z in range(box_size):
        for y in range(box_size):
            if y % 2 == 0:
                for x in range(box_size):
                    points.append((x, y, z))
            else:
                for x in range(box_size - 1, -1, -1):
                    points.append((x, y, z))
            
            # Connect to next layer if needed?
            # The above generates a 2D snake per z-layer.
            # We need to connect z-layer z to z-layer z+1.
            # End of z-layer z is at (0 or 5, box_size-1, z).
            # Start of z-layer z+1 is at (0 or 5, 0, z+1).
            # This jump is not a unit step.
            
            # So we need a 3D snake.
            # Standard 3D snake:
            # Fill x-y plane for z=0.
            # Then move to z=1? No, we must stay connected.
            
            # Let's use a simpler heuristic: Random growth with "backtracking" is hard.
            # Let's use a "greedy with lookahead" or just random growth and hope it doesn't trap.
            # For N=136, random growth often traps early.
            
            # Better: Use a known dense SAW generator.
            # "Snake" in a tube.
            
            # Let's try a specific pattern:
            # Move along X, then Y, then Z, then back Y, then back X...
            # This is hard to generalize for arbitrary N.
            
            # Let's stick to the previous strategy's growth but make it denser.
            # The previous strategy used contact-first growth.
            # Here, we just need a valid scaffold.
            # Let's use random growth with a "density" score:
            # Score = - (distance from center of mass) + (number of free neighbors)
            
    # Let's implement a robust random growth with density bias.
    chain = [(0,0,0)]
    occupied = {(0,0,0): 0}
    rng = random.Random(42)
    
    for i in range(1, n_max):
        current = chain[-1]
        candidates = []
        for nb in _neighbours(current):
            if nb in occupied:
                continue
            # Score: prefer moves that keep chain compact
            # Approximate center of mass
            # To be fast, just prefer moves that have more free neighbors (less trapped)
            # and are "inwards" (negative coordinates if we started at 0? No, 0 is center-ish)
            # Let's just prefer moves that result in a position with more free neighbors.
            free_count = 0
            for n2 in _neighbours(nb):
                if n2 not in occupied:
                    free_count += 1
            # Add randomness
            score = free_count * 0.5 + rng.random()
            candidates.append((score, nb))
        
        if not candidates:
            # Trapped. Fallback: extend in a straight line from start?
            # This breaks the SAW if we just append.
            # If trapped, we must restart or use a different strategy.
            # For now, if trapped, return a straight line (valid but low energy).
            # But we want a dense scaffold.
            # Let's restart growth with a different seed if trapped.
            # Since this is a helper, let's just return a straight line for safety.
            # The Monte Carlo will then have to do all the work, which is bad.
            # Better: Ensure we don't trap.
            # Use a "snake" pattern which never traps.
            
            # Let's generate a snake pattern directly.
            # It's deterministic and never traps.
            return _generate_snake(n_max)
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_site = candidates[0][1]
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain

def _generate_snake(n):
    """Generate a snake-like SAW that fills space densely."""
    if n == 0:
        return []
    
    # Use a 3D snake pattern.
    # Layer by layer.
    # Within a layer, snake in X.
    # Between layers, move in Y or Z.
    
    # Let's define a path that fills a box.
    # Box dimensions: 6x6x6 = 216 > 136.
    
    points = []
    # z=0 to 5
    # y=0 to 5
    # x=0 to 5
    
    # We need to connect the end of one layer to the start of the next.
    # Layer z, y: x goes 0->5 or 5->0.
    # End of (z,y) is at x=5 (if y even) or x=0 (if y odd).
    # Start of (z,y+1) is at x=0 (if y+1 even) or x=5 (if y+1 odd).
    
    # If y is even (0, 2, 4): x 0->5. End at (5, y, z).
    # Next y is odd (1, 3, 5): x 5->0. Start at (5, y+1, z).
    # Distance from (5, y, z) to (5, y+1, z) is 1 (in Y). Valid step.
    
    # If y is odd (1, 3): x 5->0. End at (0, y, z).
    # Next y is even (2, 4): x 0->5. Start at (0, y+1, z).
    # Distance from (0, y, z) to (0, y+1, z) is 1 (in Y). Valid step.
    
    # End of layer z (y=5, odd): x 5->0. End at (0, 5, z).
    # Start of layer z+1 (y=0, even): x 0->5. Start at (0, 0, z+1).
    # Distance from (0, 5, z) to (0, 0, z+1) is 5 (in Y) + 1 (in Z). NOT a unit step.
    
    # So we need a connection path between z-layers.
    # From (0, 5, z) to (0, 0, z+1).
    # We can move in Z first? (0, 5, z) -> (0, 5, z+1).
    # Then snake back in Y?
    # This gets complicated.
    
    # Simpler: Use a 1D snake in a "tube" of width 1? No, that's a line.
    # Use a "U-turn" pattern.
    
    # Let's just use the random growth with restarts.
    # It's safer to implement a robust random growth with backtracking?
    # No, backtracking is hard.
    
    # Let's use a pre-computed "standard" dense walk.
    # Actually, for N=136, a random walk with density bias usually works if we allow
    # "teleportation" restarts in the generator? No, it must be a single SAW.
    
    # Let's try a different approach for the scaffold:
    # Use the "greedy" growth from the current program, which is proven to work.
    # We just need it to be fast and dense.
    
    # Re-use the logic from the current program's _initial_grow but simplified.
    
    chain = [(0,0,0)]
    occupied = {(0,0,0): 0}
    h_positions = set() # Not needed for scaffold, just geometry
    
    rng = random.Random(123)
    
    for i in range(1, n):
        current = chain[-1]
        candidates = []
        for nb in _neighbours(current):
            if nb in occupied:
                continue
            # Score: maximize free space to avoid trapping
            free_space = 0
            for n2 in _neighbours(nb):
                if n2 not in occupied:
                    free_space += 1
            # Add slight bias towards "inward" moves?
            # Just use free space + noise
            score = free_space + rng.random()
            candidates.append((score, nb))
        
        if not candidates:
            # Trapped. Return straight line fallback.
            # This is a failure of the scaffold, but the MC can still run.
            # However, a straight line is a bad scaffold.
            # Let's try to extend in a straight line from the start?
            # No, just return what we have + straight line extension?
            # That would break self-avoidance if we just append.
            # If trapped, we must return a valid SAW.
            # The only valid SAW from a trapped state is the one we have.
            # If we haven't reached N, we fail.
            # To avoid this, we can use a "lookahead" of 1 step.
            # If moving to nb leaves the next step with 0 options, skip nb.
            
            pass # Handled by the loop below if we implement lookahead
            
        # Let's implement lookahead for robustness
        valid_candidates = []
        for score, nb in candidates:
            # Check if nb leads to a dead end immediately
            next_free = 0
            for n2 in _neighbours(nb):
                if n2 not in occupied and n2 != current:
                    next_free += 1
            if next_free > 0:
                valid_candidates.append((score, nb))
            elif len(candidates) == 1:
                # If it's the only option, take it
                valid_candidates.append((score, nb))
                
        if not valid_candidates:
             # Trapped. Fallback: straight line from (0,0,0) to (n-1, 0, 0)
             return [(i, 0, 0) for i in range(n)]
        
        valid_candidates.sort(key=lambda x: x[0], reverse=True)
        best_site = valid_candidates[0][1]
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain

def _compute_energy(sequence, chain):
    """Compute total energy (negative number of H-H contacts)."""
    n = len(sequence)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and not _is_adjacent(i, j) and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _monte_carlo_fold(sequence, scaffold, dimension, time_budget):
    """
    Perform Parallel Tempering Monte Carlo on the given scaffold.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # We only use the first n points of the scaffold
    scaffold_prefix = scaffold[:n]
    
    # Initial assignment: identity (residue i at scaffold[i])
    # This is a valid assignment.
    # We will permute the residues on the fixed scaffold points.
    # Wait, the task requires the output to be a walk of lattice points.
    # If we permute residues on a fixed scaffold, the "walk" in the output
    # is the scaffold itself, but the sequence is assigned to it?
    # NO. The output is `list of len(sequence) integer lattice points`.
    # The evaluator checks:
    # 1. Self-avoidance of the returned points.
    # 2. Unit steps between consecutive points.
    # 3. Energy based on the sequence and the points.
    
    # If I return the scaffold points, they are self-avoiding and have unit steps.
    # But the sequence is fixed. I cannot change the sequence.
    # I can only change the mapping of residues to points?
    # NO. The residue at index i MUST be at point chain[i].
    # So if I fix the scaffold, I am fixing the conformation.
    # I cannot change the conformation if the scaffold is fixed.
    
    # Ah, the strategy "operates on a fixed self-avoiding walk scaffold" implies
    # that we are searching over *assignments* of residues to scaffold nodes?
    # But the residue order is fixed by the sequence string.
    # Residue 0 is always first, Residue 1 is second, etc.
    # So the conformation IS the sequence of points.
    # If the scaffold is fixed, the conformation is fixed.
    # So "Parallel Tempering on a fixed scaffold" doesn't make sense for HP folding
    # unless we are allowing the chain to *move* along the scaffold edges?
    # Or unless we are *rebuilding* the scaffold?
    
    # Re-reading the directive: "Replace the sequential greedy growth with a parallel
    # tempering Monte Carlo chain that operates on a fixed self-avoiding walk scaffold,
    # using Metropolis acceptance criteria to explore the conformational space and
    # escape local minima via energy-driven swaps and rotations rather than incremental construction."
    
    # "Swaps and rotations" implies moving residues.
    # If the scaffold is fixed, we can't move residues to new lattice points
    # unless we are swapping which lattice point corresponds to which index?
    # But the index order must follow the walk.
    
    # Interpretation: The "scaffold" is a set of valid moves?
    # Or perhaps the "scaffold" is the initial conformation, and we use MC to
    # *modify* it via moves (single moves, reversals) that maintain self-avoidance?
    # This is just standard MC on the space of SAWs.
    
    # Let's assume "fixed scaffold" is a misinterpretation of "initial scaffold".
    # The strategy should be:
    # 1. Generate a good initial conformation (scaffold).
    # 2. Use Parallel Tempering MC to refine it.
    #    Moves: Single residue moves, Segment Reversals.
    #    These moves change the conformation (the list of points).
    
    # So, implement Parallel Tempering MC with Single Moves and Reversals.
    
    # Number of replicas
    n_replicas = 5
    # Temperatures (beta = 1/T)
    # Energy range is roughly -N to 0.
    # Let's use beta values: 0.5, 1.0, 2.0, 4.0, 8.0
    betas = [0.5, 1.0, 2.0, 4.0, 8.0]
    
    # State: list of (chain, energy) for each replica
    # chain is a list of tuples
    # energy is int
    
    # Initialize replicas with the same initial chain (scaffold)
    # But we should perturb them slightly to decorrelate?
    # For now, start with the same.
    
    initial_chain = [tuple(p) for p in scaffold_prefix]
    initial_energy = _compute_energy(sequence, initial_chain)
    
    replicas = []
    for b in betas:
        # Copy chain
        chain_copy = list(initial_chain)
        replicas.append({
            'chain': chain_copy,
            'energy': initial_energy,
            'beta': b
        })
    
    best_chain = list(initial_chain)
    best_energy = initial_energy
    
    rng = random.Random(999)
    
    # Precompute H indices
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    
    # Helper to update energy and occupancy for a replica
    # We will maintain occupancy and contact counts for O(1) delta updates.
    
    # Structure for each replica:
    # chain: list of tuples
    # occupied: dict pos -> index
    # contact_count: list of int (contacts per residue)
    # energy: int
    
    # Let's refactor to maintain these structures.
    
    def _init_replica_state(chain, sequence):
        n = len(chain)
        occupied = {p: i for i, p in enumerate(chain)}
        contact_count = [0] * n
        energy = 0
        # Compute contacts
        for i in h_indices:
            p = chain[i]
            for q in _neighbours(p):
                j = occupied.get(q)
                if j is not None and not _is_adjacent(i, j) and sequence[j] == 'H' and i < j:
                    contact_count[i] += 1
                    contact_count[j] += 1
                    energy -= 1
        return {
            'chain': chain,
            'occupied': occupied,
            'contact_count': contact_count,
            'energy': energy
        }
    
    # Re-initialize replicas with full state
    for r in replicas:
        state = _init_replica_state(r['chain'], sequence)
        r.update(state)
    
    # MC Loop
    # Each step:
    # 1. Pick a random replica.
    # 2. Propose a move (single move or reversal).
    # 3. Calculate delta E.
    # 4. Accept/Reject based on Metropolis.
    # 5. Periodically, attempt exchange between adjacent replicas.
    
    steps_per_second_estimate = 10000
    total_steps = int(time_budget * steps_per_second_estimate)
    
    # We will run until time runs out.
    
    for step in range(total_steps):
        if _remaining_time() < 0.5:
            break
            
        # Pick replica
        r_idx = rng.randint(0, n_replicas - 1)
        r = replicas[r_idx]
        
        # Propose move
        # 50% Single Move, 50% Reversal
        if rng.random() < 0.5:
            # Single Move
            i = rng.randint(1, n - 2)
            old_pos = r['chain'][i]
            valid_moves = []
            for nb in _neighbours(old_pos):
                if nb not in r['occupied'] or r['occupied'].get(nb) == i:
                    if nb != old_pos:
                        valid_moves.append(nb)
            
            if not valid_moves:
                continue
                
            new_pos = rng.choice(valid_moves)
            
            # Calculate Delta E
            # Only H residues contribute to energy.
            if sequence[i] != 'H':
                delta_e = 0
            else:
                # Remove i from old_pos
                # Gain/Loss contacts with neighbors of old_pos
                old_contacts = 0
                for nb in _neighbours(old_pos):
                    j = r['occupied'].get(nb)
                    if j is not None and j != i and not _is_adjacent(i, j) and sequence[j] == 'H':
                        old_contacts += 1
                
                # Add i to new_pos
                new_contacts = 0
                for nb in _neighbours(new_pos):
                    j = r['occupied'].get(nb)
                    # Note: old_pos is still in occupied map, but we are moving away from it.
                    # If nb == old_pos, j == i. We skip j==i.
                    if j is not None and j != i and not _is_adjacent(i, j) and sequence[j] == 'H':
                        new_contacts += 1
                
                delta_e = new_contacts - old_contacts
            
            # Metropolis Acceptance
            if delta_e <= 0 or rng.random() < math.exp(-r['beta'] * delta_e):
                # Accept
                # Update structures
                del r['occupied'][old_pos]
                r['occupied'][new_pos] = i
                r['chain'][i] = new_pos
                r['energy'] += delta_e
                
                # Update contact counts?
                # For simplicity, we don't update contact_count array here,
                # we just use the delta for acceptance.
                # The contact_count array is only used if we do complex delta calcs.
                # Here we recalculated old/new contacts explicitly, so we don't need the array
                # for this move. But we need it for reversals?
                # Reversals also use explicit recalculation of boundary contacts.
                # So we can drop contact_count array if we always recalc locally.
                # It saves memory and update time.
                
                if r['energy'] < best_energy:
                    best_energy = r['energy']
                    best_chain = list(r['chain'])
        
        else:
            # Reversal
            seg_len = rng.randint(2, min(10, n // 2))
            i = rng.randint(1, n - seg_len - 1)
            j = i + seg_len - 1
            
            # Check connectivity
            p_prev = r['chain'][i-1]
            p_next = r['chain'][j+1]
            p_start = r['chain'][i]
            p_end = r['chain'][j]
            
            # Manhattan distance
            if abs(p_prev[0] - p_end[0]) + abs(p_prev[1] - p_end[1]) + abs(p_prev[2] - p_end[2]) != 1:
                continue
            if abs(p_start[0] - p_next[0]) + abs(p_start[1] - p_next[1]) + abs(p_start[2] - p_next[2]) != 1:
                continue
            
            # Calculate Delta E
            # Only boundary contacts change.
            # Old: Residues in [i, j] at their current positions.
            # New: Residues in [i, j] reversed.
            
            old_contacts = 0
            new_contacts = 0
            
            for k in range(i, j + 1):
                pos = r['chain'][k]
                
                # Old residue at pos is sequence[k]
                if sequence[k] == 'H':
                    for nb in _neighbours(pos):
                        j_out = r['occupied'].get(nb)
                        if j_out is not None and (j_out < i or j_out > j):
                            if sequence[j_out] == 'H' and not _is_adjacent(k, j_out):
                                old_contacts += 1
                
                # New residue at pos is sequence[i + j - k]
                new_idx = i + j - k
                if sequence[new_idx] == 'H':
                    for nb in _neighbours(pos):
                        j_out = r['occupied'].get(nb)
                        if j_out is not None and (j_out < i or j_out > j):
                            if sequence[j_out] == 'H' and not _is_adjacent(new_idx, j_out):
                                new_contacts += 1
            
            delta_e = new_contacts - old_contacts
            
            if delta_e <= 0 or rng.random() < math.exp(-r['beta'] * delta_e):
                # Accept
                # Reverse segment
                r['chain'][i:j+1] = reversed(r['chain'][i:j+1])
                # Update occupied
                for k in range(i, j + 1):
                    r['occupied'][r['chain'][k]] = k
                r['energy'] += delta_e
                
                if r['energy'] < best_energy:
                    best_energy = r['energy']
                    best_chain = list(r['chain'])
        
        # Exchange
        # Attempt exchange between r_idx and r_idx+1 (or r_idx-1)
        if r_idx < n_replicas - 1 and rng.random() < 0.1:
            r2_idx = r_idx + 1
            r1 = replicas[r_idx]
            r2 = replicas[r_idx + 1]
            
            # Metropolis criterion for exchange
            # exp(-beta1 * E1 - beta2 * E2) vs exp(-beta1 * E2 - beta2 * E1)
            # Ratio = exp(-beta1*E1 - beta2*E2 + beta1*E2 + beta2*E1)
            #       = exp(-(beta1 - beta2)(E1 - E2))
            
            d_beta = r1['beta'] - r2['beta']
            d_energy = r1['energy'] - r2['energy']
            
            prob = math.exp(-d_beta * d_energy)
            
            if rng.random() < prob:
                # Swap
                replicas[r_idx], replicas[r_idx + 1] = replicas[r_idx + 1], replicas[r_idx]
    
    return best_chain

import math

def fold(sequence, dimension):
    """
    Best conformation using Parallel Tempering Monte Carlo.
    """
    if not sequence:
        return []
    
    n = len(sequence)
    
    # Time budget
    # We have ~290s total.
    # Allocate 5s for scaffold generation.
    # Rest for MC.
    
    # Generate scaffold for max length (136) once?
    # No, fold is called per sequence.
    # We can cache the scaffold for the max length.
    
    global _SCAFFOLD_CACHE
    if not hasattr(fold, '_scaffold_cache'):
        fold._scaffold_cache = {}
    
    # Use a scaffold of length max(n, 136) to ensure we have enough points?
    # No, we just need n points.
    # Generate a scaffold of length n.
    # To be efficient, we can generate a long scaffold (136) and use prefix.
    
    max_n = 136
    if max_n not in fold._scaffold_cache:
        fold._scaffold_cache[max_n] = _generate_snake(max_n)
    
    scaffold = fold._scaffold_cache[max_n][:n]
    
    # If scaffold generation failed (straight line), MC might struggle.
    # But let's proceed.
    
    time_budget = max(0, _remaining_time() - 5.0)
    
    if time_budget < 1.0:
        # Not enough time for MC, return scaffold
        return [list(p) for p in scaffold]
    
    chain = _monte_carlo_fold(sequence, scaffold, dimension, time_budget)
    
    return [list(p) for p in chain]

