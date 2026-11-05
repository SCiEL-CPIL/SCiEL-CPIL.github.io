"""Hybrid Local Search for HP Lattice Protein Folding."""
# EVOLVE-BLOCK-START
import time
import random
import sys

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # 20s safety margin before 300s kill

# Precomputed neighbor offsets for 3D cubic lattice
# (dx, dy, dz) for the 6 neighbors
_NEIGHBORS_3D = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _energy(sequence, chain):
    """
    Calculate the energy of a chain.
    Energy = - (number of H-H contacts)
    Contacts are pairs of H residues that are adjacent in lattice space 
    but not adjacent in the sequence (i.e., |i - j| > 1).
    """
    if not chain:
        return 0
    
    # Map position to index
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    contacts = 0
    # Iterate through each residue
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            # Check 6 neighbors
            for dx, dy, dz in _NEIGHBORS_3D:
                np = (p[0] + dx, p[1] + dy, p[2] + dz)
                j = pos_to_idx.get(np)
                # If neighbor exists, is not the immediate predecessor/successor, and is H
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    contacts += 1
                    # We count each contact twice (once for i, once for j). 
                    # The problem statement says "number of H-H pairs". 
                    # Standard HP model energy is usually -N_contacts.
                    # If we sum over all i, we count each pair (i,j) and (j,i).
                    # So actual pairs = contacts // 2.
                    # Let's verify the seed code logic.
                    # Seed: "contacts += 1" inside loop where j > i+1. 
                    # This counts each pair exactly once.
                    # My loop above counts each pair twice.
                    # I should divide by 2 at the end or adjust the condition.
                    # Let's stick to the seed's logic for consistency: count only if j > i + 1.
    
    # Re-calculate correctly to match seed logic (count each pair once)
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for dx, dy, dz in _NEIGHBORS_3D:
                np = (p[0] + dx, p[1] + dy, p[2] + dz)
                j = pos_to_idx.get(np)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
                    
    return -contacts

def _random_saw(n, rng):
    """Generate a random self-avoiding walk of length n."""
    if n == 0:
        return []
    
    chain = [(0, 0, 0)]
    occupied = {chain[0]}
    
    for _ in range(1, n):
        last = chain[-1]
        options = []
        for dx, dy, dz in _NEIGHBORS_3D:
            np = (last[0] + dx, last[1] + dy, last[2] + dz)
            if np not in occupied:
                options.append(np)
        
        if not options:
            return None # Stuck
            
        next_pos = rng.choice(options)
        chain.append(next_pos)
        occupied.add(next_pos)
        
    return chain

def _local_search(sequence, chain, rng, time_limit):
    """
    Perform local search (Hill Climbing with random restarts/simulated annealing-like noise)
    on the given chain.
    """
    if chain is None:
        return None, 0
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    n = len(sequence)
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    # Moves: Flip a residue (if space allows) or Pivot
    # For simplicity and speed, we will use "Move" (shift one residue) or "Flip" (change direction of a bond)
    # Actually, a robust move is to pick a random internal node and try to move it to an empty neighbor.
    # This is a "local move" that maintains self-avoidance if the new spot is empty and the old spot is not needed by neighbors.
    
    # A simpler and very effective move for SAWs is the "Pivot" move:
    # Pick a random index k. Rotate the sub-chain k..n-1 around the bond (k-1, k).
    # However, checking self-avoidance for a pivot is expensive O(N).
    
    # Alternative: "Slither" or "Move" a single bead.
    # Pick a random index i (1 to n-1). Try to move bead i to a new position.
    # This is complex because it affects neighbors i-1 and i+1.
    
    # Let's use a simpler heuristic: 
    # 1. Calculate energy.
    # 2. Try to improve by moving a single H bead to a neighbor that creates more contacts?
    # No, that breaks connectivity.
    
    # Standard approach: 
    # 1. Generate a random valid move.
    # 2. Calculate delta energy.
    # 3. Accept if better or with probability exp(-delta/T).
    
    # Efficient Move: 
    # Pick a random internal residue i.
    # Try to move it to one of its 6 neighbors.
    # The new position must be empty.
    # The old position must not be occupied by i-1 or i+1 (it isn't, by definition of SAW).
    # So, just check if new_pos is empty.
    # This is a valid SAW move!
    
    # Let's implement this "Single Bead Move" local search.
    
    start_time = time.time()
    T = 0.5 # Temperature for simulated annealing acceptance of worse moves
    # Cool down over time? Or keep fixed? 
    # Given the short time, fixed T with random restarts might be better.
    # Or simple Hill Climbing (accept only if better).
    
    # To avoid getting stuck in local minima, we will use a "First Fit" or "Best Fit" with random moves.
    # Let's try: Accept if delta < 0 (improvement). If stuck for many steps, restart.
    
    stuck_counter = 0
    max_stuck = 100
    
    while time.time() < time_limit:
        if stuck_counter > max_stuck:
            break # Restart externally
        
        # Pick a random internal index
        i = rng.randint(1, n - 2)
        pos = current_chain[i]
        
        # Try neighbors
        moved = False
        for dx, dy, dz in _NEIGHBORS_3D:
            new_pos = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
            
            # Check if new_pos is occupied
            if new_pos in occupied:
                continue
            
            # Check connectivity: 
            # The bead i is connected to i-1 and i+1.
            # Moving i to new_pos is valid if new_pos is not occupied.
            # The old pos becomes empty.
            # No other checks needed for validity of the walk structure itself, 
            # as long as new_pos is not occupied by any other bead.
            
            # Calculate delta energy
            # We need to update the energy efficiently.
            # Old energy contributions involving bead i:
            # Contacts between i and its neighbors in lattice (excluding i-1, i+1).
            # New energy contributions involving bead i at new_pos.
            
            # For simplicity, recalc energy? O(N). 
            # If N is 100, and we do 1000 moves/sec, that's 100k ops/sec. 
            # 280 seconds -> 28M ops. 28M / 100 = 280k moves. 
            # This is quite slow. We need incremental energy update.
            
            # Let's implement incremental energy update.
            
            # Get current contacts for bead i
            # We need a list of H neighbors for each bead? 
            # Or just check the 6 neighbors of pos and new_pos.
            
            # Let's precompute a set of H indices for fast lookup? 
            # No, just check sequence[j] == 'H'.
            
            # Delta = E_new - E_old
            # E_old contacts involving i:
            # 1. i with j where j is neighbor in lattice, j != i-1, j != i+1, and seq[j]=='H'
            # 2. j with i where j is neighbor in lattice, j != i-1, j != i+1, and seq[j]=='H' (same set)
            
            # Actually, let's just recalc the whole energy if N is small, 
            # but for N=136, O(N) is 136 * 6 = 816 checks. 
            # 280s / 816 checks = 343 seconds for 1 move? No.
            # 816 checks is ~0.001s in Python? Maybe 0.0001s.
            # 10,000 moves/sec is possible.
            # 280s * 10,000 = 2.8M moves.
            # This is feasible.
            
            # Let's try simple recalculation first for correctness, then optimize if needed.
            
            # Save state
            old_pos = current_chain[i]
            occupied.pop(old_pos)
            current_chain[i] = new_pos
            occupied[new_pos] = i
            
            new_energy = _energy(sequence, current_chain)
            delta = new_energy - current_energy
            
            if delta < 0:
                # Improvement
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                stuck_counter = 0
                moved = True
            else:
                # Worse or equal
                # Undo move
                current_chain[i] = old_pos
                occupied.pop(new_pos)
                occupied[old_pos] = i
                stuck_counter += 1
                
        if not moved:
            stuck_counter += 1

    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Main entry point.
    """
    if dimension != 3:
        raise ValueError("Only dimension 3 is supported")
    
    n = len(sequence)
    if n == 0:
        return []
    
    rng = random.Random(42)
    
    best_chain = None
    best_energy = 0 # Energy is negative, so 0 is "no contacts" (worst)
    
    # Time budget for this sequence
    # We don't know how many sequences will be called, but the prompt says 
    # "240 s CPU budget for all sequences together" and "300 s wall-clock".
    # We will check global deadline.
    
    # Strategy:
    # 1. Generate a few random SAWs.
    # 2. Run local search on each.
    # 3. Keep the best.
    
    # Number of restarts depends on n and time left.
    # For small n, we can do many restarts.
    # For large n, fewer restarts but longer local search?
    # Actually, local search is O(N) per move.
    
    # Let's do 10 restarts for n < 50, 5 for n < 100, 2 for n >= 100.
    # And allocate time to local search.
    
    restarts = 10
    if n > 50:
        restarts = 5
    if n > 100:
        restarts = 3
        
    for r in range(restarts):
        # Check deadline
        if time.time() > _DEADLINE:
            break
            
        # Generate initial SAW
        # Use a faster generator? 
        # Random SAW generation can fail (get stuck). 
        # If it fails, try again.
        
        init_chain = None
        for _ in range(10): # Max 10 tries to generate a valid SAW
            init_chain = _random_saw(n, rng)
            if init_chain is not None:
                break
        
        if init_chain is None:
            continue
            
        # Allocate time for local search
        # Estimate time remaining
        time_left = _DEADLINE - time.time()
        if time_left < 1:
            break
            
        # Heuristic time allocation: 
        # More time for later restarts? Or equal?
        # Let's give each restart 1/ (restarts - r) of remaining time?
        # Simpler: fixed time per restart, e.g., 5 seconds?
        # If n is large, 5 seconds is a lot of moves.
        # If n is small, 5 seconds is a lot of moves.
        
        # Let's just run local search until a sub-deadline.
        # Sub-deadline = current_time + (time_left / (restarts - r))
        # But ensure at least 1 second.
        
        sub_deadline = time.time() + max(1.0, time_left / (restarts - r))
        
        chain, energy = _local_search(sequence, init_chain, rng, sub_deadline)
        
        if chain is not None and energy < best_energy:
            best_energy = energy
            best_chain = chain

    # If no chain found (should not happen), return a straight line
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    # Convert to list of lists
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
