"""
HP lattice protein folding on the cubic lattice.
Strategy: Iterated Local Search with Simulated Annealing.
Moves: Tail Move (move last residue) and Pivot Move (rotate a segment).
"""
import random
import math
import time

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _generate_initial_chain(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n == 0:
        return []
    chain = [(0, 0, 0)]
    occupied = {chain[0]}
    for i in range(1, n):
        last = chain[-1]
        options = [p for p in _neighbours(last) if p not in occupied]
        if not options:
            # Backtracking would be complex, so we just fail or restart
            # For simplicity, return None if stuck, caller handles retry
            return None
        next_pos = rng.choice(options)
        chain.append(next_pos)
        occupied.add(next_pos)
    return chain

def _calculate_energy(sequence, chain):
    """Calculate the energy of the chain."""
    n = len(chain)
    if n == 0:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    energy = 0
    for i in range(n):
        if sequence[i] == 'H':
            for p in _neighbours(chain[i]):
                j = pos_to_idx.get(p)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    energy -= 1
    return energy

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 300  # Safety margin
    
    best_chain = None
    best_energy = float('inf')
    
    # Precompute H indices for faster checks
    is_h = [c == 'H' for c in sequence]
    
    # Helper to check if a move is valid and calculate delta energy
    # We maintain 'chain' and 'occupied' set.
    # For Pivot, we need to check the entire tail.
    
    def try_tail_move(chain, occupied, energy):
        last_idx = len(chain) - 1
        last_pos = chain[last_idx]
        prev_pos = chain[last_idx - 1]
        
        # Find empty neighbors of last_pos
        candidates = []
        for p in _neighbours(last_pos):
            if p in occupied:
                continue
            # p must be adjacent to prev_pos? No, tail move just moves last residue.
            # But the chain must remain connected.
            # The last residue is connected to prev_pos.
            # If we move last residue to p, it must be adjacent to prev_pos.
            # Wait, if we move last residue to p, is p adjacent to prev_pos?
            # If not, the chain is broken.
            # So p MUST be adjacent to prev_pos.
            # Also p must not be prev_pos itself (which is occupied).
            # And p must be empty.
            
            # Check adjacency to prev_pos
            diff = [p[i] - prev_pos[i] for i in range(dimension)]
            if sum(abs(d) for d in diff) != 1:
                continue
            
            # Check if p is empty
            # We already checked p not in occupied.
            
            candidates.append(p)
            
        if not candidates:
            return None, None, 0
            
        new_pos = rng.choice(candidates)
        
        # Calculate delta energy
        # Remove contacts involving last_idx at last_pos
        # Add contacts involving last_idx at new_pos
        
        # We need to know which residues are H.
        # Contacts are pairs (i, j) with |i-j| > 1 and dist(chain[i], chain[j]) == 1.
        
        # Let's just recalculate energy for the tail? No, that's O(N).
        # We can calculate delta by checking neighbors of last_pos and new_pos.
        
        delta = 0
        # Subtract contacts lost
        if is_h[last_idx]:
            for p in _neighbours(last_pos):
                if p in occupied:
                    j = None
                    # We need index of p. We don't have pos_to_idx in this scope efficiently?
                    # We can maintain a pos_to_idx dict.
                    pass
        
        # This incremental approach is tricky with Pivot.
        # Given N <= 136, O(N) energy calculation is fast enough.
        # Let's just use O(N) energy calculation for simplicity and correctness, 
        # and rely on high number of iterations.
        
        # New chain
        new_chain = chain.copy()
        new_chain[-1] = new_pos
        new_occupied = occupied.copy()
        new_occupied.remove(last_pos)
        new_occupied.add(new_pos)
        
        return new_chain, new_occupied, 0 # Energy calc outside

    # Simulated Annealing
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    # Initial chain
    chain = None
    for _ in range(100):
        chain = _generate_initial_chain(sequence, dimension, rng)
        if chain:
            break
    if not chain:
        # Fallback: straight line
        chain = [(i, 0, 0) for i in range(n)]
        
    occupied = set(chain)
    current_energy = _calculate_energy(sequence, chain)
    best_energy = current_energy
    best_chain = chain.copy()
    
    steps = 0
    max_steps = 500000
    
    while steps < max_steps and time.time() < deadline:
        # Choose move
        move_type = rng.random()
        
        if move_type < 0.5:
            # Tail Move
            last_idx = n - 1
            last_pos = chain[last_idx]
            prev_pos = chain[last_idx - 1]
            
            candidates = []
            for p in _neighbours(last_pos):
                if p in occupied:
                    continue
                # Check if p is adjacent to prev_pos
                diff = [p[i] - prev_pos[i] for i in range(dimension)]
                if sum(abs(d) for d in diff) == 1:
                    candidates.append(p)
            
            if candidates:
                new_pos = rng.choice(candidates)
                new_chain = chain.copy()
                new_chain[-1] = new_pos
                
                # Check self-avoidance for the new pos (already done by candidates)
                # But we must ensure new_pos is not prev_pos (it isn't, diff=1)
                
                new_energy = _calculate_energy(sequence, new_chain)
                delta = new_energy - current_energy
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    chain = new_chain
                    occupied.remove(last_pos)
                    occupied.add(new_pos)
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = chain.copy()
                        
        else:
            # Pivot Move
            # Pick random index k from 1 to n-2
            k = rng.randint(1, n - 2)
            # Rotate segment k..n-1 around k-1?
            # Actually, rotate around the bond (k-1, k).
            # The vector v = chain[k] - chain[k-1].
            # We want to change v to v' such that v' is a unit vector, v' != v.
            # And the rest of the chain shifts by v' - v.
            
            p_prev = chain[k-1]
            p_curr = chain[k]
            v = tuple(p_curr[i] - p_prev[i] for i in range(dimension))
            
            # Find 5 possible new vectors
            new_vectors = []
            for axis in range(dimension):
                for step in (-1, 1):
                    v_new = [0]*dimension
                    v_new[axis] = step
                    v_new = tuple(v_new)
                    if v_new != v:
                        new_vectors.append(v_new)
            
            if new_vectors:
                v_new = rng.choice(new_vectors)
                delta_vec = tuple(v_new[i] - v[i] for i in range(dimension))
                
                # Generate new positions for k..n-1
                new_chain = chain.copy()
                new_occupied = occupied.copy()
                
                # Remove old positions
                for i in range(k, n):
                    new_occupied.remove(chain[i])
                
                # Check if new positions are valid
                valid = True
                for i in range(k, n):
                    old_p = chain[i]
                    new_p = tuple(old_p[j] + delta_vec[j] for j in range(dimension))
                    if new_p in occupied:
                        # Collision with part of the chain that didn't move (0..k-1)
                        # Or collision with the new positions we are generating?
                        # We removed k..n-1 from new_occupied, so if new_p is in new_occupied, it's a collision.
                        # But we are adding them one by one.
                        # Better: check against original occupied minus the moving part.
                        pass
                    
                    new_chain[i] = new_p
                    new_occupied.add(new_p)
                
                # Re-check validity properly
                # The moving part is k..n-1.
                # The static part is 0..k-1.
                # New positions must not be in static part.
                # And new positions must be distinct.
                
                static_occupied = set(chain[:k])
                moving_new_positions = []
                valid = True
                for i in range(k, n):
                    old_p = chain[i]
                    new_p = tuple(old_p[j] + delta_vec[j] for j in range(dimension))
                    if new_p in static_occupied:
                        valid = False
                        break
                    moving_new_positions.append(new_p)
                
                if valid:
                    # Check distinctness of moving_new_positions
                    if len(set(moving_new_positions)) != len(moving_new_positions):
                        valid = False
                
                if valid:
                    # Apply move
                    for i in range(k, n):
                        old_p = chain[i]
                        new_p = tuple(old_p[j] + delta_vec[j] for j in range(dimension))
                        chain[i] = new_p
                        occupied.remove(old_p)
                        occupied.add(new_p)
                    
                    new_energy = _calculate_energy(sequence, chain)
                    delta = new_energy - current_energy
                    
                    if delta < 0 or rng.random() < math.exp(-delta / T):
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = chain.copy()
                    else:
                        # Revert
                        for i in range(k, n):
                            new_p = chain[i]
                            old_p = tuple(new_p[j] - delta_vec[j] for j in range(dimension))
                            chain[i] = old_p
                            occupied.remove(new_p)
                            occupied.add(old_p)
        
        steps += 1
        T *= cooling
        if T < T_min:
            T = T_min
    
    return [list(p) for p in best_chain]
