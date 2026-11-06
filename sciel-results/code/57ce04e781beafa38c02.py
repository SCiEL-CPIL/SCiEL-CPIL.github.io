"""Improved HP lattice folding: Fast growth initialization + Simulated Annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

_T0 = time.time()
_DEADLINE = 190.0  # Safety margin for 210s limit


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_contacts(sequence, chain):
    """Return number of H-H contacts and a set of contact pairs (i, j) for i < j."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    pairs = set()
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
                    pairs.add((i, j))
    return contacts, pairs


def _energy(sequence, chain):
    contacts, _ = _compute_contacts(sequence, chain)
    return -contacts


def _grow(sequence, dimension, rng):
    """Fast randomized greedy growth to get a valid self-avoiding walk."""
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
            # Heuristic: prefer contacts, then open space
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        if not options:
            return None
        # Pick best option
        best_site = max(options, key=lambda x: x[0])[1]
        occupied[best_site] = i
        chain.append(best_site)
    return chain


def _pivot_move(chain, i, direction, rng):
    """
    Attempt a pivot move at index i.
    direction: 0=right, 1=up, 2=left, 3=down (relative to previous segment)
    Returns new_chain if valid, else None.
    """
    if i <= 0 or i >= len(chain) - 1:
        return None
    
    # Get current vector from i-1 to i
    v0 = (chain[i][0] - chain[i-1][0], chain[i][1] - chain[i-1][1])
    
    # Determine rotation
    # In 2D, rotating a vector (x, y):
    # Right turn (clockwise): (y, -x)
    # Left turn (counter-clockwise): (-y, x)
    # We map direction index to rotation.
    # Let's define directions relative to the incoming vector.
    # Actually, standard pivot moves rotate the suffix.
    # We need to generate a new direction for the segment (i-1, i).
    # The original direction is v0.
    # Possible new directions for segment (i-1, i) are the 3 other neighbors of chain[i-1].
    
    # Let's just try all 3 possible new directions for the segment (i-1, i)
    # that are not the reverse of the previous segment (to avoid immediate backtracking which is always invalid for self-avoiding? No, backtracking is invalid because it hits chain[i-2]).
    # Wait, pivot move rotates the suffix starting at i. The segment (i-1, i) changes direction.
    # The new position of i will be different.
    
    # Let's implement a simpler pivot:
    # Pick a pivot point i.
    # The segment (i-1, i) has a direction d.
    # We rotate the suffix chain[i:] by 90 degrees around chain[i-1].
    # There are 2 possible 90-degree rotations (left and right).
    
    # Let's try both.
    # Vector from i-1 to i is v = (dx, dy).
    # Rotated vectors: (-dy, dx) and (dy, -dx).
    
    dx = chain[i][0] - chain[i-1][0]
    dy = chain[i][1] - chain[i-1][1]
    
    # Two possible new vectors
    new_v1 = (-dy, dx)
    new_v2 = (dy, -dx)
    
    # Choose one randomly
    if rng.random() < 0.5:
        new_dx, new_dy = new_v1
    else:
        new_dx, new_dy = new_v2
        
    # New position for i
    new_i_pos = (chain[i-1][0] + new_dx, chain[i-1][1] + new_dy)
    
    # Check if new_i_pos is valid (not occupied by any j < i-1)
    # Note: new_i_pos cannot be chain[i-2] because that would be backtracking.
    # But we must check against all j < i-1.
    
    # To be efficient, we only need to check the suffix we are moving?
    # No, the new position of i might collide with the prefix (0 to i-2).
    # The suffix (i+1 to end) moves rigidly. We need to ensure no collisions within the suffix and between new suffix and prefix.
    
    # Collision check:
    # 1. new_i_pos must not be in chain[0:i-1]
    # 2. The rest of the suffix must not collide with the prefix or each other.
    
    # This is expensive if done naively.
    # Optimization: 
    # The relative positions of the suffix remain the same.
    # We just shift/rotate the whole suffix.
    # We can check if any point in the new suffix is in the prefix (0 to i-2).
    # And if any point in the new suffix collides with another point in the new suffix? 
    # No, the suffix is rigid, so internal collisions are preserved from the original valid state.
    # Wait, the suffix is rotated. If the original suffix was self-avoiding, the rotated suffix is also self-avoiding (rigid motion).
    # So we only need to check for collisions between the new suffix and the prefix (0 to i-2).
    
    # Let's build the new suffix
    # We need to calculate the new positions for all j >= i.
    # This is O(N) per move.
    # For N=100, this is fine.
    
    # Calculate rotation matrix
    # Original vector v = (dx, dy)
    # New vector v' = (new_dx, new_dy)
    # This is a rotation.
    # We can apply the same transformation to all vectors in the suffix.
    # Let's define the transformation T such that T(v) = v'.
    # In 2D, this is a rotation + translation? No, it's a rotation around (i-1).
    
    # Actually, it's easier to just compute the new coordinates directly.
    # The vector from (i-1) to (i) changes from v to v'.
    # The vector from (i) to (i+1) was u. What is it now?
    # Since it's a rigid rotation of the suffix around (i-1), the angle of all vectors in the suffix changes by the same amount.
    
    # Angle of v: atan2(dy, dx)
    # Angle of v': atan2(new_dy, new_dx)
    # Delta angle = angle(v') - angle(v)
    # This is +/- 90 degrees.
    
    # If we rotate by 90 deg CCW: (x, y) -> (-y, x)
    # If we rotate by 90 deg CW: (x, y) -> (y, -x)
    
    # Let's determine if it's CW or CCW.
    # Cross product v x v' = dx*new_dy - dy*new_dx
    # If > 0, CCW. If < 0, CW.
    
    cross = dx * new_dy - dy * new_dx
    
    # Precompute new suffix positions
    new_chain = list(chain[:i]) # Copy prefix up to i-1
    # Wait, chain[:i] is 0 to i-1.
    # We need to append new positions for i to end.
    
    # Current position of i-1 is fixed.
    # Let's calculate positions for j >= i.
    
    # We can use the fact that the relative vectors are rotated.
    # Let vecs be the list of vectors (chain[k] - chain[k-1]) for k=i to end.
    # We rotate each vector by the same delta angle.
    
    # Instead of trig, use the rotation logic on the vectors.
    
    # If cross > 0 (CCW): (dx, dy) -> (-dy, dx)
    # If cross < 0 (CW): (dx, dy) -> (dy, -x)
    
    # Let's verify with v.
    if cross > 0:
        def rot(v):
            return (-v[1], v[0])
    else:
        def rot(v):
            return (v[1], -v[0])
    
    # Check if rot(v) == v'
    # It should be.
    
    # Now generate new positions
    curr_pos = chain[i-1]
    new_chain.append(curr_pos) # This is chain[i-1], already in prefix? 
    # My slice chain[:i] includes 0..i-1.
    # So new_chain currently has 0..i-1.
    # I need to append i..end.
    
    # Wait, I appended curr_pos which is chain[i-1]. That's a duplicate if I keep chain[:i].
    # chain[:i] is indices 0 to i-1.
    # So new_chain has indices 0 to i-1.
    # I need to start from i.
    
    # Let's restart the construction.
    new_chain = chain[:i] # 0 to i-1
    
    curr_pos = chain[i-1]
    
    for j in range(i, len(chain)):
        # Get vector from j-1 to j in original chain
        v = (chain[j][0] - chain[j-1][0], chain[j][1] - chain[j-1][1])
        # Rotate it
        rv = rot(v)
        # New position
        new_pos = (curr_pos[0] + rv[0], curr_pos[1] + rv[1])
        
        # Check collision with prefix (0 to i-2)
        # We can check against a set of prefix points
        # But building the set is O(N). We can just check if new_pos is in chain[:i-1].
        # This is O(N) per point, so O(N^2) per move.
        # For N=100, 10^4 ops per move. 10^6 moves = 10^10 ops. Too slow.
        
        # Optimization: Use a set for the prefix.
        # But we need to build it.
        # We can build the set once outside the loop? No, the prefix is fixed for a specific pivot i.
        # But we try many pivots.
        
        # Alternative: Check collision with a global occupied set?
        # The prefix is static. The suffix is moving.
        # We can maintain a set of prefix points.
        
        # Let's assume we pass in a set of prefix points.
        # Or, we can just do the O(N) check.
        # 100 residues. N^2 = 10,000.
        # If we do 10,000 moves, that's 10^8 ops. Python is slow. ~10-100s.
        # We have 190s.
        # Maybe we can be smarter.
        
        # Let's just try it. If it's too slow, we reduce moves.
        
        # Actually, we can check if new_pos is in the set of points {chain[0], ..., chain[i-2]}.
        # Let's create this set once per pivot attempt?
        # No, we can create it in the caller.
        
        # Let's modify _pivot_move to accept a prefix_set.
        
        if new_pos in prefix_set:
            return None
        
        new_chain.append(new_pos)
        curr_pos = new_pos
        
    return new_chain


def _simulated_annealing(sequence, initial_chain, rng, time_limit):
    """
    Perform simulated annealing on the chain.
    """
    n = len(sequence)
    current_chain = initial_chain
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Temperature schedule
    # Start high, end low.
    # T = T0 * (T_end / T0) ^ (t / t_max)
    # Or exponential cooling.
    
    # Let's estimate number of steps we can do.
    # Rough estimate: 1000 steps per second in Python for N=100?
    # Let's run a quick benchmark? No, just assume ~1000-5000 steps/sec.
    # Let's target 50% of time for SA.
    # We will just loop until time runs out.
    
    start_time = time.time()
    max_time = time_limit
    T = 10.0  # Initial temperature
    cooling_rate = 0.9999
    
    # Precompute prefix sets for each i?
    # No, that's O(N^2) memory.
    # We will compute prefix set on the fly for each pivot i.
    # Or, we can just check collision in the move function.
    
    while (time.time() - start_time) < max_time:
        # Pick a random pivot i
        # Weight towards middle? Uniform is fine.
        i = rng.randint(1, n - 2)
        
        # Build prefix set for collision check
        # Prefix is 0 to i-2.
        # If i=1, prefix is empty.
        if i > 1:
            prefix_set = set(chain[0:i-1] for chain in [current_chain]) # Just current_chain[0:i-1]
            prefix_set = set(current_chain[0:i-1])
        else:
            prefix_set = set()
            
        new_chain = _pivot_move(current_chain, i, 0, rng, prefix_set)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta_E = new_energy - current_energy
        
        if delta_E < 0:
            # Accept
            current_chain = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(new_chain)
        else:
            # Accept with probability
            # Avoid division by zero if T is very small
            if T > 1e-10:
                p = math.exp(-delta_E / T)
                if rng.random() < p:
                    current_chain = new_chain
                    current_energy = new_energy
            # Else reject
            
        T *= cooling_rate
        if T < 1e-10:
            # Reheat or restart?
            # Let's just break or set T to a small value.
            # Reheating might help escape local minima.
            T = 5.0
            
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence using fast growth + simulated annealing.
    """
    if dimension != 2:
        raise ValueError("Only 2D supported")
        
    rng = random.Random(42)
    
    # 1. Generate initial chain
    initial_chain = _grow(sequence, dimension, rng)
    if initial_chain is None:
        # Fallback: straight line (though growth should work)
        initial_chain = [(i, 0) for i in range(len(sequence))]
        
    # 2. Simulated Annealing
    # Allocate time
    # We have ~190s total.
    # Growth is fast.
    # Let's use 180s for SA.
    time_left = _remaining()
    sa_time = max(1.0, time_left - 5.0) # Leave 5s for overhead/other
    
    best_chain, best_energy = _simulated_annealing(sequence, initial_chain, rng, sa_time)
    
    # 3. Return result
    # Convert tuples to lists
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
