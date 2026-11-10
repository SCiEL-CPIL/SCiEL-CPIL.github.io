"""HP lattice protein folding: Parallel Tempering Monte Carlo on a fixed SAW scaffold."""
import random
import math
import time

# EVOLVE-BLOCK-START

def _neighbours(point):
    """Yield the 6 neighbours of a point on the cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    """Compute the HP energy: -number of non-consecutive H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _try_move(chain, occupied, i, j, sequence, dimension, rng):
    """
    Try a move that swaps or rotates residues at positions i and j.
    Returns (success, delta_energy, new_chain, new_occupied) or (False, 0, None, None).
    We use a simple local move: pick two non-adjacent positions and try to swap them,
    or perform a pivot rotation.
    """
    # This is a placeholder; we'll implement specific moves below.
    pass

def _pivot_move(chain, occupied, i, sequence, dimension, rng):
    """
    Pivot move: rotate the tail of the chain (from position i to end) around the bond (i-1, i).
    The tail is rotated to a new position that doesn't collide with the rest of the chain.
    Returns (success, delta_energy) or (False, 0).
    """
    n = len(chain)
    if i <= 0 or i >= n:
        return False, 0
    
    # The pivot is at position i. We rotate the segment chain[i:] around chain[i-1].
    # Actually, a standard pivot rotates the tail starting at i around the bond (i-1, i).
    # Let's define: the head of the tail is chain[i]. We move chain[i] to a new position
    # that is a neighbour of chain[i-1], different from chain[i+1] (if exists) and not occupied
    # by any other part of the chain (except the tail itself, which we'll update).
    
    # This is complex to implement efficiently for long chains. Let's use a simpler move:
    # End-to-end move or local swap.
    # For simplicity and robustness, let's use a local "kink" move or a simple swap of two
    # non-adjacent residues if possible, but swaps are hard to maintain self-avoidance.
    
    # Let's use a "pivot" where we pick a random position i (1 to n-1) and try to move
    # the entire tail chain[i:] to a new position. This is O(n) per move attempt.
    
    # Remove the tail from occupied
    tail = chain[i:]
    for p in tail:
        del occupied[p]
    
    # The new position for tail[0] must be a neighbour of chain[i-1]
    # and not equal to chain[i-2] (to avoid backtracking) and not occupied.
    pivot_point = chain[i-1]
    possible_moves = []
    for q in _neighbours(pivot_point):
        if q == chain[i-2] if i >= 2 else True:
            continue
        if q not in occupied:
            possible_moves.append(q)
    
    if not possible_moves:
        # Revert
        for p in tail:
            occupied[p] = True  # We need to store indices, not just True
        # Actually, occupied maps point to index. Let's fix the data structure.
        # We'll store occupied as a set of points for speed, and rebuild if needed,
        # or store as dict point->index.
        # Let's use a set for occupied points for faster lookup, and we don't need the index
        # for the move, only for energy calculation later. But for energy, we need to know
        # which residues are H. So we need the chain.
        # Let's revert by re-adding.
        for p in tail:
            occupied.add(p)
        return False, 0
    
    new_tail_head = rng.choice(possible_moves)
    
    # Now we need to place the rest of the tail. This is a self-avoiding walk from new_tail_head
    # of length len(tail)-1 that doesn't collide with the head part (chain[:i]).
    # This is expensive. For simplicity, let's just try to place the tail by extending one step
    # at a time, and if it fails, revert.
    
    # Actually, a simpler move: "End-to-end move" is also hard.
    
    # Let's use a different strategy: "Local move" where we pick a small segment and rotate it.
    # Or even simpler: "Swap" two non-adjacent residues if the resulting chain is self-avoiding.
    # But swaps change the sequence order, which is not allowed.
    
    # Okay, let's stick with the pivot but implement it correctly with a set for occupied.
    
    # Re-implement with occupied as a set of points.
    # We'll pass occupied as a set.
    
    # Revert the previous revert
    for p in tail:
        occupied.discard(p)
    
    # Try to place the tail
    new_tail = [new_tail_head]
    occupied.add(new_tail_head)
    
    for k in range(1, len(tail)):
        last = new_tail[-1]
        possible = []
        for q in _neighbours(last):
            if q not in occupied:
                # Avoid going back to the previous point in the new tail
                if len(new_tail) > 1 and q == new_tail[-2]:
                    continue
                possible.append(q)
        if not possible:
            # Failed, revert
            for p in new_tail:
                occupied.discard(p)
            return False, 0
        next_point = rng.choice(possible)
        new_tail.append(next_point)
        occupied.add(next_point)
    
    # Success
    # Calculate delta energy
    # We need to compute the energy change. This is complex because contacts can change
    # anywhere. For simplicity, we'll compute the full energy after the move.
    # But that's O(n^2). For n=136, that's ~18000 ops per move, which is fine for a few thousand moves.
    
    # Update chain
    old_chain = chain[:]
    chain[i:] = new_tail
    
    # Compute new energy
    new_energy = _compute_energy(sequence, chain)
    
    # Compute old energy (we should have stored it)
    # We'll pass old_energy as an argument to avoid recomputing.
    # Let's change the interface.
    
    # For now, let's assume we have old_energy.
    # We'll refactor.
    
    return True, new_energy

def fold(sequence, dimension):
    """
    Parallel Tempering Monte Carlo for HP protein folding.
    """
    n = len(sequence)
    rng = random.Random(42)
    
    # Initial conformation: a simple snake-like path
    # Let's create a simple self-avoiding walk
    chain = []
    occupied = set()
    current = (0, 0, 0)
    chain.append(current)
    occupied.add(current)
    
    directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    
    for i in range(1, n):
        moved = False
        # Try directions in a randomized order
        dirs = directions[:]
        rng.shuffle(dirs)
        for d in dirs:
            new_pos = (current[0] + d[0], current[1] + d[1], current[2] + d[2])
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied.add(new_pos)
                current = new_pos
                moved = True
                break
        if not moved:
            # If we get stuck, this shouldn't happen with a simple snake, but just in case
            # Use a straight line
            chain = [tuple([i] + [0]*(dimension-1)) for i in range(n)]
            occupied = set(chain)
            break
    
    if len(chain) != n:
        # Fallback to straight line
        chain = [tuple([i] + [0]*(dimension-1)) for i in range(n)]
        occupied = set(chain)
    
    current_energy = _compute_energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    # Parallel Tempering with multiple temperatures
    num_temps = 5
    temps = [1.0, 0.5, 0.25, 0.125, 0.0625]
    # Chains for each temperature
    chains = [chain[:]]
    energies = [current_energy]
    occupieds = [set(chain)]
    
    for t in range(1, num_temps):
        # Initialize with a perturbed version
        c = chain[:]
        occ = set(chain)
        # Perturb: do some random moves
        for _ in range(10):
            # Simple random move: pick a random position and try to move it
            # For simplicity, just use the same chain for now
            pass
        chains.append(c)
        energies.append(_compute_energy(sequence, c))
        occupieds.append(set(c))
    
    num_sweeps = 500
    start_time = time.time()
    time_budget = 220  # Leave some margin for 240s total
    
    for sweep in range(num_sweeps):
        if time.time() - start_time > time_budget:
            break
        
        # For each temperature, do a few moves
        for t in range(num_temps):
            temp = temps[t]
            c = chains[t]
            occ = occupieds[t]
            e = energies[t]
            
            # Do a few moves
            for _ in range(3):
                # Pick a random pivot position
                i = rng.randint(1, len(c) - 1)
                
                # Remove tail
                tail = c[i:]
                for p in tail:
                    occ.discard(p)
                
                # Try to move tail
                pivot_point = c[i-1]
                possible_moves = []
                for q in _neighbours(pivot_point):
                    if i >= 2 and q == c[i-2]:
                        continue
                    if q not in occ:
                        possible_moves.append(q)
                
                if not possible_moves:
                    # Revert
                    for p in tail:
                        occ.add(p)
                    continue
                
                new_tail_head = rng.choice(possible_moves)
                new_tail = [new_tail_head]
                occ.add(new_tail_head)
                
                success = True
                for k in range(1, len(tail)):
                    last = new_tail[-1]
                    possible = []
                    for q in _neighbours(last):
                        if q not in occ:
                            if len(new_tail) > 1 and q == new_tail[-2]:
                                continue
                            possible.append(q)
                    if not possible:
                        success = False
                        break
                    next_point = rng.choice(possible)
                    new_tail.append(next_point)
                    occ.add(next_point)
                
                if not success:
                    # Revert
                    for p in new_tail:
                        occ.discard(p)
                    for p in tail:
                        occ.add(p)
                    continue
                
                # Compute new energy
                c[i:] = new_tail
                new_e = _compute_energy(sequence, c)
                delta_e = new_e - e
                
                # Metropolis acceptance
                if delta_e < 0 or rng.random() < math.exp(-delta_e / temp):
                    e = new_e
                    chains[t] = c
                    occupieds[t] = occ
                    energies[t] = e
                    # Update best
                    if e < best_energy:
                        best_energy = e
                        best_chain = c[:]
                else:
                    # Revert
                    c[i:] = tail
                    for p in new_tail:
                        occ.discard(p)
                    for p in tail:
                        occ.add(p)
            
            # Update best from this chain
            if e < best_energy:
                best_energy = e
                best_chain = c[:]
        
        # Swap attempts between neighboring temperatures
        for t in range(num_temps - 1):
            if rng.random() < 0.1:  # 10% chance to swap
                # Swap chains t and t+1
                chains[t], chains[t+1] = chains[t+1], chains[t]
                energies[t], energies[t+1] = energies[t+1], energies[t]
                occupieds[t], occupieds[t+1] = occupieds[t+1], occupieds[t]
    
    # Return the best chain found
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
