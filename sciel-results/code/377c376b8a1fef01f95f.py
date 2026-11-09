"""HP lattice protein folding using Simulated Annealing on a spiral-initialized chain."""
# EVOLVE-BLOCK-START
import time
import random
import math

_START_TIME = time.time()
DEADLINE = 195.0  # Safety margin before 210s kill

def _remaining_time():
    return DEADLINE - time.time()

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _init_spiral(n, dimension):
    """
    Generate a compact spiral-like initial conformation.
    This creates a high density of potential contacts immediately,
    giving the optimizer a much better starting point than a random walk.
    """
    if n == 1:
        return [(0, 0)]
    
    # Directions: Right, Down, Left, Up
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    chain = [(0, 0)]
    occupied = {(0, 0)}
    cur = (0, 0)
    d_idx = 0
    steps_in_dir = 1
    steps_taken = 0
    dir_changes = 0
    
    # We want to fill a roughly square area.
    # A simple spiral algorithm:
    # 1 step right, 1 step down, 2 steps left, 2 steps up, 3 steps right...
    
    while len(chain) < n:
        direction = dirs[d_idx]
        for _ in range(steps_in_dir):
            if len(chain) >= n:
                break
            nx, ny = cur[0] + direction[0], cur[1] + direction[1]
            nxt = (nx, ny)
            if nxt in occupied:
                # If we hit a wall in a spiral (shouldn't happen in standard spiral, 
                # but safe guard), try to backtrack or break
                break
            chain.append(nxt)
            occupied.add(nxt)
            cur = nxt
        # Move to next direction
        d_idx = (d_idx + 1) % 4
        steps_taken += 1
        if steps_taken == steps_in_dir:
            steps_taken = 0
            # Increment step length every 2 direction changes
            dir_changes += 1
            if dir_changes % 2 == 0:
                steps_in_dir += 1
        
        # If we got stuck (break from inner loop without finishing n),
        # this simple spiral might fail for some n if it hits its own boundary
        # in a way that doesn't allow extension. 
        # For HP problems, a simple expanding rectangle is often safer.
        # Let's use a safer "fill a box" approach if the spiral fails to grow.
        if len(chain) < n and len(chain) == len(chain) and steps_in_dir > n: 
             # This condition is basically never true, but let's add a fallback
             pass
             
    # Fallback if spiral logic above gets stuck (unlikely for standard n, but possible)
    # A simpler, more robust "snake" fill for a rectangle
    if len(chain) < n:
         # Determine box size
         width = math.ceil(math.sqrt(n))
         height = math.ceil(n / width)
         chain = []
         occupied = set()
         for y in range(height):
             for x in range(width):
                 if len(chain) >= n:
                     break
                 pt = (x, y)
                 if pt not in occupied:
                     chain.append(pt)
                     occupied.add(pt)
             if len(chain) >= n:
                 break
         
         # Check self-avoidance and connectivity of this snake
         # The snake above is: (0,0)->(1,0)->...->(w-1,0) then (0,1)? No, that's not a walk.
         # A standard snake walk:
         chain = []
         occupied = set()
         for y in range(height):
             if y % 2 == 0:
                 xs = range(width)
             else:
                 xs = range(width-1, -1, -1)
             for x in xs:
                 if len(chain) >= n:
                     break
                 pt = (x, y)
                 # Ensure connectivity: (x, y-1) must be in chain
                 if y == 0 or (x, y-1) in occupied:
                     chain.append(pt)
                     occupied.add(pt)
                 else:
                     # This simple snake might have gaps if width is odd/even mismatched
                     # Let's just use a linear chain if this fails, but spiral is usually better.
                     pass
         if len(chain) < n:
             # Ultimate fallback: linear
             chain = [(i, 0) for i in range(n)]
             
    return [list(p) for p in chain]

def _compute_energy_and_contacts(chain, seq):
    """
    Returns (energy, contact_set, position_map).
    contact_set contains frozenset({i, j}) for all H-H contacts.
    """
    pos_map = {p: i for i, p in enumerate(chain)}
    contacts = set()
    energy = 0
    
    # Iterate over all pairs? No, iterate over positions.
    # For each H residue i, check neighbors.
    for i, p in enumerate(chain):
        if seq[i] == 'H':
            for q in _neighbours(p):
                j = pos_map.get(q)
                if j is not None and j > i + 1 and seq[j] == 'H':
                    # Valid contact
                    key = (i, j)
                    if key not in contacts:
                        contacts.add(key)
                        energy -= 1
                        
    return energy, contacts, pos_map

def _incremental_energy_update(chain, seq, i, j, old_pos, new_pos, pos_map, contacts):
    """
    Update energy, pos_map, and contacts when moving residue i from old_pos to new_pos.
    Assumes i and j are not adjacent in the chain (|i-j| > 1).
    Returns (new_energy, new_contacts, new_pos_map).
    """
    # 1. Remove contacts involving i
    # Contacts are stored as (min_idx, max_idx)
    to_remove = []
    to_add = []
    
    # Check neighbors of old_pos for i
    if seq[i] == 'H':
        for q in _neighbours(old_pos):
            k = pos_map.get(q)
            if k is not None and abs(i - k) > 1 and seq[k] == 'H':
                key = (min(i, k), max(i, k))
                if key in contacts:
                    to_remove.append(key)
    
    # Check neighbors of new_pos for i
    if seq[i] == 'H':
        for q in _neighbours(new_pos):
            k = pos_map.get(q)
            # k is in pos_map, but we must ensure k is not i itself (impossible) 
            # and k is not the residue currently at old_pos? 
            # Wait, if k is the residue at old_pos, that's not a neighbor of new_pos unless new_pos is adjacent to old_pos.
            # But we are moving i, so old_pos is vacated.
            # However, other residues are still there.
            if k is not None and k != i and abs(i - k) > 1 and seq[k] == 'H':
                key = (min(i, k), max(i, k))
                to_add.append(key)
                
    # 2. Handle removals
    new_contacts = set(contacts)
    delta_energy = 0
    for key in to_remove:
        if key in new_contacts:
            new_contacts.remove(key)
            delta_energy += 1 # Energy was -1, removing it adds +1
    
    # 3. Handle additions
    for key in to_add:
        if key not in new_contacts:
            # Check if this contact is valid (positions are correct)
            # The positions are correct by construction of new_pos
            new_contacts.add(key)
            delta_energy -= 1 # New contact adds -1
    
    # 4. Update pos_map
    new_pos_map = dict(pos_map)
    del new_pos_map[old_pos]
    new_pos_map[new_pos] = i
    
    # 5. Update chain
    chain[i] = new_pos
    
    return delta_energy, new_contacts, new_pos_map

def _anneal(chain, seq, time_limit):
    """
    Simulated Annealing with pivot moves.
    """
    n = len(chain)
    if n < 2:
        return chain
    
    # Initial state
    energy, contacts, pos_map = _compute_energy_and_contacts(chain, seq)
    best_chain = [list(p) for p in chain]
    best_energy = energy
    
    # SA Parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9995 # Adjust based on n?
    
    # For larger n, we need more steps or slower cooling
    # Let's aim for ~10000 steps per restart or until time limit
    steps = 0
    start_time = time.time()
    
    # To ensure we don't get stuck, we use a "pivot" move:
    # Pick a residue i, pick a direction, move it if empty.
    # Or better: "Segment reversal" or "Pivot".
    # Pivot is faster to update energy incrementally.
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Random pivot move
        # Pick i (not 0 or n-1 ideally, but any i works if we check boundaries)
        i = random.randint(1, n - 2)
        
        # Pick a random neighbor of chain[i]
        current_pos = chain[i]
        neighbors = list(_neighbours(current_pos))
        random.shuffle(neighbors)
        
        moved = False
        for new_pos in neighbors:
            if new_pos in pos_map:
                # Check if it's a valid move (not chain adjacent)
                j = pos_map[new_pos]
                if abs(i - j) > 1:
                    # Try the move
                    old_pos = current_pos
                    # Tentative update
                    # We need to check if new_pos is free. 
                    # If j exists and abs(i-j)>1, it's occupied by a non-adjacent residue.
                    # We can't move there.
                    continue
            else:
                # Free space
                # Perform move
                delta_e, new_contacts, new_pos_map = _incremental_energy_update(
                    chain, seq, i, -1, old_pos, new_pos, pos_map, contacts
                )
                new_energy = energy + delta_e
                
                if new_energy < energy or random.random() < math.exp(-delta_e / T):
                    # Accept
                    energy = new_energy
                    contacts = new_contacts
                    pos_map = new_pos_map
                    moved = True
                    
                    if energy < best_energy:
                        best_energy = energy
                        best_chain = [list(p) for p in chain]
                    break
                else:
                    # Reject, but we modified chain/pos_map/contacts in the update function?
                    # My _incremental_energy_update modifies chain in place!
                    # I need to revert if rejected.
                    chain[i] = old_pos
                    # Revert pos_map
                    # The function returned new_pos_map which is a copy. 
                    # But it also modified chain.
                    # So if rejected, we just keep the old state variables (energy, contacts, pos_map)
                    # and reset chain[i] to old_pos.
                    pass

        if not moved:
            # No valid move found or all rejected?
            # If no valid empty neighbor, try another i? Or just cool down.
            pass
            
        T *= cooling_rate
        steps += 1
        
        # Check time
        if time.time() - start_time > time_limit:
            break
            
    return best_chain

def fold(sequence, dimension):
    """
    Fold the sequence using spiral initialization and simulated annealing.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    # Determine time budget per sequence
    # Total budget is ~190s. 10 sequences. ~19s each.
    # But we should allocate based on difficulty (length).
    # Let's use a fixed budget per call, say 15s, to be safe.
    # If there are 10 calls, 150s total. Leaves margin.
    time_per_seq = 15.0
    
    # Initial conformation
    chain = _init_spiral(n, dimension)
    
    # Validate initial chain (just in case spiral logic had a bug)
    # Check self-avoidance
    if len(set(chain)) != n:
        # Fallback to simple line
        chain = [[i, 0] for i in range(n)]
        
    # Check connectivity
    for i in range(1, n):
        dx = abs(chain[i][0] - chain[i-1][0])
        dy = abs(chain[i][1] - chain[i-1][1])
        if dx + dy != 1:
             # Invalid, fallback
             chain = [[i, 0] for i in range(n)]
             break

    # Run Annealing
    best_chain = _anneal(chain, sequence, time_per_seq)
    
    # Final verification
    # Ensure self-avoidance and connectivity
    if len(set(best_chain)) != n:
         return chain # Return the initial valid one if annealing broke it (shouldn't happen with correct SA)
         
    for i in range(1, n):
        dx = abs(best_chain[i][0] - best_chain[i-1][0])
        dy = abs(best_chain[i][1] - best_chain[i-1][1])
        if dx + dy != 1:
             return chain
             
    return best_chain

# EVOLVE-BLOCK-END
