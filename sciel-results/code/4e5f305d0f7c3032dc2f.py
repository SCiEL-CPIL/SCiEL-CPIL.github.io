"""HP lattice protein folding on the cubic lattice - pivot MC with incremental energy."""
import time
import random
import math
import sys

# EVOLVE-BLOCK-START

_NEIGHBORS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]

def _get_deadline():
    if not hasattr(_get_deadline, '_start'):
        _get_deadline._start = time.time()
    return _get_deadline._start + 325.0

def _time_left():
    return max(0.0, _get_deadline() - time.time())

def _delta_move(chain, sequence, i, new_pos):
    """
    Compute the delta energy (change in energy) when moving residue i to new_pos.
    Returns (delta, valid) where valid is True if the move is self-avoiding.
    chain is a list of tuples, sequence is a string.
    """
    n = len(chain)
    old_pos = chain[i]
    
    # Check if new_pos is occupied by another residue
    for j in range(n):
        if j != i and chain[j] == new_pos:
            return 0, False
    
    # Check unit step constraints with neighbors
    if i > 0:
        prev = chain[i-1]
        d = abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) + abs(new_pos[2] - prev[2])
        if d != 1:
            return 0, False
    if i < n - 1:
        nxt = chain[i+1]
        d = abs(nxt[0] - new_pos[0]) + abs(nxt[1] - new_pos[1]) + abs(nxt[2] - new_pos[2])
        if d != 1:
            return 0, False
    
    # Compute delta energy
    # Energy = -number of H-H contacts (non-consecutive)
    # When we move residue i, contacts involving i change.
    # Also, if i was adjacent to some H residue j (non-consecutive), that contact is lost.
    # If i is H and new_pos is adjacent to some H residue j (non-consecutive), that contact is gained.
    
    is_h = (sequence[i] == 'H')
    delta = 0
    
    if is_h:
        # Check old position: for each j, if j is H, j != i, |j-i|>1, and chain[j] adjacent to old_pos
        # then that contact is lost (delta += 1, meaning energy becomes more negative, so delta_energy = -1)
        # Wait: energy = -contacts. If we lose a contact, energy increases (becomes less negative), so delta = +1.
        # If we gain a contact, energy decreases, delta = -1.
        
        for j in range(n):
            if j == i or abs(j - i) <= 1:
                continue
            if sequence[j] != 'H':
                continue
            pos_j = chain[j]
            d = abs(pos_j[0] - old_pos[0]) + abs(pos_j[1] - old_pos[1]) + abs(pos_j[2] - old_pos[2])
            if d == 1:
                # Lost contact
                delta += 1
        
        # Check new position
        for j in range(n):
            if j == i or abs(j - i) <= 1:
                continue
            if sequence[j] != 'H':
                continue
            pos_j = chain[j]
            d = abs(pos_j[0] - new_pos[0]) + abs(pos_j[1] - new_pos[1]) + abs(pos_j[2] - new_pos[2])
            if d == 1:
                # Gained contact
                delta -= 1
    
    # Also, if i is P, moving it doesn't directly change energy, BUT if moving i breaks a contact between two other H residues?
    # No, i is P, so it was never part of a contact. Moving it doesn't affect H-H contacts unless it was blocking them?
    # No, contacts are based on lattice positions, not blocking.
    # So if i is P, delta is always 0?
    # Wait, what if moving i changes the position of a neighbor? No, we're only moving i.
    # So if sequence[i] == 'P', delta = 0.
    
    return delta, True

def _pivot_move(chain, sequence, pivot, k, rng):
    """
    Perform a pivot move: rotate the subchain from k+1 to n-1 around the bond (k, k+1) or (k-1, k).
    Actually, standard pivot: pick pivot index p, rotate the part of the chain from p to n-1 around the bond (p-1, p).
    The rotation is a 90-degree rotation in a random plane.
    Returns (new_chain, valid, delta_energy)
    """
    n = len(chain)
    if n <= 2:
        return chain, False, 0
    
    # Choose pivot point p (1 to n-2)
    # We rotate the tail (p..n-1) around the bond (p-1, p).
    # The bond vector is chain[p] - chain[p-1].
    # We pick a random axis perpendicular to the bond and rotate by 90 degrees.
    
    p = rng.randint(1, n - 2)
    a = chain[p-1]
    b = chain[p]
    
    # Bond vector
    bond = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    
    # Find all possible rotation axes perpendicular to the bond
    # The bond is a unit vector along one of the axes.
    # There are 4 other axes, and for each, 2 rotations (90 and -90).
    # Actually, we can just pick a random axis from the 6, and if it's parallel to bond, pick another.
    
    axes = []
    for ax in _NEIGHBORS:
        # Check if ax is perpendicular to bond
        dot = ax[0]*bond[0] + ax[1]*bond[1] + ax[2]*bond[2]
        if dot == 0:
            axes.append(ax)
    
    if not axes:
        return chain, False, 0
    
    axis = rng.choice(axes)
    # Choose rotation direction: +90 or -90
    # Rotation matrix for 90 degrees around axis (1,0,0): (x,y,z) -> (x,-z,y)
    # For (0,1,0): (x,y,z) -> (z,y,-x)
    # For (0,0,1): (x,y,z) -> (-y,x,z)
    # We can implement a general rotation function.
    
    def rotate90(pt, axis, sign=1):
        x, y, z = pt
        ax, ay, az = axis
        if ax == 1:
            return (x, -sign*z, sign*y)
        elif ax == -1:
            return (x, sign*z, -sign*y)
        elif ay == 1:
            return (sign*z, y, -sign*x)
        elif ay == -1:
            return (-sign*z, y, sign*x)
        elif az == 1:
            return (-sign*y, sign*x, z)
        elif az == -1:
            return (sign*y, -sign*x, z)
        return pt
    
    sign = rng.choice([1, -1])
    
    # Apply rotation to all points in the tail (p..n-1) relative to a
    new_chain = list(chain[:p])
    for i in range(p, n):
        pt = chain[i]
        rel = (pt[0] - a[0], pt[1] - a[1], pt[2] - a[2])
        new_rel = rotate90(rel, axis, sign)
        new_pt = (a[0] + new_rel[0], a[1] + new_rel[1], a[2] + new_rel[2])
        new_chain.append(new_pt)
    
    # Check self-avoidance
    if len(set(new_chain)) != n:
        return chain, False, 0
    
    # Check unit steps
    for i in range(1, n):
        d = abs(new_chain[i][0] - new_chain[i-1][0]) + abs(new_chain[i][1] - new_chain[i-1][1]) + abs(new_chain[i][2] - new_chain[i-1][2])
        if d != 1:
            return chain, False, 0
    
    # Compute delta energy
    # This is expensive if done from scratch, but for correctness, let's do it.
    # We can optimize later.
    old_energy = _compute_energy(chain, sequence)
    new_energy = _compute_energy(new_chain, sequence)
    delta = new_energy - old_energy
    
    return new_chain, True, delta

def _compute_energy(chain, sequence):
    n = len(chain)
    pos_to_idx = {chain[i]: i for i in range(n)}
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        for nb in _NEIGHBORS:
            pt = (chain[i][0]+nb[0], chain[i][1]+nb[1], chain[i][2]+nb[2])
            j = pos_to_idx.get(pt)
            if j is not None and j > i + 1:
                contacts += 1
    return -contacts

def _grow_greedy(sequence, rng):
    n = len(sequence)
    if n <= 1:
        return [(0,0,0)]
    
    chain = [(0,0,0)]
    occupied = {(0,0,0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        best_score = -1
        best_sites = []
        
        for site in _NEIGHBORS:
            pos = (last[0]+site[0], last[1]+site[1], last[2]+site[2])
            if pos in occupied:
                continue
            score = 0
            if sequence[i] == 'H':
                for j in range(i-1, -1, -1):
                    if i - j > 2:
                        break
                    if sequence[j] == 'H':
                        d = abs(pos[0]-chain[j][0]) + abs(pos[1]-chain[j][1]) + abs(pos[2]-chain[j][2])
                        if d == 1:
                            score += 1
            else:
                for j in range(i-1, -1, -1):
                    if i - j > 2:
                        break
                    if sequence[j] == 'H':
                        d = abs(pos[0]-chain[j][0]) + abs(pos[1]-chain[j][1]) + abs(pos[2]-chain[j][2])
                        if d == 1:
                            score += 1
            if score > best_score:
                best_score = score
                best_sites = [pos]
            elif score == best_score:
                best_sites.append(pos)
        
        if not best_sites:
            return None
        
        pos = rng.choice(best_sites)
        chain.append(pos)
        occupied[pos] = i
    
    return chain

def _local_search_pivot(chain, sequence, time_budget, rng):
    n = len(chain)
    if n <= 2:
        return chain
    
    energy = _compute_energy(chain, sequence)
    deadline = time.time() + time_budget
    
    # Simulated annealing-like
    T = 1.0
    T_min = 0.1
    cooling_rate = 0.999
    
    while time.time() < deadline:
        new_chain, valid, delta = _pivot_move(chain, sequence, 0, 0, rng)
        if not valid:
            continue
        
        if delta < 0:
            chain = new_chain
            energy += delta
        else:
            # Metropolis criterion
            if random.random() < math.exp(-delta / T):
                chain = new_chain
                energy += delta
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    return chain

def _fold_single(sequence, time_budget, rng):
    n = len(sequence)
    if n <= 2:
        return [(i, 0, 0) for i in range(n)]
    
    best_chain = None
    best_energy = float('inf')
    
    deadline = time.time() + time_budget
    restarts = 0
    max_restarts = 50
    
    while time.time() < deadline - 2 and restarts < max_restarts:
        chain = _grow_greedy(sequence, rng)
        if chain is None:
            continue
        
        # Local search
        remaining = deadline - time.time() - 1
        if remaining > 0.5:
            chain = _local_search_pivot(chain, sequence, remaining, rng)
        
        energy = _compute_energy(chain, sequence)
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
        
        restarts += 1
    
    if best_chain is None:
        return [(i, 0, 0) for i in range(n)]
    return best_chain

def _fallback_chain(n):
    return [(i, 0, 0) for i in range(n)]

def fold(sequence, dimension):
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    remaining = _time_left()
    if remaining < 1.0:
        return [list(p) for p in _fallback_chain(n)]
    
    per_call_budget = min(remaining * 0.9, 30.0)
    
    if n <= 48:
        # More restarts for shorter
        pass
    else:
        pass
    
    rng = random.Random(42 + n * 7)
    
    best_chain = _fold_single(sequence, per_call_budget, rng)
    
    # Validate
    if len(set(best_chain)) != n:
        best_chain = _fallback_chain(n)
    
    for i in range(1, n):
        d = abs(best_chain[i][0] - best_chain[i-1][0]) + \
            abs(best_chain[i][1] - best_chain[i-1][1]) + \
            abs(best_chain[i][2] - best_chain[i-1][2])
        if d != 1:
            best_chain = _fallback_chain(n)
            break
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
