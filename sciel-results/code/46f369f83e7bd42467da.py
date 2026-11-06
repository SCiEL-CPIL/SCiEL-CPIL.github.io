"""HP lattice protein folding using simulated annealing with global moves."""
# EVOLVE-BLOCK-START
import time
import random
import math

_TIME_LIMIT = 200.0  # safety margin before 210s kill
_DEADLINE = time.time() + _TIME_LIMIT


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _build_index(chain):
    return {tuple(p): i for i, p in enumerate(chain)}


def _energy_from_index(sequence, chain, where):
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _energy(sequence, chain):
    where = _build_index(chain)
    return _energy_from_index(sequence, chain, where)


def _is_valid_move(sequence, chain, i, j):
    """Check if reversing segment [i, j] (inclusive) keeps self-avoidance."""
    if j - i < 2:
        return False
    # The reversed segment will occupy the same positions but in reverse order.
    # We need to ensure no new contacts with the rest of the chain are created
    # that would cause overlap. Actually, reversal doesn't change the set of
    # positions occupied, so self-avoidance is automatically preserved IF the
    # segment is reversed in place. Wait, that's not right. The positions are
    # the same, but the chain order changes. The set of occupied lattice sites
    # is identical. So self-avoidance is trivially preserved.
    # However, we must ensure that the move is a valid "crankshaft" or "pivot".
    # A pivot rotates a segment around a pivot point. A crankshaft rotates two
    # segments around two pivots. Reversal is a specific type of move.
    # Let's implement a simpler global move: pick two non-adjacent indices i, j
    # and try to reverse the segment between them. Since the set of positions
    # doesn't change, self-avoidance is preserved. But does it change energy?
    # Yes, because the adjacency relationships change.
    # Actually, reversing a segment of a self-avoiding walk on a lattice is not
    # always a valid move because the endpoints might not connect properly if
    # we just reverse the list. Let's think.
    # Chain: ... A - S1 - S2 - ... - Sk - B ...
    # Reverse S1..Sk: ... A - Sk - ... - S2 - S1 - B ...
    # For this to be a valid walk, A must be adjacent to Sk, and S1 must be
    # adjacent to B. In the original, A is adjacent to S1, and Sk is adjacent to B.
    # So we need A adjacent to Sk and S1 adjacent to B.
    # This is a "reversal" move. It is valid if those two adjacency conditions hold.
    return True  # We'll check the actual adjacency in the move function


def _try_reversal(sequence, chain, i, j, where):
    """Try to reverse segment [i, j]. Returns new chain if valid, else None."""
    if j - i < 2:
        return None
    # Check adjacency conditions
    # Original: chain[i-1] -- chain[i] ... chain[j] -- chain[j+1]
    # New: chain[i-1] -- chain[j] ... chain[i] -- chain[j+1]
    # So we need chain[i-1] adjacent to chain[j] and chain[i] adjacent to chain[j+1]
    if i > 0:
        a = chain[i-1]
        b = chain[j]
        if abs(a[0]-b[0]) + abs(a[1]-b[1]) != 1:
            return None
    if j < len(chain) - 1:
        a = chain[i]
        b = chain[j+1]
        if abs(a[0]-b[0]) + abs(a[1]-b[1]) != 1:
            return None
    
    new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
    return new_chain


def _try_pivot(sequence, chain, i, where):
    """Try a pivot move at index i. Rotates segment [i, end] by 90 or 180 degrees."""
    if i <= 0 or i >= len(chain) - 1:
        return None
    pivot = chain[i]
    segment = chain[i+1:]
    
    # Try 90-degree clockwise and counter-clockwise rotations, and 180-degree.
    # Rotation is around the pivot point.
    # Vector from pivot to next point: v = chain[i+1] - pivot
    # Rotate v, then new point = pivot + rotated_v
    # But this only works for the first point. The rest of the segment must be
    # translated accordingly. Actually, a pivot move rotates the entire tail.
    
    # Let's use a simpler approach: try to move the tail by rotating the vector
    # from pivot to the next point.
    vx = chain[i+1][0] - pivot[0]
    vy = chain[i+1][1] - pivot[1]
    
    # Possible rotations: (vx, vy) -> (-vy, vx), (vy, -vx), (-vx, -vy)
    rotations = [(-vy, vx), (vy, -vx), (-vx, -vy)]
    
    for rx, ry in rotations:
        new_segment = []
        new_segment.append((pivot[0] + rx, pivot[1] + ry))
        
        # For the rest of the segment, we need to apply the same rotation.
        # This is complex. Let's skip this and use a simpler global move.
        # For now, let's just try the reversal move and a "kick" move.
        pass
    
    return None


def _try_kick(sequence, chain, i, where):
    """Try to move residue i to an adjacent empty site."""
    if i == 0 or i == len(chain) - 1:
        return None
    
    p = chain[i]
    for q in _neighbours(p):
        if q in where and where[q] != i:
            continue
        # Check if moving i to q is valid (no self-intersection)
        # The new position q must not be occupied by any other residue.
        # We already checked that.
        # But we also need to ensure that the new chain is self-avoiding.
        # Since we're only moving one residue, and q was empty, the only
        # potential conflict is if q is adjacent to some other residue that
        # is not adjacent in the chain. But that's allowed in HP model.
        # Wait, self-avoidance means no two residues occupy the same site.
        # Since q was empty, and we're moving i to q, and i was at p,
        # the new set of occupied sites is (old set - {p}) + {q}.
        # Since q was not in old set, and p was, the new set has the same size.
        # And no two residues are at the same site. So self-avoidance is preserved.
        # But we must ensure that the chain is still a valid walk.
        # The new chain has residue i at q. Its neighbors in the chain are
        # i-1 and i+1. We need q to be adjacent to chain[i-1] and chain[i+1].
        # Actually, no. The chain is a sequence of points. The move changes
        # chain[i] from p to q. For the chain to be a valid walk,
        # chain[i-1] must be adjacent to q, and q must be adjacent to chain[i+1].
        if i > 0 and abs(chain[i-1][0]-q[0]) + abs(chain[i-1][1]-q[1]) != 1:
            continue
        if i < len(chain) - 1 and abs(chain[i+1][0]-q[0]) + abs(chain[i+1][1]-q[1]) != 1:
            continue
        
        new_chain = chain[:]
        new_chain[i] = q
        return new_chain
    
    return None


def _initial_conformation(sequence, dimension, rng):
    """Generate an initial random self-avoiding walk."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                options.append(site)
        if not options:
            # Backtrack or restart
            return None
        site = rng.choice(options)
        occupied[site] = i
        chain.append(site)
    
    return chain


def _simulated_annealing(sequence, dimension, rng, time_budget):
    """Simulated annealing with global moves."""
    n = len(sequence)
    
    # Generate initial conformation
    chain = None
    for _ in range(1000):
        chain = _initial_conformation(sequence, dimension, rng)
        if chain is not None:
            break
    if chain is None:
        # Fallback: straight line
        chain = [(i, 0) for i in range(n)]
    
    where = _build_index(chain)
    current_energy = _energy_from_index(sequence, chain, where)
    best_chain = chain[:]
    best_energy = current_energy
    
    # Simulated annealing parameters
    T = 1.0
    cooling_rate = 0.995
    min_T = 0.01
    
    start_time = time.time()
    while T > min_T and time.time() - start_time < time_budget:
        # Try a move
        move_type = rng.random()
        
        new_chain = None
        if move_type < 0.5:
            # Kick move
            i = rng.randint(1, n - 2)
            new_chain = _try_kick(sequence, chain, i, where)
        elif move_type < 0.8:
            # Reversal move
            if n >= 4:
                i = rng.randint(1, n - 3)
                j = rng.randint(i + 2, n - 2)
                new_chain = _try_reversal(sequence, chain, i, j, where)
        
        if new_chain is None:
            continue
        
        new_where = _build_index(new_chain)
        new_energy = _energy_from_index(sequence, new_chain, new_where)
        
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            where = new_where
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
        
        T *= cooling_rate
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Find a low-energy conformation using simulated annealing with global moves."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    # Time budget per sequence
    time_budget = max(5.0, _TIME_LIMIT / 10.0 - 5.0)
    
    # Run simulated annealing
    best_chain, best_energy = _simulated_annealing(sequence, dimension, rng, time_budget)
    
    # Convert to list of lists
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
