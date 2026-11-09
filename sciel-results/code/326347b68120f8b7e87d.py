"""Rewritten HP lattice folding program using simulated annealing with 2-opt moves."""
import random
import math
from collections import defaultdict


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_contact_map(sequence, chain):
    """Precompute which positions are H-H contacts (non-adjacent along chain)."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    # contact[i] = set of j > i+1 such that i and j are H and adjacent on lattice
    contact = [set() for _ in range(n)]
    for i, p in enumerate(chain):
        if sequence[i] != 'H':
            continue
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contact[i].add(j)
    return contact


def _energy_from_contact(contact, sequence):
    """Compute energy from contact map."""
    count = 0
    for i in range(len(contact)):
        count += len(contact[i])
    return -count


def _try_move_2opt(chain, i, j, sequence, contact):
    """
    Attempt a 2-opt move: reverse the segment from i to j.
    Returns (new_chain, new_contact, valid) if move is valid and changes energy.
    We only consider moves where i and j are not adjacent along the chain (j > i+1)
    and the resulting chain is self-avoiding.
    """
    n = len(chain)
    if j <= i + 1:
        return None, None, False
    
    # The segment [i..j] is reversed.
    # Check self-avoidance: the new positions at i..j are chain[j], chain[j-1], ..., chain[i]
    # We need to ensure no two of these new positions collide with each other (they don't, since original was self-avoiding and we're just permuting)
    # But we need to check that the new position at k (for k in i..j) doesn't collide with any position outside i..j.
    
    # Positions outside [i, j] remain the same.
    # For k in [i, j], new position is chain[i + j - k].
    
    # Build set of occupied positions outside [i, j]
    occupied_outside = set()
    for k in range(n):
        if k < i or k > j:
            occupied_outside.add(chain[k])
    
    # Check each new position in [i, j]
    new_segment = []
    for k in range(i, j + 1):
        new_pos = chain[i + j - k]
        if new_pos in occupied_outside:
            return None, None, False
        new_segment.append(new_pos)
    
    # Also check within the segment: since we're reversing a self-avoiding path, 
    # consecutive elements in new_segment were non-consecutive in original (except at boundaries),
    # but actually the internal structure is preserved in reverse, so no self-collision within segment.
    # However, we must check that new_segment doesn't have duplicates (it won't since chain is self-avoiding).
    
    # Build new chain
    new_chain = chain[:i] + new_segment + chain[j+1:]
    
    # Recompute contact map for the new chain
    new_contact = _compute_contact_map(sequence, new_chain)
    
    return new_chain, new_contact, True


def _try_insertion_move(chain, i, j, sequence, contact):
    """
    Attempt an insertion move: remove residue at position i and insert it at position j.
    i < j, and j >= i+2 (to avoid trivial moves).
    """
    n = len(chain)
    if j <= i + 1:
        return None, None, False
    
    # Remove chain[i], shift i+1..j-1 left, insert chain[i] at position j
    # New chain: [0..i-1], [i+1..j], [i], [j+1..n-1]
    
    # Check self-avoidance
    # Positions outside the affected region remain the same.
    # Affected region: indices i to j in the new chain.
    # New positions at indices i..j: chain[i+1], chain[i+2], ..., chain[j], chain[i]
    
    occupied_outside = set()
    for k in range(n):
        if k < i or k > j:
            occupied_outside.add(chain[k])
    
    new_segment = []
    for k in range(i + 1, j + 1):
        new_segment.append(chain[k])
    new_segment.append(chain[i])
    
    for pos in new_segment:
        if pos in occupied_outside:
            return None, None, False
    
    new_chain = chain[:i] + new_segment + chain[j+1:]
    
    # Recompute contact map
    new_contact = _compute_contact_map(sequence, new_chain)
    
    return new_chain, new_contact, True


def fold(sequence, dimension):
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    # Initial chain: a simple zig-zag or straight line
    # Use a compact initial conformation: spiral-like
    chain = []
    x, y = 0, 0
    direction = 0  # 0:right, 1:down, 2:left, 3:up
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]
    
    # Build a simple self-avoiding walk
    for i in range(n):
        chain.append((x, y))
        # Try to continue in current direction, else turn
        found = False
        for d in range(4):
            nx = x + dx[d]
            ny = y + dy[d]
            if (nx, ny) not in chain:
                x, y = nx, ny
                direction = d
                found = True
                break
        if not found:
            # Shouldn't happen for reasonable n with this strategy, but fallback
            x += 1
    
    # Ensure self-avoidance
    if len(set(chain)) != n:
        # Fallback to straight line
        chain = [(i, 0) for i in range(n)]
    
    # Compute initial contact map and energy
    contact = _compute_contact_map(sequence, chain)
    best_chain = chain
    best_contact = contact
    best_energy = _energy_from_contact(contact, sequence)
    
    # Simulated annealing
    rng = random.Random(42 + len(sequence))
    start_temp = 5.0
    end_temp = 0.01
    cooling = 0.995
    max_iter = 50000
    
    temp = start_temp
    current_chain = best_chain
    current_contact = best_contact
    current_energy = best_energy
    
    for it in range(max_iter):
        if temp <= end_temp:
            break
        
        # Choose a random move
        move_type = rng.random()
        
        if move_type < 0.5:
            # 2-opt move
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i > j:
                i, j = j, i
            if j <= i + 1:
                continue
            new_chain, new_contact, valid = _try_move_2opt(current_chain, i, j, sequence, current_contact)
        else:
            # Insertion move
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i > j:
                i, j = j, i
            if j <= i + 1:
                continue
            new_chain, new_contact, valid = _try_insertion_move(current_chain, i, j, sequence, current_contact)
        
        if not valid:
            continue
        
        new_energy = _energy_from_contact(new_contact, sequence)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            current_chain = new_chain
            current_contact = new_contact
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain
                best_contact = current_contact
        
        temp *= cooling
    
    return [list(p) for p in best_chain]
