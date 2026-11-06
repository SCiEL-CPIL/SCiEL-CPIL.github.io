"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def _paley_83():
    """Return the 83-vertex Paley tournament as a list of bits."""
    p = 83
    squares = {x * x % p for x in range(1, p)}
    return [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]

def _check_validity(orientation, n):
    """
    Check if every 4-set has a common dominator.
    orientation: list of bits for pairs (i,j), i<j, in lexicographic order.
    Returns True if valid.
    """
    # Build adjacency: adj[i][j] = 1 if i -> j
    adj = [[0]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                adj[i][j] = 1
            else:
                adj[j][i] = 1
    
    # For each 4-set, check if there's a vertex that dominates all others in the set
    # A vertex v dominates the set if adj[v][u] == 1 for all u in set, u != v
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        # Check if any of the 4 vertices dominates the other 3
        if adj[a][b] and adj[a][c] and adj[a][d]:
            continue
        if adj[b][a] and adj[b][c] and adj[b][d]:
            continue
        if adj[c][a] and adj[c][b] and adj[c][d]:
            continue
        if adj[d][a] and adj[d][b] and adj[d][c]:
            continue
        return False
    return True

def _violation_count(orientation, n):
    """
    Count the number of 4-sets that do NOT have a common dominator.
    Uses bitmask representation for speed.
    """
    # Build out-neighborhood bitmasks
    out_mask = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                out_mask[i] |= (1 << j)
            else:
                out_mask[j] |= (1 << i)
    
    violations = 0
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        # Check if any vertex in the quad dominates all others in the quad
        ok = False
        if (out_mask[a] >> b) & 1 and (out_mask[a] >> c) & 1 and (out_mask[a] >> d) & 1:
            ok = True
        elif (out_mask[b] >> a) & 1 and (out_mask[b] >> c) & 1 and (out_mask[b] >> d) & 1:
            ok = True
        elif (out_mask[c] >> a) & 1 and (out_mask[c] >> b) & 1 and (out_mask[c] >> d) & 1:
            ok = True
        elif (out_mask[d] >> a) & 1 and (out_mask[d] >> b) & 1 and (out_mask[d] >> c) & 1:
            ok = True
        if not ok:
            violations += 1
    return violations

def _remove_vertex(orientation, n, v):
    """Remove vertex v from the tournament, returning new orientation list and new n."""
    new_n = n - 1
    new_orientation = []
    idx = 0
    for i in range(n):
        if i == v:
            continue
        for j in range(i+1, n):
            if j == v:
                continue
            new_orientation.append(orientation[idx])
            idx += 1
        # Skip all pairs where i is the current vertex and j > i
        # Actually, we need to be more careful. Let me redo this.
    
    # Better approach: rebuild from scratch
    new_orientation = []
    for i in range(n):
        if i == v:
            continue
        for j in range(i+1, n):
            if j == v:
                continue
            # Find the index in original orientation
            # Original index for pair (i,j) is i*n - i*(i+1)//2 + (j - i - 1)
            orig_idx = i * n - i*(i+1)//2 + (j - i - 1)
            new_orientation.append(orientation[orig_idx])
    return new_orientation, new_n

def construct():
    """
    Construct a tournament where every 4-set has a common dominator.
    Strategy: Start with Paley tournament on 83 vertices, remove vertices one by one,
    then use local search to repair violations.
    """
    deadline = time.time() + 110  # Safety margin
    
    # Start with Paley 83
    orientation = _paley_83()
    n = 83
    
    # Try to remove vertices greedily
    # For each vertex, check if removing it still leaves a valid tournament
    # If so, remove it
    # We want to minimize n, so keep removing as long as valid
    
    # First, verify the starting tournament is valid (it should be for Paley 83)
    # Actually, Paley tournament on p ≡ 3 mod 4 is a tournament, but we need to check
    # the 4-dominator property. Let's just proceed with removal.
    
    # Greedy removal: try to remove each vertex, check validity
    # This is expensive, so we'll do a limited number of removals
    
    # Actually, let's try a different approach:
    # Start from n=83, try to remove vertices one at a time
    # For each candidate vertex, remove it and count violations
    # If violations == 0, keep it removed
    # If violations > 0, try to fix with local search
    
    # For efficiency, let's use the violation count
    
    # Try removing vertices from 0 to n-1
    removed = set()
    for v in range(n):
        if time.time() > deadline - 10:
            break
        if v in removed:
            continue
        # Temporarily remove vertex v
        new_orient, new_n = _remove_vertex(orientation, n, v)
        # Check if valid
        if _check_validity(new_orient, new_n):
            orientation = new_orient
            n = new_n
            removed.add(v)
            # Restart the loop since indices changed
            removed = set()
            # Adjust removed set for the new indexing
            # Actually, this is tricky. Let's just restart the whole removal process
            break
    
    # Now do local search to try to remove more vertices or improve
    # Actually, let's just try to get a valid tournament with as few vertices as possible
    
    # If the greedy removal didn't work well, let's try a different approach:
    # Use local search on the orientation to find a valid tournament with fewer vertices
    
    # For now, return what we have
    # The Paley 83 tournament should be valid, so we should be able to remove some vertices
    
    # Let's do a more careful greedy removal
    # Restart with fresh Paley 83
    orientation = _paley_83()
    n = 83
    
    # Try removing vertices in a specific order
    # For each vertex v, check if the tournament without v is valid
    # If yes, remove it
    # Repeat until no more vertices can be removed
    
    for iteration in range(20):  # Limit iterations
        if time.time() > deadline - 10:
            break
        removed_this_round = False
        for v in range(n):
            if time.time() > deadline - 10:
                break
            new_orient, new_n = _remove_vertex(orientation, n, v)
            if _check_validity(new_orient, new_n):
                orientation = new_orient
                n = new_n
                removed_this_round = True
                break  # Restart the inner loop since indices changed
        if not removed_this_round:
            break
    
    # Verify final result
    if not _check_validity(orientation, n):
        # Fallback to the original Paley 83
        orientation = _paley_83()
        n = 83
    
    return {'n': n, 'orientation': orientation}

# EVOLVE-BLOCK-END
