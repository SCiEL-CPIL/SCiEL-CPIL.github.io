import time
import random
import itertools

N = 16
DEADLINE = 300.0  # safety margin under 330s

# Precompute all points
POINTS = [(x, y) for x in range(N) for y in range(N)]
IDX = {p: i for i, p in enumerate(POINTS)}
M = len(POINTS)  # 256

def _dist_sq(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2

# Precompute isosceles relationships
# iso_mask[i][j] is a bitmask of points k such that (i, j, k) is isosceles.
# This allows O(1) check: is (i, j, k) isosceles? -> (iso_mask[i][j] >> k) & 1
iso_mask = [[0] * M for _ in range(M)]
for i in range(M):
    for j in range(i + 1, M):
        a = POINTS[i]
        b = POINTS[j]
        d_ab = _dist_sq(a, b)
        mask = 0
        for k in range(M):
            if k == i or k == j:
                continue
            c = POINTS[k]
            d_ac = _dist_sq(a, c)
            d_bc = _dist_sq(b, c)
            if d_ac == d_bc or d_ac == d_ab or d_bc == d_ab:
                mask |= (1 << k)
        iso_mask[i][j] = mask
        iso_mask[j][i] = mask

def _is_valid_set(chosen_indices):
    """Check if the set of indices forms a no-isosceles set."""
    n = len(chosen_indices)
    for i in range(n):
        a = chosen_indices[i]
        for j in range(i + 1, n):
            b = chosen_indices[j]
            # Check if any other point c in the set forms an isosceles triangle with a, b
            mask = iso_mask[a][b]
            # We need to check if (mask & chosen_bitmask) has any bits set
            # But we don't have a global bitmask for the set in this function signature easily without passing it.
            # Let's just iterate for simplicity in the check, or pass bitmask.
            # For speed in minimization, we will use a different check.
            pass
    return True # Placeholder, will use optimized check in main loop

def _is_maximal(chosen_indices, chosen_set):
    """Check if the set is maximal: no other point can be added."""
    for p_idx in range(M):
        if p_idx in chosen_set:
            continue
        # Can we add p_idx?
        # It can be added if for all pairs (a, b) in chosen_set, (a, b, p_idx) is NOT isosceles.
        # Equivalently, p_idx is NOT in iso_mask[a][b] for any pair (a, b).
        ok = True
        n = len(chosen_indices)
        for i in range(n):
            a = chosen_indices[i]
            for j in range(i + 1, n):
                b = chosen_indices[j]
                if (iso_mask[a][b] >> p_idx) & 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return False # Not maximal, p_idx could be added
    return True

def _greedy_build(order):
    """Build a valid set using the given order."""
    chosen = []
    chosen_set = set()
    for idx in order:
        p = POINTS[idx]
        ok = True
        # Check against all pairs in chosen
        for ii in range(len(chosen)):
            a = chosen[ii]
            for jj in range(ii + 1, len(chosen)):
                b = chosen[jj]
                if (iso_mask[idx][a] >> b) & 1:
                    ok = False
                    break
                if (iso_mask[idx][b] >> a) & 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen.append(idx)
            chosen_set.add(idx)
    return chosen, chosen_set

def _minimize_by_deletion(chosen_indices, chosen_set):
    """Try to remove points while maintaining maximality."""
    improved = True
    while improved:
        improved = False
        # Try removing each point
        for p in list(chosen_indices):
            # Tentatively remove p
            new_indices = [x for x in chosen_indices if x != p]
            new_set = set(new_indices)
            
            # Check if new_set is maximal
            # Optimization: If removing p makes it non-maximal, some q was only blocked by p (in pairs with others).
            # We just run the full check for simplicity first.
            if _is_maximal(new_indices, new_set):
                # Accept removal
                chosen_indices = new_indices
                chosen_set = new_set
                improved = True
                # Break to restart loop with smaller set? Or continue?
                # Restarting is safer to avoid complex state updates.
                break
    return chosen_indices, chosen_set

def construct():
    start_time = time.time()
    best_set = None
    best_len = M + 1
    
    rng = random.Random(12345)
    
    # Initial orderings
    orderings = []
    # Row-major
    orderings.append(list(range(M)))
    # Column-major
    orderings.append([IDX[(y, x)] for x in range(N) for y in range(N)])
    # Diagonal
    orderings.append([IDX[(x, y)] for d in range(2 * N - 1) for x in range(N) for y in range(N) if x + y == d])
    
    # Randomized restarts
    attempts = 0
    while time.time() - start_time < DEADLINE - 30:
        if attempts < len(orderings):
            order = orderings[attempts]
        else:
            order = list(range(M))
            rng.shuffle(order)
        
        attempts += 1
        
        # 1. Greedy Build
        chosen_indices, chosen_set = _greedy_build(order)
        
        # 2. Minimize by Deletion
        chosen_indices, chosen_set = _minimize_by_deletion(chosen_indices, chosen_set)
        
        if len(chosen_indices) < best_len:
            best_len = len(chosen_indices)
            best_set = list(chosen_indices)
            
            # Optional: If we find a very small set, we can tighten the deadline or stop early?
            # Theoretical min is likely small. If we hit 6 or 7, we can be more aggressive.
            
        # If we've found a set of size 7 or less, we might stop early to save time?
        # But we want the MINIMUM. Let's keep going unless time is up.
        
    # Convert indices back to points
    if best_set is None:
        # Fallback: return an empty set? No, must be valid.
        # If no set found, return a single point (which is always valid and maximal? 
        # A single point is valid. Is it maximal? 
        # If S={p}, is it maximal? 
        # It is maximal if no q can be added. 
        # q can be added if {p, q} is not part of an isosceles triangle with any other point in S? 
        # S has only p. So {p, q} is just a pair. 
        # The condition "no isosceles triangle" applies to triples. 
        # So any pair is valid. 
        # Thus S={p} is NOT maximal, because we can add q. 
        # So we must return a maximal set.
        # Greedy should always find at least one.
        return []
        
    return [POINTS[i] for i in best_set]
