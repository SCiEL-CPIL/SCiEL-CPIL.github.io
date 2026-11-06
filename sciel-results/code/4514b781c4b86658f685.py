import itertools
import random
import time
import sys

N = 16
NUM_POINTS = N * N
POINTS = [(x, y) for y in range(N) for x in range(N)]
POINT_TO_IDX = {p: i for i, p in enumerate(POINTS)}

# Precomputation
# dist2[i][j] = squared distance between point i and point j
# block_mask[i][j] = bitmask of points x such that {i, j, x} is isosceles
# For efficiency, we use integers as bitmasks.

def precompute():
    dist2 = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
    for i in range(NUM_POINTS):
        xi, yi = POINTS[i]
        for j in range(NUM_POINTS):
            xj, yj = POINTS[j]
            dist2[i][j] = (xi - xj) ** 2 + (yi - yj) ** 2

    block_mask = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
    for i in range(NUM_POINTS):
        for j in range(i + 1, NUM_POINTS):
            mask = 0
            d_ij = dist2[i][j]
            for x in range(NUM_POINTS):
                if x == i or x == j:
                    continue
                d_ix = dist2[i][x]
                d_jx = dist2[j][x]
                if d_ix == d_jx or d_ix == d_ij or d_jx == d_ij:
                    mask |= (1 << x)
            block_mask[i][j] = mask
            block_mask[j][i] = mask
    return dist2, block_mask

DIST2, BLOCK_MASK = precompute()

def construct():
    start_time = time.time()
    deadline = start_time + 285  # Safety margin for 300s limit
    
    best_set = None
    best_size = float('inf')
    
    # Helper: Check if a set of indices is a valid no-isosceles set
    # S is a set of indices
    def is_valid(S):
        s_list = list(S)
        for i in range(len(s_list)):
            a = s_list[i]
            for j in range(i + 1, len(s_list)):
                b = s_list[j]
                # Check if any k in S (k != a, k != b) forms isosceles with a, b
                # If k is in block_mask[a][b], it's invalid.
                # We need to check if S contains any k in block_mask[a][b].
                # Since S is small, we can just check intersection.
                # But bitset intersection is faster.
                # Create a bitmask for S? That's expensive if done every time.
                # Instead, iterate through S?
                # Let's use the property: if S is valid, no two points in S can "see" a third point in S via block_mask.
                # Actually, the condition is: for all distinct a, b, c in S, {a, b, c} is not isosceles.
                # This is equivalent to: for all pairs (a, b) in S, (block_mask[a][b] & S_mask) == 0, where S_mask excludes a and b?
                # No, block_mask[a][b] contains x such that {a, b, x} is isosceles.
                # If x is in S, and x != a, x != b, then invalid.
                # So we need to check if (block_mask[a][b] & S_mask) has any bit set that is not a or b.
                # Since a, b are in S, we must exclude them from the check.
                # Actually, block_mask[a][b] never contains a or b (by construction in precompute, x != i, x != j).
                # So we just need to check if (block_mask[a][b] & S_mask) is non-zero.
                
                # Efficient check:
                # We can precompute a bitmask for S?
                # For small S, iterating is fine.
                for k in s_list:
                    if k == a or k == b:
                        continue
                    if (BLOCK_MASK[a][b] >> k) & 1:
                        return False
        return True

    # Helper: Check if a set is maximal
    # S is a set of indices
    def is_maximal(S):
        s_list = list(S)
        # For every point x not in S, there must be a pair (a, b) in S such that x is in block_mask[a][b]
        for x in range(NUM_POINTS):
            if x in S:
                continue
            blocked = False
            for i in range(len(s_list)):
                a = s_list[i]
                for j in range(i + 1, len(s_list)):
                    b = s_list[j]
                    if (BLOCK_MASK[a][b] >> x) & 1:
                        blocked = True
                        break
                if blocked:
                    break
            if not blocked:
                return False
        return True

    # Heuristic 1: Greedy Construction with Coverage
    # We want to find a minimal maximal set.
    # Strategy: 
    # 1. Generate a maximal set (using greedy insertion).
    # 2. Reduce it (remove points while maintaining maximality).
    
    def run_greedy_reduce(seed_val):
        random.seed(seed_val)
        
        # Step 1: Greedy construction of a maximal independent set (no-isosceles)
        # We maintain the set S.
        # To ensure maximality, we must add points until no more can be added.
        # However, a simple greedy might not be minimal.
        # We use a "conflict" based approach.
        
        S = set()
        # We need to track which points are "blocked" by S.
        # A point x is blocked if there exist a, b in S such that x is in block_mask[a][b].
        # We can maintain a count of how many pairs in S block each point x.
        # counts[x] = number of pairs (a, b) in S such that x is in block_mask[a][b].
        
        counts = [0] * NUM_POINTS
        
        # Order of insertion:
        # To get a smaller set, we might want to insert points that block many others?
        # But we are building an independent set.
        # Let's just try random order first, then optimize.
        
        indices = list(range(NUM_POINTS))
        random.shuffle(indices)
        
        for p_idx in indices:
            # Can we add p_idx?
            # It must not be blocked by any pair in S.
            # i.e., counts[p_idx] must be 0.
            # Wait, counts[p_idx] counts pairs in S that block p_idx.
            # If counts[p_idx] > 0, then there is a pair (a, b) in S such that {a, b, p_idx} is isosceles.
            # So we cannot add p_idx.
            
            if counts[p_idx] == 0:
                S.add(p_idx)
                # Update counts for all x
                # Adding p_idx creates new pairs (p_idx, q) for all q in S.
                # These new pairs might block some x.
                # For each q in S:
                #   For each x in block_mask[p_idx][q]:
                #       counts[x] += 1
                # Note: block_mask is symmetric.
                for q_idx in S:
                    if q_idx == p_idx:
                        continue
                    mask = BLOCK_MASK[p_idx][q_idx]
                    # Iterate through set bits in mask
                    m = mask
                    while m:
                        # Get least significant set bit
                        lsb = m & -m
                        x_idx = lsb.bit_length() - 1
                        counts[x_idx] += 1
                        m ^= lsb
        
        # Now S is a maximal independent set?
        # Yes, because we considered all points in random order and added them if possible.
        # Any point not in S was blocked at the time we considered it.
        # But wait, could a point that was blocked become unblocked later?
        # No, because we only add points to S, which only increases counts.
        # So if counts[x] > 0 at time t, it remains > 0.
        # So S is maximal.
        
        # Step 2: Reduction
        # Try to remove points from S while maintaining maximality.
        # We maintain counts.
        # If we remove p_idx:
        #   We remove pairs (p_idx, q) for all q in S \ {p_idx}.
        #   For each such pair, we decrement counts[x] for x in block_mask[p_idx][q].
        #   After removal, we must check if S \ {p_idx} is still maximal.
        #   This means:
        #   1. p_idx must still be blocked by S \ {p_idx}. (i.e., counts[p_idx] > 0 after update)
        #   2. All x not in S \ {p_idx} must be blocked.
        #      x not in S \ {p_idx} is (x not in S) U {p_idx}.
        #      For x not in S, they were blocked by S. Some of those blocking pairs might have involved p_idx.
        #      If all blocking pairs for x involved p_idx, then counts[x] might drop to 0.
        #      So we need to check if counts[x] > 0 for all x not in S \ {p_idx}.
        
        improved = True
        while improved:
            improved = False
            s_list = list(S)
            random.shuffle(s_list)
            
            for p_idx in s_list:
                if p_idx not in S:
                    continue
                
                # Simulate removal of p_idx
                # Calculate delta_counts: how much counts[x] changes
                # delta_counts[x] = - (number of q in S \ {p_idx} such that x is in block_mask[p_idx][q])
                
                # We can just update counts in place, check, and revert if fail.
                # Or compute new_counts.
                
                # Let's update in place for speed, but be careful.
                
                # First, record the changes we will make.
                # Actually, just do it.
                
                # Save old counts? No, we can revert.
                # Let's collect the list of (x_idx, delta) to revert.
                changes = []
                
                for q_idx in S:
                    if q_idx == p_idx:
                        continue
                    mask = BLOCK_MASK[p_idx][q_idx]
                    m = mask
                    while m:
                        lsb = m & -m
                        x_idx = lsb.bit_length() - 1
                        counts[x_idx] -= 1
                        changes.append(x_idx)
                        m ^= lsb
                
                # Check maximality of S \ {p_idx}
                # 1. Check p_idx is blocked
                if counts[p_idx] == 0:
                    # Revert
                    for x_idx in changes:
                        counts[x_idx] += 1
                    continue
                
                # 2. Check all x not in S \ {p_idx} are blocked
                # x not in S \ {p_idx} means x not in S OR x == p_idx.
                valid = True
                for x_idx in range(NUM_POINTS):
                    if x_idx in S and x_idx != p_idx:
                        continue # x is in the new set, doesn't need to be blocked
                    # x is not in the new set
                    if counts[x_idx] == 0:
                        valid = False
                        break
                
                if not valid:
                    # Revert
                    for x_idx in changes:
                        counts[x_idx] += 1
                    continue
                
                # Removal is valid
                S.remove(p_idx)
                # Counts are already updated (decremented)
                improved = True
                break # Restart loop
        
        return len(S), S

    # Run multiple restarts
    restarts = 0
    while time.time() < deadline:
        size, s = run_greedy_reduce(restarts)
        if size < best_size:
            best_size = size
            best_set = s
        restarts += 1
        
        # Optional: Try a different heuristic?
        # The current heuristic is: Greedy Insertion (random order) + Reduction (random order).
        # We can try to bias the insertion order.
        # E.g., insert points with high "potential" to block others?
        # But for independent set, we want to avoid conflicts.
        
    if best_set is None:
        # Fallback: Simple greedy
        S = set()
        for p in range(NUM_POINTS):
            valid = True
            for q in S:
                for r in S:
                    if r == q: continue
                    if (BLOCK_MASK[q][r] >> p) & 1:
                        valid = False
                        break
                if not valid: break
            if valid:
                S.add(p)
        best_set = S

    return [POINTS[i] for i in best_set]

if __name__ == "__main__":
    s = construct()
    print(len(s))
