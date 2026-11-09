import time
import random
import math
import numpy as np

N = 64
NUM_POINTS = N * N
MAX_D2 = (N-1)**2 + (N-1)**2  # 7840 for N=64
# We need to represent a set of distances from 0 to 7840.
# 7841 bits. 7841 / 64 = 122.5 -> 123 uint64s.
# Using uint64 for efficient bitwise ops.
NUM_WORDS = (MAX_D2 + 1 + 63) // 64
BIT_COUNT = NUM_WORDS * 64

# Precompute all coordinates
COORDS = [(i // N, i % N) for i in range(NUM_POINTS)]

# Precompute squared distances between all pairs
# dist2_mat[i, j] = squared distance from i to j
# Memory: 4096 * 4096 * 8 bytes (int64) = ~128 MB.
xs = np.array([i // N for i in range(NUM_POINTS)])
ys = np.array([i % N for i in range(NUM_POINTS)])

dx_all = xs[:, None] - xs[None, :]
dy_all = ys[:, None] - ys[None, :]
dist2_mat = (dx_all * dx_all + dy_all * dy_all).astype(np.int32)

# Precompute bitmasks for each point's distances to every other point?
# That would be 4096 * 4096 * 123 * 8 bytes ~ 16 GB. Too much.
# Instead, we compute the bitmask on the fly for a candidate point p against the current set S.
# But we can precompute the "distance bits" for each point p relative to a fixed reference? No, relative to all points.

# Alternative:
# For a candidate point p, we need to check if it can be added to set S.
# Condition 1: For all a in S, d2(p, a) is not in used_d2[a].
# Condition 2: All d2(p, a) for a in S are distinct.

# Let's use a different bitset strategy.
# We maintain `used_d2_bits` as a list of arrays of shape (NUM_WORDS,) of uint64, one for each point in S?
# No, `used_d2` is per point in the grid, but only non-zero for points in S.
# Actually, `used_d2[a]` is the set of distances from `a` to other points in S.
# So `used_d2_bits[a]` is a bitmask of length NUM_WORDS.

# To check if p can be added:
# 1. Check distinctness of distances from p to S.
#    We can compute the set of distances from p to S.
#    If any distance appears more than once, it's invalid.
#    Also, if any distance d is in used_d2[a] for the corresponding a, it's invalid.
#    Wait, the condition "d2(p, a) not in used_d2[a]" means that the distance from p to a is not already a distance from a to some other point in S.
#    This is equivalent to: The set of distances from p to S must be disjoint from the union of used_d2[a] for a in S?
#    No. For each a in S, d2(p, a) must NOT be in used_d2[a].
#    This is a pairwise check.

# Let's stick to the bitset optimization for the *distinctness* check and the *pairwise* check.
# Actually, the pairwise check is:
# For each a in S:
#   d = d2(p, a)
#   if d is in used_d2[a]: return False
#   if d is in seen_distances: return False
#   seen_distances.add(d)

# Can we vectorize this?
# Let S be an array of indices.
# d2s = dist2_mat[p, S]  # array of distances from p to all points in S
# We need to check if any d2s[i] is in used_d2_bits[S[i]].
# This is hard to vectorize directly because used_d2_bits is a 2D array (NUM_POINTS, NUM_WORDS).
# used_d2_bits[S[i], :] is the bitmask for S[i].
# We need to check if bit d2s[i] is set in used_d2_bits[S[i], :].

# Let's try a different representation.
# Instead of checking per point, what if we maintain a global "forbidden" structure?
# No, the constraint is local to the triangle.

# Let's optimize the Python loop by using NumPy for the distance retrieval and then a tight loop or vectorized bit check.
# If S is large (e.g., 1000 points), the loop is 1000 iterations.
# 1000 iterations * 1000 candidates = 1,000,000 checks per restart.
# If each check is fast, it's okay.

# Let's precompute `dist2_mat` as a list of arrays? No, 2D array is fine.
# Let's precompute the bit position for each distance.
# bit_idx = d2 // 64
# bit_mask = 1 << (d2 % 64)

# We will maintain `used_d2_bits` as a numpy array of shape (NUM_POINTS, NUM_WORDS) of dtype uint64.
# Initially all zeros.
# When adding point p:
#   For each a in S:
#     d = dist2_mat[p, a]
#     word_idx = d // 64
#     bit_mask = 1 << (d % 64)
#     used_d2_bits[a, word_idx] |= bit_mask
#     used_d2_bits[p, word_idx] |= bit_mask
#   Note: used_d2_bits[p] should also be updated with distances to all a in S.
#   And used_d2_bits[a] for a in S should be updated with distance to p.

# When checking if p can be added:
#   d2s = dist2_mat[p, S]  # array of length |S|
#   # Check 1: Distinctness
#   # If len(set(d2s)) < len(d2s): return False
#   # This can be done by sorting or using a set. For small |S|, set is fast.
#   # For large |S|, sorting is O(N log N).
#   # Let's use a set for simplicity first, then optimize if needed.
#   
#   # Check 2: Pairwise forbidden
#   # For each i, a = S[i], d = d2s[i]
#   # Check if bit d is set in used_d2_bits[a]
#   # This is: (used_d2_bits[a, d//64] & (1 << (d%64))) != 0
#   # We can vectorize this:
#   # a_indices = S
#   # d_indices = d2s
#   # word_indices = d_indices // 64
#   # bit_masks = 1 << (d_indices % 64)
#   # current_bits = used_d2_bits[a_indices, word_indices]
#   # conflicts = (current_bits & bit_masks) != 0
#   # if np.any(conflicts): return False

# This vectorization is promising!
# Let's implement this.

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    # used_d2_bits: (NUM_POINTS, NUM_WORDS) uint64 array
    used_d2_bits = np.zeros((NUM_POINTS, NUM_WORDS), dtype=np.uint64)
    in_set = np.zeros(NUM_POINTS, dtype=bool)
    S = []  # List of indices in S
    
    def can_add(p):
        if in_set[p]:
            return False
        if not S:
            return True
        
        S_arr = np.array(S, dtype=np.int32)
        d2s = dist2_mat[p, S_arr]
        
        # Check distinctness
        # If |S| is small, set is faster. If large, sort is faster.
        # Let's use set for now, as |S| is likely < 1000.
        if len(set(d2s.tolist())) < len(d2s):
            return False
            
        # Check pairwise forbidden
        word_indices = d2s // 64
        bit_masks = 1 << (d2s % 64)
        # Fetch the relevant bits from used_d2_bits for each point in S
        # used_d2_bits[S_arr, word_indices] gives an array of the specific 64-bit words
        current_bits = used_d2_bits[S_arr, word_indices]
        conflicts = (current_bits & bit_masks) != 0
        if np.any(conflicts):
            return False
            
        return True

    def add_point(p):
        in_set[p] = True
        S.append(p)
        if not S[:-1]:
            return
            
        S_prev = np.array(S[:-1], dtype=np.int32)
        d2s = dist2_mat[p, S_prev]
        
        word_indices = d2s // 64
        bit_masks = 1 << (d2s % 64)
        
        # Update used_d2_bits for points in S_prev
        used_d2_bits[S_prev, word_indices] |= bit_masks
        # Update used_d2_bits for p
        used_d2_bits[p, word_indices] |= bit_masks

    def remove_point(p):
        idx = S.index(p)
        S.pop(idx)
        in_set[p] = False
        
        if not S:
            used_d2_bits[p] = 0
            return
            
        S_remaining = np.array(S, dtype=np.int32)
        d2s = dist2_mat[p, S_remaining]
        
        word_indices = d2s // 64
        bit_masks = 1 << (d2s % 64)
        
        # Remove bits from used_d2_bits for points in S_remaining
        used_d2_bits[S_remaining, word_indices] &= ~bit_masks
        # Clear used_d2_bits for p
        used_d2_bits[p] = 0

    # ---- Phase 1: Greedy with multiple random orderings ----
    best_S = []
    best_size = 0
    
    def greedy_one():
        S_local = []
        in_set_local = np.zeros(NUM_POINTS, dtype=bool)
        used_d2_bits_local = np.zeros((NUM_POINTS, NUM_WORDS), dtype=np.uint64)
        
        def can_add_local(p):
            if in_set_local[p]:
                return False
            if not S_local:
                return True
            
            S_arr = np.array(S_local, dtype=np.int32)
            d2s = dist2_mat[p, S_arr]
            
            if len(set(d2s.tolist())) < len(d2s):
                return False
                
            word_indices = d2s // 64
            bit_masks = 1 << (d2s % 64)
            current_bits = used_d2_bits_local[S_arr, word_indices]
            conflicts = (current_bits & bit_masks) != 0
            if np.any(conflicts):
                return False
                
            return True

        def add_local(p):
            in_set_local[p] = True
            S_local.append(p)
            if not S_local[:-1]:
                return
                
            S_prev = np.array(S_local[:-1], dtype=np.int32)
            d2s = dist2_mat[p, S_prev]
            
            word_indices = d2s // 64
            bit_masks = 1 << (d2s % 64)
            
            used_d2_bits_local[S_prev, word_indices] |= bit_masks
            used_d2_bits_local[p, word_indices] |= bit_masks

        order = list(range(NUM_POINTS))
        random.shuffle(order)
        for p in order:
            if can_add_local(p):
                add_local(p)
        
        return S_local

    # Run several greedy passes
    for _ in range(50):
        if time.time() > deadline:
            break
        S_g = greedy_one()
        if len(S_g) > best_size:
            best_size = len(S_g)
            best_S = S_g[:]

    # Initialize current state from best_S
    for p in best_S:
        add_point(p)

    # ---- Phase 2: Local search ----
    def try_add_random():
        for _ in range(100):
            p = random.randint(0, NUM_POINTS - 1)
            if not in_set[p] and can_add(p):
                add_point(p)
                return True
        return False

    def try_swap():
        if not S:
            return False
        q = random.choice(S)
        remove_point(q)
        added = False
        for _ in range(200):
            p = random.randint(0, NUM_POINTS - 1)
            if not in_set[p] and can_add(p):
                add_point(p)
                added = True
                break
        if not added:
            add_point(q)
        return added

    stagnation = 0
    max_stagnation = 50

    while time.time() < deadline:
        moved = False
        r = random.random()
        if r < 0.6:
            moved = try_add_random()
        else:
            moved = try_swap()
        
        if moved:
            stagnation = 0
            if len(S) > best_size:
                best_size = len(S)
                best_S = S[:]
        else:
            stagnation += 1
            if stagnation > max_stagnation:
                # Perturbation
                num_remove = max(1, len(S) // 10)
                for _ in range(num_remove):
                    if S:
                        q = random.choice(S)
                        remove_point(q)
                stagnation = 0

    return [COORDS[i] for i in best_S]
