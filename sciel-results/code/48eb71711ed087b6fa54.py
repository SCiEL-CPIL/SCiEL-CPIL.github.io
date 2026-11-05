"""Minimum maximal no-isosceles set on [0,N)^2 using vectorized bitsets."""
import numpy as np
import time
import random

N = 16
N_PTS = N * N

# Precompute grid coordinates
coords = np.array([(i // N, i % N) for i in range(N_PTS)], dtype=np.int32)

# Precompute squared distances: dist2[i, j] = squared distance between point i and point j
# dist2 is N_PTS x N_PTS
dx = coords[:, 0:1].astype(np.int64) - coords[1:0:-1, 0:1].astype(np.int64)
dy = coords[:, 1:2].astype(np.int64) - coords[1:0:-1, 1:2].astype(np.int64)
# Actually, let's just do it simply and correctly for clarity, N=16 is small (256x256)
# But let's use the broadcast method properly.
dx = coords[:, 0:1] - coords[1:0:-1, 0:1]  # This is wrong shape if not careful. Let's just use standard broadcasting.
# coords is (256, 2). coords[:, 0:1] is (256, 1). coords[1:0:-1, 0:1] is (255, 1). 
# Let's stick to a simple, robust precomputation.
diff_x = coords[:, 0:1].astype(np.int64) - coords[1:0:-1, 0:1].astype(np.int64) # No, this is messy.
# Let's just do:
dx = coords[:, 0:1] - coords[:, 0:1].T  # (256, 256)
dy = coords[:, 1:2] - coords[:, 1:2].T  # (256, 256)
dist2 = dx**2 + dy**2

# For each pair (i, j), we need to know which points k form an isosceles triangle with i and j.
# Condition: dist2[i, k] == dist2[j, k] OR dist2[i, k] == dist2[i, j] OR dist2[j, k] == dist2[i, j].
# If (i, j) is in the set, no k can be added.
# forbidden[i, j] is a bitset of all k such that {i, j, k} is isosceles.

# To speed up, we can precompute for each point i and distance d, the set of points at distance d.
# But direct bitset computation is fine for N=16.

forbidden = np.zeros((N_PTS, N_PTS, N_PTS // 64 + 1), dtype=np.uint64)

# We only need forbidden[i, j] for i < j, but let's compute all for simplicity or just upper triangle.
# Actually, we can just compute it in a loop. 256*256 = 65536 iterations. Each is vectorized over 256 points.
# This might be slow in Python loop. Let's optimize.

# Alternative: 
# For a fixed pair (i, j), the forbidden set is:
# F = {k | d(i,k) == d(j,k)} U {k | d(i,k) == d(i,j)} U {k | d(j,k) == d(i,j)}
# Note: i and j themselves are in these sets usually, but we don't care about adding i or j if they are already in the set.
# We care about adding k not in {i, j}.

# Let's precompute distance sets.
# dist_to[i] is an array of distances from i to all other points.
# This is just dist2[i, :].

# To make it fast, let's use numpy broadcasting to compute the forbidden masks for all pairs at once?
# Memory: 256 * 256 * 256 bits = 16M bits = 2 MB. That's fine.
# But computing it...
# mask1[i, j, k] = (dist2[i, k] == dist2[j, k])
# mask2[i, j, k] = (dist2[i, k] == dist2[i, j])
# mask3[i, j, k] = (dist2[j, k] == dist2[i, j])
# forbidden_mask[i, j, k] = mask1 | mask2 | mask3

# This is 256^3 boolean ops. 16M ops. NumPy can handle this quickly if done in chunks or fully vectorized.
# Let's try fully vectorized.
# dist2 is (256, 256).
# We need dist2[i, k] for all i, k. That's dist2.
# We need dist2[j, k] for all j, k. That's dist2.T? No.
# Let's reshape.
# d_ik = dist2[:, :, None]  # (256, 256, 1) -> i, k, dummy
# d_jk = dist2[None, :, :]  # (1, 256, 256) -> dummy, j, k
# d_ij = dist2[:, :, None] * dist2[None, :, :] # No, this is not right. d_ij is (256, 256).
# d_ij_exp = dist2[:, :, None]  # (256, 256, 1) -> i, j, dummy

# Let's do:
# d_ik: index [i, k]
# d_jk: index [j, k]
# d_ij: index [i, j]

# We want a tensor M[i, j, k] which is True if (dist2[i,k] == dist2[j,k]) or (dist2[i,k] == dist2[i,j]) or (dist2[j,k] == dist2[i,j])

# Let's compute the three conditions separately.
# cond1: dist2[i,k] == dist2[j,k]
# dist2 is (N, N). 
# d_ik = dist2[:, :, None] # shape (N, N, 1), indices i, k, ? No, dist2[i, k]. So dist2[:, :, None] is i, k, 1? No.
# dist2 is indexed [row, col]. So dist2[i, k].
# To get shape (N, N, N) for i, j, k:
# d_ik: we want dist2[i, k]. 
#   dist2[:, :, None] is (N, N, 1). If we broadcast against (N, N, N) for i, j, k...
#   Let's use np.newaxis.
#   d_ik = dist2[:, None, :]  # (N, 1, N) -> i, 1, k. Broadcasts to (N, N, N).
#   d_jk = dist2[None, :, :]  # (1, N, N) -> 1, j, k. Broadcasts to (N, N, N).
#   d_ij = dist2[:, :, None]  # (N, N, 1) -> i, j, 1. Broadcasts to (N, N, N).

# Memory check: 3 arrays of 256^3 booleans. 3 * 16M * 1 byte = 48 MB. Acceptable.
# Time check: 3 comparisons + 2 ORs. ~5 * 16M ops. Should be < 1s.

# Let's do it.
d_ik = dist2[:, None, :]
d_jk = dist2[None, :, :]
d_ij = dist2[:, :, None]

cond1 = (d_ik == d_jk)
cond2 = (d_ik == d_ij)
cond3 = (d_jk == d_ij)

forbidden_bool = cond1 | cond2 | cond3

# Now pack forbidden_bool into uint64 bitsets.
# forbidden_bool shape: (N_PTS, N_PTS, N_PTS)
# We want forbidden[i, j] to be a bitset of k.
# Reshape last dim to (N_PTS // 64, 64)
N_WORDS = N_PTS // 64 + 1
# Pad if necessary? N_PTS = 256, 256/64 = 4. So exactly 4 words.
# forbidden_bool is (256, 256, 256).
# Reshape to (256, 256, 4, 64).
forbidden_bool_reshaped = forbidden_bool.reshape(N_PTS, N_PTS, N_WORDS, 64)

# Convert bool to uint64.
# We need to pack 64 bools into a uint64.
# bit 0 is LSB.
# forbidden_bool_reshaped[..., 0] is bit 0, ..., forbidden_bool_reshaped[..., 63] is bit 63.
# We can use view or packing.
# Let's just loop over the 64 bits? No, that's slow.
# Use numpy bit operations.
# Actually, since N_PTS=256 is small, we can just keep the boolean array if memory allows?
# 256 * 256 * 256 * 1 byte = 16 MB.
# Checking membership: forbidden_bool[i, j, k] is O(1).
# But bitwise AND is faster with packed ints.
# Let's pack it.

# Create an empty uint64 array
forbidden_bits = np.zeros((N_PTS, N_PTS, N_WORDS), dtype=np.uint64)

# Pack bits
# For each word w, and each bit b in 0..63:
# If forbidden_bool[i, j, w*64+b] is True, set bit b in forbidden_bits[i, j, w].
# We can do this by iterating over the 64 bits, but vectorized over i, j.
for b in range(64):
    # Extract the b-th bit plane
    # Index: [..., w, b]
    mask = forbidden_bool_reshaped[:, :, :, b]  # Shape (N_PTS, N_PTS, N_WORDS)
    # Set bit b in forbidden_bits where mask is True
    # Bit mask for position b
    bit_val = np.uint64(1) << np.uint64(b)
    forbidden_bits[mask] |= bit_val

# Now forbidden_bits[i, j, w] contains the bits for k in [w*64, w*64+63].
# To check if any k is forbidden given a set S (bitset):
# For each word w, if (forbidden_bits[i, j, w] & S[w]) != 0, then some k in S is forbidden?
# Wait. forbidden_bits[i, j, w] has 1s where k is forbidden.
# If (forbidden_bits[i, j, w] & S[w]) != 0, it means there is a k in S that is forbidden.
# We want to know if ALL k not in S are forbidden? No.
# We want to add p. p is valid if p is not in S, and for all pairs (i, j) in S, p is NOT in forbidden[i, j].
# So we need to check if p is in the union of forbidden[i, j] for all (i, j) in S.
# Union of bitsets can be computed by OR-ing them.
# But we need to do this for each candidate p?
# No, we can precompute the "union of forbidden sets" for the current set S.
# Let U be the bitset union of forbidden[i, j] for all i, j in S.
# Then a point p can be added iff p is not in S AND p is not in U.
# So valid candidates = ~S & ~U.
# If valid candidates is empty, S is maximal.

# This is a huge speedup.
# To compute U:
# U = OR of forbidden_bits[i, j, :] for all i, j in S.
# We can do this by iterating over pairs in S?
# If S has size K, there are K*(K-1)/2 pairs.
# For each pair, we OR a vector of 4 uint64s.
# This is fast if K is small.
# In our search, K will be small (target < 14).
# So this is extremely fast.

def _compute_union(S_indices):
    """Compute the union of forbidden bitsets for all pairs in S_indices."""
    U = np.zeros(N_WORDS, dtype=np.uint64)
    n = len(S_indices)
    for a in range(n):
        i = S_indices[a]
        for b in range(a + 1, n):
            j = S_indices[b]
            # OR forbidden_bits[i, j, :] into U
            U |= forbidden_bits[i, j, :]
    return U

def _is_valid_set(S_indices):
    """Check if S_indices is a no-isosceles set.
    This is true iff for all pairs (i, j) in S, there is no k in S such that (i, j, k) is isosceles.
    Equivalent to: S & U is empty? No.
    If S is a no-isosceles set, then for any pair (i, j) in S, no other point in S is in forbidden[i, j].
    So S & U should be empty? 
    U contains all points k that are forbidden by some pair in S.
    If S is valid, then no point in S is forbidden by any pair in S.
    So S & U should be empty.
    """
    U = _compute_union(S_indices)
    S_bits = np.zeros(N_WORDS, dtype=np.uint64)
    for i in S_indices:
        word = i // 64
        bit = i % 64
        S_bits[word] |= (np.uint64(1) << np.uint64(bit))
    
    # Check if any bit is set in both S_bits and U
    for w in range(N_WORDS):
        if (S_bits[w] & U[w]) != 0:
            return False
    return True

def _get_valid_candidates(S_indices, S_bits):
    """Get bitset of points that can be added to S_indices."""
    U = _compute_union(S_indices)
    # Candidates are points not in S and not in U.
    # valid = ~S_bits & ~U
    valid = np.zeros(N_WORDS, dtype=np.uint64)
    for w in range(N_WORDS):
        valid[w] = ~(S_bits[w] | U[w])
    # Mask out bits beyond N_PTS
    # Last word might have extra bits.
    # N_PTS = 256, N_WORDS = 4. 4*64=256. No extra bits.
    return valid

def _bit_count(arr):
    """Count bits in a numpy array of uint64."""
    return sum(bin(x).count('1') for x in arr)

def _iter_bits(arr):
    """Iterate over indices of set bits in a numpy array of uint64."""
    for w in range(len(arr)):
        x = arr[w]
        while x:
            # Get lowest set bit
            lsb = x & -x
            idx = lsb.bit_length() - 1
            yield w * 64 + idx
            x ^= lsb

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    best_set = None
    best_size = float('inf')
    
    # Initial random sets
    # We want to start with small sets.
    # Greedy construction is hard to do fast with bitsets if we don't have the union precomputed dynamically.
    # But we can do a simple greedy:
    # Start with empty set.
    # At each step, pick a random valid candidate.
    
    def _greedy_construct():
        S = []
        S_bits = np.zeros(N_WORDS, dtype=np.uint64)
        
        # Candidates initially all points
        # But we need to pick one.
        # Let's just pick random points and check if they can be added.
        # This is slow if we check one by one.
        # Better: maintain the valid candidates bitset.
        
        valid = _get_valid_candidates([], S_bits)
        
        while True:
            # If no valid candidates, we are maximal.
            if _bit_count(valid) == 0:
                break
            
            # Pick a random valid candidate
            candidates = list(_iter_bits(valid))
            p = random.choice(candidates)
            
            # Add p to S
            S.append(p)
            word = p // 64
            bit = p % 64
            S_bits[word] |= (np.uint64(1) << np.uint64(bit))
            
            # Update valid candidates
            valid = _get_valid_candidates(S, S_bits)
            
        return S
    
    # Run multiple restarts
    n_restarts = 0
    while time.time() < deadline - 50.0:
        # 1. Construct a maximal set using greedy
        S = _greedy_construct()
        
        # 2. Inclusion-minimal reduction
        # Try to remove each point. If the set remains maximal, remove it.
        improved = True
        while improved:
            if time.time() > deadline - 10.0:
                break
            improved = False
            for i in range(len(S)):
                if time.time() > deadline - 10.0:
                    break
                test_S = S[:i] + S[i+1:]
                # Check if test_S is maximal
                # Compute valid candidates for test_S
                test_bits = np.zeros(N_WORDS, dtype=np.uint64)
                for p in test_S:
                    w = p // 64
                    b = p % 64
                    test_bits[w] |= (np.uint64(1) << np.uint64(b))
                
                test_valid = _get_valid_candidates(test_S, test_bits)
                if _bit_count(test_valid) == 0:
                    # test_S is maximal, so we can remove S[i]
                    S = test_S
                    improved = True
                    break
        
        # 3. Iterated Local Search (Perturb + Repair)
        # We want to find smaller sets.
        # Perturb: Remove a random point, then repair to maximality.
        # Or: Swap a point.
        
        for _ in range(50):  # Max 50 ILS iterations per restart
            if time.time() > deadline - 10.0:
                break
            
            # Perturb: Remove a random point from S
            if len(S) > 1:
                idx_to_remove = random.randint(0, len(S) - 1)
                S_perturbed = S[:idx_to_remove] + S[idx_to_remove+1:]
            else:
                S_perturbed = list(S)
            
            # Repair: Add random valid candidates until maximal
            S_repaired = list(S_perturbed)
            S_bits = np.zeros(N_WORDS, dtype=np.uint64)
            for p in S_repaired:
                w = p // 64
                b = p % 64
                S_bits[w] |= (np.uint64(1) << np.uint64(b))
            
            valid = _get_valid_candidates(S_repaired, S_bits)
            
            # Add random points until maximal
            while _bit_count(valid) > 0:
                candidates = list(_iter_bits(valid))
                p = random.choice(candidates)
                S_repaired.append(p)
                w = p // 64
                b = p % 64
                S_bits[w] |= (np.uint64(1) << np.uint64(b))
                valid = _get_valid_candidates(S_repaired, S_bits)
            
            # Inclusion-minimal reduction on repaired set
            improved = True
            while improved:
                if time.time() > deadline - 10.0:
                    break
                improved = False
                for i in range(len(S_repaired)):
                    if time.time() > deadline - 10.0:
                        break
                    test_S = S_repaired[:i] + S_repaired[i+1:]
                    test_bits = np.zeros(N_WORDS, dtype=np.uint64)
                    for p in test_S:
                        w = p // 64
                        b = p % 64
                        test_bits[w] |= (np.uint64(1) << np.uint64(b))
                    
                    test_valid = _get_valid_candidates(test_S, test_bits)
                    if _bit_count(test_valid) == 0:
                        S_repaired = test_S
                        improved = True
                        break
            
            # Update best
            if len(S_repaired) < best_size:
                best_size = len(S_repaired)
                best_set = S_repaired
                S = S_repaired  # Continue from the new best
        
        n_restarts += 1
    
    # Final cleanup: try to reduce best_set one more time
    if best_set is not None:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_set)):
                if time.time() > deadline:
                    break
                test_S = best_set[:i] + best_set[i+1:]
                test_bits = np.zeros(N_WORDS, dtype=np.uint64)
                for p in test_S:
                    w = p // 64
                    b = p % 64
                    test_bits[w] |= (np.uint64(1) << np.uint64(b))
                
                test_valid = _get_valid_candidates(test_S, test_bits)
                if _bit_count(test_valid) == 0:
                    best_set = test_S
                    improved = True
                    break
    
    # Convert indices to (x, y) tuples
    if best_set is None:
        # Fallback
        best_set = _greedy_construct()
        
    result = [tuple(coords[i]) for i in best_set]
    return result

# Precomputation happens at import time
# To ensure it's done before construct is called, we can do it here.
# But the code above defines forbidden_bits inside the module scope?
# No, I defined it in the module scope in the thought process, but in the code block I need to make sure it's initialized.

# Let's move the precomputation to module level.

# --- Precomputation ---
N = 16
N_PTS = N * N
coords = np.array([(i // N, i % N) for i in range(N_PTS)], dtype=np.int32)

dx = coords[:, 0:1].astype(np.int64) - coords[:, 0:1].T.astype(np.int64)
dy = coords[:, 1:2].astype(np.int64) - coords[:, 1:2].T.astype(np.int64)
dist2 = dx**2 + dy**2

N_WORDS = N_PTS // 64 + 1
forbidden_bits = np.zeros((N_PTS, N_PTS, N_WORDS), dtype=np.uint64)

# Compute forbidden masks
d_ik = dist2[:, None, :]
d_jk = dist2[None, :, :]
d_ij = dist2[:, :, None]

cond1 = (d_ik == d_jk)
cond2 = (d_ik == d_ij)
cond3 = (d_jk == d_ij)

forbidden_bool = cond1 | cond2 | cond3
forbidden_bool_reshaped = forbidden_bool.reshape(N_PTS, N_PTS, N_WORDS, 64)

for b in range(64):
    mask = forbidden_bool_reshaped[:, :, :, b]
    bit_val = np.uint64(1) << np.uint64(b)
    forbidden_bits[mask] |= bit_val

# Free memory
del d_ik, d_jk, d_ij, cond1, cond2, cond3, forbidden_bool, forbidden_bool_reshaped

# --- End Precomputation ---

# Re-define construct to use the global forbidden_bits
# (The previous definition of construct in the code block was inside the thought process, 
#  I need to output the complete program.)

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    best_set = None
    best_size = float('inf')
    
    def _greedy_construct():
        S = []
        S_bits = np.zeros(N_WORDS, dtype=np.uint64)
        
        valid = _get_valid_candidates([], S_bits)
        
        while True:
            if _bit_count(valid) == 0:
                break
            
            candidates = list(_iter_bits(valid))
            p = random.choice(candidates)
            
            S.append(p)
            word = p // 64
            bit = p % 64
            S_bits[word] |= (np.uint64(1) << np.uint64(bit))
            
            valid = _get_valid_candidates(S, S_bits)
            
        return S
    
    n_restarts = 0
    while time.time() < deadline - 50.0:
        S = _greedy_construct()
        
        # Inclusion-minimal reduction
        improved = True
        while improved:
            if time.time() > deadline - 10.0:
                break
            improved = False
            for i in range(len(S)):
                if time.time() > deadline - 10.0:
                    break
                test_S = S[:i] + S[i+1:]
                test_bits = np.zeros(N_WORDS, dtype=np.uint64)
                for p in test_S:
                    w = p // 64
                    b = p % 64
                    test_bits[w] |= (np.uint64(1) << np.uint64(b))
                
                test_valid = _get_valid_candidates(test_S, test_bits)
                if _bit_count(test_valid) == 0:
                    S = test_S
                    improved = True
                    break
        
        # ILS
        for _ in range(50):
            if time.time() > deadline - 10.0:
                break
            
            if len(S) > 1:
                idx_to_remove = random.randint(0, len(S) - 1)
                S_perturbed = S[:idx_to_remove] + S[idx_to_remove+1:]
            else:
                S_perturbed = list(S)
            
            S_repaired = list(S_perturbed)
            S_bits = np.zeros(N_WORDS, dtype=np.uint64)
            for p in S_repaired:
                w = p // 64
                b = p % 64
                S_bits[w] |= (np.uint64(1) << np.uint64(b))
            
            valid = _get_valid_candidates(S_repaired, S_bits)
            
            while _bit_count(valid) > 0:
                candidates = list(_iter_bits(valid))
                p = random.choice(candidates)
                S_repaired.append(p)
                w = p // 64
                b = p % 64
                S_bits[w] |= (np.uint64(1) << np.uint64(b))
                valid = _get_valid_candidates(S_repaired, S_bits)
            
            improved = True
            while improved:
                if time.time() > deadline - 10.0:
                    break
                improved = False
                for i in range(len(S_repaired)):
                    if time.time() > deadline - 10.0:
                        break
                    test_S = S_repaired[:i] + S_repaired[i+1:]
                    test_bits = np.zeros(N_WORDS, dtype=np.uint64)
                    for p in test_S:
                        w = p // 64
                        b = p % 64
                        test_bits[w] |= (np.uint64(1) << np.uint64(b))
                    
                    test_valid = _get_valid_candidates(test_S, test_bits)
                    if _bit_count(test_valid) == 0:
                        S_repaired = test_S
                        improved = True
                        break
            
            if len(S_repaired) < best_size:
                best_size = len(S_repaired)
                best_set = S_repaired
                S = S_repaired
        
        n_restarts += 1
    
    if best_set is not None:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_set)):
                if time.time() > deadline:
                    break
                test_S = best_set[:i] + best_set[i+1:]
                test_bits = np.zeros(N_WORDS, dtype=np.uint64)
                for p in test_S:
                    w = p // 64
                    b = p % 64
                    test_bits[w] |= (np.uint64(1) << np.uint64(b))
                
                test_valid = _get_valid_candidates(test_S, test_bits)
                if _bit_count(test_valid) == 0:
                    best_set = test_S
                    improved = True
                    break
    
    if best_set is None:
        best_set = _greedy_construct()
        
    result = [tuple(coords[i]) for i in best_set]
    return result

