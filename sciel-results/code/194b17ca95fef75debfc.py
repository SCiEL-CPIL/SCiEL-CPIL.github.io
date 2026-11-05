import time
import random
import math
import numpy as np

N = 256

# Precompute 3-AP neighbors
f3 = [[] for _ in range(N + 1)]
m3 = [[] for _ in range(N + 1)]
l3 = [[] for _ in range(N + 1)]

for x in range(1, N + 1):
    for d in range(1, (N - x) // 2 + 1):
        f3[x].append((x + d, x + 2 * d))
    for d in range(1, min(x - 1, N - x) + 1):
        m3[x].append((x - d, x + d))
    for d in range(1, (x - 1) // 2 + 1):
        l3[x].append((x - 2 * d, x - d))

# Precompute 4-AP conflict masks as numpy arrays for fast vectorized checks
# f4_masks[x]: list of bitmasks
f4_masks = [[] for _ in range(N + 1)]

def to_mask(vals):
    m = 0
    for v in vals:
        m |= (1 << (v - 1))
    return m

for x in range(1, N + 1):
    for d in range(1, (N - x) // 3 + 1):
        f4_masks[x].append(to_mask([x + d, x + 2 * d, x + 3 * d]))
    for d in range(1, min(x - 1, (N - x) // 2) + 1):
        f4_masks[x].append(to_mask([x - d, x + d, x + 2 * d]))
    for d in range(1, min((x - 1) // 2, N - x) + 1):
        f4_masks[x].append(to_mask([x - 2 * d, x - d, x + d]))
    for d in range(1, (x - 1) // 3 + 1):
        f4_masks[x].append(to_mask([x - 3 * d, x - 2 * d, x - d]))

# Convert to numpy arrays for vectorized operations
# f4_mask_arrays[x] = np.array of masks, or empty array
f4_mask_arrays = [np.array(masks, dtype=object) if masks else np.array([], dtype=object) for masks in f4_masks]
f4_mask_counts = np.array([len(m) for m in f4_masks], dtype=np.int32)

# Precompute 3-AP contribution matrices for fast gain/loss
# For each x, precompute which elements pair with x to form 3-APs
# gain_matrix[x] = list of (a, b) pairs where {a,b} in A and x completes a 3-AP
# We'll use a different approach: precompute for each pair (a,c) the middle b
# Then gain_add(A, x) = number of pairs (a,c) in A such that x is the middle
#   + number of pairs (a,b) in A such that x is the last
#   + number of pairs (b,c) in A such that x is the first

# Precompute: for each x, the list of (a, c) where x is middle of 3-AP (a, x, c)
# and (a, b) where x is last, and (b, c) where x is first
# Actually let's just precompute the total 3-AP count differently.

# For counting 3-APs in set A:
# A 3-AP is (a, a+d, a+2d). Count = sum over all (a,d) of [a in A and a+d in A and a+2d in A]
# We can precompute all possible 3-APs as triples.

all_3APs = []  # list of (a, b, c) where a < b < c and b-a == c-b
for a in range(1, N + 1):
    for d in range(1, (N - a) // 2 + 1):
        all_3APs.append((a, a + d, a + 2 * d))

# For each element x, precompute which 3-APs contain x
# ap_containing[x] = list of indices into all_3APs where x appears
ap_containing = [[] for _ in range(N + 1)]
for i, (a, b, c) in enumerate(all_3APs):
    ap_containing[a].append(i)
    ap_containing[b].append(i)
    ap_containing[c].append(i)

# Precompute as numpy arrays
ap_containing_arr = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in ap_containing]
ap_containing_counts = np.array([len(lst) for lst in ap_containing], dtype=np.int32)

# For each 3-AP (a,b,c), precompute the three elements
ap_a = np.array([t[0] for t in all_3APs], dtype=np.int32)
ap_b = np.array([t[1] for t in all_3APs], dtype=np.int32)
ap_c = np.array([t[2] for t in all_3APs], dtype=np.int32)

def is_in_set(A, x):
    return bool((A >> (x - 1)) & 1)

def creates_4AP(A, x):
    masks = f4_masks[x]
    if not masks:
        return False
    for mask in masks:
        if (A & mask) == mask:
            return True
    return False

def count_3APs(A):
    """Count 3-APs using numpy vectorization."""
    if A == 0:
        return 0
    # Create boolean array of membership
    A_bool = np.zeros(N + 1, dtype=np.bool_)
    temp = A
    while temp:
        lsb = temp & -temp
        b = lsb.bit_length()
        A_bool[b] = True
        temp ^= lsb
    
    # For each 3-AP (a, b, c), check if all three are in A
    # Vectorized: check ap_a, ap_b, ap_c against A_bool
    in_a = A_bool[ap_a]
    in_b = A_bool[ap_b]
    in_c = A_bool[ap_c]
    return int(np.sum(in_a & in_b & in_c))

def gain_add(A, x):
    """Compute gain from adding x to A, using precomputed AP containing x."""
    idxs = ap_containing_arr[x]
    if len(idxs) == 0:
        return 0
    # For each 3-AP containing x, check if the other two elements are in A
    a_idx = ap_a[idxs]
    b_idx = ap_b[idxs]
    c_idx = ap_c[idxs]
    
    # Determine which of a, b, c is x, and which are the other two
    # x is at position 0 (a), 1 (b), or 2 (c) in the AP
    # For each AP, the other two elements are the ones != x
    # We need to check if both other elements are in A
    
    # Create arrays for the two "other" elements
    other1 = np.where(x == a_idx, b_idx, a_idx)  # one of the other two
    other2 = np.where(x == a_idx, c_idx, np.where(x == b_idx, a_idx, b_idx))
    
    # Actually, let's be more careful:
    # If x == a_idx: others are b_idx, c_idx
    # If x == b_idx: others are a_idx, c_idx  
    # If x == c_idx: others are a_idx, b_idx
    
    is_a = (x == a_idx)
    is_b = (x == b_idx)
    is_c = (x == c_idx)
    
    other1 = np.where(is_a, b_idx, np.where(is_b, a_idx, a_idx))
    other2 = np.where(is_a, c_idx, np.where(is_b, c_idx, b_idx))
    
    # Check if both other elements are in set A
    temp = A
    A_bool = np.zeros(N + 1, dtype=np.bool_)
    while temp:
        lsb = temp & -temp
        b_val = lsb.bit_length()
        A_bool[b_val] = True
        temp ^= lsb
    
    in_other1 = A_bool[other1]
    in_other2 = A_bool[other2]
    return int(np.sum(in_other1 & in_other2))

def loss_remove(A, x):
    """Compute loss from removing x from A."""
    return gain_add(A, x)  # Same computation

def greedy_seed(rng):
    A = 0
    order = list(range(1, N + 1))
    rng.shuffle(order)
    for x in order:
        if not creates_4AP(A, x):
            A |= (1 << (x - 1))
    return A

def local_search(A, deadline):
    best_score = count_3APs(A)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try Adding
        best_gain = 0
        best_x = -1
        for x in range(1, N + 1):
            if not is_in_set(A, x) and not creates_4AP(A, x):
                g = gain_add(A, x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
        if best_x != -1:
            A |= (1 << (best_x - 1))
            best_score += best_gain
            improved = True
            continue
        
        # 2. Try Swapping (1-out, 1-in)
        best_net = 0
        best_r = -1
        best_x = -1
        for r in range(1, N + 1):
            if not is_in_set(A, r):
                continue
            loss = loss_remove(A, r)
            A_temp = A & ~(1 << (r - 1))
            for x in range(1, N + 1):
                if x == r or is_in_set(A, x):
                    continue
                if creates_4AP(A_temp, x):
                    continue
                g = gain_add(A_temp, x)
                net = g - loss
                if net > best_net:
                    best_net = net
                    best_r = r
                    best_x = x
        if best_r != -1 and best_x != -1:
            A = A & ~(1 << (best_r - 1))
            A |= (1 << (best_x - 1))
            best_score += best_net
            improved = True
            continue
            
    return best_score, A

def iterated_local_search(A, deadline, rng):
    best_score, best_A = local_search(A, deadline)
    
    while time.time() < deadline - 2.0:
        A_curr = best_A
        # Perturb: remove k random elements
        k = rng.randint(3, 15)
        elems = [x for x in range(1, N + 1) if is_in_set(A_curr, x)]
        if len(elems) < k:
            k = max(1, len(elems) // 2)
        if k == 0:
            break
        to_remove = rng.sample(elems, k)
        for x in to_remove:
            A_curr &= ~(1 << (x - 1))
        
        sub_deadline = min(time.time() + 3.0, deadline - 1.0)
        score_curr, A_curr = local_search(A_curr, sub_deadline)
        
        if score_curr > best_score:
            best_score = score_curr
            best_A = A_curr

    return best_score, best_A

def construct():
    start = time.time()
    deadline = start + 55.0
    
    best_score = -1
    best_A = 0
    
    rng = random.Random(42)
    
    # Multiple greedy seeds with different shuffles
    for trial in range(3):
        A0 = greedy_seed(rng)
        score0, A0 = local_search(A0, start + 8.0 + trial * 2.0)
        if score0 > best_score:
            best_score = score0
            best_A = A0
    
    # ILS
    s, A_curr = iterated_local_search(best_A, deadline, rng)
    
    if s > best_score:
        best_score = s
        best_A = A_curr

    # Final polish
    if time.time() < deadline - 1.0 and best_A != 0:
        final_score, best_A = local_search(best_A, deadline)
        best_score = final_score

    # Convert to list
    A_list = []
    temp = best_A
    while temp:
        lsb = temp & -temp
        x = lsb.bit_length()
        A_list.append(x)
        temp ^= lsb
        
    if not A_list:
        A_list = [1]
        
    return A_list
