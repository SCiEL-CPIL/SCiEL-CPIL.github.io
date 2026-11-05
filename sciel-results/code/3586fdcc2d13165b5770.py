import time
import random

N = 256

# Precompute 3-AP pair masks for each element
# For element x, store masks of the two other elements needed to form a 3-AP
# pair_masks[x] = list of (mask1, mask2) where both must be in A to form a 3-AP with x
# Actually, let's store it differently: for each x, list of 2-element masks
# where if both bits are set in A, then x completes a 3-AP

# For x as first term: pairs (x+d, x+2d) -> mask = (1<<(x+d-1)) | (1<<(x+2d-1))
# For x as middle: pairs (x-d, x+d)
# For x as last: pairs (x-2d, x-d)

pair_masks = [[] for _ in range(N + 1)]

for x in range(1, N + 1):
    # x first: x, x+d, x+2d
    for d in range(1, (N - x) // 2 + 1):
        mask = (1 << (x + d - 1)) | (1 << (x + 2 * d - 1))
        pair_masks[x].append(mask)
    # x middle: x-d, x, x+d
    for d in range(1, min(x - 1, N - x) + 1):
        mask = (1 << (x - d - 1)) | (1 << (x + d - 1))
        pair_masks[x].append(mask)
    # x last: x-2d, x-d, x
    for d in range(1, (x - 1) // 2 + 1):
        mask = (1 << (x - 2 * d - 1)) | (1 << (x - d - 1))
        pair_masks[x].append(mask)

# Precompute 4-AP conflict masks: for each x, list of masks of the other 3 elements
f4_masks = [[] for _ in range(N + 1)]

def to_mask(vals):
    m = 0
    for v in vals:
        m |= (1 << (v - 1))
    return m

for x in range(1, N + 1):
    # pos 1: x, x+d, x+2d, x+3d
    for d in range(1, (N - x) // 3 + 1):
        f4_masks[x].append(to_mask([x + d, x + 2 * d, x + 3 * d]))
    # pos 2: x-d, x, x+d, x+2d
    for d in range(1, min(x - 1, (N - x) // 2) + 1):
        f4_masks[x].append(to_mask([x - d, x + d, x + 2 * d]))
    # pos 3: x-2d, x-d, x, x+d
    for d in range(1, min((x - 1) // 2, N - x) + 1):
        f4_masks[x].append(to_mask([x - 2 * d, x - d, x + d]))
    # pos 4: x-3d, x-2d, x-d, x
    for d in range(1, (x - 1) // 3 + 1):
        f4_masks[x].append(to_mask([x - 3 * d, x - 2 * d, x - d]))

def is_in_set(A, x):
    return bool((A >> (x - 1)) & 1)

def creates_4AP(A, x):
    for mask in f4_masks[x]:
        if (A & mask) == mask:
            return True
    return False

def gain_add(A, x):
    """Number of 3-APs formed by adding x to A."""
    g = 0
    for mask in pair_masks[x]:
        if (A & mask) == mask:
            g += 1
    return g

def loss_remove(A, x):
    """Number of 3-APs lost by removing x from A."""
    l = 0
    for mask in pair_masks[x]:
        if (A & mask) == mask:
            l += 1
    return l

def count_3AP_total(A):
    """Count total 3-APs in set A."""
    count = 0
    temp = A
    while temp:
        lsb = temp & -temp
        b = lsb.bit_length()
        temp ^= lsb
        for mask in pair_masks[b]:
            if (A & mask) == mask:
                count += 1
    return count

def greedy_seed(rng):
    A = 0
    order = list(range(1, N + 1))
    rng.shuffle(order)
    for x in order:
        if not creates_4AP(A, x):
            A |= (1 << (x - 1))
    return A

def local_search(A, deadline):
    best_score = count_3AP_total(A)
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try single additions
        best_gain = 0
        best_x = -1
        for x in range(1, N + 1):
            if is_in_set(A, x):
                continue
            if creates_4AP(A, x):
                continue
            g = gain_add(A, x)
            if g > best_gain:
                best_gain = g
                best_x = x
        
        if best_x != -1:
            A |= (1 << (best_x - 1))
            best_score += best_gain
            improved = True
            continue
        
        # 2. Try single swaps
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
        
        if not improved:
            break
             
    return best_score, A

def construct():
    start = time.time()
    deadline = start + 55.0
    rng = random.Random(12345)

    best_A = 0
    best_score = -1

    while time.time() < deadline - 2.0:
        A = greedy_seed(rng)
        score, A = local_search(A, deadline)
        if score > best_score:
            best_score = score
            best_A = A

    if time.time() < deadline - 0.5 and best_A != 0:
        score, best_A = local_search(best_A, deadline)
        best_score = score

    if best_A == 0:
        best_A = 1

    A_list = [x for x in range(1, N + 1) if is_in_set(best_A, x)]
    
    if not A_list:
        A_list = [1]
        
    return A_list

if __name__ == "__main__":
    res = construct()
    print(len(res))
