import time
import random

N = 256

# Precompute pairs for 3-APs
# f3[x]: list of (y, z) such that x, y, z is a 3-AP (x first)
# m3[x]: list of (y, z) such that y, x, z is a 3-AP (x middle)
# l3[x]: list of (y, z) such that y, z, x is a 3-AP (x last)
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

# Precompute masks for 4-AP conflicts
# f4_masks[x]: list of bitmasks. Each mask has bits set for the other 3 elements in a 4-AP containing x.
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
    """Calculate number of 3-APs formed by adding x to A."""
    g = 0
    for a, b in f3[x]:
        if is_in_set(A, a) and is_in_set(A, b):
            g += 1
    for a, b in m3[x]:
        if is_in_set(A, a) and is_in_set(A, b):
            g += 1
    for a, b in l3[x]:
        if is_in_set(A, a) and is_in_set(A, b):
            g += 1
    return g

def gain_remove(A, x):
    """Calculate number of 3-APs destroyed by removing x from A."""
    # Note: This is the same as gain_add(x) if x was in A?
    # No, gain_add checks neighbors in A.
    # If we remove x, we lose all 3-APs containing x.
    # The count is the number of pairs (a,b) in the lists such that a,b are in A.
    # This is identical to gain_add(A without x, x).
    # But we can just compute it directly on current A (which contains x).
    # The pairs (a,b) do not include x. So checking is_in_set(A, a) is same as is_in_set(A\{x}, a).
    g = 0
    for a, b in f3[x]:
        if is_in_set(A, a) and is_in_set(A, b):
            g += 1
    for a, b in m3[x]:
        if is_in_set(A, a) and is_in_set(A, b):
            g += 1
    for a, b in l3[x]:
        if is_in_set(A, a) and is_in_set(A, b):
            g += 1
    return g

def count_3AP_total(A):
    """Count total 3-APs in set A (bitmask)."""
    count = 0
    # Iterate through all x in A
    # To iterate bits: while A: lsb = A & -A; x = lsb.bit_length(); ...
    # But we need to count each 3-AP once.
    # A 3-AP is (a, b, c) with a < b < c.
    # We can count by iterating middle term b.
    # For each b in A, count pairs (a, c) in A such that a+b=2c? No, a+c=2b.
    # Using m3[b] which lists (b-d, b+d).
    for b in range(1, N + 1):
        if is_in_set(A, b):
            for a, c in m3[b]:
                if is_in_set(A, a) and is_in_set(A, c):
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
            continue # Restart loop to re-evaluate

        # 2. Try single swaps (remove r, add x)
        best_net = 0
        best_r = -1
        best_x = -1
        # Optimization: Only consider swaps if we haven't improved in a while?
        # For now, try them.
        for r in range(1, N + 1):
            if not is_in_set(A, r):
                continue
            A_temp = A & ~(1 << (r - 1))
            loss = gain_remove(A, r) # Pairs in A\{r} involving r's neighbors? 
                                     # Actually gain_remove(A, r) counts 3-APs containing r.
                                     # When we remove r, we lose these.
                                     # The new set is A\{r}.
                                     # The gain of adding x to A\{r} is gain_add(A\{r}, x).
                                     # Net = gain_add(A\{r}, x) - loss.
            
            # We need to be careful. gain_remove(A, r) calculates 3-APs in A that use r.
            # These are lost.
            # gain_add(A\{r}, x) calculates 3-APs in A\{r} that use x.
            # These are gained.
            
            for x in range(1, N + 1):
                if is_in_set(A, x) and x != r:
                    continue
                if x == r:
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

        # 3. Try 2-swaps (remove r1, r2; add x1, x2)
        # This is O(N^4) which is 256^4 ~ 4e9. Too slow.
        # We need to restrict candidates.
        # Only try 2-swaps if we are stuck.
        # And only try promising swaps.
        
        if not improved:
             # Try a few random 2-swaps? Or limited 2-swaps.
             # Let's try a limited set of 2-swaps:
             # Pick r1, r2 from present. Pick x1, x2 from absent.
             # To keep it fast, we can limit the number of candidates.
             pass # Skip for now to ensure speed, or implement a very restricted version.
        
        if not improved:
             # Try removing elements that contribute little?
             # Or just break if no improvement.
             # Let's try "kick" moves: random swap to escape local optimum.
             # But we must maintain validity.
             # Let's just break for now. The 1-swap is quite powerful.
             break
             
    return best_score

def construct():
    start = time.time()
    deadline = start + 55.0
    rng = random.Random(12345)

    best_A = 0
    best_score = -1

    while time.time() < deadline - 2.0:
        A = greedy_seed(rng)
        score = local_search(A, deadline - 1.0)
        if score > best_score:
            best_score = score
            best_A = A

    # Final polish
    if time.time() < deadline - 0.5:
        score = local_search(best_A, deadline - 0.1)
        if score > best_score:
            best_score = score

    # Convert bitmask to list
    A_list = []
    for i in range(1, N + 1):
        if is_in_set(best_A, i):
            A_list.append(i)
            
    if not A_list:
        A_list = [1]
        
    return A_list
