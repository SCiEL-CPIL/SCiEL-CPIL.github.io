import time
import random
import math

N = 256

# Precompute 3-AP neighbors
# f3[x]: (y, z) such that x, y, z is a 3-AP (x first)
# m3[x]: (y, z) such that y, x, z is a 3-AP (x middle)
# l3[x]: (y, z) such that y, z, x is a 3-AP (x last)
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

# Precompute 4-AP conflict masks
# f4_masks[x]: list of bitmasks. If (A & mask) == mask, adding x creates a 4-AP.
f4_masks = [[] for _ in range(N + 1)]

def to_mask(vals):
    m = 0
    for v in vals:
        m |= (1 << (v - 1))
    return m

for x in range(1, N + 1):
    # x is 1st term
    for d in range(1, (N - x) // 3 + 1):
        f4_masks[x].append(to_mask([x + d, x + 2 * d, x + 3 * d]))
    # x is 2nd term
    for d in range(1, min(x - 1, (N - x) // 2) + 1):
        f4_masks[x].append(to_mask([x - d, x + d, x + 2 * d]))
    # x is 3rd term
    for d in range(1, min((x - 1) // 2, N - x) + 1):
        f4_masks[x].append(to_mask([x - 2 * d, x - d, x + d]))
    # x is 4th term
    for d in range(1, (x - 1) // 3 + 1):
        f4_masks[x].append(to_mask([x - 3 * d, x - 2 * d, x - d]))

def is_in_set(A, x):
    return bool((A >> (x - 1)) & 1)

def creates_4AP(A, x):
    for mask in f4_masks[x]:
        if (A & mask) == mask:
            return True
    return False

def count_3APs(A):
    count = 0
    temp = A
    while temp:
        lsb = temp & -temp
        b = lsb.bit_length()
        temp ^= lsb
        for a, c in m3[b]:
            if is_in_set(A, a) and is_in_set(A, c):
                count += 1
    return count

def gain_add(A, x):
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

def loss_remove(A, x):
    l = 0
    for a, b in f3[x]:
        if is_in_set(A, a) and is_in_set(A, b):
            l += 1
    for a, b in m3[x]:
        if is_in_set(A, a) and is_in_set(A, b):
            l += 1
    for a, b in l3[x]:
        if is_in_set(A, a) and is_in_set(A, b):
            l += 1
    return l

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
            if loss == 0:
                # Removing r is free, try to add something? 
                # Actually, if loss is 0, we just remove it? No, we want to maximize 3-APs.
                # If loss is 0, removing it doesn't hurt score. But we want to add.
                # The swap logic: remove r, add x. Net = gain(x) - loss(r).
                pass
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
            
        # 3. Try 2-Swapping (2-out, 2-in) - Structural Enhancement
        # This is expensive O(N^4) naive, but we can restrict or sample.
        # Given time constraints, we might skip full 2-opt if 1-opt is fast enough, 
        # but 2-opt often breaks plateaus.
        # Let's try a limited 2-opt: pick 2 to remove, best 2 to add.
        # To keep it fast, we only do this if 1-opt fails.
        # We'll sample a few pairs of removals.
        
        found_2opt = False
        # Sample removal pairs
        elems = [x for x in range(1, N + 1) if is_in_set(A, x)]
        if len(elems) >= 2:
            # Try a few random pairs or best loss pairs
            # For speed, let's just try the top 5 loss elements for removal?
            # No, low loss elements are better to keep. We want to remove elements that 
            # "block" many additions.
            
            # Simplified 2-opt: 
            # Try removing r1, r2. Then find best x1, x2 to add.
            # This is very slow in Python. 
            # Alternative: Just do more 1-opt passes? 
            # Let's stick to 1-opt for now but ensure it's exhaustive.
            # The current 1-opt is exhaustive.
            
            pass

    return best_score, A

def iterated_local_search(A, deadline, rng):
    best_score, best_A = local_search(A, deadline)
    
    # ILS Loop
    while time.time() < deadline - 1.0:
        # Perturb
        A_curr = best_A
        # Remove k random elements
        k = rng.randint(5, 20)
        elems = [x for x in range(1, N + 1) if is_in_set(A_curr, x)]
        if len(elems) < k:
            k = len(elems)
        to_remove = rng.sample(elems, k)
        for x in to_remove:
            A_curr &= ~(1 << (x - 1))
            
        # Local Search
        sub_deadline = min(time.time() + 5.0, deadline - 1.0)
        score_curr, A_curr = local_search(A_curr, sub_deadline)
        
        if score_curr > best_score:
            best_score = score_curr
            best_A = A_curr
        else:
            # Accept with some probability if not improved (to escape local minima)
            # Or just restart from best? Standard ILS usually accepts worse to explore.
            # Given we want max, let's accept worse with prob exp(-delta/T) or just random.
            # Simpler: If not improved, keep exploring from the perturbed version?
            # No, usually you go back to best and perturb again.
            # Let's just perturb from best_A again.
            pass

    return best_score, best_A

def construct():
    start = time.time()
    deadline = start + 55.0
    
    best_score = -1
    best_A = 0
    
    rng = random.Random(42)
    
    # Initial Greedy
    A0 = greedy_seed(rng)
    score0, A0 = local_search(A0, start + 5.0)
    if score0 > best_score:
        best_score = score0
        best_A = A0
        
    # ILS
    s, A_curr = iterated_local_search(best_A, deadline, rng)
    
    if s > best_score:
        best_score = s
        best_A = A_curr

    # Final polish
    if time.time() < deadline - 0.5 and best_A != 0:
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

# Global rng for SA
rng = random.Random(12345)

if __name__ == "__main__":
    res = construct()
    print(len(res))
