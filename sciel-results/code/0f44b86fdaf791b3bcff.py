import time
import random

N = 256

# Precompute 3-AP and 4-AP structures
f4 = [[] for _ in range(N + 1)]
masks3 = [[] for _ in range(N + 1)]
masks4 = [[] for _ in range(N + 1)]
third = [[-1] * (N + 1) for _ in range(N + 1)]

# Build 3-AP masks and third element map
for a in range(1, N + 1):
    for d in range(1, (N - a) // 2 + 1):
        b = a + d
        c = a + 2 * d
        # 3-AP: a, b, c
        masks3[a].append((1 << b) | (1 << c))
        masks3[b].append((1 << a) | (1 << c))
        masks3[c].append((1 << a) | (1 << b))
        
        third[a][b] = c
        third[b][a] = c
        third[a][c] = b
        third[c][a] = b
        third[b][c] = a
        third[c][b] = a

# Build 4-AP masks and f4 list
for a in range(1, N + 1):
    for d in range(1, (N - a) // 3 + 1):
        b = a + d
        c = a + 2 * d
        e = a + 3 * d
        # 4-AP: a, b, c, e
        f4[a].append((b, c, e))
        f4[b].append((a, c, e))
        f4[c].append((a, b, e))
        f4[e].append((a, b, c))
        
        masks4[a].append((1 << b) | (1 << c) | (1 << e))
        masks4[b].append((1 << a) | (1 << c) | (1 << e))
        masks4[c].append((1 << a) | (1 << b) | (1 << e))
        masks4[e].append((1 << a) | (1 << b) | (1 << c))

def count_3AP_total(mask):
    s = 0
    for x in range(1, N + 1):
        if (mask >> x) & 1:
            for m in masks3[x]:
                if (mask & m) == m:
                    s += 1
    return s // 3

def greedy_seed(rng):
    mask = 0
    order = list(range(1, N + 1))
    rng.shuffle(order)
    for x in order:
        valid = True
        for m in masks4[x]:
            if (mask & m) == m:
                valid = False
                break
        if valid:
            mask |= (1 << x)
    return mask

def local_search(mask, deadline, rng):
    best_mask = mask
    best_score = count_3AP_total(mask)
    
    while time.time() < deadline:
        # Precompute gains/losses
        # gain[x] if x not in mask, else -loss[x]
        val = [0] * (N + 1)
        for x in range(1, N + 1):
            if (mask >> x) & 1:
                # Loss
                l = 0
                for m in masks3[x]:
                    if (mask & m) == m:
                        l += 1
                val[x] = -l
            else:
                # Check validity
                valid = True
                for m in masks4[x]:
                    if (mask & m) == m:
                        valid = False
                        break
                if valid:
                    g = 0
                    for m in masks3[x]:
                        if (mask & m) == m:
                            g += 1
                    val[x] = g
                else:
                    val[x] = -10**9
        
        best_delta = 0
        best_swap = None
        
        # Find best swap
        # Iterate r in mask, x not in mask
        # To reduce overhead, we can just iterate all pairs
        for r in range(1, N + 1):
            if not ((mask >> r) & 1):
                continue
            loss_r = -val[r]
            for x in range(1, N + 1):
                if (mask >> x) & 1:
                    continue
                if val[x] < -10**8:
                    continue
                gain_x = val[x]
                
                shared = 0
                z = third[x][r]
                if z != -1:
                    if (mask >> z) & 1:
                        shared = 1
                
                delta = gain_x - loss_r - shared
                if delta > best_delta:
                    best_delta = delta
                    best_swap = (r, x)
        
        if best_swap is None or best_delta <= 0:
            # Try random perturbation
            improved = False
            for _ in range(5):
                r = rng.randint(1, N)
                if not ((mask >> r) & 1):
                    continue
                x = rng.randint(1, N)
                if (mask >> x) & 1:
                    continue
                temp_mask = mask & ~(1 << r)
                valid = True
                for m in masks4[x]:
                    if (temp_mask & m) == m:
                        valid = False
                        break
                if not valid:
                    continue
                new_mask = temp_mask | (1 << x)
                new_score = count_3AP_total(new_mask)
                if new_score > best_score:
                    mask = new_mask
                    best_mask = new_mask
                    best_score = new_score
                    improved = True
                    break
            if not improved:
                break
        
        r, x = best_swap
        mask = (mask & ~(1 << r)) | (1 << x)
        new_score = count_3AP_total(mask)
        if new_score > best_score:
            best_score = new_score
            best_mask = mask
        # If new_score <= best_score, we still update mask to explore?
        # Usually best improvement local search requires improvement.
        # If delta > 0, it should improve.
        
    return best_mask, best_score

def construct():
    start = time.time()
    deadline = start + 52.0
    rng = random.Random(12345)
    
    best_mask = 0
    best_score = -1
    
    while time.time() < deadline - 5.0:
        mask = greedy_seed(rng)
        mask, score = local_search(mask, deadline, rng)
        if score > best_score:
            best_score = score
            best_mask = mask
            
    if time.time() < deadline - 0.5 and best_mask != 0:
        mask, score = local_search(best_mask, deadline, rng)
        if score > best_score:
            best_score = score
            best_mask = mask
            
    A = [x for x in range(1, N + 1) if (best_mask >> x) & 1]
    if not A:
        A = [1]
        
    # Hard validity check
    present = [False] * (N + 1)
    for x in A:
        present[x] = True
    for x in list(A):
        for (a, b, c) in f4[x]:
            if present[a] and present[b] and present[c]:
                present[x] = False
                A.remove(x)
                break
    if not A:
        A = [1]
        
    return A
