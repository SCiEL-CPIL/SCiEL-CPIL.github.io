import numpy as np
import time
import random

def _count_4aps(arr):
    """Return number of nonconstant 4-term APs in arr (bool array len 256)."""
    n = len(arr)
    total = 0
    for d in range(1, n // 4 + 1):
        # positions i such that i, i+d, i+2d, i+3d all in [0, n)
        # i ranges 0..n-1-3d
        limit = n - 3 * d
        if limit <= 0:
            break
        # arr[0:limit], arr[d:d+limit], arr[2d:2d+limit], arr[3d:3d+limit]
        a = arr[0:limit]
        b = arr[d:d + limit]
        c = arr[2 * d:2 * d + limit]
        e = arr[3 * d:3 * d + limit]
        total += int(np.sum(a & b & c & e))
    return total

def _count_3aps(arr):
    """Return number of 3-term APs (positive diff) in arr."""
    n = len(arr)
    total = 0
    for d in range(1, n // 2 + 1):
        limit = n - 2 * d
        if limit <= 0:
            break
        a = arr[0:limit]
        b = arr[d:d + limit]
        c = arr[2 * d:2 * d + limit]
        total += int(np.sum(a & b & c))
    return total

def _greedy_init(n=256):
    """Greedy: add elements one by one, choosing the one that adds the most 3APs while keeping 4AP-free."""
    arr = np.zeros(n, dtype=bool)
    # Precompute for each candidate x and each d whether adding x would create a 4AP
    # We'll just check validity after each tentative add.
    # To speed up, we track current 3AP count.
    cur_3ap = 0
    remaining = list(range(n))
    random.shuffle(remaining)
    # Actually, let's do a smarter greedy: at each step, for each remaining x, compute gain in 3APs
    # and check 4AP validity. Pick best.
    # This is O(n^3) which is fine for n=256.
    for _ in range(n):
        best_x = -1
        best_gain = -1
        best_new_3ap = 0
        for x in remaining:
            if arr[x]:
                continue
            # Tentatively add x
            arr[x] = True
            # Check 4AP
            if _count_4aps(arr) > 0:
                arr[x] = False
                continue
            # Count 3APs
            new_3ap = _count_3aps(arr)
            gain = new_3ap - cur_3ap
            if gain > best_gain:
                best_gain = gain
                best_x = x
                best_new_3ap = new_3ap
            arr[x] = False
        if best_x == -1:
            break
        arr[best_x] = True
        cur_3ap = best_new_3ap
        remaining.remove(best_x)
    return arr, cur_3ap

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 75  # 75s safety margin before 90s kill
    
    random.seed(42)
    np.random.seed(42)
    
    # Greedy initialization
    arr, cur_3ap = _greedy_init(n)
    best_arr = arr.copy()
    best_3ap = cur_3ap
    
    # Simulated annealing
    T = 5.0
    cooling = 0.9995
    
    while time.time() < deadline:
        # Decide move type: flip or swap
        if random.random() < 0.5:
            # Flip move
            x = random.randrange(n)
            if arr[x]:
                # Try removing x
                arr[x] = False
                if _count_4aps(arr) == 0:
                    new_3ap = _count_3aps(arr)
                    delta = new_3ap - cur_3ap
                    if delta >= 0 or random.random() < np.exp(delta / T):
                        cur_3ap = new_3ap
                        if cur_3ap > best_3ap:
                            best_3ap = cur_3ap
                            best_arr = arr.copy()
                    else:
                        arr[x] = True
                else:
                    arr[x] = True
            else:
                # Try adding x
                arr[x] = True
                if _count_4aps(arr) == 0:
                    new_3ap = _count_3aps(arr)
                    delta = new_3ap - cur_3ap
                    if delta >= 0 or random.random() < np.exp(delta / T):
                        cur_3ap = new_3ap
                        if cur_3ap > best_3ap:
                            best_3ap = cur_3ap
                            best_arr = arr.copy()
                    else:
                        arr[x] = False
                else:
                    arr[x] = False
        else:
            # Swap move: pick two indices, one in set, one not
            in_set = np.where(arr)[0]
            not_in_set = np.where(~arr)[0]
            if len(in_set) == 0 or len(not_in_set) == 0:
                T *= cooling
                continue
            x = in_set[random.randrange(len(in_set))]
            y = not_in_set[random.randrange(len(not_in_set))]
            arr[x] = False
            arr[y] = True
            if _count_4aps(arr) == 0:
                new_3ap = _count_3aps(arr)
                delta = new_3ap - cur_3ap
                if delta >= 0 or random.random() < np.exp(delta / T):
                    cur_3ap = new_3ap
                    if cur_3ap > best_3ap:
                        best_3ap = cur_3ap
                        best_arr = arr.copy()
                else:
                    arr[x] = True
                    arr[y] = False
            else:
                arr[x] = True
                arr[y] = False
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    # Ensure best_arr is valid
    if _count_4aps(best_arr) > 0:
        # Fallback to greedy
        best_arr, best_3ap = _greedy_init(n)
    
    result = [int(i + 1) for i in range(n) if best_arr[i]]
    return sorted(result)

