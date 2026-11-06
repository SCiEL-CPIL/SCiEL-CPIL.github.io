"""B_3[1] set construction using bytearray for O(1) sum lookups and recursive backtracking."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    n_max = 4096
    max_sum = 3 * n_max  # maximum possible triple sum = 12288
    best = []
    best_len = 0

    def make_sums_array():
        """Create a bytearray where index s indicates if sum s is already used."""
        return bytearray(max_sum + 1)

    def try_add(selected, sums_arr, x):
        """Try adding x to selected. Returns True if successful, updates sums_arr."""
        s = 3 * x
        if sums_arr[s]:
            return False
        for a in selected:
            s = 2 * x + a
            if sums_arr[s]:
                return False
        n = len(selected)
        for i in range(n):
            ai = selected[i]
            for j in range(i, n):
                s = x + ai + selected[j]
                if sums_arr[s]:
                    return False
        sums_arr[3 * x] = 1
        for a in selected:
            sums_arr[2 * x + a] = 1
        for i in range(n):
            ai = selected[i]
            for j in range(i, n):
                sums_arr[x + ai + selected[j]] = 1
        return True

    def remove_element_sums(selected, sums_arr, idx):
        """Remove all triple sums involving selected[idx]."""
        x = selected[idx]
        n = len(selected)
        # Remove 3*x
        sums_arr[3 * x] = 0
        # Remove 2*x + a for each a != x
        for i in range(n):
            if i == idx:
                continue
            a = selected[i]
            sums_arr[2 * x + a] = 0
        # Remove x + ai + aj for i < j, neither is idx
        for i in range(n):
            if i == idx:
                continue
            ai = selected[i]
            for j in range(i, n):
                if j == idx:
                    continue
                aj = selected[j]
                if i < j:
                    sums_arr[x + ai + aj] = 0

    def greedy_build(start_val, max_iter=50000):
        """Greedy build from start_val, incrementing by 1."""
        selected = []
        sums_arr = make_sums_array()
        x = start_val
        count = 0
        while x <= n_max and count < max_iter:
            if time.time() > deadline:
                break
            if try_add(selected, sums_arr, x):
                selected.append(x)
            x += 1
            count += 1
        return selected

    def backtrack(selected, sums_arr, min_val, max_add, max_depth=30):
        """Backtracking search to add up to max_add more elements."""
        nonlocal best, best_len
        if time.time() > deadline:
            return
        if len(selected) > best_len:
            best = list(selected)
            best_len = len(best)
        
        if len(selected) >= max_depth or max_add <= 0:
            return
        
        # Try adding the smallest valid element first
        for x in range(min_val, min_val + 200):
            if time.time() > deadline:
                return
            if x > n_max:
                break
            if try_add(selected, sums_arr, x):
                selected.append(x)
                backtrack(selected, sums_arr, x + 1, max_add - 1, max_depth)
                selected.pop()
                # Remove sums of x
                n = len(selected)
                sums_arr[3 * x] = 0
                for a in selected:
                    sums_arr[2 * x + a] = 0
                for i in range(n):
                    ai = selected[i]
                    for j in range(i, n):
                        sums_arr[x + ai + selected[j]] = 0

    def local_improve(selected, max_iters=200):
        """Try to improve by removing one element and trying to add a larger one."""
        nonlocal best, best_len
        if time.time() > deadline:
            return selected
        
        improved = True
        iteration = 0
        while improved and iteration < max_iters:
            if time.time() > deadline:
                break
            improved = False
            iteration += 1
            sums_arr = make_sums_array()
            n = len(selected)
            # Rebuild sums_arr
            for i in range(n):
                ai = selected[i]
                for j in range(i, n):
                    aj = selected[j]
                    for k in range(j, n):
                        ak = selected[k]
                        sums_arr[ai + aj + ak] = 1
            
            for i in range(n - 1, -1, -1):
                if time.time() > deadline:
                    break
                x = selected[i]
                remove_element_sums(selected, sums_arr, i)
                selected.pop(i)
                
                found = False
                for new_x in range(x + 1, min(x + 300, n_max + 1)):
                    if try_add(selected, sums_arr, new_x):
                        selected.append(new_x)
                        selected.sort()
                        found = True
                        improved = True
                        break
                
                if not found:
                    selected.insert(i, x)
                    n2 = len(selected)
                    sums_arr[3 * x] = 1
                    for j in range(n2):
                        if j == i:
                            continue
                        a = selected[j]
                        sums_arr[2 * x + a] = 1
                    for j in range(n2):
                        if j == i:
                            continue
                        aj = selected[j]
                        for k in range(j, n2):
                            if k == i:
                                continue
                            sums_arr[x + aj + selected[k]] = 1
        
        if len(selected) > best_len:
            best = list(selected)
            best_len = len(best)
        
        return selected

    # === Strategy 1: Linear greedy from various starts ===
    for start in range(1, 50):
        if time.time() > deadline:
            break
        selected = greedy_build(start)
        if len(selected) > best_len:
            best = list(selected)
            best_len = len(selected)
    
    # === Strategy 2: Backtracking from small starts ===
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline:
            break
        start = random.randint(1, 30)
        selected = []
        sums_arr = make_sums_array()
        # Greedily build a small initial set
        for x in range(start, start + 20):
            if try_add(selected, sums_arr, x):
                selected.append(x)
            if len(selected) >= 5:
                break
        if selected:
            backtrack(selected, sums_arr, selected[-1] + 1, max_add=8, max_depth=30)
    
    # === Strategy 3: Exponential greedy ===
    for base in [1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0]:
        if time.time() > deadline:
            break
        selected = []
        sums_arr = make_sums_array()
        x = 1
        while x <= n_max:
            if time.time() > deadline:
                break
            if try_add(selected, sums_arr, x):
                selected.append(x)
            x = int(x * base) + 1
        if len(selected) > best_len:
            best = list(selected)
            best_len = len(selected)
    
    # === Strategy 4: Random skip greedy ===
    for restart in range(60):
        if time.time() > deadline:
            break
        selected = []
        sums_arr = make_sums_array()
        current = random.randint(1, 50)
        while current <= n_max:
            if time.time() > deadline:
                break
            if try_add(selected, sums_arr, current):
                selected.append(current)
            skip = random.randint(1, 15)
            current += skip
        if len(selected) > best_len:
            best = list(selected)
            best_len = len(selected)
    
    # === Strategy 5: Local improvement on best ===
    if best and time.time() < deadline - 5:
        local_improve(list(best), max_iters=200)
    
    return best
# EVOLVE-BLOCK-END
