import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    def check_new_element(a, S_sorted):
        """Check if adding a to S_sorted (sorted list) creates any violation."""
        n = len(S_sorted)
        # Case 1: a is the smallest (a < b <= c)
        # Need: exists b,c in S with a < b <= c and a | (b+c)
        # b+c is a multiple of a, b+c >= 2b >= 2*(a+1)
        # b+c <= 2*S_sorted[-1]
        # k*a in [2*(a+1), 2*S_sorted[-1]], k >= 2
        for k in range(2, 2*S_sorted[-1]//a + 1):
            target = k * a
            # Need b in S, a < b, c = target - b, b <= c, c in S
            # b <= target/2, c = target - b >= b
            # b in [a+1, target//2]
            # Use binary search
            lo = 0
            hi = n
            # Find elements in S in range [a+1, target//2]
            while lo < hi:
                mid = (lo + hi) // 2
                if S_sorted[mid] <= a:
                    lo = mid + 1
                else:
                    hi = mid
            start_idx = lo
            # Upper bound: target//2
            hi = n
            while lo < hi:
                mid = (lo + hi) // 2
                if S_sorted[mid] <= target // 2:
                    lo = mid + 1
                else:
                    hi = mid
            end_idx = lo
            for idx in range(start_idx, end_idx):
                b = S_sorted[idx]
                c = target - b
                if c < b:
                    break
                # Binary search for c in S_sorted
                lo2 = 0
                hi2 = n
                while lo2 < hi2:
                    mid2 = (lo2 + hi2) // 2
                    if S_sorted[mid2] < c:
                        lo2 = mid2 + 1
                    else:
                        hi2 = mid2
                if lo2 < n and S_sorted[lo2] == c:
                    return False
        # Case 2: a is the middle (x < a <= c), x in S, c in S, x | (a+c)
        # x < a, so x in S_sorted elements < a
        # c >= a, c in S
        # a+c is multiple of x
        # Find all x in S with x < a
        lo = 0
        hi = n
        while lo < hi:
            mid = (lo + hi) // 2
            if S_sorted[mid] < a:
                lo = mid + 1
            else:
                hi = mid
        x_end = lo
        for xi in range(x_end):
            x = S_sorted[xi]
            # c = k*x - a, c >= a, c in S
            # k*x >= 2a, k >= ceil(2a/x)
            # k*x - a <= S_sorted[-1], k <= (S_sorted[-1]+a)/x
            k_min = (2*a + x - 1) // x
            k_max = (S_sorted[-1] + a) // x
            for k in range(k_min, k_max + 1):
                c = k * x - a
                if c < a:
                    continue
                # Binary search for c
                lo2 = 0
                hi2 = n
                while lo2 < hi2:
                    mid2 = (lo2 + hi2) // 2
                    if S_sorted[mid2] < c:
                        lo2 = mid2 + 1
                    else:
                        hi2 = mid2
                if lo2 < n and S_sorted[lo2] == c:
                    return False
        # Case 3: a is the largest (x < y <= a), x,y in S, x | (y+a)
        # x < y <= a, x,y in S
        # y+a is multiple of x
        # y in [x+1, a], y in S
        # y+a = k*x, k >= 2 (since y+a > 2x when y > x... not necessarily)
        # Actually y+a >= (x+1)+a, and k*x = y+a
        # Let's just iterate
        for xi in range(n):
            x = S_sorted[xi]
            if x >= a:
                break
            # y in S, x < y <= a, (y+a) % x == 0
            # y = k*x - a, x < y <= a
            # k*x - a > x => k*x > a+x => k > (a+x)/x = a/x + 1
            # k*x - a <= a => k*x <= 2a => k <= 2a/x
            k_min = (a + x + 1) // x  # k*x > a+x
            k_max = (2*a) // x
            for k in range(k_min, k_max + 1):
                y = k * x - a
                if y <= x or y > a:
                    continue
                # Binary search for y
                lo2 = 0
                hi2 = n
                while lo2 < hi2:
                    mid2 = (lo2 + hi2) // 2
                    if S_sorted[mid2] < y:
                        lo2 = mid2 + 1
                    else:
                        hi2 = mid2
                if lo2 < n and S_sorted[lo2] == y:
                    return False
        return True
    
    # Start with [129, 256]
    best_set = set(range(129, 257))
    best_sorted = sorted(best_set)
    
    # Verify and fix initial set
    # For [129,256], check for violations
    n = len(best_sorted)
    for _ in range(100):
        if time.time() - start_time > DEADLINE:
            break
        found_violation = False
        for i in range(n):
            a = best_sorted[i]
            for j in range(i+1, n):
                b = best_sorted[j]
                for k in range(j, n):
                    c = best_sorted[k]
                    if (b+c) % a == 0:
                        best_set.discard(a)
                        best_sorted = sorted(best_set)
                        n = len(best_sorted)
                        found_violation = True
                        break
                if found_violation:
                    break
            if found_violation:
                break
        if not found_violation:
            break
    
    # Greedy add from [1,128]
    candidates = list(range(1, 129))
    random.shuffle(candidates)
    for cand in candidates:
        if time.time() - start_time > DEADLINE:
            break
        if cand not in best_set:
            if check_new_element(cand, best_sorted):
                best_set.add(cand)
                best_sorted.append(cand)
                best_sorted.sort()
    
    # Local search
    for _ in range(3000):
        if time.time() - start_time > DEADLINE:
            break
        cand = random.randint(1, 256)
        if cand not in best_set:
            if check_new_element(cand, best_sorted):
                best_set.add(cand)
                best_sorted.append(cand)
                best_sorted.sort()
    
    return list(best_set)
