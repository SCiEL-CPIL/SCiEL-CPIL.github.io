import time
import random

class Verifier:
    __slots__ = ("n", "dp", "acc", "valid", "max_k")

    def __init__(self, n):
        self.n = n
        self.dp = [0] * (n + 1)
        self.dp[0] = 1
        self.acc = 1
        self.valid = True
        self.max_k = 0

    def add(self, x):
        if not self.valid:
            return False
        n = self.n
        prev = self.dp
        ndp = list(prev)
        new_max_k = min(self.max_k + 1, n)
        for k in range(new_max_k, 0, -1):
            ndp[k] = prev[k] | (prev[k - 1] << x)
        for k in range(1, new_max_k + 1):
            new_sums = ndp[k] & ~prev[k]
            if new_sums & self.acc:
                self.valid = False
                return False
        for k in range(1, new_max_k + 1):
            self.acc |= ndp[k]
        self.dp = ndp
        self.max_k = new_max_k
        return True

def _check(arr):
    n = len(arr)
    if n < 1 or n > 60:
        return False
    seen = set()
    for v in arr:
        if not isinstance(v, int) or isinstance(v, bool):
            return False
        if v < 1 or v > 256:
            return False
        if v in seen:
            return False
        seen.add(v)
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    acc = 0
    for k in range(n + 1):
        if dp[k] & acc:
            return False
        acc |= dp[k]
    return True

def _calculate_overlap(arr):
    """Calculates the sum of overlaps between ranges [Min_k, Max_k]."""
    n = len(arr)
    if n == 0: return 0
    sorted_arr = sorted(arr)
    min_sums = [0] * (n + 1)
    max_sums = [0] * (n + 1)
    min_sums[0] = 0
    max_sums[0] = 0
    for k in range(1, n + 1):
        min_sums[k] = min_sums[k-1] + sorted_arr[k-1]
        max_sums[k] = max_sums[k-1] + sorted_arr[n-k]
    
    overlap = 0
    for k in range(1, n + 1):
        if max_sums[k-1] >= min_sums[k]:
            overlap += (max_sums[k-1] - min_sums[k] + 1)
    return overlap

def _sa_search(target_size, deadline):
    """Simulated Annealing to find a valid set of target_size."""
    if target_size > 256:
        return None
        
    # Initialize with a high-range set
    start_val = 256 - target_size + 1
    if start_val < 1:
        start_val = 1
    cur = list(range(start_val, start_val + target_size))
    
    # If start_val is too low, we might have overlaps.
    # We want to minimize overlap.
    
    cur_overlap = _calculate_overlap(cur)
    best_overlap = cur_overlap
    best_arr = list(cur)
    
    t = 1000.0
    cool = 0.99
    
    rng = random.Random(42)
    
    while time.monotonic() < deadline and t > 1e-3:
        # Pick a random element to perturb
        idx = rng.randint(0, target_size - 1)
        old_val = cur[idx]
        
        # Propose a new value
        # Bias towards keeping it in range and distinct
        # Simple random walk
        delta = rng.randint(-10, 10)
        new_val = old_val + delta
        if new_val < 1 or new_val > 256:
            continue
            
        # Check distinctness
        if new_val in cur and cur.count(new_val) > 1:
             # This simple check is O(N). For N=32, it's fine.
             # But we need to ensure it's not the same index
             if new_val == old_val:
                 continue
             # If new_val is in cur at another index, it's a duplicate
             if new_val in cur:
                 # If it's a duplicate, we can't just swap values easily without handling indices
                 # Let's just skip if it creates a duplicate
                 pass
             
        # To handle duplicates properly in SA:
        # If new_val is in cur, we must swap it with the element that has new_val?
        # Or just reject.
        # Rejecting duplicates is safer.
        if new_val in cur:
            continue
            
        # Apply change
        cur[idx] = new_val
        new_overlap = _calculate_overlap(cur)
        
        delta_e = new_overlap - cur_overlap
        if delta_e < 0 or rng.random() < pow(2.718, -delta_e / t):
            cur_overlap = new_overlap
            if new_overlap < best_overlap:
                best_overlap = new_overlap
                best_arr = list(cur)
                if best_overlap == 0:
                    # Check if actually valid
                    if _check(best_arr):
                        return best_arr
        else:
            cur[idx] = old_val
            
        t *= cool
        
    if best_overlap == 0 and _check(best_arr):
        return best_arr
    return None

def construct():
    t0 = time.monotonic()
    deadline = t0 + 80.0 # Safety margin
    
    # 1. Try to find a set of size 32 using SA
    # If found, return it.
    res_32 = _sa_search(32, deadline)
    if res_32 is not None:
        return res_32
        
    # 2. Fallback to size 31
    # The highest range consecutive integers for 31 is 226..256
    # We know this is valid.
    res_31 = list(range(226, 257))
    if _check(res_31):
        return res_31
        
    # 3. If even 31 fails (shouldn't), try greedy
    # (Omitted for brevity, but 31 is guaranteed)
    
    return res_31
