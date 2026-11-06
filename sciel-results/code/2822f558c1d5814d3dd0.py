import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 50  # safety margin

    # --- Helper: fast validity check using DP per element ---
    def is_valid_set(S):
        if len(S) < 2:
            return True
        n = len(S)
        # For each x in S, check if x is mean of any subset of size >=2 from S\{x}
        # Target sum for k elements: k*x
        # We use DP: dp[k] is a Python int as bitset of achievable sums
        # Max sum <= 512 * 48 = 24576, so bitset fits in ~24576 bits
        for x in S:
            others = [v for v in S if v != x]
            if len(others) < 2:
                continue
            m = len(others)
            max_sum = sum(others)
            # We only care about sums up to m*x (the target for k=m is m*x)
            # But we need dp[k] for k from 2 to m
            dp = [0] * (m + 1)
            dp[0] = 1  # sum 0
            for val in others:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k - 1] << val)
            # Check targets
            for k in range(2, m + 1):
                target = k * x
                if target > max_sum:
                    break
                if (dp[k] >> target) & 1:
                    return False
        return True

    # --- Generate base-3 candidates (3-AP-free) ---
    base3 = []
    for i in range(1, 513):
        n = i
        ok = True
        while n > 0:
            if n % 3 == 2:
                ok = False
                break
            n //= 3
        if ok:
            base3.append(i)
    base3.sort()

    # --- Strategy 1: Try the full base3 set or its prefixes ---
    # If base3 has >= 48 elements, try to find a valid subset of size 48
    best_set = []
    
    # First, check if the smallest 48 base3 elements are valid
    if len(base3) >= 48:
        S = base3[:48]
        if is_valid_set(S):
            return S
    
    # --- Strategy 2: Greedy construction from base3 ---
    # This is the main workhorse. We greedily add elements, checking validity.
    # We try multiple orders: ascending, descending, and a few shuffles.
    
    def greedy_construct(candidates, order="asc", deadline=None):
        if order == "asc":
            cands = sorted(candidates)
        elif order == "desc":
            cands = sorted(candidates, reverse=True)
        else:
            cands = candidates[:]
            random.shuffle(cands)
        
        S = []
        for x in cands:
            if len(S) >= 48:
                break
            if deadline and time.time() > deadline:
                break
            new_S = S + [x]
            if is_valid_set(new_S):
                S.append(x)
        return S

    # Try ascending greedy
    best_set = greedy_construct(base3, "asc", deadline)
    if len(best_set) == 48:
        return best_set

    # Try descending greedy
    if time.time() < deadline:
        S2 = greedy_construct(base3, "desc", deadline)
        if len(S2) > len(best_set):
            best_set = S2
        if len(best_set) == 48:
            return best_set

    # Try a few random shuffles
    random.seed(12345)
    for _ in range(10):
        if time.time() >= deadline:
            break
        S3 = greedy_construct(base3, "rand", deadline)
        if len(S3) > len(best_set):
            best_set = S3
        if len(best_set) == 48:
            return best_set

    # --- Strategy 3: Local search / repair ---
    # If we have a valid set of size < 48, try to replace elements to free up space
    # or add new ones from the pool.
    if len(best_set) < 48 and time.time() < deadline:
        # Create a pool of unused base3 elements
        used = set(best_set)
        pool = [x for x in base3 if x not in used]
        random.shuffle(pool)
        
        improved = True
        while improved and len(best_set) < 48 and time.time() < deadline:
            improved = False
            # Try to add an element
            for x in pool:
                if len(best_set) >= 48:
                    break
                new_S = best_set + [x]
                if is_valid_set(new_S):
                    best_set.append(x)
                    pool.remove(x)
                    improved = True
                    break
            
            # If we couldn't add, try a swap
            if not improved and len(best_set) < 48:
                # Try swapping one element in best_set with one from pool
                # This is expensive, so limit iterations
                for i in range(min(20, len(pool))):
                    if time.time() >= deadline:
                        break
                    x_new = pool[i]
                    for j, x_old in enumerate(best_set):
                        new_S = best_set[:j] + best_set[j+1:] + [x_new]
                        if is_valid_set(new_S):
                            best_set = new_S
                            pool.remove(x_new)
                            # Mark old as available? No, we swapped, so x_old is no longer in set.
                            # But we need to be careful with pool.
                            # Actually, let's just break and re-evaluate.
                            improved = True
                            break
                    if improved:
                        break

    # --- Strategy 4: Expand pool to all 1..512 if still stuck ---
    # This is risky because non-base3 elements introduce 3-APs, but maybe we can
    # still find a valid set of size 48 by mixing.
    if len(best_set) < 48 and time.time() < deadline:
        used = set(best_set)
        all_nums = [x for x in range(1, 513) if x not in used]
        random.shuffle(all_nums)
        
        for x in all_nums:
            if len(best_set) >= 48 or time.time() >= deadline:
                break
            new_S = best_set + [x]
            if is_valid_set(new_S):
                best_set.append(x)

    # Final check
    if is_valid_set(best_set):
        return best_set
    
    # Fallback: return empty or smallest valid set
    # But the task requires distinct positive integers <= 512.
    # If we fail, return [] or a small valid set.
    # Let's ensure we return at least something valid.
    # The empty set is valid.
    return best_set

# Local test
if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    print(f"Time: {time.time()-t0:.2f}s, Size: {len(res)}")
    print(res[:20], "...")
