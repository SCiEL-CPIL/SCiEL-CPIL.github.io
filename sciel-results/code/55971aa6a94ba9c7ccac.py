# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0  # Safety margin (90s kill)
    N = 512
    MAXC = 48
    rng = random.Random(12345)

    # Precompute conflict structure:
    # For each element a, find all pairs (b,c) with b+c = 2a (b != c, b != a, c != a)
    # Actually, we need: a is the mean of subset {b,c,...} of size >= 2 from OTHER elements.
    # For size 2: 2a = b + c
    # For size 3: 3a = b + c + d
    # etc.
    
    # Strategy: 
    # 1. Build a graph where edges represent "2-uniform" conflicts (a is mean of pair)
    # 2. Use greedy to build a large set with no 2-conflicts
    # 3. Then check for larger subset conflicts and remove violators
    # 4. Local improvement: try to add back removed elements or swap
    
    # Precompute pair-sum conflicts using bitsets
    # conflict[i] = bitset of j such that s[i] is the mean of {s[j], s[k]} for some k
    # But we need this dynamically as we build the set...
    
    # Alternative approach: 
    # Phase 1: Greedy with 2-conflict check (fast, O(n) per addition)
    # Phase 2: Full validation with bitset DP, remove violators
    # Phase 3: Local search - try to add back elements, try swaps
    
    def is_2_valid_addition(s, x, s_set):
        """Check if adding x to s maintains no 2-element mean constraint."""
        # x is mean of pair in s: 2x = a + b
        for a in s:
            b = 2 * x - a
            if b in s_set and b != a and b != x:
                return False
        # Some a in s becomes mean of pair involving x: 2a = b + x => b = 2a - x
        for a in s:
            b = 2 * a - x
            if b in s_set and b != a and b != x:
                return False
        return True

    def full_validate(s):
        """Return True if s has no element that is the mean of any subset (size>=2) of others."""
        n = len(s)
        if n < 3:
            return True
        for i in range(n):
            a = s[i]
            # DP: dp[k] = bitset of achievable sums of exactly k elements from others
            dp = [0] * (n + 1)
            dp[0] = 1
            for j in range(n):
                if j == i:
                    continue
                v = s[j]
                for k in range(n, 0, -1):
                    dp[k] |= dp[k - 1] << v
            for k in range(2, n):
                if (dp[k] >> (k * a)) & 1:
                    return False
        return True

    def find_violator(s):
        """Return index of first element that is a mean of some subset, or -1."""
        n = len(s)
        for i in range(n):
            a = s[i]
            dp = [0] * (n + 1)
            dp[0] = 1
            for j in range(n):
                if j == i:
                    continue
                v = s[j]
                for k in range(n, 0, -1):
                    dp[k] |= dp[k - 1] << v
            for k in range(2, n):
                if (dp[k] >> (k * a)) & 1:
                    return i
        return -1

    # Phase 1: Greedy construction with 2-conflict avoidance
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    best = []
    best_set = set()
    
    for x in candidates:
        if len(best) >= MAXC:
            break
        if is_2_valid_addition(best, x, best_set):
            best.append(x)
            best_set.add(x)
    
    best.sort()
    
    # Phase 2: Validate and repair
    repair_count = 0
    while time.time() < DEADLINE and repair_count < 100:
        if full_validate(best):
            break
        v_idx = find_violator(best)
        if v_idx == -1:
            break
        # Remove the violator
        removed = best.pop(v_idx)
        best_set.discard(removed)
        repair_count += 1
    
    current_len = len(best)
    
    # Phase 3: Local improvement - try to add back elements
    # Try elements from 1 to 512 that aren't in best
    if time.time() < DEADLINE:
        for x in range(1, N + 1):
            if time.time() >= DEADLINE:
                break
            if x in best_set:
                continue
            if len(best) >= MAXC:
                break
            if is_2_valid_addition(best, x, best_set):
                # Tentatively add and check full validity
                test_best = best + [x]
                test_best.sort()
                if full_validate(test_best):
                    best = test_best
                    best_set.add(x)
    
    # Phase 4: Local search with swaps
    if time.time() < DEADLINE:
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for i in range(len(best)):
                if time.time() >= DEADLINE:
                    break
                # Try removing best[i] and adding a different element
                for x in range(1, N + 1):
                    if time.time() >= DEADLINE:
                        break
                    if x in best_set:
                        continue
                    # Remove best[i], add x
                    test_best = [v for j, v in enumerate(best) if j != i] + [x]
                    test_best.sort()
                    if len(test_best) <= len(best):
                        continue
                    if full_validate(test_best):
                        best = test_best
                        best_set = set(best)
                        improved = True
                        break
                if improved:
                    break
    
    return best

# EVOLVE-BLOCK-END
