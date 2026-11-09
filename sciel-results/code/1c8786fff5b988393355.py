import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    N = 512
    MAXC = 48

    # Precompute pair conflicts for danger scoring
    pair_conflicts = defaultdict(int)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            s = a + b
            if s % 2 == 0:
                m = s // 2
                if 1 <= m <= N and m != a and m != b:
                    pair_conflicts[m] += 1

    constituent_conflicts = defaultdict(int)
    for a in range(1, N + 1):
        for b in range(1, N + 1):
            if a == b:
                continue
            s = a + b
            if s % 2 == 0:
                m = s // 2
                if 1 <= m <= N and m != a and m != b:
                    constituent_conflicts[a] += 1

    danger = {}
    for x in range(1, N + 1):
        danger[x] = pair_conflicts.get(x, 0) + constituent_conflicts.get(x, 0)

    all_nums = list(range(1, N + 1))
    all_nums.sort(key=lambda x: (danger[x], x))

    # DP-based verification
    def build_dp(S):
        n = len(S)
        dp = [0] * (n + 1)
        dp[0] = 1
        for a in S:
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k + 1] |= dp[k] << a
        return dp

    def verify(S):
        S = list(S)
        n = len(S)
        for i in range(n):
            a = S[i]
            sub = S[:i] + S[i + 1:]
            m = len(sub)
            if m < 2:
                continue
            dp = build_dp(sub)
            maxs = sum(sub)
            for k in range(2, m + 1):
                t = k * a
                if t > maxs:
                    break
                if (dp[k] >> t) & 1:
                    return False
        return True

    def test_add(S, x):
        """Check if x can be added to S without violation."""
        n = len(S)
        
        # Check 1: x is mean of subset of S
        if n >= 2:
            dp_full = build_dp(S)
            maxfull = sum(S)
            for k in range(2, n + 1):
                t = k * x
                if t > maxfull:
                    break
                if (dp_full[k] >> t) & 1:
                    return False
        
        # Check 2: some a in S becomes mean of subset involving x
        for a in S:
            sub = [s for s in S if s != a]
            m = len(sub)
            if m < 1:
                continue
            dp_sub = build_dp(sub)
            max_sub = sum(sub)
            for j in range(1, m + 1):
                t = (j + 1) * a - x
                if t < 0 or t > max_sub:
                    continue
                if (dp_sub[j] >> t) & 1:
                    return False
        return True

    best = []
    best_sz = 0

    # Multiple greedy passes with different orderings
    orders = [
        all_nums,
        all_nums[::-1],
    ]
    
    for _ in range(5):
        mixed = all_nums[:]
        random.shuffle(mixed)
        orders.append(mixed)
    
    for order in orders:
        if time.time() > deadline - 15:
            break
        if best_sz >= MAXC:
            break
        
        cur = []
        for x in order:
            if len(cur) >= MAXC:
                break
            if time.time() > deadline - 15:
                break
            if test_add(cur, x):
                cur.append(x)
        
        if len(cur) > best_sz:
            if verify(cur):
                best_sz = len(cur)
                best = cur[:]
    
    # Local search: try to add more elements
    if best_sz < MAXC and time.time() < deadline - 10:
        S = best[:]
        Sset = set(S)
        unused = [x for x in range(1, N + 1) if x not in Sset]
        unused.sort(key=lambda x: danger[x])
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for u in unused[:]:
                if len(S) >= MAXC:
                    break
                if time.time() > deadline - 5:
                    break
                if test_add(S, u):
                    S.append(u)
                    Sset.add(u)
                    unused.remove(u)
                    improved = True
                    break
        
        if verify(S) and len(S) > best_sz:
            best_sz = len(S)
            best = S[:]
    
    # Local search: try swaps
    if best_sz < MAXC and time.time() < deadline - 5:
        S = best[:]
        Sset = set(S)
        unused = [x for x in range(1, N + 1) if x not in Sset]
        unused.sort(key=lambda x: danger[x])
        
        improved = True
        no_improve = 0
        while improved and time.time() < deadline - 3 and no_improve < 10:
            improved = False
            swapped = False
            for u in unused[:30]:
                if time.time() > deadline - 3:
                    break
                for i, s in enumerate(S):
                    newS = S[:i] + S[i + 1:]
                    if test_add(newS, u):
                        S = newS + [u]
                        Sset.discard(s)
                        Sset.add(u)
                        unused.remove(u)
                        unused.append(s)
                        improved = True
                        swapped = True
                        break
                if swapped:
                    break
            if not swapped:
                no_improve += 1
                random.shuffle(unused)
        
        if verify(S) and len(S) > best_sz:
            best_sz = len(S)
            best = S[:]
    
    # Final safety check
    if not verify(best):
        for sz in range(len(best), 0, -1):
            if verify(best[:sz]):
                best = best[:sz]
                break
        else:
            best = []
    
    return best

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} elements")
    print(res)
