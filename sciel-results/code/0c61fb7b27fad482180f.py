import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Precompute divisibility table for speed
    # div[a][s] = True if a divides s
    div = [[False] * (2 * n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for s in range(2 * a, 2 * n + 1, a):
            div[a][s] = True

    def can_add(x, S_set):
        """Check if x can be added to S_set. S_set is a set of ints."""
        # 1. x is a: check all pairs (b,c) in S_set with b > x, c >= b
        for b in S_set:
            if b <= x:
                continue
            for c in S_set:
                if c < b:
                    continue
                if div[x][b + c]:
                    return False

        # 2. x is b: check all pairs (a,c) in S_set with a < x, c >= x
        for a in S_set:
            if a >= x:
                continue
            for c in S_set:
                if c < x:
                    continue
                if div[a][x + c]:
                    return False

        # 3. x is c: check all pairs (a,b) in S_set with a < b <= x
        for a in S_set:
            if a >= x:
                continue
            for b in S_set:
                if b <= a or b > x:
                    continue
                if div[a][b + x]:
                    return False

        return True

    # 1. Greedy Initialization
    S = set(range(171, n + 1))
    candidates = list(range(170, 0, -1))

    for x in candidates:
        if time.time() > deadline - 10:
            break
        if can_add(x, S):
            S.add(x)

    best_S = set(S)
    best_size = len(S)

    # 2. Local Search
    R = set(range(1, n + 1)) - S

    improved = True
    while improved and time.time() < deadline - 15:
        improved = False

        # Try adding elements from R
        for r in list(R):
            if time.time() > deadline - 15:
                break
            if can_add(r, S):
                S.add(r)
                R.discard(r)
                improved = True

        # Try 1-for-2 swaps: remove s from S, add r1 and r2 from R
        if time.time() > deadline - 15:
            break

        s_list = sorted(S)
        for s in s_list:
            if time.time() > deadline - 15:
                break

            S_minus_s = S - {s}

            # Find all r in R that can be added to S_minus_s
            addable = []
            for r in R:
                if can_add(r, S_minus_s):
                    addable.append(r)

            # Try to add two from addable
            for i in range(len(addable)):
                if time.time() > deadline - 15:
                    break
                r1 = addable[i]
                S_temp = S_minus_s | {r1}
                for j in range(i + 1, len(addable)):
                    r2 = addable[j]
                    if can_add(r2, S_temp):
                        S = S_temp | {r2}
                        R.discard(r1)
                        R.discard(r2)
                        R.add(s)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Final polish: try adding any remaining elements
    for r in list(R):
        if time.time() > deadline - 5:
            break
        if can_add(r, S):
            S.add(r)
            R.discard(r)

    if len(S) > len(best_S):
        best_S = S

    return sorted(best_S)

# Test harness
if __name__ == "__main__":
    t0 = time.time()
    S = construct()
    t1 = time.time()
    print(f"Size: {len(S)}, Time: {t1-t0:.4f}s")
    valid = True
    L = len(S)
    for i in range(L):
        a = S[i]
        for j in range(i+1, L):
            b = S[j]
            for k in range(j, L):
                c = S[k]
                if (b+c) % a == 0:
                    valid = False
                    print(f"Invalid: {a} divides {b}+{c}")
                    break
            if not valid: break
        if not valid: break
    print("Valid" if valid else "Invalid")
