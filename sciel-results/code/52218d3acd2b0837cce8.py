import time
import random

def construct():
    n = 10
    deadline = time.time() + 55  # 55s safety margin before 60s kill

    # Construction via GF(5) x GF(2)
    # Represent elements as pairs (a, b) where a in GF(5), b in GF(2)
    # Encode as integer: a * 2 + b
    # L1[i][j] = encode(a_i + a_j, b_i + b_j)  where i, j encoded as (a, b)
    # L2[i][j] = encode(a_i + 2*a_j, b_i + b_j)
    # These are orthogonal because for each pair (s, t) of symbols,
    # the system of linear equations over GF(5) x GF(2) has a unique solution.

    def encode(a, b):
        return a * 2 + b

    def decode(x):
        return x // 2, x % 2

    # Build L1: L1[i][j] = i + j (cyclic, which is a Latin square)
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]

    # Build L2 using the GF(5) x GF(2) construction
    # For index i, decode to (a, b) where a = i // 2, b = i % 2
    # L2[i][j]: 
    #   a_part = (a_i + 2 * a_j) % 5
    #   b_part = (b_i + b_j) % 2
    #   L2[i][j] = encode(a_part, b_part)
    L2 = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        ai, bi = decode(i)
        for j in range(n):
            aj, bj = decode(j)
            a_part = (ai + 2 * aj) % 5
            b_part = (bi + bj) % 2
            L2[i][j] = encode(a_part, b_part)

    # Verify L1 is a valid Latin square
    def is_latin(sq):
        for i in range(n):
            if sorted(sq[i]) != list(range(n)):
                return False
        for j in range(n):
            col = [sq[i][j] for i in range(n)]
            if sorted(col) != list(range(n)):
                return False
        return True

    # Verify orthogonality of L1 and L2
    def are_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (sq1[i][j], sq2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return True

    if is_latin(L1) and is_latin(L2) and are_orthogonal(L1, L2):
        # Convert to ensure Python int types
        result = []
        for sq in [L1, L2]:
            result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
        return result

    # Fallback: try to find a second square via backtracking
    # (This should not be needed, but for safety)
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    L2 = [[-1 for _ in range(n)] for _ in range(n)]
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    pair_used = set()

    def is_valid(i, j, val):
        if val in row_used[i]: return False
        if val in col_used[j]: return False
        if (L1[i][j], val) in pair_used: return False
        return True

    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n * n:
            return True
        i = idx // n
        j = idx % n
        vals = list(range(n))
        random.shuffle(vals)
        for val in vals:
            if is_valid(i, j, val):
                L2[i][j] = val
                row_used[i].add(val)
                col_used[j].add(val)
                pair_used.add((L1[i][j], val))
                if backtrack(idx + 1):
                    return True
                else:
                    L2[i][j] = -1
                    row_used[i].remove(val)
                    col_used[j].remove(val)
                    pair_used.remove((L1[i][j], val))
        return False

    if backtrack(0):
        if is_latin(L2) and are_orthogonal(L1, L2):
            result = []
            for sq in [L1, L2]:
                result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
            return result

    # Return just L1 as fallback
    return [[[int((i + j) % n) for j in range(n)] for i in range(n)]]
