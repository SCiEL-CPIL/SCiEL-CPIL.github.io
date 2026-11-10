import math
import time
from itertools import combinations

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255,
    with gcd(a, b) > 1 for all distinct pairs.
    Maximizes cardinality.
    """
    N = 255
    deadline = time.time() + 55.0

    # 255 = 3 * 5 * 17
    # Every element must share a factor with 255, so must be divisible by 3, 5, or 17.
    candidates = [x for x in range(1, 256) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    n_cand = len(candidates)
    cand_index = {x: i for i, x in enumerate(candidates)}
    idx_255 = cand_index[255]

    # Build adjacency as bitmasks for fast intersection
    adj_mask = [0] * n_cand
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)

    # Bron-Kerbosch with pivot for maximum clique
    best_clique = [idx_255]
    best_size = 1

    def bron_kerbosch(R_mask, P_mask, X_mask):
        nonlocal best_clique, best_size
        if time.time() > deadline:
            return
        if P_mask == 0 and X_mask == 0:
            size = bin(R_mask).count('1')
            if size > best_size:
                best_size = size
                best_clique = [i for i in range(n_cand) if (R_mask >> i) & 1]
            return

        # Upper bound: if |R| + |P| <= best_size, prune
        if bin(R_mask).count('1') + bin(P_mask).count('1') <= best_size:
            return

        # Choose pivot: vertex in P∪X with maximum neighbors in P
        PX = P_mask | X_mask
        pivot = -1
        max_common = -1
        # Iterate through vertices in PX
        temp = PX
        while temp:
            v = (temp & (-temp)).bit_length() - 1
            temp &= temp - 1
            common = P_mask & adj_mask[v]
            cnt = bin(common).count('1')
            if cnt > max_common:
                max_common = cnt
                pivot = v
            if cnt == bin(P_mask).count('1'):
                break  # Can't do better

        # Candidates are P minus neighbors of pivot
        candidates_mask = P_mask & ~(adj_mask[pivot] if pivot >= 0 else 0)

        temp = candidates_mask
        while temp:
            v = (temp & (-temp)).bit_length() - 1
            temp &= temp - 1
            # Recurse with v added to R
            new_R = R_mask | (1 << v)
            new_P = P_mask & adj_mask[v]
            new_X = X_mask & adj_mask[v]
            bron_kerbosch(new_R, new_P, new_X)
            P_mask &= ~(1 << v)
            X_mask |= (1 << v)

    # Initialize: R = {255}, P = neighbors of 255, X = empty
    initial_R = 1 << idx_255
    initial_P = adj_mask[idx_255]
    initial_X = 0
    bron_kerbosch(initial_R, initial_P, initial_X)

    # Convert back to values
    result = [candidates[i] for i in best_clique]
    result.sort()

    # Verify 255 is in the set
    if 255 not in result:
        result.append(255)

    return result
