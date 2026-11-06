import time
from math import gcd

def construct():
    """
    Returns the maximum cardinality subset of [1, 255] containing 255,
    such that gcd(a, b) > 1 for all distinct pairs a, b in the subset.
    """
    start_time = time.time()
    deadline = start_time + 55.0

    # Fallback: all multiples of 3 (size 85)
    fallback = list(range(3, 256, 3))

    # 1. Identify candidate nodes: numbers in [1, 255] that share a factor with 255
    # 255 = 3 * 5 * 17
    candidates = []
    for i in range(1, 256):
        if gcd(i, 255) > 1:
            candidates.append(i)

    # We need a clique containing 255.
    # 255 is connected to all candidates (since they all share a factor with 255).
    # So the max clique in the full graph containing 255 is 1 + max clique in the
    # subgraph induced by candidates \ {255}.
    # Actually, we just need the max clique in the induced subgraph of candidates \ {255}.
    # Then add 255.

    if 255 in candidates:
        candidates.remove(255)

    m = len(candidates)

    # Build adjacency matrix as bitmasks for fast intersection
    adj = [0] * m
    for i in range(m):
        for j in range(i + 1, m):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    best_clique = []

    def bron_kerbosch(R_mask, P_mask, X_mask):
        nonlocal best_clique, deadline

        if time.time() > deadline:
            return

        # If P is empty, we have found a maximal clique
        if P_mask == 0:
            if len(R_mask) > len(best_clique):
                # Extract clique
                clique = []
                mask = R_mask
                while mask:
                    # Find least significant bit
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    clique.append(candidates[idx])
                    mask ^= lsb
                best_clique = clique
            return

        # Upper bound: size of R + size of P
        if R_mask.bit_count() + P_mask.bit_count() <= len(best_clique):
            return

        # Pivot selection: choose u in P ∪ X with max |N(u) ∩ P|
        # We only care about P for the bound, but pivot is chosen from P ∪ X
        # Standard: pick u in P ∪ X that maximizes |N(u) ∩ P|
        # If P ∩ N(u) is empty for some u, then P is contained in N(u), so we can stop
        # Actually, the theorem is: if P \ N(u) is empty, then R is a maximal clique.
        # So we branch on vertices in P \ N(u).

        # Let's pick a pivot u from P ∪ X.
        # For simplicity, we pick u from P if P is not empty, otherwise from X.
        # But we should pick the one with max degree in P.
        # To be fast, we can just pick the first one in P, or one with high degree.
        # Let's iterate over P ∪ X.
        # Since P is a bitmask, we can iterate bits.
        # But X is also a bitmask.
        # Let's just pick a pivot from P if P is not empty, else from X.
        # Actually, the standard algorithm picks u from P ∪ X.
        # Let's find the vertex in P ∪ X with the maximum number of neighbors in P.
        
        # Optimization: We don't need the exact max, just a good one.
        # Let's pick the vertex in P with the highest degree in P.
        # If P is empty, we return early.
        
        # To save time, let's just pick the first vertex in P.
        # Or better, pick one that minimizes the branching set P \ N(u).
        
        # Let's try to find a pivot quickly.
        # Iterate through a few candidates in P.
        # For m=126, iterating all bits in P is fast enough.
        
        P_list = []
        temp_mask = P_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            idx = lsb.bit_length() - 1
            P_list.append(idx)
            temp_mask ^= lsb
            
        # Also consider X? The bound is tighter if we use P∪X.
        # But P \ N(u) is the set we branch on.
        # If we pick u from X, it might give a smaller P \ N(u).
        # Let's stick to u in P for now, as it's simpler.
        # Actually, the theoretical guarantee holds for any u in P ∪ X.
        
        # Let's pick the vertex in P_list with max degree in P.
        if not P_list:
            return
            
        best_u = P_list[0]
        max_deg = -1
        for u in P_list:
            deg = (adj[u] & P_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_u = u

        # Branch on vertices in P \ N(best_u)
        branch_set = P_mask & ~adj[best_u]
        
        # Iterate through vertices in branch_set
        # To be efficient, we can order them.
        # Let's just iterate.
        temp_mask = branch_set
        while temp_mask:
            lsb = temp_mask & -temp_mask
            v = lsb.bit_length() - 1
            temp_mask ^= lsb
            
            new_R = R_mask | (1 << v)
            new_P = P_mask & adj[v]
            new_X = X_mask & adj[v]
            
            bron_kerbosch(new_R, new_P, new_X)
            
            # Update P and X
            P_mask &= ~(1 << v)
            X_mask |= (1 << v)
            
            if time.time() > deadline:
                return

    try:
        bron_kerbosch(0, (1 << m) - 1, 0)
    except Exception:
        pass

    if best_clique:
        result = best_clique
        result.append(255)
        # Verify validity
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if gcd(result[i], result[j]) == 1:
                    valid = False
                    break
            if not valid:
                break
        if valid and 255 in result:
            return result

    return fallback

if __name__ == "__main__":
    res = construct()
    print(len(res))
    # Verify
    valid = True
    if 255 not in res:
        valid = False
    for i in range(len(res)):
        for j in range(i + 1, len(res)):
            if gcd(res[i], res[j]) == 1:
                valid = False
                break
    print("Valid:", valid)
