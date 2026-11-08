import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Rank = 105 - (number of disjoint 2x2x2 Strassen blocks).
    Uses a weighted greedy heuristic (Set Cover style) to find a strong initial
    packing of disjoint Strassen blocks, followed by local search (SA + 2-opt)
    to maximize the number of blocks.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21
    total = n * m * p  # 105

    # Precompute all blocks: each = (params, frozenset of (i,j,k), bitmask)
    blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            mask = 0
                            triples = []
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        idx = (i * m + j) * p + k
                                        mask |= (1 << idx)
                                        triples.append((i, j, k))
                            blocks.append(((i1, i2, j1, j2, k1, k2), frozenset(triples), mask))

    B = len(blocks)  # 168

    # comp[a] = bitmask of blocks b (b != a) that are DISJOINT from a
    comp = [0] * B
    for a in range(B):
        ma = blocks[a][2]
        bits = 0
        for b in range(B):
            if a != b and (ma & blocks[b][2]) == 0:
                bits |= (1 << b)
        comp[a] = bits

    def popcount(x):
        return bin(x).count('1')

    def greedy_weighted(rng):
        """
        Greedy heuristic based on Set Cover logic.
        At each step, select the block that minimizes the "cost" of reducing
        the future search space. Cost is approximated by the number of blocks
        that become incompatible.
        """
        used = 0
        res = []
        remaining = list(range(B))
        
        # Precompute conflict counts for initial state? 
        # Actually, we just need to evaluate candidates dynamically.
        
        while remaining:
            best_a = None
            best_score = -1
            
            # To speed up, we can sample if remaining is large, but B=168 is small.
            for a in remaining:
                if (used & blocks[a][2]) != 0:
                    continue
                
                # Calculate the "cost" of picking a:
                # How many other remaining blocks become incompatible?
                # A block d is incompatible with a if (blocks[a][2] & blocks[d][2]) != 0
                # Which is equivalent to: d is NOT in comp[a]
                
                # Let's count how many remaining blocks are compatible with a.
                # compatible_mask = comp[a] & ~used
                # We want to maximize the number of remaining blocks that are compatible.
                
                compat_mask = comp[a] & ~used
                # Count how many of the 'remaining' blocks are in compat_mask
                # We can iterate through remaining or use bit operations if we had a bitmask for remaining.
                # Let's maintain a bitmask for remaining to make this fast.
                
                # For now, simple loop is fine for B=168.
                count_compat = 0
                for d in remaining:
                    if d == a:
                        continue
                    if (used & blocks[d][2]) == 0: # d is still available
                        if (blocks[a][2] & blocks[d][2]) == 0:
                            count_compat += 1
                            
                # We want to maximize count_compat (keep options open)
                # Or minimize (len(remaining) - count_compat)
                
                # Add a small random tie-breaker to diversify
                score = count_compat + rng.random() * 0.1
                
                if score > best_score:
                    best_score = score
                    best_a = a
            
            if best_a is None:
                break
            
            used |= blocks[best_a][2]
            res.append(best_a)
            remaining.remove(best_a)
            
        return res

    def local_search(packing, max_iters=500, rng=None):
        """Fast local search with 1-for-1, 2-for-1, 2-for-2, 2-opt + SA."""
        if rng is None:
            rng = random.Random(hash(tuple(sorted(packing))) % (2**31))

        used = 0
        in_pack = set(packing)
        for b in packing:
            used |= blocks[b][2]

        cur_len = len(packing)
        best_packing = list(packing)
        best_len = cur_len
        best_used = used

        temperature = 5.0
        cooling = 0.98
        min_temp = 0.01

        for iteration in range(max_iters):
            if time.time() > deadline - 8.0:
                break
            if temperature < min_temp:
                break

            improved = False
            move_type = rng.choice(['11', '22', '2opt'])
            # '21' (remove 2 add 1) reduces rank, so we only do it if we are stuck 
            # and hope to escape, but it's risky. Let's stick to moves that maintain or improve.
            # Actually, standard SA allows worsening moves.

            if move_type == '11':
                # remove one, add one
                for c in packing:
                    used2 = used & ~blocks[c][2]
                    cand_mask = comp[c] & ~used2
                    for d in in_pack:
                        if d != c:
                            cand_mask &= ~(1 << d)
                    if cand_mask == 0:
                        continue
                    cm = cand_mask
                    chosen = []
                    while cm:
                        low = cm & (-cm)
                        chosen.append(low.bit_length() - 1)
                        cm ^= low
                        if len(chosen) >= 20:
                            break
                    a = chosen[rng.randrange(len(chosen))]
                    new_used = used2 | blocks[a][2]
                    # Accept if same length (neutral) or if it helps (it doesn't change length)
                    # In SA, we accept neutral moves to explore.
                    if rng.random() < temperature * 0.3:
                        packing = [x for x in packing if x != c] + [a]
                        in_pack = set(packing)
                        used = new_used
                        improved = True
                        break
                        
            elif move_type == '22':
                # remove two, add two
                L = len(packing)
                if L < 2:
                    continue
                i1 = rng.randrange(L)
                i2 = rng.randrange(L)
                if i1 == i2:
                    continue
                c1 = packing[i1]
                c2 = packing[i2]
                used2 = used & ~blocks[c1][2] & ~blocks[c2][2]
                cand_mask = comp[c1] & comp[c2] & ~used2
                for d in in_pack:
                    if d != c1 and d != c2:
                        cand_mask &= ~(1 << d)
                cm = cand_mask
                chosen = []
                while cm:
                    low = cm & (-cm)
                    chosen.append(low.bit_length() - 1)
                    cm ^= low
                    if len(chosen) >= 30:
                        break
                if len(chosen) < 2:
                    continue
                a = chosen[rng.randrange(len(chosen))]
                used3 = used2 | blocks[a][2]
                cand_mask2 = comp[a] & ~used3
                for d in in_pack:
                    if d != c1 and d != c2:
                        cand_mask2 &= ~(1 << d)
                cm2 = cand_mask2
                chosen2 = []
                while cm2:
                    low = cm2 & (-cm2)
                    chosen2.append(low.bit_length() - 1)
                    cm2 ^= low
                    if len(chosen2) >= 30:
                        break
                if not chosen2:
                    continue
                b = chosen2[rng.randrange(len(chosen2))]
                new_used = used3 | blocks[b][2]
                if rng.random() < temperature * 0.2:
                    packing = [x for x in packing if x != c1 and x != c2] + [a, b]
                    in_pack = set(packing)
                    used = new_used
                    improved = True
                    break
            else:  # 2opt: swap one in for one out (reposition)
                for c in packing:
                    used2 = used & ~blocks[c][2]
                    cand_mask = comp[c] & ~used2
                    for d in in_pack:
                        if d != c:
                            cand_mask &= ~(1 << d)
                    if cand_mask == 0:
                        continue
                    cm = cand_mask
                    chosen = []
                    while cm:
                        low = cm & (-cm)
                        chosen.append(low.bit_length() - 1)
                        cm ^= low
                        if len(chosen) >= 20:
                            break
                    a = chosen[rng.randrange(len(chosen))]
                    new_used = used2 | blocks[a][2]
                    if rng.random() < temperature * 0.15:
                        packing = [x for x in packing if x != c] + [a]
                        in_pack = set(packing)
                        used = new_used
                        improved = True
                        break

            if improved:
                if len(packing) > best_len:
                    best_len = len(packing)
                    best_packing = list(packing)
                    best_used = used
            else:
                temperature *= cooling

        # Greedy top-up
        used = 0
        in_best = set(best_packing)
        for b in best_packing:
            used |= blocks[b][2]
        order = list(range(B))
        rng.shuffle(order)
        for b in order:
            if b not in in_best and (used & blocks[b][2]) == 0:
                best_packing.append(b)
                in_best.add(b)
                used |= blocks[b][2]

        return best_packing

    best = []
    best_len = 0
    seen_packing = set()

    rng = random.Random(42)

    # Phase 0: Weighted Greedy (Set Cover style)
    for _ in range(5):
        cand = greedy_weighted(rng)
        key = frozenset(cand)
        if key not in seen_packing:
            seen_packing.add(key)
            if len(cand) > best_len:
                best = list(cand)
                best_len = len(cand)
        
        # Local search on the greedy result
        cand_ls = local_search(list(cand), max_iters=300, rng=rng)
        key_ls = frozenset(cand_ls)
        if key_ls not in seen_packing:
            seen_packing.add(key_ls)
            if len(cand_ls) > best_len:
                best = list(cand_ls)
                best_len = len(cand_ls)
                if best_len == 13: # Theoretical max guess, break early if found
                    break

    # Phase 1: Random Greedy + Local Search
    while time.time() < deadline - 20.0:
        order = list(range(B))
        rng.shuffle(order)
        used = 0
        cand = []
        for b in order:
            if (used & blocks[b][2]) == 0:
                used |= blocks[b][2]
                cand.append(b)
        
        key = frozenset(cand)
        if key in seen_packing:
            continue
        seen_packing.add(key)
        
        if len(cand) > best_len:
            best = list(cand)
            best_len = len(cand)
        
        # Only run local search if this candidate is competitive
        if len(cand) >= best_len - 1:
            cand_ls = local_search(list(cand), max_iters=200, rng=rng)
            key_ls = frozenset(cand_ls)
            if key_ls not in seen_packing:
                seen_packing.add(key_ls)
                if len(cand_ls) > best_len:
                    best = list(cand_ls)
                    best_len = len(cand_ls)
                    if best_len == 13:
                        break

    # Phase 2: Perturb Best
    if best_len < 13 and time.time() < deadline - 10.0:
        for _ in range(1000):
            if time.time() > deadline - 5.0:
                break
            perturbed = list(best)
            # Remove 1-2 random blocks
            for _ in range(rng.randint(1, 2)):
                if perturbed:
                    idx = rng.randint(0, len(perturbed) - 1)
                    perturbed.pop(idx)
            
            # Fill greedily
            used = 0
            for b in perturbed:
                used |= blocks[b][2]
            order = list(range(B))
            rng.shuffle(order)
            for b in order:
                if (used & blocks[b][2]) == 0:
                    perturbed.append(b)
                    used |= blocks[b][2]
            
            key = frozenset(perturbed)
            if key in seen_packing:
                continue
            seen_packing.add(key)
            
            cand_ls = local_search(list(perturbed), max_iters=150, rng=rng)
            key_ls = frozenset(cand_ls)
            if key_ls not in seen_packing:
                seen_packing.add(key_ls)
                if len(cand_ls) > best_len:
                    best = list(cand_ls)
                    best_len = len(cand_ls)
                    if best_len == 13:
                        break

    # Build matrices
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2

        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2

        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2

        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    covered = set()
    for bidx in best:
        i1, i2, j1, j2, k1, k2 = blocks[bidx][0]
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for t in blocks[bidx][1]:
            covered.add(t)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)

    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
