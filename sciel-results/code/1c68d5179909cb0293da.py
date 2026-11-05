import itertools
import sys

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if not (q == 3 and n == 7 and radius == 1):
        total = q ** n
        def get_neighbors(idx):
            digits = []
            tmp = idx
            for _ in range(n):
                digits.append(tmp % q)
                tmp //= q
            neighbors = [idx]
            for k in range(n):
                orig = digits[k]
                for s in range(q):
                    if s == orig: continue
                    new_d = digits[:]
                    new_d[k] = s
                    j = 0
                    mult = 1
                    for val in new_d:
                        j += val * mult
                        mult *= q
                    neighbors.append(j)
            return neighbors

        cover_masks = [0] * total
        for i in range(total):
            mask = 0
            for nb in get_neighbors(i):
                mask |= (1 << nb)
            cover_masks[i] = mask

        uncovered = (1 << total) - 1
        code = []
        while uncovered:
            best_c = -1
            best_count = -1
            for c in range(total):
                cnt = (uncovered & cover_masks[c]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
            code.append(best_c)
            uncovered &= ~cover_masks[best_c]
        
        def idx_to_list(idx):
            res = []
            for _ in range(n):
                res.append(idx % q)
                idx //= q
            return res
        
        return [idx_to_list(i) for i in code]

    # Specific solver for q=3, n=7, radius=1
    total = 2187
    all_digits = []
    for i in range(total):
        d = []
        tmp = i
        for _ in range(7):
            d.append(tmp % 3)
            tmp //= 3
        all_digits.append(d)

    # Precompute cover masks (dist <= 1)
    cover_masks = [0] * total
    # Precompute dist2 masks (for lower bound packing)
    dist2_masks = [0] * total
    
    for i in range(total):
        d_i = all_digits[i]
        mask1 = 1 << i
        mask2 = 1 << i
        
        # Dist 1
        for k in range(7):
            orig = d_i[k]
            for s in range(3):
                if s == orig: continue
                new_d = d_i[:]
                new_d[k] = s
                j = 0
                mult = 1
                for val in new_d:
                    j += val * mult
                    mult *= 3
                mask1 |= (1 << j)
                mask2 |= (1 << j)
        
        # Dist 2
        for k1 in range(7):
            for k2 in range(k1 + 1, 7):
                orig1 = d_i[k1]
                orig2 = d_i[k2]
                for s1 in range(3):
                    if s1 == orig1: continue
                    for s2 in range(3):
                        if s2 == orig2: continue
                        new_d = d_i[:]
                        new_d[k1] = s1
                        new_d[k2] = s2
                        j = 0
                        mult = 1
                        for val in new_d:
                            j += val * mult
                            mult *= 3
                        mask2 |= (1 << j)
        
        cover_masks[i] = mask1
        dist2_masks[i] = mask2

    # Greedy to find initial UB
    # We force the greedy to pick 0 first to align with WLOG
    def greedy_solve():
        uncovered = (1 << total) - 1
        code = []
        # Force 0
        code.append(0)
        uncovered &= ~cover_masks[0]
        
        while uncovered:
            best_c = -1
            best_count = -1
            for c in range(total):
                cnt = (uncovered & cover_masks[c]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
            code.append(best_c)
            uncovered &= ~cover_masks[best_c]
        return code

    initial_code = greedy_solve()
    best_size = len(initial_code)
    best_code = initial_code

    # B&B
    sys.setrecursionlimit(10000)
    node_limit = 200000 # Increase limit slightly
    nodes_visited = 0
    aborted = False

    def packing_lb(mask):
        lb = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            lb += 1
            m &= ~dist2_masks[idx]
        return lb

    def solve(mask, depth, current_code):
        nonlocal best_size, best_code, nodes_visited, aborted
        if aborted:
            return
        nodes_visited += 1
        if nodes_visited > node_limit:
            aborted = True
            return

        if mask == 0:
            if depth < best_size:
                best_size = depth
                best_code = current_code[:]
            return

        lb = packing_lb(mask)
        if depth + lb >= best_size:
            return

        # Pivot: first uncovered word
        lsb = mask & -mask
        pivot_idx = lsb.bit_length() - 1
        
        # Candidates: words that cover pivot
        # We can restrict candidates to those that cover pivot
        # Precompute list of coverers for each index?
        # coverers[i] = list of c such that i in cover_masks[c]
        # This is the reverse of cover_masks.
        
        candidates = []
        # To speed up, we can iterate all c and check if pivot is covered
        # But we can precompute coverers
        # For now, iterate all and filter
        for c in range(total):
            if (cover_masks[c] & lsb):
                candidates.append(c)
                
        # Sort by coverage of remaining mask
        candidates.sort(key=lambda c: (mask & cover_masks[c]).bit_count(), reverse=True)

        # Symmetry/Pruning: 
        # If we have already tried a candidate c1, and c2 covers a subset of c1's coverage in the current mask, skip c2?
        # This is expensive to check.
        
        for c in candidates:
            new_mask = mask & ~cover_masks[c]
            if new_mask == mask:
                continue
            current_code.append(c)
            solve(new_mask, depth + 1, current_code)
            current_code.pop()
            if aborted:
                return

    # WLOG: Start with 0
    full_mask = (1 << total) - 1
    initial_mask = full_mask & ~cover_masks[0]
    
    # Initial best_size from greedy (which included 0)
    # If greedy didn't include 0, we force it in the B&B start
    # The greedy above forces 0.
    
    solve(initial_mask, 1, [0])

    return [all_digits[idx] for idx in best_code]
