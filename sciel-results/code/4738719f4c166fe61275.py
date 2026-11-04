import sys

def construct(q, n, radius):
    if not (q == 3 and n == 7 and radius == 1):
        # Fallback for general case
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

    cover_masks = [0] * total
    dist2_masks = [0] * total
    dist1_neighbors = [[] for _ in range(total)]

    for i in range(total):
        d_i = all_digits[i]
        mask1 = 1 << i
        mask2 = 1 << i
        neigh1 = [i]
        
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
                neigh1.append(j)
        
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
        dist1_neighbors[i] = neigh1

    # Symmetry Breaking: Fix first codeword to 0
    # This is valid because the problem is symmetric under coordinate permutations and symbol permutations.
    # We can assume the all-zero vector is in the code.
    
    sys.setrecursionlimit(10000)
    node_limit = 200000
    nodes_visited = 0
    aborted = False
    best_size = 1000
    best_code = []

    def packing_lb(mask):
        # Greedy packing lower bound
        lb = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            lb += 1
            m &= ~dist2_masks[idx]
        
        # Sphere covering bound
        count = mask.bit_count()
        # Max coverage per word is 15 (1 + 7*2)
        lb_sphere = (count + 14) // 15
        
        return max(lb, lb_sphere)

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

        # Heuristic: Find the codeword that covers the most uncovered words
        best_c = -1
        best_count = -1
        # Optimization: Only consider codewords that are "close" to the uncovered set?
        # No, any codeword can cover any uncovered word if they are close.
        # But we can filter candidates: A codeword c is a candidate if (cover_masks[c] & mask) != 0
        # And we want to maximize (cover_masks[c] & mask).bit_count()
        
        # To speed up, we can precompute for each uncovered word, which codewords cover it?
        # That's the reverse index.
        # For now, iterate all 2187.
        for c in range(total):
            cnt = (mask & cover_masks[c]).bit_count()
            if cnt > best_count:
                best_count = cnt
                best_c = c
        
        if best_c == -1 or best_count == 0:
            # Should not happen if mask != 0
            return

        # Branching:
        # We pick best_c as the "pivot" logic?
        # Actually, in covering code B&B, we usually pick a specific uncovered word (pivot)
        # and branch on which codeword covers it.
        # This reduces the branching factor from 2187 to ~15.
        
        # Let's switch to the Pivot Strategy.
        # Pick an uncovered word `p`.
        # We MUST pick a codeword `c` such that `p` is in `cover_masks[c]`.
        # This is much faster.
        
        # Choose pivot: The uncovered word with the fewest candidates?
        # Or just the first one?
        # Let's pick the one that has the fewest codewords covering it.
        # Precompute `reverse_cover`: for each word w, list of codewords c that cover w.
        
        # We didn't precompute reverse_cover. Let's do it now?
        # It's 2187 * 15 entries. Small.
        
        # For the sake of the edit, I will add the reverse cover precomputation
        # and use the pivot strategy.
        
        pass # Placeholder, I will implement this in the replacement

    # Precompute reverse cover: reverse_cover[w] = list of c such that w in cover_masks[c]
    reverse_cover = [[] for _ in range(total)]
    for c in range(total):
        m = cover_masks[c]
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            reverse_cover[w].append(c)
            m ^= lsb

    # Modify solve to use pivot strategy
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

        # Pick pivot: uncovered word with fewest candidates
        lsb = mask & -mask
        pivot = lsb.bit_length() - 1
        
        # Sort candidates for this pivot by how much they cover
        candidates = reverse_cover[pivot]
        candidates.sort(key=lambda c: (mask & cover_masks[c]).bit_count(), reverse=True)

        for c in candidates:
            new_mask = mask & ~cover_masks[c]
            if new_mask == mask:
                continue
            current_code.append(c)
            solve(new_mask, depth + 1, current_code)
            current_code.pop()
            if aborted:
                return

    # Initialize with symmetry breaking
    # Fix codeword 0
    initial_code = [0]
    initial_mask = ((1 << total) - 1) & ~cover_masks[0]
    
    # Get a good initial upper bound using Greedy
    def greedy_solve(mask):
        code = []
        m = mask
        while m:
            best_c = -1
            best_count = -1
            for c in range(total):
                cnt = (m & cover_masks[c]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
            if best_c == -1: break
            code.append(best_c)
            m &= ~cover_masks[best_c]
        return code

    greedy_code = greedy_solve(initial_mask)
    best_size = 1 + len(greedy_code)
    best_code = initial_code + greedy_code

    solve(initial_mask, 1, initial_code)

    return [all_digits[idx] for idx in best_code]
