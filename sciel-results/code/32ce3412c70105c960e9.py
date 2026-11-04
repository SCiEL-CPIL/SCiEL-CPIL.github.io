import itertools
import sys

def construct(q, n, radius):
    # Fallback for non-standard parameters to ensure validity
    if not (q == 3 and n == 7 and radius == 1):
        # Use the seed logic if possible, otherwise a simple greedy
        # The seed logic in the prompt is specific to ternary length 4 base.
        # For general case, we'll just use a basic greedy to be safe, 
        # though the task implies 3,7,1.
        
        total = q ** n
        # Precompute neighbors for greedy
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

        # Simple Greedy
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
        
        # Convert to lists
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
                neigh1.append(j)
        
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
        dist1_neighbors[i] = neigh1

    # Greedy with Symmetry Breaking (fix first codeword to 0)
    def greedy_solve():
        uncovered = (1 << total) - 1
        uncovered &= ~cover_masks[0]
        code = [0]
        while uncovered:
            best_c = -1
            best_count = -1
            for c in range(total):
                if c == 0:
                    continue
                cnt = (uncovered & cover_masks[c]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
            if best_c == -1: break
            code.append(best_c)
            uncovered &= ~cover_masks[best_c]
        return code

    initial_code = greedy_solve()
    best_size = len(initial_code)
    best_code = initial_code[:]

    sys.setrecursionlimit(10000)
    node_limit = 200000
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

        lsb = mask & -mask
        pivot_idx = lsb.bit_length() - 1
        
        candidates = dist1_neighbors[pivot_idx]
        valid_candidates = []
        for c in candidates:
            if c == 0:
                continue
            new_mask = mask & ~cover_masks[c]
            if new_mask != mask:
                cnt = (mask & cover_masks[c]).bit_count()
                valid_candidates.append((cnt, c))
                
        valid_candidates.sort(reverse=True)

        for cnt, c in valid_candidates:
            new_mask = mask & ~cover_masks[c]
            current_code.append(c)
            solve(new_mask, depth + 1, current_code)
            current_code.pop()
            if aborted:
                return

    full_mask = (1 << total) - 1
    initial_mask = full_mask & ~cover_masks[0]
    
    solve(initial_mask, 1, [0])

    return [all_digits[idx] for idx in best_code]
