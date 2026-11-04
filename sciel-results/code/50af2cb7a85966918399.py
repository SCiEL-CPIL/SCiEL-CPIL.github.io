import sys

def construct(q, n, radius):
    # Fallback for non-standard parameters to ensure validity
    if not (q == 3 and n == 7 and radius == 1):
        total = q ** n
        
        def idx_to_list(idx):
            res = []
            for _ in range(n):
                res.append(idx % q)
                idx //= q
            return res

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

    # Greedy to find initial UB
    def greedy_solve():
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
        return code

    initial_code = greedy_solve()
    best_size = len(initial_code)
    best_code = initial_code

    # B&B with Symmetry Breaking
    # We fix the first codeword to be 0 (all zeros)
    # This is valid because the problem is symmetric under coordinate and symbol permutations.
    # Any optimal code can be transformed to contain the all-zero word.
    
    sys.setrecursionlimit(10000)
    node_limit = 200000 # Increased limit to allow more search time
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
        
        # Filter candidates: only those that cover the pivot (which they all do by definition of dist1_neighbors)
        # Sort by coverage count in the current mask
        candidates.sort(key=lambda c: (mask & cover_masks[c]).bit_count(), reverse=True)
        
        # Dominance Pruning:
        # If candidate A's coverage of uncovered points is a subset of candidate B's,
        # and B is processed before A, we can skip A.
        # We process in order of decreasing coverage, so larger sets come first.
        # We maintain a list of seen coverage masks.
        # If (mask & cover_masks[c]) is a subset of any previously seen mask, skip.
        
        seen_masks = []
        
        for c in candidates:
            c_mask = mask & cover_masks[c]
            
            # Check if c_mask is subset of any seen mask
            dominated = False
            for s_mask in seen_masks:
                if (c_mask & s_mask) == c_mask:
                    dominated = True
                    break
            
            if dominated:
                continue
            
            # Add to seen (optional optimization: only keep maximal ones, but list is small)
            seen_masks.append(c_mask)
            
            new_mask = mask & ~cover_masks[c]
            current_code.append(c)
            solve(new_mask, depth + 1, current_code)
            current_code.pop()
            if aborted:
                return

    # Initialize with 0 as the first codeword
    first_codeword = 0
    initial_mask = (1 << total) - 1
    initial_mask &= ~cover_masks[first_codeword]
    
    # The depth starts at 1 because we already picked one codeword
    solve(initial_mask, 1, [first_codeword])

    return [all_digits[idx] for idx in best_code]
