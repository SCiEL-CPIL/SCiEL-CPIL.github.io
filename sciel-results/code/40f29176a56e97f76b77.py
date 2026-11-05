"""Improved program: bitmask-based maximum clique search with multi-start hubs."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin under 90s
    N = 512

    # Precompute compatibility: compat[a] is bitmask of b in 1..N with lcm(a,b) <= N
    # Index: number x maps to bit (x-1)
    compat = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        for b in range(1, N + 1):
            if a * b // gcd(a, b) <= N:
                mask |= (1 << (b - 1))
        compat[a] = mask

    # Precompute popcount for 16-bit chunks
    popcount16 = [bin(i).count('1') for i in range(1 << 16)]

    def popcount(mask):
        """Fast popcount using precomputed table for 16-bit chunks."""
        total = 0
        while mask:
            total += popcount16[mask & 0xFFFF]
            mask >>= 16
        return total

    def get_divisors(x):
        divs = []
        for d in range(1, int(x**0.5) + 1):
            if x % d == 0:
                divs.append(d)
                if d != x // d:
                    divs.append(x // d)
        return divs

    def count_divisors(x):
        return len(get_divisors(x))

    # Score each number by its divisor count (proxy for hub quality)
    div_counts = [(count_divisors(k), k) for k in range(1, N + 1)]
    div_counts.sort(key=lambda x: (-x[0], x[1]))

    # Greedy expansion: given a starting set (as bitmask), add compatible numbers
    def expand(start_mask):
        s = start_mask
        # Numbers already in set
        in_set = [False] * (N + 1)
        for x in range(1, N + 1):
            if (s >> (x - 1)) & 1:
                in_set[x] = True

        # Find all numbers compatible with entire set
        candidates_mask = (1 << N) - 1
        for x in range(1, N + 1):
            if in_set[x]:
                candidates_mask &= compat[x]

        # Greedy: repeatedly add the number with most connections among remaining candidates
        changed = True
        while changed:
            changed = False
            # Among candidates not in set, find best
            best_x = 0
            best_score = -1
            for x in range(1, N + 1):
                if in_set[x]:
                    continue
                if not ((candidates_mask >> (x - 1)) & 1):
                    continue
                # Score: number of other candidates it's compatible with
                score = popcount(candidates_mask & compat[x])
                if score > best_score:
                    best_score = score
                    best_x = x
            if best_x == 0:
                break
            s |= (1 << (best_x - 1))
            in_set[best_x] = True
            candidates_mask &= compat[best_x]
            changed = True
        return s

    # Local search: 1-swap and 2-swap improvements
    def local_search(s):
        improved = True
        while improved:
            if time.time() > deadline - 3:
                break
            improved = False
            in_set = [False] * (N + 1)
            for x in range(1, N + 1):
                if (s >> (x - 1)) & 1:
                    in_set[x] = True

            # Compute candidates compatible with all in set
            candidates_mask = (1 << N) - 1
            for x in range(1, N + 1):
                if in_set[x]:
                    candidates_mask &= compat[x]

            # Try 1-swap: remove one element, add a better one
            for rem in range(1, N + 1):
                if not in_set[rem]:
                    continue
                s_without = s & ~(1 << (rem - 1))
                # Candidates compatible with s_without
                cm = (1 << N) - 1
                for x in range(1, N + 1):
                    if (s_without >> (x - 1)) & 1:
                        cm &= compat[x]
                # Find best addable
                best_add = 0
                best_score = -1
                for add in range(1, N + 1):
                    if (s >> (add - 1)) & 1:
                        continue
                    if not ((cm >> (add - 1)) & 1):
                        continue
                    score = popcount(cm & compat[add])
                    if score > best_score:
                        best_score = score
                        best_add = add
                if best_add != 0:
                    new_s = s_without | (1 << (best_add - 1))
                    if popcount(new_s) > popcount(s):
                        s = new_s
                        improved = True
                        break
                if time.time() > deadline - 3:
                    break

            # Try 2-add: add two compatible elements at once
            if not improved:
                cm = (1 << N) - 1
                for x in range(1, N + 1):
                    if in_set[x]:
                        cm &= compat[x]
                addable = [x for x in range(1, N + 1)
                           if not in_set[x] and ((cm >> (x - 1)) & 1)]
                if len(addable) >= 2:
                    for i in range(len(addable)):
                        if time.time() > deadline - 3:
                            break
                        for j in range(i + 1, len(addable)):
                            a, b = addable[i], addable[j]
                            if (compat[a] >> (b - 1)) & 1:
                                new_s = s | (1 << (a - 1)) | (1 << (b - 1))
                                if popcount(new_s) > popcount(s):
                                    s = new_s
                                    improved = True
                                    break
                        if improved:
                            break
        return s

    # Optimized Branch and Bound maximum clique
    def max_clique_bnb(candidates, timeout=2.0):
        """Find maximum clique in the subgraph induced by candidates.
        candidates is a list of numbers (1..N). Returns the clique as a bitmask.
        Uses Tomita's algorithm with coloring bound.
        """
        n = len(candidates)
        if n == 0:
            return 0

        # Build adjacency among candidates (0-based indices)
        # adj[i] is bitmask of neighbors of i in 0..n-1
        adj = [0] * n
        for i in range(n):
            ai = candidates[i]
            for j in range(i + 1, n):
                aj = candidates[j]
                if (compat[ai] >> (aj - 1)) & 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)

        # Order vertices by degree descending to improve pruning
        degrees = [popcount(adj[i]) for i in range(n)]
        order = sorted(range(n), key=lambda i: -degrees[i])
        
        # Remap adjacency to new order
        adj2 = [0] * n
        # map_old_to_new: old_index -> new_index
        map_old_to_new = [0] * n
        for new_idx, old_idx in enumerate(order):
            map_old_to_new[old_idx] = new_idx
            
        for i in range(n):
            old_i = order[i]
            mask = adj[old_i]
            new_mask = 0
            for j in range(n):
                if (mask >> j) & 1:
                    new_j = map_old_to_new[j]
                    new_mask |= (1 << new_j)
            adj2[i] = new_mask

        best_size = 0
        best_clique_mask = 0 # Bitmask of vertices in the best clique (0-based)

        # Coloring bound: returns list of (vertex, color) sorted by color
        # We use a simple greedy coloring.
        def color_sort(vertices_mask):
            # vertices_mask is a bitmask of vertices to color
            # Returns a list of vertices sorted by color (descending color first for branching? 
            # Actually, standard Tomita sorts by color ascending, and branches in that order.
            # The bound is the max color.
            # Let's return (sorted_list, max_color)
            
            # Extract vertices
            verts = []
            m = vertices_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                verts.append(idx)
                m ^= lsb
            
            if not verts:
                return [], 0
            
            colors = [0] * n
            uncolored = verts[:]
            color = 0
            # To make it faster, we can just assign colors greedily
            # Standard Tomita: 
            # 1. Initialize colors to 0
            # 2. For each vertex in order, assign smallest available color
            
            # Let's use a simple O(n^2) greedy coloring
            # Sort by degree? No, the input order is already by degree.
            # But we need to color the subgraph induced by 'verts'.
            
            # Greedy coloring:
            # available_colors = [False] * (len(verts)+1)
            
            # A faster heuristic:
            # Iterate through verts. For each v, check neighbors in verts that are already colored.
            # Find smallest color not used by neighbors.
            
            # This is still O(n^2).
            
            colored_order = []
            for v in verts:
                # Check neighbors of v that are in verts
                neigh_mask = adj2[v] & vertices_mask
                used_colors = set()
                nm = neigh_mask
                while nm:
                    lsb = nm & -nm
                    ni = lsb.bit_length() - 1
                    if colors[ni] > 0:
                        used_colors.add(colors[ni])
                    nm ^= lsb
                
                c = 1
                while c in used_colors:
                    c += 1
                colors[v] = c
                colored_order.append((v, c))
            
            # Sort by color descending? 
            # Tomita's algorithm: 
            # The list P is sorted by color. 
            # The bound is the max color.
            # We branch on vertices in the order of the list.
            # Usually, we want to branch on high color vertices first? 
            # Actually, the bound is max_color. 
            # If we branch on v, the new bound is max_color of the remaining set.
            
            # Sort by color descending for Tomita's bound
            colored_order.sort(key=lambda x: -x[1])
            
            max_c = colored_order[0][1] if colored_order else 0
            
            return colored_order, max_c

        def bronk(clique_mask, clique_size, candidates_mask):
            nonlocal best_size, best_clique_mask
            
            if time.time() > timeout:
                return
            
            if candidates_mask == 0:
                if clique_size > best_size:
                    best_size = clique_size
                    best_clique_mask = clique_mask
                return
            
            # Get coloring and bound
            colored, bound = color_sort(candidates_mask)
            
            # If current clique size + bound <= best_size, prune
            if clique_size + bound <= best_size:
                return
            
            # Branch on vertices in colored order
            for v, c in colored:
                if time.time() > timeout:
                    return
                
                # If v is no longer in candidates_mask, skip
                if not (candidates_mask & (1 << v)):
                    continue
                
                # Pruning bound
                if clique_size + c <= best_size:
                    break
                
                # Add v to clique
                new_clique = clique_mask | (1 << v)
                
                # New candidates: neighbors of v in candidates_mask, excluding v
                new_candidates = candidates_mask & adj2[v]
                new_candidates &= ~(1 << v)
                
                bronk(new_clique, clique_size + 1, new_candidates)
                
                # Remove v from candidates
                candidates_mask &= ~(1 << v)
                
                # Optional: if candidates_mask becomes empty, break
                if candidates_mask == 0:
                    break

        # Start BnB
        full_set = (1 << n) - 1
        bronk(0, 0, full_set)

        # Convert best_clique_mask (0-based indices) back to original numbers
        result_mask = 0
        m = best_clique_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            num = candidates[order[idx]]
            result_mask |= (1 << (num - 1))
            m ^= lsb
        return result_mask

    best_mask = 0
    best_size = 0

    # Multi-start: try each top hub's divisor set
    for cnt, hub in div_counts[:30]:
        if time.time() > deadline - 10:
            break
        divs = get_divisors(hub)
        start_mask = 0
        for d in divs:
            if d <= N:
                start_mask |= (1 << (d - 1))
        expanded = expand(start_mask)
        expanded = local_search(expanded)
        sz = popcount(expanded)
        if sz > best_size:
            best_size = sz
            best_mask = expanded

    # Also try single-element starts for high-degree nodes
    degree = [(popcount(compat[x]), x) for x in range(1, N + 1)]
    degree.sort(key=lambda x: (-x[0], x[1]))
    for deg, x in degree[:20]:
        if time.time() > deadline - 10:
            break
        start_mask = 1 << (x - 1)
        expanded = expand(start_mask)
        expanded = local_search(expanded)
        sz = popcount(expanded)
        if sz > best_size:
            best_size = sz
            best_mask = expanded

    # Try branch-and-bound on the full candidate set (all numbers)
    # Allocate remaining time to BnB
    elapsed = time.time() - start_time
    remaining_time = deadline - 5 - elapsed
    if remaining_time > 2.0:
        all_numbers = list(range(1, N + 1))
        bnb_result = max_clique_bnb(all_numbers, timeout=remaining_time)
        bnb_size = popcount(bnb_result)
        if bnb_size > best_size:
            best_size = bnb_size
            best_mask = bnb_result

    # Convert to list
    result = []
    for x in range(1, N + 1):
        if (best_mask >> (x - 1)) & 1:
            result.append(x)

    if not result:
        result = [1]

    return result
# EVOLVE-BLOCK-END
