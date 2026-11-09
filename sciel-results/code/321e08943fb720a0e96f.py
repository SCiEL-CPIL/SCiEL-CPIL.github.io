import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start = time.time()
    deadline = start + 78  # Safety margin before 90s kill

    n = P['n']
    edges = P['edges']

    # Build adjacency as bitmasks
    adj_mask = [0] * n
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edge_set.add((u, v))

    total_edges = len(edge_set)
    all_mask = (1 << n) - 1

    def popcount(x):
        return bin(x).count('1')

    def bits_to_list(x):
        s = []
        i = 0
        while x:
            if x & 1:
                s.append(i)
            x >>= 1
            i += 1
        return s

    def make_edge(u, v):
        return (min(u, v), max(u, v))

    def expand_biclique(L_mask, R_mask, rem_adj, n):
        """Expand biclique to maximal via fixpoint iteration using bitmasks."""
        changed = True
        while changed:
            changed = False
            # Expand L: vertices connected to all in R
            common = all_mask
            r_bits = R_mask
            while r_bits:
                lsb = r_bits & (-r_bits)
                r = lsb.bit_length() - 1
                common &= rem_adj[r]
                r_bits ^= lsb
            new_L = common & ~(L_mask | R_mask)
            if new_L:
                L_mask |= new_L
                changed = True

            # Expand R: vertices connected to all in L
            common = all_mask
            l_bits = L_mask
            while l_bits:
                lsb = l_bits & (-l_bits)
                l = lsb.bit_length() - 1
                common &= rem_adj[l]
                l_bits ^= lsb
            new_R = common & ~(L_mask | R_mask)
            if new_R:
                R_mask |= new_R
                changed = True

        return L_mask, R_mask

    def find_bicliques_from_edges(remaining, rem_adj, n, max_seeds=30):
        """Find candidate bicliques by expanding from edge seeds."""
        candidates = []
        seen = set()
        
        # Sort edges by some heuristic
        edge_list = list(remaining)
        random.shuffle(edge_list)
        
        for (u, v) in edge_list[:max_seeds]:
            L_mask = 1 << u
            R_mask = 1 << v
            
            L_mask, R_mask = expand_biclique(L_mask, R_mask, rem_adj, n)
            
            size = popcount(L_mask) * popcount(R_mask)
            key = (L_mask, R_mask)
            if key not in seen and size > 0:
                seen.add(key)
                # Count edges covered
                covered = 0
                l_bits = L_mask
                while l_bits:
                    lsb = l_bits & (-l_bits)
                    l = lsb.bit_length() - 1
                    r_bits = R_mask
                    while r_bits:
                        rlsb = r_bits & (-r_bits)
                        r = rlsb.bit_length() - 1
                        if l != r and make_edge(l, r) in remaining:
                            covered += 1
                        r_bits ^= rlsb
                    l_bits ^= lsb
                candidates.append((L_mask, R_mask, covered))
        
        return candidates

    def find_bicliques_from_vertices(remaining, rem_adj, n, max_vertices=15):
        """Find candidate bicliques by expanding from vertex seeds."""
        candidates = []
        seen = set()
        
        # Get degrees
        deg = [0] * n
        for (u, v) in remaining:
            deg[u] += 1
            deg[v] += 1
        
        sorted_vertices = sorted(range(n), key=lambda v: -deg[v])
        
        for u in sorted_vertices[:max_vertices]:
            if deg[u] == 0:
                continue
            
            R_mask = rem_adj[u]
            if R_mask == 0:
                continue
            
            L_mask = 1 << u
            L_mask, R_mask = expand_biclique(L_mask, R_mask, rem_adj, n)
            
            size = popcount(L_mask) * popcount(R_mask)
            key = (L_mask, R_mask)
            if key not in seen and size > 0:
                seen.add(key)
                covered = 0
                l_bits = L_mask
                while l_bits:
                    lsb = l_bits & (-l_bits)
                    l = lsb.bit_length() - 1
                    r_bits = R_mask
                    while r_bits:
                        rlsb = r_bits & (-r_bits)
                        r = rlsb.bit_length() - 1
                        if l != r and make_edge(l, r) in remaining:
                            covered += 1
                        r_bits ^= rlsb
                    l_bits ^= lsb
                candidates.append((L_mask, R_mask, covered))
        
        return candidates

    def build_rem_adj(remaining, n):
        """Build adjacency bitmask for remaining edges."""
        rem_adj = [0] * n
        for (u, v) in remaining:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
        return rem_adj

    def greedy_set_cover(remaining, adj_mask, n):
        """Set cover greedy: pick biclique covering most uncovered edges."""
        bicliques = []
        uncovered = set(remaining)
        
        while uncovered and time.time() < deadline:
            rem_adj = build_rem_adj(uncovered, n)
            
            # Get candidates from both strategies
            candidates = find_bicliques_from_edges(uncovered, rem_adj, n, max_seeds=25)
            candidates += find_bicliques_from_vertices(uncovered, rem_adj, n, max_vertices=12)
            
            if not candidates:
                # Fallback: single edge
                (u, v) = next(iter(uncovered))
                bicliques.append((1 << u, 1 << v))
                uncovered.discard(make_edge(u, v))
                continue
            
            # Pick candidate covering most uncovered edges
            best_idx = 0
            best_covered = 0
            for idx, (L, R, cov) in enumerate(candidates):
                # Recount covered in current uncovered
                covered = 0
                l_bits = L
                while l_bits:
                    lsb = l_bits & (-l_bits)
                    l = lsb.bit_length() - 1
                    r_bits = R
                    while r_bits:
                        rlsb = r_bits & (-r_bits)
                        r = rlsb.bit_length() - 1
                        if l != r and make_edge(l, r) in uncovered:
                            covered += 1
                        r_bits ^= rlsb
                    l_bits ^= lsb
                if covered > best_covered:
                    best_covered = covered
                    best_idx = idx
            
            L, R, _ = candidates[best_idx]
            bicliques.append((L, R))
            
            # Remove covered edges
            to_remove = set()
            l_bits = L
            while l_bits:
                lsb = l_bits & (-l_bits)
                l = lsb.bit_length() - 1
                r_bits = R
                while r_bits:
                    rlsb = r_bits & (-r_bits)
                    r = rlsb.bit_length() - 1
                    if l != r:
                        e = make_edge(l, r)
                        if e in uncovered:
                            to_remove.add(e)
                    r_bits ^= rlsb
                l_bits ^= lsb
            uncovered -= to_remove
        
        return bicliques, uncovered

    def try_merge(bicliques, edge_set):
        """Try to merge pairs of bicliques. Returns merged list or None."""
        n_bics = len(bicliques)
        for i in range(n_bics):
            for j in range(i + 1, n_bics):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]

                candidates = []

                # Option 1: Union both sides
                newL = L1 | L2
                newR = R1 | R2
                if not (newL & newR):
                    candidates.append((newL, newR))

                # Option 2: Cross merge
                newL = L1 | R2
                newR = R1 | L2
                if not (newL & newR):
                    candidates.append((newL, newR))

                # Option 3: L2 subset of L1
                if L2 & ~L1 == 0:
                    newR = R1 | R2
                    if not (L1 & newR):
                        candidates.append((L1, newR))

                # Option 4: R2 subset of R1
                if R2 & ~R1 == 0:
                    newL = L1 | L2
                    if not (newL & R1):
                        candidates.append((newL, R1))

                # Option 5: L1 subset of L2
                if L1 & ~L2 == 0:
                    newR = R1 | R2
                    if not (L2 & newR):
                        candidates.append((L2, newR))

                # Option 6: R1 subset of R2
                if R1 & ~R2 == 0:
                    newL = L1 | L2
                    if not (newL & R2):
                        candidates.append((newL, R2))

                for newL, newR in candidates:
                    if newL == 0 or newR == 0:
                        continue
                    if newL & newR:
                        continue

                    valid = True
                    l_bits = newL
                    while l_bits and valid:
                        lsb = l_bits & (-l_bits)
                        l = lsb.bit_length() - 1
                        r_bits = newR
                        while r_bits and valid:
                            rlsb = r_bits & (-r_bits)
                            r = rlsb.bit_length() - 1
                            if l != r:
                                e = make_edge(l, r)
                                if e not in edge_set:
                                    valid = False
                            r_bits ^= rlsb
                        l_bits ^= lsb

                    if valid:
                        result = list(bicliques)
                        result.pop(j)
                        result.pop(i)
                        result.append((newL, newR))
                        return result

        return None

    def local_search(bicliques, edge_set):
        """Local search: try moving vertices between bicliques to enable merges."""
        if time.time() > deadline:
            return bicliques

        improved = True
        while improved and time.time() < deadline:
            improved = False
            current = list(bicliques)

            merged = try_merge(current, edge_set)
            if merged is not None:
                current = merged
                improved = True
                continue

            for i in range(len(current)):
                if time.time() > deadline:
                    break
                L_i, R_i = current[i]

                # Try moving a vertex from L to R
                l_bits = L_i
                while l_bits:
                    lsb = l_bits & (-l_bits)
                    v = lsb.bit_length() - 1
                    newL = L_i & ~lsb
                    newR = R_i | lsb
                    if newL == 0 or newR == 0 or (newL & newR):
                        l_bits ^= lsb
                        continue

                    valid = True
                    lb = newL
                    while lb and valid:
                        lsb2 = lb & (-lb)
                        l = lsb2.bit_length() - 1
                        rb = newR
                        while rb and valid:
                            rlsb2 = rb & (-rb)
                            r = rlsb2.bit_length() - 1
                            if l != r:
                                e = make_edge(l, r)
                                if e not in edge_set:
                                    valid = False
                            rb ^= rlsb2
                        lb ^= lsb2

                    if valid:
                        new_bicliques = list(current)
                        new_bicliques[i] = (newL, newR)
                        merged2 = try_merge(new_bicliques, edge_set)
                        if merged2 is not None and len(merged2) < len(current):
                            current = merged2
                            improved = True
                            break

                    l_bits ^= lsb
                if improved:
                    break

                # Try moving a vertex from R to L
                r_bits = R_i
                while r_bits:
                    rlsb = r_bits & (-r_bits)
                    v = rlsb.bit_length() - 1
                    newL = L_i | rlsb
                    newR = R_i & ~rlsb
                    if newL == 0 or newR == 0 or (newL & newR):
                        r_bits ^= rlsb
                        continue

                    valid = True
                    lb = newL
                    while lb and valid:
                        lsb2 = lb & (-lb)
                        l = lsb2.bit_length() - 1
                        rb = newR
                        while rb and valid:
                            rlsb2 = rb & (-rb)
                            r = rlsb2.bit_length() - 1
                            if l != r:
                                e = make_edge(l, r)
                                if e not in edge_set:
                                    valid = False
                            rb ^= rlsb2
                        lb ^= lsb2

                    if valid:
                        new_bicliques = list(current)
                        new_bicliques[i] = (newL, newR)
                        merged2 = try_merge(new_bicliques, edge_set)
                        if merged2 is not None and len(merged2) < len(current):
                            current = merged2
                            improved = True
                            break

                    r_bits ^= rlsb
                if improved:
                    break

        return current

    # Run multiple strategies
    best_result = None
    best_count = total_edges + 1

    random.seed(42)

    for restart in range(20):
        if time.time() > deadline:
            break

        # Greedy set cover
        bicliques, leftover = greedy_set_cover(set(edge_set), adj_mask, n)

        if leftover:
            for (u, v) in leftover:
                bicliques.append((1 << u, 1 << v))

        # Local search
        bicliques = local_search(bicliques, edge_set)

        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_result = bicliques

        # Try more merging on best
        improved = True
        while improved and time.time() < deadline:
            improved = False
            merged = try_merge(best_result, edge_set)
            if merged is not None:
                best_result = merged
                best_count = len(merged)
                improved = True

    # Final conversion to required format
    result = []
    for L, R in best_result:
        result.append([bits_to_list(L), bits_to_list(R)])

    # Validate the result
    covered_edges = set()
    valid = True
    for bc in result:
        if not isinstance(bc, list) or len(bc) != 2:
            valid = False
            break
        left, right = bc
        if not isinstance(left, list) or not isinstance(right, list):
            valid = False
            break
        if len(left) == 0 or len(right) == 0:
            valid = False
            break
        if set(left) & set(right):
            valid = False
            break
        for l in left:
            if not isinstance(l, int) or isinstance(l, bool):
                valid = False
                break
            for r in right:
                if not isinstance(r, int) or isinstance(r, bool):
                    valid = False
                    break
                if l == r:
                    valid = False
                    break
                e = make_edge(l, r)
                if e not in edge_set:
                    valid = False
                    break
                if e in covered_edges:
                    valid = False
                    break
                covered_edges.add(e)
        if not valid:
            break

    if not valid or covered_edges != edge_set:
        # Fallback: use simple approach
        result = []
        for u in range(n):
            neighbors = bits_to_list(adj_mask[u])
            if neighbors:
                result.append([[u], neighbors])

    return result
