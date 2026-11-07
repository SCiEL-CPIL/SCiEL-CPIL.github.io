"""Cochromatic partition via exhaustive enumeration of homogeneous sets + B&B."""
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 85.0

    # Build adjacency as bitmasks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Complement adjacency
    full_mask = (1 << n) - 1
    comp_adj = [0] * n
    for v in range(n):
        comp_adj[v] = full_mask ^ (1 << v) ^ adj[v]

    # Enumerate ALL cliques (not just maximal) using Bron-Kerbosch
    def enumerate_all_cliques(adj_matrix):
        cliques = []
        def bronk(R, P_set, X):
            if P_set == 0 and X == 0:
                if R:  # non-empty
                    cliques.append(R)
                return
            if P_set == 0:
                return
            # Choose pivot
            union = P_set | X
            if union == 0:
                return
            u = -1
            max_deg = -1
            temp = union
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = (adj_matrix[v] & union).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    u = v
                temp ^= lsb
            candidates = P_set & ~adj_matrix[u] if u >= 0 else P_set
            while candidates:
                lsb = candidates & (-candidates)
                v = lsb.bit_length() - 1
                bronk(R | lsb, P_set & adj_matrix[v], X & adj_matrix[v])
                P_set ^= lsb
                X |= lsb
                candidates ^= lsb
        bronk(0, full_mask, 0)
        return cliques

    # Enumerate all cliques
    all_cliques = enumerate_all_cliques(adj)
    # Enumerate all independent sets (cliques in complement)
    all_indeps = enumerate_all_cliques(comp_adj)

    # All homogeneous sets
    all_sets = all_cliques + all_indeps

    # Remove duplicates and sort by size descending
    seen = set()
    unique_sets = []
    for s in all_sets:
        if s not in seen:
            seen.add(s)
            unique_sets.append(s)

    # Sort by size descending
    unique_sets.sort(key=lambda x: -x.bit_count())

    num_sets = len(unique_sets)

    # Precompute for each vertex, which sets contain it
    vertex_to_sets = [[] for _ in range(n)]
    for i, s in enumerate(unique_sets):
        temp = s
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            vertex_to_sets[v].append(i)
            temp ^= lsb

    # Branch and bound
    best_parts = None
    best_count = n + 1

    def time_check():
        return time.time() > deadline

    def solve(remaining, current_parts, count):
        nonlocal best_parts, best_count
        if time_check():
            return
        if count >= best_count:
            return
        if remaining == 0:
            best_parts = current_parts[:]
            best_count = count
            return

        # Lower bound: for each vertex in remaining, it needs at least one set
        # Better LB: maximum independent set in the "conflict" graph of remaining vertices
        # where two vertices conflict if no homogeneous set contains both
        # Simpler LB: ceil(popcount(remaining) / max_set_size)
        max_sz = 0
        for s in unique_sets:
            if (s & remaining) == s:
                sz = s.bit_count()
                if sz > max_sz:
                    max_sz = sz
        if max_sz == 0:
            return  # can't cover any vertex
        lb = (remaining.bit_count() + max_sz - 1) // max_sz

        # Stronger LB: greedy coloring lower bound on the "incompatibility" graph
        # Two vertices are incompatible if no homogeneous set contains both
        # Actually, let's use a simpler but effective LB:
        # Count how many vertices must each get their own set (isolated in compatibility)
        rem_bits = remaining
        forced = 0
        while rem_bits:
            lsb = rem_bits & (-rem_bits)
            v = lsb.bit_length() - 1
            # Check if v can be in a set with any other remaining vertex
            compat = 0
            for i in vertex_to_sets[v]:
                s = unique_sets[i]
                if (s & remaining) == s:
                    compat |= s
            compat &= remaining
            compat &= ~lsb
            if compat == 0:
                forced += 1
                rem_bits ^= lsb
            else:
                rem_bits ^= lsb

        # LB is at least forced, and at least ceil(n_rem / max_sz)
        lb = max(lb, forced)

        if count + lb >= best_count:
            return

        # Pick vertex with fewest valid sets
        best_v = -1
        min_opts = n + 1
        temp = remaining
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            cnt = 0
            for i in vertex_to_sets[v]:
                s = unique_sets[i]
                if (s & remaining) == s:
                    cnt += 1
                    if cnt >= min_opts:
                        break
            if cnt < min_opts:
                min_opts = cnt
                best_v = v
                if cnt <= 1:
                    break
            temp ^= lsb

        if best_v == -1:
            return

        v_bit = 1 << best_v

        # Try each set containing best_v that fits in remaining
        for i in vertex_to_sets[best_v]:
            s = unique_sets[i]
            if (s & remaining) != s:
                continue
            new_remaining = remaining & ~s
            # Build vertex list
            verts = []
            temp_s = s
            while temp_s:
                lsb = temp_s & (-temp_s)
                verts.append(lsb.bit_length() - 1)
                temp_s ^= lsb
            current_parts.append(verts)
            solve(new_remaining, current_parts, count + 1)
            current_parts.pop()
            if time_check():
                return

    solve(full_mask, [], 0)

    if best_parts is not None:
        result = []
        for part in best_parts:
            result.append([int(v) for v in part])
        return result

    # Fallback: greedy
    parts = []
    es = {tuple(e) for e in P['edges']}
    for v in range(n):
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                break
        else:
            parts.append([v])
    return parts
