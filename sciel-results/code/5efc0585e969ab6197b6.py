import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}


def construct():
    n = P['n']
    edges = P['edges']

    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    all_mask = (1 << n) - 1
    non_adj = [0] * n
    for v in range(n):
        non_adj[v] = all_mask & ~(adj[v] | (1 << v))

    # Precompute clique and independent-set feasibility for every subset
    # clique_ok[S]: S is a clique
    # indep_ok[S]: S is an independent set
    clique_ok = [False] * (1 << n)
    indep_ok = [False] * (1 << n)
    clique_ok[0] = True
    indep_ok[0] = True
    for s in range(1, 1 << n):
        # lowest set bit
        lb = s & (-s)
        v = lb.bit_length() - 1
        rest = s ^ lb
        clique_ok[s] = clique_ok[rest] and (adj[v] & rest) == rest
        indep_ok[s] = indep_ok[rest] and (non_adj[v] & rest) == rest

    # Precompute, for each subset S, whether it can be a valid part (clique or independent)
    valid_part = [False] * (1 << n)
    for s in range(1 << n):
        valid_part[s] = clique_ok[s] or indep_ok[s]

    # For a valid part S and a vertex v not in S, can we add v to S?
    # We need S∪{v} to be a valid part.
    # Precompute addable[S][v] as a bitmask of vertices v such that S∪{v} is valid.
    # But that's expensive. Instead, during search, we check valid_part[S | (1<<v)].

    # Order vertices by degree descending for better pruning
    order = sorted(range(n), key=lambda v: -bin(adj[v]).count('1'))
    order_idx = [0] * n
    for i, v in enumerate(order):
        order_idx[v] = i

    deadline = time.time() + 85

    def solve_k(k):
        """
        Try to partition vertices into k parts.
        Each part must be a clique or an independent set.
        """
        # State: parts[0..k-1], each is a bitmask
        # assigned: bitmask of all assigned vertices
        parts = [0] * k
        assigned = 0

        # Symmetry breaking: parts are ordered. We only allow placing v into part i
        # if i is the first empty part (when parts[i]==0) or parts[i]!=0.
        # Additionally, for non-empty parts, we break symmetry by requiring that
        # if parts[i] == parts[j] for i<j, we skip j. But since parts are subsets,
        # equality is rare. A simpler symmetry break: only place into the first empty part.

        # Also: if two parts have the same set of vertices, skip the later one.
        # We track used part masks to break symmetry: only allow a part mask if it's the first time.
        # This is complex. Let's use a simpler approach: enforce that parts[0] <= parts[1] <= ... in some ordering.
        # Actually, the simplest: when placing v, if part i is empty, skip if any earlier part is empty.
        # For non-empty parts, no additional symmetry break needed for now.

        # Additional pruning: if a part is already determined to be clique or independent,
        # we can pre-filter.

        # Let's use a recursive backtracking with the following state:
        # - parts: list of k bitmasks
        # - idx: next vertex in order to place
        # - assigned: bitmask of assigned vertices

        # Precompute for each part mask, what type it is: 0=empty, 1=clique-only, 2=indep-only, 3=both (size 0 or 1)
        # For size 0: both. For size 1: both. For size >=2: exactly one.
        def part_type(s):
            if s == 0:
                return 3  # both
            if s & (s - 1) == 0:  # single vertex
                return 3
            if clique_ok[s]:
                return 1
            if indep_ok[s]:
                return 2
            return 0  # invalid

        # For a part with mask s and type t, compute the set of vertices that can be added
        # such that s∪{v} is still valid.
        # If t==1 (clique): v must be adjacent to all in s.
        # If t==2 (indep): v must be non-adjacent to all in s.
        # If t==3 (size 0 or 1): any v is fine (s∪{v} will be size 1 or 2, and we check validity).
        #   Actually for size 1, s∪{v} is size 2, which is either clique or indep. Both are valid.
        #   So any v not in s can be added.
        # We precompute add_mask[s] for each valid s.

        # This is (1<<n) entries, each an n-bit mask. That's 2^14 * 14 bits, manageable.
        add_mask = [0] * (1 << n)
        for s in range(1 << n):
            if not valid_part[s]:
                continue
            t = part_type(s)
            if t == 3:
                # Can add any vertex not in s
                add_mask[s] = all_mask ^ s
            elif t == 1:
                # Must be adjacent to all in s
                m = 0
                for v in range(n):
                    if s & (1 << v):
                        m &= adj[v]
                add_mask[s] = m & (all_mask ^ s)
            elif t == 2:
                # Must be non-adjacent to all in s
                m = 0
                for v in range(n):
                    if s & (1 << v):
                        m &= non_adj[v]
                add_mask[s] = m & (all_mask ^ s)
            else:
                add_mask[s] = 0

        # Now backtrack
        best_result = [None]

        def backtrack(idx):
            if time.time() > deadline:
                return False

            if idx == n:
                # All vertices assigned
                res = []
                for i in range(k):
                    if parts[i]:
                        part_verts = []
                        m = parts[i]
                        for v in range(n):
                            if m & (1 << v):
                                part_verts.append(v)
                        res.append(part_verts)
                best_result[0] = res
                return True

            v = order[idx]
            v_bit = 1 << v

            first_empty = -1
            for i in range(k):
                if parts[i] == 0:
                    first_empty = i
                    break

            for i in range(k):
                # Symmetry breaking for empty parts
                if parts[i] == 0:
                    if i != first_empty:
                        continue

                # Check if v can be added to part i
                if not (add_mask[parts[i]] & v_bit):
                    continue

                # Also check that v is not already assigned
                if assigned & v_bit:
                    continue

                old_part = parts[i]
                old_assigned = assigned
                parts[i] |= v_bit
                assigned |= v_bit

                if backtrack(idx + 1):
                    return True

                parts[i] = old_part
                assigned = old_assigned

            return False

        if backtrack(0):
            return best_result[0]
        return None

    # Try k from 1 upward
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        res = solve_k(k)
        if res is not None:
            # Ensure all elements are Python ints and lists
            final = []
            for part in res:
                final.append([int(v) for v in part])
            return final

    # Fallback: each vertex in its own part (valid: single vertex is both clique and independent)
    return [[v] for v in range(n)]
