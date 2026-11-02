# EVOLVE-BLOCK-START
import time
from collections import defaultdict
from fractions import Fraction

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    Uses a block-Strassen / Winograd hybrid with aggressive Winograd-style merging
    to minimize the rank.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    n, m, p = 2, 4, 5

    def add_strassen_2x2x2(a_offset, b_offset, c_offset, sign=1):
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset

        terms = [
            ({(i0, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1},
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            ({(i0, j0): 1},
             {(j0, k0+1): 1, (j0+1, k0+1): -1},
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0+1): 1},
             {(j0+1, k0): 1, (j0, k0): -1},
             {(i0, k0): 1, (i0+1, k0): 1}),
            ({(i0, j0): 1, (i0, j0+1): 1},
             {(j0+1, k0+1): 1},
             {(i0, k0): -1, (i0, k0+1): 1}),
            ({(i0+1, j0): 1},
             {(j0, k0): 1, (j0, k0+1): 1},
             {(i0+1, k0+1): 1}),
            ({(i0, j0+1): 1, (i0+1, j0+1): -1},
             {(j0+1, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1}),
        ]

        for u, v, w in terms:
            u_scaled = {k: val * sign for k, val in u.items()}
            v_scaled = {k: val * sign for k, val in v.items()}
            w_scaled = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_scaled, v_scaled, w_scaled))

    def add_schoolbook_2x2x1(a_offset, b_offset, c_offset, sign=1):
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset

        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
            ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
            ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
        ]

        for u, v, w in terms:
            u_scaled = {k: val * sign for k, val in u.items()}
            v_scaled = {k: val * sign for k, val in v.items()}
            w_scaled = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_scaled, v_scaled, w_scaled))

    def build_base_decomposition():
        """Build the base block-Strassen decomposition."""
        terms = []

        def add_strassen_local(a_offset, b_offset, c_offset, sign=1):
            i0, j0 = a_offset
            j1, k0 = b_offset
            c_i0, c_k0 = c_offset

            local_terms = [
                ({(i0, j0): 1, (i0+1, j0+1): 1},
                 {(j0, k0): 1, (j0+1, k0+1): 1},
                 {(i0, k0): 1, (i0+1, c_k0+1): 1}),
                ({(i0+1, j0): 1, (i0+1, j0+1): 1},
                 {(j0, k0): 1},
                 {(i0+1, k0): 1, (i0+1, k0+1): -1}),
                ({(i0, j0): 1},
                 {(j0, k0+1): 1, (j0+1, k0+1): -1},
                 {(i0, k0+1): 1, (i0+1, k0+1): 1}),
                ({(i0+1, j0+1): 1},
                 {(j0+1, k0): 1, (j0, k0): -1},
                 {(i0, k0): 1, (i0+1, k0): 1}),
                ({(i0, j0): 1, (i0, j0+1): 1},
                 {(j0+1, k0+1): 1},
                 {(i0, k0): -1, (i0, k0+1): 1}),
                ({(i0+1, j0): 1},
                 {(j0, k0): 1, (j0, k0+1): 1},
                 {(i0+1, k0+1): 1}),
                ({(i0, j0+1): 1, (i0+1, j0+1): -1},
                 {(j0+1, k0): 1, (j0+1, k0+1): 1},
                 {(i0, k0): 1}),
            ]

            for u, v, w in local_terms:
                u_scaled = {k: val * sign for k, val in u.items()}
                v_scaled = {k: val * sign for k, val in v.items()}
                w_scaled = {k: val * sign for k, val in w.items()}
                terms.append((u_scaled, v_scaled, w_scaled))

        def add_schoolbook_local(a_offset, b_offset, c_offset, sign=1):
            i0, j0 = a_offset
            j1, k0 = b_offset
            c_i0, c_k0 = c_offset

            local_terms = [
                ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
                ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
                ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
                ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
            ]

            for u, v, w in local_terms:
                u_scaled = {k: val * sign for k, val in u.items()}
                v_scaled = {k: val * sign for k, val in v.items()}
                w_scaled = {k: val * sign for k, val in w.items()}
                terms.append((u_scaled, v_scaled, w_scaled))

        # Block decomposition for 2x4 * 4x5 -> 2x5
        # A is 2x4, B is 4x5, C is 2x5
        # We can think of this as:
        # C[:,0:2] = A[:,0:2] * B[0:2,0:2] + A[:,2:4] * B[2:4,0:2]  (2x2 * 2x2 blocks)
        # C[:,2:4] = A[:,0:2] * B[0:2,2:4] + A[:,2:4] * B[2:4,2:4]  (2x2 * 2x2 blocks)
        # C[:,4:5] = A[:,0:2] * B[0:2,4:5] + A[:,2:4] * B[2:4,4:5]  (2x2 * 2x1 blocks)

        # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (Strassen 2x2x2)
        add_strassen_local((0, 0), (0, 0), (0, 0))

        # Block 2: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2] (Strassen 2x2x2)
        add_strassen_local((0, 2), (2, 0), (0, 0))

        # Block 3: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4] (Strassen 2x2x2)
        add_strassen_local((0, 0), (0, 2), (0, 2))

        # Block 4: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4] (Strassen 2x2x2)
        add_strassen_local((0, 2), (2, 2), (0, 2))

        # Block 5: A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5] (Schoolbook 2x2x1)
        add_schoolbook_local((0, 0), (0, 4), (0, 4))

        # Block 6: A[0:2,2:4] * B[2:4,4:5] -> C[0:2,4:5] (Schoolbook 2x2x1)
        add_schoolbook_local((0, 2), (2, 4), (0, 4))

        return terms

    def verify_decomposition(terms):
        """Verify that the tensor decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for (u, v, w) in terms:
                        total += u.get((i, j), 0) * v.get((j, k), 0) * w.get((i, k), 0)
                    if total != 1:
                        return False
        return True

    def merge_terms(terms):
        """
        Winograd-style merge: group terms by (U, W) pairs.
        Terms with identical U and W factors can have their V factors summed.
        Also try grouping by (U, V) and summing W, and (V, W) and summing U.
        """
        # First pass: merge by (U, W)
        groups = defaultdict(list)
        for idx, (u, v, w) in enumerate(terms):
            u_key = tuple(sorted(u.items()))
            w_key = tuple(sorted(w.items()))
            groups[(u_key, w_key)].append((idx, v))

        merged = []
        used = set()

        for (u_key, w_key), members in groups.items():
            u = dict(u_key)
            w = dict(w_key)
            v_sum = {}

            for idx, v in members:
                if idx in used:
                    continue
                for (jk, coeff) in v.items():
                    v_sum[jk] = v_sum.get(jk, 0) + coeff
                used.add(idx)

            # Remove zero entries
            v_sum = {k: val for k, val in v_sum.items() if val != 0}

            if v_sum:
                merged.append((u, v_sum, w))

        # Second pass: try merging by (U, V)
        groups2 = defaultdict(list)
        for idx, (u, v, w) in enumerate(merged):
            u_key = tuple(sorted(u.items()))
            v_key = tuple(sorted(v.items()))
            groups2[(u_key, v_key)].append((idx, w))

        merged2 = []
        used2 = set()

        for (u_key, v_key), members in groups2.items():
            u = dict(u_key)
            v = dict(v_key)
            w_sum = {}

            for idx, w in members:
                if idx in used2:
                    continue
                for (ik, coeff) in w.items():
                    w_sum[ik] = w_sum.get(ik, 0) + coeff
                used2.add(idx)

            # Remove zero entries
            w_sum = {k: val for k, val in w_sum.items() if val != 0}

            if w_sum:
                merged2.append((u, v, w_sum))

        # Third pass: try merging by (V, W)
        groups3 = defaultdict(list)
        for idx, (u, v, w) in enumerate(merged2):
            v_key = tuple(sorted(v.items()))
            w_key = tuple(sorted(w.items()))
            groups3[(v_key, w_key)].append((idx, u))

        merged3 = []
        used3 = set()

        for (v_key, w_key), members in groups3.items():
            v = dict(v_key)
            w = dict(w_key)
            u_sum = {}

            for idx, u in members:
                if idx in used3:
                    continue
                for (ij, coeff) in u.items():
                    u_sum[ij] = u_sum.get(ij, 0) + coeff
                used3.add(idx)

            # Remove zero entries
            u_sum = {k: val for k, val in u_sum.items() if val != 0}

            if u_sum:
                merged3.append((u_sum, v, w))

        # Return the best (smallest) result
        best = min([merged, merged2, merged3], key=len)
        return best

    # Build the base decomposition
    rank_terms = build_base_decomposition()

    # Verify it's correct
    if not verify_decomposition(rank_terms):
        # Fallback to schoolbook
        rank_terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    rank_terms.append(
                        ({(i, j): 1}, {(j, k): 1}, {(i, k): 1})
                    )

    # Apply Winograd-style merging
    merged_terms = merge_terms(rank_terms)

    # Verify the merged decomposition is still correct
    if verify_decomposition(merged_terms):
        rank_terms = merged_terms

    rank = len(rank_terms)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][col] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
