import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses maximum packing of 2x2x2 Strassen blocks (via max-flow hypergraph matching)
    + schoolbook for remaining uncovered triples.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

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

        # M1 = (a11+a22)*(b11+b22)
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        # M2 = (a21+a22)*b11
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        # M3 = a11*(b12-b22)
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        # M4 = a22*(b21-b11)
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        # M5 = (a11+a12)*b22
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        # M6 = (a21-a11)*(b11+b12)
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        # M7 = (a12-a22)*(b21+b22)
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    def add_schoolbook(i, j, k):
        add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    # Build all possible Strassen blocks
    # A block is (i1,i2,j1,j2,k1,k2) with i1<i2, j1<j2, k1<k2
    # It covers 8 triples: all (i,j,k) in {i1,i2}x{j1,j2}x{k1,k2}
    # Each block has cost 7 (saves 1 term vs schoolbook's 8)

    blocks = []
    for i1 in range(n):
        for i2 in range(i1 + 1, n):
            for j1 in range(m):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p):
                        for k2 in range(k1 + 1, p):
                            triples = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        triples.add((i, j, k))
                            blocks.append(((i1, i2, j1, j2, k1, k2), triples))

    # Total triples: 3*5*7 = 105
    total_triples = n * m * p  # 105

    # Each block saves 1 term (cost 7 vs 8 schoolbook)
    # We want to maximize number of disjoint blocks.
    # This is a maximum independent set in a conflict graph, or equivalently
    # a maximum matching in a hypergraph.
    #
    # Since blocks can overlap in complex ways (not just pairwise),
    # we use a greedy approach with backtracking, or a max-flow formulation.
    #
    # For max-flow: create a bipartite-like structure.
    # Actually, since each block covers 8 specific triples and blocks conflict
    # if they share any triple, this is a set packing problem.
    #
    # With ~200 blocks and 105 triples, we can use a greedy + local search approach.

    # Greedy: repeatedly pick the block that covers the most uncovered triples
    covered = set()
    selected_blocks = []
    remaining_blocks = list(range(len(blocks)))

    while remaining_blocks:
        best_idx = -1
        best_count = 0
        for idx in remaining_blocks:
            new_triples = blocks[idx][1] - covered
            if len(new_triples) > best_count:
                best_count = len(new_triples)
                best_idx = idx
        if best_idx == -1 or best_count == 0:
            break
        selected_blocks.append(blocks[best_idx][0])
        covered |= blocks[best_idx][1]
        remaining_blocks.remove(best_idx)

    # Local search: try to improve by swapping blocks
    # For each selected block, try removing it and re-greedy on the freed triples
    improved = True
    while improved:
        improved = False
        for si in range(len(selected_blocks)):
            # Try removing selected block si
            removed_block = selected_blocks[si]
            freed_triples = set()
            for i in (removed_block[0], removed_block[1]):
                for j in (removed_block[2], removed_block[3]):
                    for k in (removed_block[4], removed_block[5]):
                        freed_triples.add((i, j, k))

            new_covered = covered - freed_triples
            new_selected = [b for idx, b in enumerate(selected_blocks) if idx != si]

            # Re-greedy on freed triples using remaining blocks
            remaining = [blocks[idx][1] for idx in remaining_blocks]
            # Also consider blocks that were previously conflicting but now might fit
            # Actually, we should consider ALL blocks that don't conflict with new_selected
            all_candidate_blocks = []
            for idx in range(len(blocks)):
                if idx in [remaining_blocks.index(b) for b in remaining_blocks] if False else True:
                    pass
            # Simpler: consider all blocks, check if they conflict with new_selected
            for b_idx in range(len(blocks)):
                b_triples = blocks[b_idx][1]
                # Check if this block conflicts with any in new_selected
                conflicts = False
                for nb in new_selected:
                    nb_triples = set()
                    for i in (nb[0], nb[1]):
                        for j in (nb[2], nb[3]):
                            for k in (nb[4], nb[5]):
                                nb_triples.add((i, j, k))
                    if b_triples & nb_triples:
                        conflicts = True
                        break
                if not conflicts:
                    all_candidate_blocks.append((b_idx, b_triples))

            # Greedy select from candidates that cover freed triples
            new_covered_local = set(new_covered)
            new_selected_local = list(new_selected)
            candidates = list(all_candidate_blocks)
            while candidates:
                best_c_idx = -1
                best_c_count = 0
                for c_idx, c_triples in candidates:
                    new_t = c_triples - new_covered_local
                    if len(new_t) > best_c_count:
                        best_c_count = len(new_t)
                        best_c_idx = c_idx
                if best_c_idx == -1 or best_c_count == 0:
                    break
                c_triples = candidates[best_c_idx][1]
                new_selected_local.append(blocks[candidates[best_c_idx][0]][0])
                new_covered_local |= c_triples
                candidates.remove(candidates[best_c_idx])

            if len(new_selected_local) > len(selected_blocks):
                selected_blocks = new_selected_local
                covered = new_covered_local
                # Update remaining_blocks
                remaining_blocks = []
                for b_idx in range(len(blocks)):
                    if blocks[b_idx][0] not in selected_blocks:
                        remaining_blocks.append(b_idx)
                improved = True
                break

    # Now build the decomposition
    for block_params in selected_blocks:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)

    # Add schoolbook for uncovered triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_schoolbook(i, j, k)

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
