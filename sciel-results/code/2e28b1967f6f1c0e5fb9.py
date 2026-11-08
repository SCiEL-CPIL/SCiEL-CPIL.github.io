import time
import random
from itertools import combinations

DIMENSIONS = [3, 4, 7]
_DEADLINE = 135.0

def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS
    total_terms = n * m * p  # 84

    # Generate all possible 2x2x2 Strassen blocks
    blocks = []
    block_terms_list = []
    for i0 in range(n):
        for i1 in range(i0 + 1, n):
            for j0 in range(m):
                for j1 in range(j0 + 1, m):
                    for k0 in range(p):
                        for k1 in range(k0 + 1, p):
                            terms = frozenset((i, j, k) for i in (i0, i1) for j in (j0, j1) for k in (k0, k1))
                            blocks.append((i0, i1, j0, j1, k0, k1))
                            block_terms_list.append(terms)

    nb = len(blocks)

    # Precompute bitmasks for fast intersection
    term_to_idx = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                term_to_idx[(i, j, k)] = len(term_to_idx)

    block_mask = []
    for terms in block_terms_list:
        mask = 0
        for t in terms:
            mask |= (1 << term_to_idx[t])
        block_mask.append(mask)

    def eval_rank(selected):
        """Compute rank for a set of selected blocks."""
        covered = 0
        for idx in selected:
            covered |= block_mask[idx]
        covered_count = bin(covered).count('1')
        return 7 * len(selected) + (total_terms - covered_count)

    # Fixed known-good configuration: 6 disjoint blocks
    fixed_tuples = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]
    fixed_indices = []
    for ft in fixed_tuples:
        for idx, b in enumerate(blocks):
            if b == ft:
                fixed_indices.append(idx)
                break

    # Verify fixed blocks are disjoint
    for i in range(len(fixed_indices)):
        for j in range(i + 1, len(fixed_indices)):
            if block_mask[fixed_indices[i]] & block_mask[fixed_indices[j]]:
                raise ValueError("Fixed blocks not disjoint")

    best_rank = eval_rank(fixed_indices)
    best_config = list(fixed_indices)

    def greedy_pack(seed):
        """Greedy set packing with random ordering."""
        rng = random.Random(seed)
        remaining = list(range(nb))
        rng.shuffle(remaining)
        selected = []
        covered = 0
        for idx in remaining:
            if (block_mask[idx] & covered) == 0:
                selected.append(idx)
                covered |= block_mask[idx]
        return selected

    def local_search(selected, rng, max_iters=300):
        """Improve via add, remove, and swap moves."""
        selected = list(selected)
        rank = eval_rank(selected)
        sel_set = set(selected)

        for _ in range(max_iters):
            if time.time() - _t0 > _DEADLINE - 10:
                break
            improved = False

            covered = 0
            for s in selected:
                covered |= block_mask[s]

            # Try adding
            add_cands = [i for i in range(nb) if i not in sel_set and (block_mask[i] & covered) == 0]
            if add_cands:
                for pick in rng.sample(add_cands, min(len(add_cands), 15)):
                    new_rank = eval_rank(selected + [pick])
                    if new_rank < rank:
                        selected.append(pick)
                        sel_set.add(pick)
                        rank = new_rank
                        improved = True
                        break

            if not improved:
                # Try removing
                if len(selected) > 1:
                    for pick in rng.sample(selected, min(len(selected), 8)):
                        new_sel = [s for s in selected if s != pick]
                        new_rank = eval_rank(new_sel)
                        if new_rank < rank:
                            selected = new_sel
                            sel_set.discard(pick)
                            rank = new_rank
                            improved = True
                            break

            if not improved:
                # Try single swap
                if selected:
                    for rem in rng.sample(selected, min(len(selected), 5)):
                        new_covered = 0
                        for s in selected:
                            if s != rem:
                                new_covered |= block_mask[s]
                        add_cands = [i for i in range(nb) if i not in sel_set and i != rem and (block_mask[i] & new_covered) == 0]
                        if add_cands:
                            for add in rng.sample(add_cands, min(len(add_cands), 10)):
                                new_sel = [s for s in selected if s != rem] + [add]
                                new_rank = eval_rank(new_sel)
                                if new_rank < rank:
                                    selected = new_sel
                                    sel_set.discard(rem)
                                    sel_set.add(add)
                                    rank = new_rank
                                    improved = True
                                    break
                        if improved:
                            break

            if not improved:
                break

        return selected, rank

    def exhaustive_single_swap(selected, current_rank):
        """Try all single swaps exhaustively."""
        selected = list(selected)
        rank = current_rank
        sel_set = set(selected)
        improved = False

        for rem in list(selected):
            if time.time() - _t0 > _DEADLINE - 8:
                break
            new_covered = 0
            for s in selected:
                if s != rem:
                    new_covered |= block_mask[s]
            for add in range(nb):
                if add in sel_set or add == rem:
                    continue
                if block_mask[add] & new_covered:
                    continue
                new_sel = [s for s in selected if s != rem] + [add]
                new_rank = eval_rank(new_sel)
                if new_rank < rank:
                    selected = new_sel
                    sel_set.discard(rem)
                    sel_set.add(add)
                    rank = new_rank
                    improved = True
                    break
            if improved:
                break

        return selected, rank, improved

    # Iterated local search
    num_restarts = 500
    for r in range(num_restarts):
        if time.time() - _t0 > _DEADLINE - 20:
            break

        if r % 5 == 0:
            selected = greedy_pack(r * 31 + 7)
        elif r % 5 == 1:
            selected = list(fixed_indices)
        elif r % 5 == 2:
            selected = list(best_config)
        elif r % 5 == 3:
            rng_init = random.Random(r * 13 + 5)
            k = rng_init.randint(max(1, len(fixed_indices) - 2), len(fixed_indices))
            selected = rng_init.sample(fixed_indices, k)
            covered = 0
            for s in selected:
                covered |= block_mask[s]
            for idx in range(nb):
                if idx not in selected and (block_mask[idx] & covered) == 0:
                    selected.append(idx)
                    covered |= block_mask[idx]
        else:
            selected = greedy_pack(r * 17 + 3)

        rng = random.Random(r * 7919 + 1)
        improved, improved_rank = local_search(selected, rng, max_iters=150)

        if improved_rank < best_rank:
            best_rank = improved_rank
            best_config = list(improved)

    # Final polish
    if time.time() - _t0 < _DEADLINE - 15:
        final, final_rank, _ = exhaustive_single_swap(best_config, best_rank)
        if final_rank < best_rank:
            best_rank = final_rank
            best_config = list(final)

        rng = random.Random(12345)
        final, final_rank = local_search(best_config, rng, max_iters=200)
        if final_rank < best_rank:
            best_rank = final_rank
            best_config = list(final)

        if time.time() - _t0 < _DEADLINE - 10:
            final, final_rank, _ = exhaustive_single_swap(best_config, best_rank)
            if final_rank < best_rank:
                best_rank = final_rank
                best_config = list(final)

    # Exhaustive 2-opt check if time permits
    if time.time() - _t0 < _DEADLINE - 5 and len(best_config) >= 2:
        sel_set = set(best_config)
        found_improvement = True
        while found_improvement and time.time() - _t0 < _DEADLINE - 5:
            found_improvement = False
            for i, rem1 in enumerate(best_config):
                if time.time() - _t0 > _DEADLINE - 5:
                    break
                for rem2 in best_config[i + 1:]:
                    if time.time() - _t0 > _DEADLINE - 5:
                        break
                    new_covered = 0
                    for s in best_config:
                        if s != rem1 and s != rem2:
                            new_covered |= block_mask[s]
                    add_cands = [a for a in range(nb) if a not in sel_set and a != rem1 and a != rem2 and (block_mask[a] & new_covered) == 0]
                    for a1_idx, a1 in enumerate(add_cands):
                        if time.time() - _t0 > _DEADLINE - 5:
                            break
                        for a2 in add_cands[a1_idx + 1:]:
                            if block_mask[a1] & block_mask[a2]:
                                continue
                            new_sel = [s for s in best_config if s != rem1 and s != rem2] + [a1, a2]
                            new_rank = eval_rank(new_sel)
                            if new_rank < best_rank:
                                best_config = new_sel
                                sel_set = set(best_config)
                                best_rank = new_rank
                                found_improvement = True
                                break
                        if found_improvement:
                            break
                    if found_improvement:
                        break
                if found_improvement:
                    break

    # Construct matrices
    rank = best_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    covered_terms = set()

    for b_idx in best_config:
        b = blocks[b_idx]
        i0, i1, j0, j1, k0, k1 = b

        # M1
        U[i0 * m + j0][col] = 1
        U[i1 * m + j1][col] = 1
        V[j0 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1

        # M2
        U[i1 * m + j0][col] = 1
        U[i1 * m + j1][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + i1][col] = 1
        W[k1 * n + i1][col] = -1
        col += 1

        # M3
        U[i0 * m + j0][col] = 1
        V[j0 * p + k1][col] = 1
        V[j1 * p + k1][col] = -1
        W[k1 * n + i0][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1

        # M4
        U[i1 * m + j1][col] = 1
        V[j1 * p + k0][col] = 1
        V[j0 * p + k0][col] = -1
        W[k0 * n + i0][col] = 1
        W[k0 * n + i1][col] = 1
        col += 1

        # M5
        U[i0 * m + j0][col] = 1
        U[i0 * m + j1][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = -1
        W[k1 * n + i0][col] = 1
        col += 1

        # M6
        U[i1 * m + j0][col] = 1
        U[i0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1
        V[j0 * p + k1][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1

        # M7
        U[i0 * m + j1][col] = 1
        U[i1 * m + j1][col] = -1
        V[j1 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = 1
        col += 1

        for i in (i0, i1):
            for j in (j0, j1):
                for k in (k0, k1):
                    covered_terms.add((i, j, k))

    # Fill remaining terms with trivial rank-1 tensors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered_terms:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    assert col == rank, f"Rank mismatch: {col} != {rank}"

    # Verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if total != 1:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k})")

    return {"U": U, "V": V, "W": W}
