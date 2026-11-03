import itertools
import time
import random
import numpy as np
import heapq
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 115.0

    # 1. Precomputation
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # block_tsubs[i] = list of t-subset indices covered by block i
    block_tsubs = [None] * n_blocks
    # tsub_blocks[j] = list of block indices covering t-subset j
    tsub_blocks = [[] for _ in range(n_tsubsets)]

    for bi, block in enumerate(all_blocks):
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs[bi] = tsubs
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)

    # Precompute set versions for fast intersection
    block_tsub_sets = [set(tsubs) for tsubs in block_tsubs]
    tsub_block_sets = [set(lst) for lst in tsub_blocks]

    # 2. LP relaxation: maximize minimum coverage (bottleneck formulation)
    dual_weights = np.ones(n_tsubsets, dtype=np.float64)
    try:
        from scipy.sparse import csr_matrix
        from scipy.optimize import linprog

        # Build sparse incidence: M[ts_idx, bi] = 1
        rows = []
        cols = []
        data = []
        for bi in range(n_blocks):
            for ts_idx in block_tsubs[bi]:
                rows.append(ts_idx)
                cols.append(bi)
                data.append(1.0)
        M = csr_matrix((data, (rows, cols)), shape=(n_tsubsets, n_blocks))

        # Maximize z subject to:
        #   M @ x >= z * 1 (each t-subset covered at least z times)
        #   0 <= x <= 1
        #   z >= 0
        # Convert to min -z:
        #   -M @ x + z * 1 <= 0
        #   x <= 1
        #   z >= 0

        n_vars = n_blocks + 1  # x_0..x_{n-1}, z
        c = np.zeros(n_vars)
        c[-1] = -1.0  # minimize -z

        # Constraints: -M @ x + z <= 0  (n_tsubsets constraints)
        # A_ub shape: (n_tsubsets + n_blocks, n_vars)
        A_ub = np.zeros((n_tsubsets + n_blocks, n_vars))
        b_ub = np.zeros(n_tsubsets + n_blocks)

        # -M @ x + z <= 0
        A_ub[:n_tsubsets, :n_blocks] = -M.todense()
        A_ub[:n_tsubsets, -1] = 1.0
        b_ub[:n_tsubsets] = 0.0

        # x <= 1
        A_ub[n_tsubsets:, :n_blocks] = np.eye(n_blocks)
        b_ub[n_tsubsets:] = 1.0

        bounds = [(0, 1)] * n_blocks + [(0, None)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'maxiter': 5000, 'presolve': True})
        if res.success:
            # Dual variables from inequality constraints give us weights
            # The first n_tsubsets duals correspond to coverage constraints
            dw = np.abs(res.ineqlin.marginals[:n_tsubsets])
            if np.sum(dw) > 0:
                dual_weights = dw
    except Exception:
        pass

    # 3. Greedy cover using dual weights
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        # score[bi] = sum of dual_weights of uncovered t-subsets in block bi
        score = np.zeros(n_blocks, dtype=np.float64)
        for bi in range(n_blocks):
            score[bi] = np.sum(dual_weights[block_tsubs[bi]])

        # Max-heap with lazy deletion
        heap = [(-score[bi], bi) for bi in range(n_blocks)]
        heapq.heapify(heap)
        cur_score = score.copy()

        selected = []
        while True:
            if time.time() > deadline - 5:
                break

            # Pop until valid
            best_bi = -1
            best_s = 0.0
            while heap:
                neg_s, bi = heapq.heappop(heap)
                s = -neg_s
                if abs(s - cur_score[bi]) < 1e-12 and s > 1e-9:
                    best_bi = bi
                    best_s = s
                    break
            if best_bi < 0 or best_s <= 1e-9:
                break

            selected.append(best_bi)

            # Update scores for newly covered t-subsets
            for ts_idx in block_tsubs[best_bi]:
                if not uncovered[ts_idx]:
                    continue
                uncovered[ts_idx] = False
                w_sub = dual_weights[ts_idx]
                if w_sub == 0:
                    continue
                for bj in tsub_blocks[ts_idx]:
                    if bj == best_bi:
                        continue
                    cur_score[bj] -= w_sub
                    if cur_score[bj] < 0:
                        cur_score[bj] = 0.0
                    heapq.heappush(heap, (-cur_score[bj], bj))
            cur_score[best_bi] = -1.0

        return selected

    # 4. Local search refinement
    def refine(block_indices, seed):
        rng = random.Random(seed)
        block_set = set(block_indices)
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        for bi in block_set:
            cover_count[block_tsubs[bi]] += 1

        def find_replacement(unique_tsubs, exclude):
            """Find a block (not in exclude) covering all unique_tsubs."""
            if not unique_tsubs:
                return None
            # Start from the rarest t-subset
            order = sorted(unique_tsubs, key=lambda x: len(tsub_blocks[x]))
            candidates = set(tsub_block_sets[order[0]])
            for ts_idx in order[1:]:
                if not candidates:
                    break
                candidates &= tsub_block_sets[ts_idx]
                if not candidates:
                    break
            candidates -= exclude
            if not candidates:
                return None
            cand_list = list(candidates)
            rng.shuffle(cand_list)
            return cand_list

        def do_remove(bi):
            block_set.discard(bi)
            cover_count[block_tsubs[bi]] -= 1

        def do_add(bi):
            block_set.add(bi)
            cover_count[block_tsubs[bi]] += 1

        best_set = set(block_set)
        best_count = len(block_set)

        max_rounds = 80
        for rnd in range(max_rounds):
            if time.time() > deadline - 2:
                break

            improved = False

            # Phase 1: systematic redundancy removal
            to_remove = []
            for bi in list(block_set):
                redundant = True
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(bi)
            if to_remove:
                for bi in to_remove:
                    do_remove(bi)
                improved = True
                if len(block_set) < best_count:
                    best_count = len(block_set)
                    best_set = set(block_set)
                continue

            # Phase 2: single-swap (replace one block with another)
            cur_list = list(block_set)
            rng.shuffle(cur_list)
            swapped = False
            for bi in cur_list:
                if time.time() > deadline - 2:
                    break
                if bi not in block_set:
                    continue
                unique_tsubs = [ts_idx for ts_idx in block_tsubs[bi]
                                if cover_count[ts_idx] == 1]
                if not unique_tsubs:
                    do_remove(bi)
                    improved = True
                    swapped = True
                    break
                cands = find_replacement(unique_tsubs, exclude={bi})
                if not cands:
                    continue
                bj = cands[0]
                do_remove(bi)
                do_add(bj)
                improved = True
                swapped = True
                break
            if swapped:
                if len(block_set) < best_count:
                    best_count = len(block_set)
                    best_set = set(block_set)
                continue

            # Phase 3: 2-for-1 pair replacement (net -1)
            if len(block_set) >= 2:
                cur_list = list(block_set)
                rng.shuffle(cur_list)
                pair_done = False
                # Try multiple pairs per round
                for attempt in range(min(5, len(cur_list))):
                    if time.time() > deadline - 2:
                        break
                    i = rng.randint(0, len(cur_list) - 1)
                    j = rng.randint(0, len(cur_list) - 1)
                    if i == j:
                        continue
                    b1, b2 = cur_list[i], cur_list[j]
                    if b1 not in block_set or b2 not in block_set:
                        continue

                    set1 = block_tsub_sets[b1]
                    set2 = block_tsub_sets[b2]
                    unique = set()
                    for ts_idx in set1:
                        if cover_count[ts_idx] == 1:
                            unique.add(ts_idx)
                    for ts_idx in set2:
                        if cover_count[ts_idx] == 1:
                            unique.add(ts_idx)
                    unique_list = list(unique)

                    if not unique_list:
                        do_remove(b1)
                        do_remove(b2)
                        improved = True
                        pair_done = True
                        break

                    cands = find_replacement(unique_list, exclude={b1, b2})
                    if not cands:
                        continue
                    bk = cands[0]
                    do_remove(b1)
                    do_remove(b2)
                    do_add(bk)
                    improved = True
                    pair_done = True
                    break

                if pair_done:
                    if len(block_set) < best_count:
                        best_count = len(block_set)
                        best_set = set(block_set)
                    continue

            if not improved:
                # Random kick to escape local opt
                if block_set and rng.random() < 0.25:
                    ci = rng.randint(0, n_blocks - 1)
                    if ci not in block_set:
                        do_add(ci)
                else:
                    break

        # Final redundancy cleanup
        to_remove = []
        for bi in list(block_set):
            redundant = True
            for ts_idx in block_tsubs[bi]:
                if cover_count[ts_idx] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(bi)
        for bi in to_remove:
            do_remove(bi)
        if len(block_set) < best_count:
            best_count = len(block_set)
            best_set = set(block_set)

        return list(best_set)

    best_blocks = None
    best_len = float('inf')

    # Multiple restarts with dynamic budget
    start_time = time.time()
    seed = 0
    while time.time() < deadline - 15:
        init_blocks = greedy_cover(seed)
        if not init_blocks:
            seed += 1
            continue
        refined = refine(init_blocks, seed)
        if len(refined) < best_len:
            best_len = len(refined)
            best_blocks = [all_blocks[bi] for bi in refined]
        seed += 1

    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = set(range(n_tsubsets))
        blocks_used = []
        gain = [len(block_tsubs[i]) for i in range(n_blocks)]
        while uncovered:
            max_gain = max(gain)
            if max_gain == 0:
                break
            chosen = gain.index(max_gain)
            blocks_used.append(chosen)
            for ts_idx in block_tsubs[chosen]:
                if ts_idx in uncovered:
                    uncovered.discard(ts_idx)
                    for bi in tsub_blocks[ts_idx]:
                        if gain[bi] > 0:
                            gain[bi] -= 1
            gain[chosen] = 0
        best_blocks = [all_blocks[bi] for bi in blocks_used]

    return best_blocks
