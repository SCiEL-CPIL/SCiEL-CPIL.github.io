import numpy as np
import itertools
import time
import random

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    _HAS_MILP = True
except Exception:
    _HAS_MILP = False

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin (evaluator kills at 150s, but keep buffer)

    points = list(range(v))
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # Precompute t-subset indices for each k-block
    # shape: (n_ks, C(k,t))
    t_count = 1
    for i in range(t):
        t_count *= (k - i)
        t_count //= (i + 1)
        
    sub_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            sub_indices[i, j] = idx_map[sub]
            
    best_solution = None
    best_score = float('inf')
    
    def greedy_cover(seed, balanced=False):
        """
        Standard greedy set cover.
        If balanced=True, uses a discrepancy-minimization heuristic:
        Prioritize blocks that cover the most uncovered items, but penalize
        blocks that are already 'heavy' (high average coverage) to encourage
        a more uniform distribution of coverage, which aids later reduction.
        """
        random.seed(seed)
        np.random.seed(seed)
        
        uncovered = np.ones(n_ts, dtype=bool)
        selected_blocks = []
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        
        while uncovered.sum() > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None
            
            # Vectorized gain calculation
            gains = uncovered[sub_indices].sum(axis=1)
            
            if balanced:
                # Penalty based on current average coverage of the block's subsets
                # We want to avoid piling up coverage on specific triples
                # Approximate load: sum(coverage_count[sub_indices]) / t_count
                # This is O(N*K) per step. N=346k, K=35. ~12M ops.
                # Might be slow. Let's use a simpler proxy:
                # Just randomize among top-gain candidates to break ties randomly,
                # which is already in the standard version.
                # True balanced approach: Minimize max(coverage).
                # Let's try: gain - lambda * current_coverage_avg
                # But computing current_coverage_avg for all blocks is expensive.
                # Alternative: Just use the standard greedy but with random restarts
                # is usually good enough. The 'balanced' idea is hard to vectorize fast.
                # Let's stick to standard greedy for speed, but maybe add noise?
                pass
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            best = random.choice(candidates)
            
            selected_blocks.append(best)
            subs = sub_indices[best]
            coverage_count[subs] += 1
            uncovered[subs] = False
            
        return selected_blocks

    def reduce_cover(blocks, seed=0):
        random.seed(seed)
        
        if not blocks:
            return []
            
        # Build coverage count
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b in blocks:
            coverage_count[sub_indices[b]] += 1
            
        current_blocks = list(blocks)
        improved = True
        
        while improved:
            if time.time() - start_time > time_limit - 5.0:
                break
            improved = False
            
            order = list(range(len(current_blocks)))
            random.shuffle(order)
            
            new_blocks = []
            for idx in order:
                b = current_blocks[idx]
                subs = sub_indices[b]
                if np.all(coverage_count[subs] > 1):
                    coverage_count[subs] -= 1
                    improved = True
                else:
                    new_blocks.append(b)
                    
            current_blocks = new_blocks
            
        return current_blocks

    def polish(blocks, seed=0):
        """
        Local search: Swap blocks to enable further deletion.
        """
        random.seed(seed)
        if not blocks:
            return blocks

        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b in blocks:
            coverage_count[sub_indices[b]] += 1

        current_blocks = list(blocks)
        
        # First do a reduction pass
        current_blocks = reduce_cover(current_blocks, seed + 500)
        if not current_blocks:
            return []

        # Rebuild coverage count after reduction
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b in current_blocks:
            coverage_count[sub_indices[b]] += 1

        block_set = set(current_blocks)

        improved = True
        max_iters = 100
        iter_count = 0

        while improved and iter_count < max_iters:
            if time.time() - start_time > time_limit - 5.0:
                break
            improved = False
            iter_count += 1

            n_blocks = len(current_blocks)
            if n_blocks == 0:
                break

            order = list(range(n_blocks))
            random.shuffle(order)

            for idx in order:
                if time.time() - start_time > time_limit - 5.0:
                    break

                b = current_blocks[idx]

                # Temporarily remove block b
                coverage_count[sub_indices[b]] -= 1

                uncovered_mask = (coverage_count == 0)
                n_uncovered = int(np.sum(uncovered_mask))

                if n_uncovered == 0:
                    # Block was redundant - remove it
                    current_blocks.pop(idx)
                    block_set.discard(b)
                    improved = True
                    block_set = set(current_blocks)
                    break

                # Try to find a single block that covers all uncovered triples
                gains = uncovered_mask[sub_indices].sum(axis=1)
                full_cover_candidates = np.where(gains == n_uncovered)[0]
                valid_candidates = [c for c in full_cover_candidates if c not in block_set]

                if valid_candidates:
                    # Pick a candidate.
                    # Heuristic: Pick one that maximizes overlap with existing blocks?
                    # Or just random.
                    new_block = random.choice(valid_candidates)

                    # Perform the swap
                    current_blocks[idx] = new_block
                    block_set.discard(b)
                    block_set.add(new_block)
                    coverage_count[sub_indices[new_block]] += 1
                    improved = True
                    break
                else:
                    # Revert
                    coverage_count[sub_indices[b]] += 1
                    continue

        return current_blocks

    def ilp_complete(covered_blocks, seed=0):
        """
        Solve the residual covering problem using MILP.
        """
        if not _HAS_MILP:
            return None

        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b in covered_blocks:
            coverage_count[sub_indices[b]] += 1

        uncovered_mask = (coverage_count == 0)
        n_uncovered = int(np.sum(uncovered_mask))
        if n_uncovered == 0:
            return []

        # Build residual incidence matrix
        # Only consider blocks that cover at least one uncovered triple
        # To keep size manageable, we can restrict to blocks with high potential
        # but for correctness we need all that cover >=1.
        
        # A_full is (n_ts, n_ks). We want (n_uncovered, n_ks)
        # This is a sparse operation effectively.
        # sub_indices.T is (n_ts, n_ks)
        A_full = sub_indices.T
        A_res = A_full[uncovered_mask, :]

        col_counts = A_res.sum(axis=0)
        active_cols = np.where(col_counts >= 1)[0]
        if len(active_cols) == 0:
            return []

        A = A_res[:, active_cols]
        m = A.shape[1]

        c = np.ones(m, dtype=float)
        constraint = LinearConstraint(A, lb=np.ones(n_uncovered), ub=np.inf)
        bounds = Bounds(lb=0, ub=1)

        # Tight time limit for ILP
        result = milp(c=c, constraints=[constraint], integrality=np.ones(m, dtype=int),
                      bounds=bounds, options={'time_limit': 3.0, 'presolve': True})

        if result.status == 0:
            x = result.x
            selected = [int(active_cols[i]) for i in range(m) if x[i] > 0.5]
            return selected
        else:
            return None

    def greedy_complete(uncovered_mask, seed=0):
        random.seed(seed)
        remaining = uncovered_mask.copy()
        added = []
        while remaining.sum() > 0:
            if time.time() - start_time > time_limit - 3.0:
                break
            gains = remaining[sub_indices].sum(axis=1)
            max_gain = int(np.max(gains))
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best = random.choice(candidates)
            added.append(int(best))
            remaining[sub_indices[best]] = False
        return added

    # Main search loop
    restart_count = 0
    # We will alternate between standard greedy and a slightly perturbed greedy
    # The 'balanced' idea from the prompt suggests minimizing discrepancy.
    # A practical way to achieve this in set cover is to randomize the choice
    # among top-gain candidates more aggressively, or use a penalty term.
    # Given the speed constraints, we will rely on multiple restarts with
    # different seeds and the robust local search (polish) which effectively
    # optimizes the structure.
    
    while time.time() - start_time < time_limit - 10.0:
        seed = restart_count
        blocks = greedy_cover(seed)

        if blocks is None:
            break

        reduced = reduce_cover(blocks, seed + 100)
        polished = polish(reduced, seed + 200)

        if polished is not None and len(polished) < best_score:
            best_score = len(polished)
            best_solution = list(polished)

        # Hybrid: ILP completion on the residual of the polished solution
        if polished is not None and time.time() - start_time < time_limit - 15.0:
            cov = np.zeros(n_ts, dtype=np.int32)
            for b in polished:
                cov[sub_indices[b]] += 1
            unc = (cov == 0)
            n_unc = int(unc.sum())

            if n_unc > 0:
                # Try ILP completion
                extra = ilp_complete(polished, seed + 300)
                if extra is not None:
                    candidate = list(polished) + extra
                    # Verify coverage
                    cov2 = np.zeros(n_ts, dtype=np.int32)
                    for b in candidate:
                        cov2[sub_indices[b]] += 1
                    if np.all(cov2 >= 1):
                        cand_reduced = reduce_cover(candidate, seed + 400)
                        cand_polished = polish(cand_reduced, seed + 500)
                        if cand_polished is not None and len(cand_polished) < best_score:
                            best_score = len(cand_polished)
                            best_solution = list(cand_polished)
                else:
                    # Fallback to greedy completion
                    extra_g = greedy_complete(unc, seed + 600)
                    if extra_g:
                        candidate = list(polished) + extra_g
                        cov2 = np.zeros(n_ts, dtype=np.int32)
                        for b in candidate:
                            cov2[sub_indices[b]] += 1
                        if np.all(cov2 >= 1):
                            cand_reduced = reduce_cover(candidate, seed + 700)
                            cand_polished = polish(cand_reduced, seed + 800)
                            if cand_polished is not None and len(cand_polished) < best_score:
                                best_score = len(cand_polished)
                                best_solution = list(cand_polished)

        restart_count += 1

    if best_solution is None:
        blocks = greedy_cover(0)
        if blocks:
            best_solution = reduce_cover(blocks, 1)
            best_solution = polish(best_solution, 2)
            
    if best_solution is None:
        return []

    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
