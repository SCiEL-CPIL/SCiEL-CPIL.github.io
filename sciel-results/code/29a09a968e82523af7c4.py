"""Combined: Lagrangian-guided WalkSAT + vectorized greedy + pair-replacement refinement.
Uses dynamic penalty weights on uncovered t-subsets to guide stochastic local search,
with pair-replacement moves for structural improvements beyond single-block flips."""
import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from collections import defaultdict
from scipy.sparse import csr_matrix

_RNG = random.Random(20250101)


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0

    # Precompute all t-subsets
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {ts: i for i, ts in enumerate(t_subs)}
    n_t = len(t_subs)

    # Precompute all k-subsets and their covered t-subset indices
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)
    block_covers = []
    for ks in k_subs:
        covered = []
        for ts in itertools.combinations(ks, t):
            covered.append(t_sub_index[ts])
        block_covers.append(covered)

    # Inverse: t-subset -> list of blocks covering it
    t_to_blocks = [[] for _ in range(n_t)]
    for bi, cov in enumerate(block_covers):
        for ti in cov:
            t_to_blocks[ti].append(bi)

    # Convert to numpy for speed where possible
    block_covers_np = [np.array(c, dtype=np.int32) for c in block_covers]
    t_to_blocks_np = [np.array(b, dtype=np.int32) for b in t_to_blocks]

    # Precompute block overlap counts for faster pair-replacement
    overlap_cache = {}
    def get_overlap(bi, bj):
        key = (min(bi, bj), max(bi, bj))
        if key in overlap_cache:
            return overlap_cache[key]
        a = block_covers_np[bi]
        b = block_covers_np[bj]
        overlap = len(set(a.tolist()) & set(b.tolist()))
        overlap_cache[key] = overlap
        return overlap

    # ---------------- Primal-Dual Approximation for Minimum Weight Vertex Cover ----------------
    # Formulate as: vertices = k-blocks, hyperedges = t-subsets
    # Dual variables y_e >= 0 for each hyperedge e
    # Primal: minimize sum x_v s.t. for each e, sum_{v in e} x_v >= 1, x_v in {0,1}
    # Primal-dual: raise y_e for uncovered e, select v when sum_{e: v in e} y_e >= cost(v) = 1
    
    def primal_dual_cover():
        """Primal-dual approximation algorithm for hypergraph vertex cover."""
        dual_vars = np.zeros(n_t, dtype=np.float64)  # y_e for each t-subset
        selected = set()
        x = np.zeros(n_k, dtype=bool)  # primal variables
        cover_count = np.zeros(n_t, dtype=np.int32)  # how many selected blocks cover each t-subset
        
        # Track which t-subsets are uncovered
        uncovered_mask = np.ones(n_t, dtype=bool)
        
        # Precompute for each block: which t-subsets it covers (already have block_covers_np)
        
        # Main loop: raise dual variables, select blocks when dual constraint tight
        max_iterations = 50000
        for iteration in range(max_iterations):
            if time.time() > deadline - 60.0:
                break
            
            # If all covered, break
            if not np.any(uncovered_mask):
                break
            
            # Find uncovered t-subsets
            unsat_indices = np.where(uncovered_mask)[0]
            if len(unsat_indices) == 0:
                break
            
            # Strategy: raise dual variables for uncovered hyperedges
            # Use a random subset to avoid bias
            batch_size = min(50, len(unsat_indices))
            batch = _RNG.sample(unsat_indices.tolist(), batch_size)
            
            # For each uncovered hyperedge in batch, raise its dual variable
            # and check if any block becomes "tight" (dual constraint satisfied)
            for ti in batch:
                # Raise dual variable
                dual_vars[ti] += 1.0
                
                # Check all blocks that cover this t-subset
                for bi in t_to_blocks[ti]:
                    if x[bi]:
                        continue
                    
                    # Compute dual objective for block bi: sum of y_e for all e covered by bi
                    # Only need to check if this newly raised y_ti makes it tight
                    # Efficient: track current dual sum for each block
                    # For simplicity, compute on the fly for blocks covering ti
                    
                    # Get all t-subsets covered by bi
                    bi_covers = block_covers_np[bi]
                    
                    # Sum dual variables for all t-subsets covered by bi
                    dual_sum = np.sum(dual_vars[bi_covers])
                    
                    # If dual_sum >= 1 (cost of selecting bi), select it
                    if dual_sum >= 1.0:
                        # Select block bi
                        x[bi] = True
                        selected.add(bi)
                        
                        # Update cover counts
                        for tj in bi_covers:
                            cover_count[tj] += 1
                            if cover_count[tj] >= 1:
                                uncovered_mask[tj] = False
                        
                        # Break inner loop since we found a block to select
                        break
        
        return selected, cover_count

    # ---------------- Vectorized Greedy with random tie-breaking ----------------
    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        gain = np.zeros(n_k, dtype=np.int32)
        for bi in range(n_k):
            gain[bi] = np.sum(uncovered[block_covers_np[bi]])
        gain = gain.copy()
        blocks_used = []
        while np.any(uncovered):
            if time.time() > deadline - 30.0:
                break
            max_gain = gain.max()
            if max_gain == 0:
                break
            candidates = np.where(gain == max_gain)[0]
            chosen = candidates[rng.randint(0, len(candidates) - 1)]
            blocks_used.append(chosen)
            for ts_idx in block_covers_np[chosen]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    for bi in t_to_blocks_np[ts_idx]:
                        gain[bi] -= 1
            gain[chosen] = -1
        return blocks_used

    # Run multiple greedy restarts, keep best
    best_greedy = None
    best_greedy_size = n_k + 1
    for seed in range(12):
        if time.time() > deadline - 40.0:
            break
        blocks = greedy_cover(seed)
        if len(blocks) < best_greedy_size:
            best_greedy_size = len(blocks)
            best_greedy = blocks
        if len(blocks) <= 135:  # Good enough
            break

    if best_greedy is None:
        best_greedy = greedy_cover(0)

    # Also try primal-dual approximation
    pd_selected, pd_cover = primal_dual_cover()
    pd_size = len(pd_selected)
    
    # Choose the better initial solution
    if pd_size < best_greedy_size:
        selected = pd_selected
        cover_count = pd_cover
    else:
        selected = set(best_greedy)
        cover_count = np.zeros(n_t, dtype=np.int32)
        for bi in selected:
            cover_count[block_covers_np[bi]] += 1

    # Build cover_count
    cover_count = np.zeros(n_t, dtype=np.int32)
    for bi in selected:
        cover_count[block_covers_np[bi]] += 1

    # ---------------- Deletion reduction ----------------
    def reduce_blocks(selected_set, cc):
        changed = True
        while changed:
            changed = False
            for bi in list(selected_set):
                if time.time() > deadline - 20.0:
                    return selected_set, cc
                cc[block_covers_np[bi]] -= 1
                if np.any(cc[block_covers_np[bi]] == 0):
                    cc[block_covers_np[bi]] += 1
                else:
                    selected_set.discard(bi)
                    changed = True
        return selected_set, cc

    # Reduce the selected solution
    selected, cover_count = reduce_blocks(selected, cover_count)
    
    # Also reduce best_greedy and keep the better
    if best_greedy is not None:
        gg_set = set(best_greedy)
        gg_cc = np.zeros(n_t, dtype=np.int32)
        for bi in gg_set:
            gg_cc[block_covers_np[bi]] += 1
        gg_set, gg_cc = reduce_blocks(gg_set, gg_cc)
        
        if len(gg_set) < len(selected):
            selected = gg_set
            cover_count = gg_cc

    # ---------------- Pair replacement: try to replace 2 blocks with 1 ----------------
    def pair_replace(selected_set, cc, max_trials=800):
        rng = _RNG
        n_sel = len(selected_set)
        if n_sel < 2:
            return selected_set, cc
        sel_list = list(selected_set)
        for trial in range(max_trials):
            if time.time() > deadline - 25.0:
                break
            i, j = rng.sample(range(n_sel), 2)
            bi, bj = sel_list[i], sel_list[j]
            # Union of t-subsets covered by bi and bj
            union_t = set(block_covers_np[bi].tolist()) | set(block_covers_np[bj].tolist())
            if len(union_t) > 35:  # Can't be covered by one block
                continue
            # Check if any single block covers all t-subsets in union_t
            # For each t-subset in union_t, find common blocks
            common = None
            for ti in union_t:
                blocks_covering = set(t_to_blocks[ti])
                if common is None:
                    common = blocks_covering
                else:
                    common &= blocks_covering
                if not common:
                    break
            if not common:
                continue
            # Try each candidate block
            for cand in common:
                if cand == bi or cand == bj:
                    continue
                # Check if replacing bi, bj with cand maintains coverage
                # Temporarily remove bi, bj
                cc[block_covers_np[bi]] -= 1
                cc[block_covers_np[bj]] -= 1
                cc[block_covers_np[cand]] += 1
                if np.all(cc >= 1):
                    # Valid replacement
                    selected_set.discard(bi)
                    selected_set.discard(bj)
                    selected_set.add(cand)
                    n_sel = len(selected_set)
                    sel_list = list(selected_set)
                    # Reduce again
                    selected_set, cc = reduce_blocks(selected_set, cc)
                    n_sel = len(selected_set)
                    sel_list = list(selected_set)
                    break
                else:
                    # Restore
                    cc[block_covers_np[bi]] += 1
                    cc[block_covers_np[bj]] += 1
                    cc[block_covers_np[cand]] -= 1
        return selected_set, cc

    # Apply pair replacement
    selected, cover_count = pair_replace(selected, cover_count)

    # ---------------- Lagrangian-guided WalkSAT ----------------
    # State: boolean array
    x = np.array([bi in selected for bi in range(n_k)], dtype=bool)
    n_true = x.sum()

    # Clause satisfaction counts
    clause_sat = np.zeros(n_t, dtype=np.int32)
    for bi in range(n_k):
        if x[bi]:
            clause_sat[block_covers_np[bi]] += 1

    # Penalty weights for Lagrangian relaxation
    penalty = np.ones(n_t, dtype=np.float64)

    # Precompute penalty-weighted soft deficit for fast cost updates
    def compute_soft_deficit(cs):
        safe = np.maximum(cs, 1).astype(np.float64)
        return np.where(cs > 0, 1.0 / (safe + 1.0), 1.0)

    soft_deficit = compute_soft_deficit(clause_sat)
    current_pen_cost = np.sum(penalty * soft_deficit)

    def smooth_cost_fast(n_true_val, pen_cost_val):
        return n_true_val + 50.0 * pen_cost_val

    best_x = x.copy()
    best_cost = smooth_cost_fast(n_true, current_pen_cost)
    best_n_true = n_true

    # WalkSAT with Lagrangian guidance
    remaining = deadline - time.time() - 15.0
    if remaining > 5.0:
        max_steps = int(min(remaining * 15000, 800000))
        max_steps = max(max_steps, 5000)

        # Maintain unsat count for quick check
        unsat_count = int((clause_sat == 0).sum())

        true_blocks = np.where(x)[0].tolist()
        
        for step in range(max_steps):
            if time.time() > deadline - 15.0:
                break

            if unsat_count > 0:
                # Pick an unsatisfied clause, weighted by penalty
                unsat_indices = np.where(clause_sat == 0)[0]
                if len(unsat_indices) == 0:
                    unsat_count = 0
                    continue
                weights = penalty[unsat_indices]
                total_w = weights.sum()
                if total_w == 0:
                    ti = unsat_indices[_RNG.randint(0, len(unsat_indices) - 1)]
                else:
                    probs = weights / total_w
                    ti = unsat_indices[_RNG.choice(len(unsat_indices), p=probs)]

                # Pick a block covering this clause
                cands = t_to_blocks[ti]
                bi = cands[_RNG.randint(0, len(cands) - 1)]

                # Flip
                old = x[bi]
                x[bi] = not old
                delta = 1 if x[bi] else -1
                affected = block_covers_np[bi]
                
                # Update clause_sat and track changes to soft_deficit
                old_sat = clause_sat[affected].copy()
                clause_sat[affected] += delta
                
                # Update unsat_count
                if delta == 1:
                    newly_sat = (clause_sat[affected] == 1)
                    unsat_count -= int(newly_sat.sum())
                else:
                    newly_unsat = (clause_sat[affected] == 0)
                    unsat_count += int(newly_unsat.sum())

                # Update penalty: increase for still-unsat, decrease for sat
                unsat_mask = clause_sat == 0
                penalty[unsat_mask] = np.minimum(penalty[unsat_mask] * 1.15, 50.0)
                sat_mask = ~unsat_mask
                penalty[sat_mask] = np.maximum(penalty[sat_mask] * 0.97, 0.1)

                # Update soft_deficit and pen_cost incrementally
                old_sd = compute_soft_deficit(old_sat)
                new_sd = compute_soft_deficit(clause_sat[affected])
                delta_sd = new_sd - old_sd
                current_pen_cost += np.sum(penalty[affected] * delta_sd)
                
                n_true += delta
                c = smooth_cost_fast(n_true, current_pen_cost)
                if c < best_cost:
                    best_cost = c
                    best_x = x.copy()
                    best_n_true = n_true
            else:
                # Satisfied: try to remove a block
                if not true_blocks:
                    break
                bi = true_blocks[_RNG.randint(0, len(true_blocks) - 1)]
                # Try removal
                affected = block_covers_np[bi]
                old_sat = clause_sat[affected].copy()
                clause_sat[affected] -= 1
                failed = np.any(clause_sat[affected] == 0)
                if failed:
                    x[bi] = True
                    clause_sat[affected] = old_sat
                else:
                    x[bi] = False
                    true_blocks.remove(bi)
                    n_true -= 1
                    # Update soft_deficit
                    new_sd = compute_soft_deficit(clause_sat[affected])
                    old_sd = compute_soft_deficit(old_sat)
                    delta_sd = new_sd - old_sd
                    current_pen_cost += np.sum(penalty[affected] * delta_sd)
                    
                    c = smooth_cost_fast(n_true, current_pen_cost)
                    if c < best_cost:
                        best_cost = c
                        best_x = x.copy()
                        best_n_true = n_true

                # Also try 2-block swap: remove one true block, add one false block
                if _RNG.random() < 0.3 and true_blocks:
                    bi_remove = true_blocks[_RNG.randint(0, len(true_blocks) - 1)]
                    # Find a candidate block to add - prefer blocks that cover many clauses
                    false_mask = ~x
                    if false_mask.any():
                        # Score false blocks by how many low-sat clauses they cover
                        false_indices = np.where(false_mask)[0]
                        # Sample a subset for efficiency
                        if len(false_indices) > 200:
                            sample_idx = _RNG.sample(range(len(false_indices)), 200)
                            sample_blocks = false_indices[sample_idx]
                        else:
                            sample_blocks = false_indices
                        
                        best_add = -1
                        best_score = -1
                        for bi_add in sample_blocks:
                            aff_add = block_covers_np[bi_add]
                            score = int(np.sum(clause_sat[aff_add] == 1))
                            if score > best_score:
                                best_score = score
                                best_add = bi_add
                        
                        if best_add >= 0:
                            bi_add = best_add
                            aff_rm = block_covers_np[bi_remove]
                            aff_add = block_covers_np[bi_add]
                            old_sat_rm = clause_sat[aff_rm].copy()
                            old_sat_add = clause_sat[aff_add].copy()
                            clause_sat[aff_rm] -= 1
                            clause_sat[aff_add] += 1
                            if np.all(clause_sat >= 1):
                                # Valid swap
                                x[bi_remove] = False
                                x[bi_add] = True
                                true_blocks.remove(bi_remove)
                                true_blocks.append(bi_add)
                                # Update soft_deficit
                                new_sd_rm = compute_soft_deficit(clause_sat[aff_rm])
                                old_sd_rm = compute_soft_deficit(old_sat_rm)
                                new_sd_add = compute_soft_deficit(clause_sat[aff_add])
                                old_sd_add = compute_soft_deficit(old_sat_add)
                                current_pen_cost += np.sum(penalty[aff_rm] * (new_sd_rm - old_sd_rm))
                                current_pen_cost += np.sum(penalty[aff_add] * (new_sd_add - old_sd_add))
                                
                                c = smooth_cost_fast(n_true, current_pen_cost)
                                if c < best_cost:
                                    best_cost = c
                                    best_x = x.copy()
                                    best_n_true = n_true
                            else:
                                # Restore
                                clause_sat[aff_rm] = old_sat_rm
                                clause_sat[aff_add] = old_sat_add

        # Use best found
        x = best_x
        clause_sat = np.zeros(n_t, dtype=np.int32)
        for bi in range(n_k):
            if x[bi]:
                clause_sat[block_covers_np[bi]] += 1
        selected = set(int(bi) for bi in range(n_k) if x[bi])
        cover_count = clause_sat

    # ---------------- Final reduction ----------------
    selected, cover_count = reduce_blocks(selected, cover_count)

    # ---------------- MILP polish (if time permits) ----------------
    if time.time() < deadline - 35.0:
        try:
            A_rows, A_cols, A_data = [], [], []
            for ti in range(n_t):
                for bi in t_to_blocks[ti]:
                    A_rows.append(ti)
                    A_cols.append(bi)
                    A_data.append(1)
            A = csr_matrix((A_data, (A_rows, A_cols)), shape=(n_t, n_k))
            c_obj = np.ones(n_k)
            lb = np.ones(n_t)
            ub = np.full(n_t, np.inf)
            integrality = np.ones(n_k)
            bounds = Bounds(lb=np.zeros(n_k), ub=np.ones(n_k))
            constraints = LinearConstraint(A, lb, ub)
            remaining_for_milp = max(8, deadline - 35.0 - time.time())
            res = milp(c_obj, constraints=constraints, integrality=integrality,
                       bounds=bounds, options={"time_limit": remaining_for_milp})
            if res.status == 0 and res.x is not None:
                milp_sel = set(int(bi) for bi in range(n_k) if res.x[bi] > 0.5)
                if len(milp_sel) < len(selected):
                    # Verify
                    cc = np.zeros(n_t, dtype=np.int32)
                    for bi in milp_sel:
                        cc[block_covers_np[bi]] += 1
                    if np.all(cc >= 1):
                        selected = milp_sel
                        selected, cover_count = reduce_blocks(selected, cc)
        except Exception:
            pass

    # Final reduction
    if selected:
        cc = np.zeros(n_t, dtype=np.int32)
        for bi in selected:
            cc[block_covers_np[bi]] += 1
        selected, cover_count = reduce_blocks(selected, cc)

    # Build output
    blocks = [list(k_subs[bi]) for bi in sorted(selected)]

    # Final validation
    if blocks:
        cc_final = np.zeros(n_t, dtype=np.int32)
        for bi in selected:
            cc_final[block_covers_np[bi]] += 1
        if np.any(cc_final == 0):
            # Fallback: rebuild from best_greedy
            selected = set(best_greedy)
            blocks = [list(k_subs[bi]) for bi in sorted(selected)]

    return blocks
