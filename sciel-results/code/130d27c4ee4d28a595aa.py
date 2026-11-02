"""Combined approach: LP-relaxation-guided greedy + WalkSAT local search + deletion reduction.
The LP relaxation provides dual variables that weight uncovered t-subsets by scarcity,
guiding the greedy phase to select blocks covering the most "difficult" subsets first.
WalkSAT refines the solution, and deletion reduction polishes it."""
import time
import random
import itertools
import numpy as np
from scipy.optimize import linprog
from collections import defaultdict

_RNG = random.Random(20250101)


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin under 150s

    # Precompute all t-subsets and map each to an index
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {ts: i for i, ts in enumerate(t_subs)}
    n_t = len(t_subs)

    # Precompute all k-subsets (candidate blocks) and the t-subsets they cover
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)
    block_covers = []
    for ks in k_subs:
        covered = []
        for ts in itertools.combinations(ks, t):
            covered.append(t_sub_index[ts])
        block_covers.append(covered)

    # For each t-subset, list of block indices that cover it
    t_to_blocks = [[] for _ in range(n_t)]
    for bi, cov in enumerate(block_covers):
        for ti in cov:
            t_to_blocks[ti].append(bi)

    # ---------------- LP Relaxation for dual guidance ----------------
    # Solve: min sum(x_i) s.t. sum_{i: j in block_i} x_i >= 1 for all j, 0 <= x_i <= 1
    # We use linprog with the dual to get weights on t-subsets.
    # Due to size (n_k ~ 50334, n_t ~ 3876), we solve a reduced LP if needed.
    # Actually, let's try solving the full LP with HiGHS (default in scipy >= 1.7).
    
    # Build constraint matrix in CSR format for efficiency
    A_rows = []
    A_cols = []
    A_data = []
    for ti in range(n_t):
        for bi in t_to_blocks[ti]:
            A_rows.append(ti)
            A_cols.append(bi)
            A_data.append(1)
    
    from scipy.sparse import csr_matrix
    A = csr_matrix((A_data, (A_rows, A_cols)), shape=(n_t, n_k))
    
    # Objective: minimize sum(x_i)
    c = np.ones(n_k)
    
    # Constraints: A @ x >= 1  =>  -A @ x <= -1
    lp_A = -A.tocsr()
    lp_b = -np.ones(n_t)
    
    bounds = [(0, 1)] * n_k
    
    dual_weights = np.ones(n_t)  # fallback: uniform weights
    
    if time.time() < deadline - 80.0:
        try:
            res = linprog(c, A_ub=lp_A, b_ub=lp_b, bounds=bounds, method='highs',
                          options={'time_limit': 30.0, 'presolve': True})
            if res.status == 0 and res.ineqlin is not None and len(res.ineqlin.slack) > 0:
                # Dual variables from inequality constraints (A_ub @ x <= b_ub)
                # res.ineqlin.marginals gives the dual values
                # For min c^T x s.t. A_ub x <= b_ub, duals are >= 0
                # The dual for constraint i corresponds to t-subset i
                if hasattr(res, 'ineqlin') and res.ineqlin is not None:
                    marginals = np.array(res.ineqlin.marginals)
                    if len(marginals) == n_t:
                        # Dual weights: higher means more "scarce"
                        dual_weights = np.maximum(marginals, 0.0)
                        # Normalize
                        total = dual_weights.sum()
                        if total > 0:
                            dual_weights = dual_weights / total
                        else:
                            dual_weights = np.ones(n_t) / n_t
        except Exception:
            pass  # Fall back to uniform weights

    # ---------------- Dual-weighted greedy initial solution ----------------
    # At each step, pick the uncovered t-subset with highest dual weight,
    # then choose the block covering the most weighted uncovered t-subsets.
    uncovered = [True] * n_t
    uncovered_count = n_t
    selected = set()
    
    # For efficiency, maintain a list of uncovered indices sorted by weight
    # But n_t is only ~3876, so scanning is fine.
    
    while uncovered_count > 0:
        if time.time() > deadline - 5.0:
            break
        # Pick the uncovered t-subset with the highest dual weight
        target = -1
        best_w = -1
        for i in range(n_t):
            if uncovered[i]:
                if dual_weights[i] > best_w:
                    best_w = dual_weights[i]
                    target = i
        if target < 0:
            break
        
        # Choose the block covering the most weighted uncovered t-subsets
        best_block = None
        best_gain = -1
        for bi in t_to_blocks[target]:
            gain = 0.0
            for ti in block_covers[bi]:
                if uncovered[ti]:
                    gain += dual_weights[ti]
            if gain > best_gain:
                best_gain = gain
                best_block = bi
        if best_block is None:
            break
        selected.add(best_block)
        for ti in block_covers[best_block]:
            if uncovered[ti]:
                uncovered[ti] = False
                uncovered_count -= 1

    # Ensure full coverage (safety net)
    while uncovered_count > 0:
        if time.time() > deadline - 2.0:
            break
        target = -1
        for i in range(n_t):
            if uncovered[i]:
                target = i
                break
        if target < 0:
            break
        best_block = None
        best_gain = -1
        for bi in t_to_blocks[target]:
            gain = 0
            for ti in block_covers[bi]:
                if uncovered[ti]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = bi
        if best_block is None:
            break
        selected.add(best_block)
        for ti in block_covers[best_block]:
            if uncovered[ti]:
                uncovered[ti] = False
                uncovered_count -= 1

    # Final safety: brute force any remaining
    while uncovered_count > 0:
        target = -1
        for i in range(n_t):
            if uncovered[i]:
                target = i
                break
        if target < 0:
            break
        for bi in t_to_blocks[target]:
            if bi not in selected:
                selected.add(bi)
                for ti in block_covers[bi]:
                    if uncovered[ti]:
                        uncovered[ti] = False
                        uncovered_count -= 1
                break

    # ---------------- Deletion reduction (first pass) ----------------
    cover_count = [0] * n_t
    for bi in selected:
        for ti in block_covers[bi]:
            cover_count[ti] += 1

    for bi in list(selected):
        if time.time() > deadline - 5.0:
            break
        # Try to remove bi
        can_remove = True
        for ti in block_covers[bi]:
            if cover_count[ti] == 1:
                can_remove = False
                break
        if can_remove:
            for ti in block_covers[bi]:
                cover_count[ti] -= 1
            selected.discard(bi)

    # ---------------- WalkSAT-style local search ----------------
    x = [bi in selected for bi in range(n_k)]
    
    clause_sat_count = [0] * n_t
    for ti in range(n_t):
        cnt = 0
        for bi in t_to_blocks[ti]:
            if x[bi]:
                cnt += 1
        clause_sat_count[ti] = cnt

    unsat_clauses = [ti for ti in range(n_t) if clause_sat_count[ti] == 0]
    
    # Track number of true variables
    num_true = sum(1 for xi in x if xi)
    
    best_x = x[:]
    best_num_true = num_true
    best_unsat = len(unsat_clauses)
    
    def cost():
        return num_true + 1000 * len(unsat_clauses)
    
    best_cost = cost()
    
    def update_best():
        nonlocal best_x, best_num_true, best_unsat, best_cost
        c = num_true + 1000 * len(unsat_clauses)
        if c < best_cost:
            best_cost = c
            best_num_true = num_true
            best_unsat = len(unsat_clauses)
            best_x = x[:]
    
    update_best()
    
    remaining_time = deadline - 15.0 - time.time()
    
    if remaining_time >= 5.0:
        # WalkSAT loop
        # Use a list of currently selected blocks for efficient random selection
        selected_list = [bi for bi in range(n_k) if x[bi]]
        
        max_steps = int(remaining_time * 8000)
        max_steps = max(5000, min(max_steps, 400000))
        
        for step in range(max_steps):
            if time.time() > deadline - 15.0:
                break
            
            if unsat_clauses:
                # Pick a random unsatisfied clause
                ti = _RNG.choice(unsat_clauses)
                candidates = t_to_blocks[ti]
                if candidates:
                    # With probability 0.5, pick from currently selected blocks in this clause
                    # Otherwise pick a random candidate
                    sel_in_clause = [bi for bi in candidates if x[bi]]
                    if sel_in_clause and _RNG.random() < 0.5:
                        bi = _RNG.choice(sel_in_clause)
                    else:
                        bi = _RNG.choice(candidates)
                    
                    old_val = x[bi]
                    x[bi] = not old_val
                    delta = 1 if x[bi] else -1
                    num_true += delta
                    
                    for ti2 in block_covers[bi]:
                        clause_sat_count[ti2] += delta
                    
                    # Update unsat_clauses incrementally
                    # Remove from unsat any that became satisfied
                    # Add to unsat any that became unsatisfied
                    new_unsat = []
                    for ti2 in block_covers[bi]:
                        if clause_sat_count[ti2] == 0:
                            new_unsat.append(ti2)
                    
                    # Rebuild unsat_clauses: remove those now satisfied, add new unsat
                    # This is O(n_t) but n_t is small
                    unsat_clauses = [ti2 for ti2 in range(n_t) if clause_sat_count[ti2] == 0]
                    
                    # Update selected_list
                    if x[bi]:
                        if bi not in selected_list:
                            selected_list.append(bi)
                    else:
                        if bi in selected_list:
                            selected_list.remove(bi)
                    
                    update_best()
            else:
                # Satisfied: try to remove a random selected block
                if selected_list:
                    idx = _RNG.randrange(len(selected_list))
                    bi = selected_list[idx]
                    # Try to remove
                    can_remove = True
                    for ti2 in block_covers[bi]:
                        if clause_sat_count[ti2] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        x[bi] = False
                        num_true -= 1
                        for ti2 in block_covers[bi]:
                            clause_sat_count[ti2] -= 1
                        selected_list.remove(bi)
                        update_best()
                    else:
                        # Try adding a random unselected block that helps
                        # Pick a random block not in selected_list
                        # This is expensive; instead, just continue
                        pass
                else:
                    # No selected blocks but no unsat? Shouldn't happen
                    pass

    # Use best found
    final_selected = set(bi for bi in range(n_k) if best_x[bi])
    
    # ---------------- Final deletion reduction ----------------
    cover_count = [0] * n_t
    for bi in final_selected:
        for ti in block_covers[bi]:
            cover_count[ti] += 1
    
    # Verify validity
    if any(c == 0 for c in cover_count):
        # Fall back to greedy solution
        final_selected = set(selected)
        cover_count = [0] * n_t
        for bi in final_selected:
            for ti in block_covers[bi]:
                cover_count[ti] += 1
    
    # Multiple passes of deletion reduction
    for _pass in range(3):
        if time.time() > deadline - 2.0:
            break
        changed = False
        for bi in list(final_selected):
            if time.time() > deadline - 2.0:
                break
            can_remove = True
            for ti in block_covers[bi]:
                if cover_count[ti] == 1:
                    can_remove = False
                    break
            if can_remove:
                for ti in block_covers[bi]:
                    cover_count[ti] -= 1
                final_selected.discard(bi)
                changed = True
        if not changed:
            break

    # Build output blocks
    blocks = [list(k_subs[bi]) for bi in sorted(final_selected)]
    
    return blocks
