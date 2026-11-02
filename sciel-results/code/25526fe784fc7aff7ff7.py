"""Structural change: Max-SAT style stochastic local search (WalkSAT) over block-selection
variables with clause = t-subset coverage. We minimise number of selected blocks while
satisfying all coverage clauses. We combine with a greedy initialisation, a MILP phase,
and a deletion-reduction pass. Explicit deadline guard keeps us under the wall-clock limit."""
import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from collections import defaultdict

# Deterministic seed for reproducibility of stochastic choices
_RNG = random.Random(20250101)


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin under 120s

    # Precompute all t-subsets and map each to an index for fast lookup
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {ts: i for i, ts in enumerate(t_subs)}
    n_t = len(t_subs)

    # Precompute all k-subsets (candidate blocks) and the t-subsets they cover
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)
    # For each k-subset, list of t-subset indices it covers
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

    # ---------------- Greedy initial solution ----------------
    uncovered = [True] * n_t
    uncovered_count = n_t
    selected = set()

    while uncovered_count > 0:
        if time.time() > deadline - 5.0:
            break
        # Pick an uncovered t-subset (first one for determinism)
        target = None
        for i in range(n_t):
            if uncovered[i]:
                target = i
                break
        if target is None:
            break
        # Choose the block covering the most uncovered t-subsets among candidates
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
            # Should not happen
            break
        selected.add(best_block)
        for ti in block_covers[best_block]:
            if uncovered[ti]:
                uncovered[ti] = False
                uncovered_count -= 1

    # Ensure we have a valid covering (if time ran out during greedy, finish it)
    while uncovered_count > 0:
        if time.time() > deadline - 2.0:
            break
        target = None
        for i in range(n_t):
            if uncovered[i]:
                target = i
                break
        if target is None:
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

    # If still not covered (shouldn't happen), add brute-force blocks
    while uncovered_count > 0:
        target = None
        for i in range(n_t):
            if uncovered[i]:
                target = i
                break
        if target is None:
            break
        # Pick any block that covers it
        for bi in t_to_blocks[target]:
            if bi not in selected:
                selected.add(bi)
                for ti in block_covers[bi]:
                    if uncovered[ti]:
                        uncovered[ti] = False
                        uncovered_count -= 1
                break

    # ---------------- Deletion reduction ----------------
    # Try to remove each selected block if coverage is still satisfied
    def is_covered(selected_set):
        # Build coverage count array
        cover_count = [0] * n_t
        for bi in selected_set:
            for ti in block_covers[bi]:
                cover_count[ti] += 1
        return all(c > 0 for c in cover_count)

    # We'll maintain cover_count incrementally for speed
    cover_count = [0] * n_t
    for bi in selected:
        for ti in block_covers[bi]:
            cover_count[ti] += 1

    def try_remove(bi):
        nonlocal cover_count
        for ti in block_covers[bi]:
            cover_count[ti] -= 1
        # Check if any went to 0
        for ti in block_covers[bi]:
            if cover_count[ti] < 0:
                break
        # Actually check properly
        ok = True
        for ti in block_covers[bi]:
            if cover_count[ti] == 0:
                ok = False
                break
        if ok:
            return True
        # Restore
        for ti in block_covers[bi]:
            cover_count[ti] += 1
        return False

    for bi in list(selected):
        if time.time() > deadline - 5.0:
            break
        if try_remove(bi):
            selected.discard(bi)

    # ---------------- MILP phase (optional, time-limited) ----------------
    # Formulate: minimize sum(x_i) subject to for each t-subset j, sum_{i: j in block_i} x_i >= 1
    # Use scipy.optimize.milp with a time limit
    milp_start = time.time()
    milp_budget = 40.0  # seconds for MILP
    milp_solution = None
    
    if time.time() < deadline - 60.0:  # Only if we have enough time left
        try:
            # Build constraint matrix A: n_t x n_k, A[j,i] = 1 if t-subset j is covered by block i
            # This is the transpose of t_to_blocks structure
            # For memory efficiency, build sparse representation
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
            
            # Constraints: A @ x >= 1 for all t-subsets
            # milp requires LinearConstraint with lb, ub
            lb = np.ones(n_t)
            ub = np.full(n_t, np.inf)
            
            # Bounds: x_i in {0, 1}
            integrality = np.ones(n_k)  # all integer
            bounds = Bounds(lb=np.zeros(n_k), ub=np.ones(n_k))
            
            constraints = LinearConstraint(A, lb, ub)
            
            # Solve with time limit
            res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds,
                      options={"time_limit": milp_budget})
            
            milp_elapsed = time.time() - milp_start
            if res.status == 0 and res.x is not None:
                # Extract solution
                sol = res.x
                milp_solution = set(int(bi) for bi in range(n_k) if sol[bi] > 0.5)
                
                # Check if this is better than current
                if len(milp_solution) < len(selected):
                    selected = milp_solution
                    # Rebuild cover_count
                    cover_count = [0] * n_t
                    for bi in selected:
                        for ti in block_covers[bi]:
                            cover_count[ti] += 1
                    # Deletion reduction on MILP solution
                    for bi in list(selected):
                        if time.time() > deadline - 10.0:
                            break
                        for ti in block_covers[bi]:
                            cover_count[ti] -= 1
                        ok = True
                        for ti in block_covers[bi]:
                            if cover_count[ti] == 0:
                                ok = False
                                break
                        if ok:
                            selected.discard(bi)
                        else:
                            for ti in block_covers[bi]:
                                cover_count[ti] += 1
                    
        except Exception:
            pass  # Fall back to greedy solution if MILP fails
    
    # ---------------- WalkSAT-style local search ----------------
    # State: boolean array of length n_k, True = selected
    x = [bi in selected for bi in range(n_k)]
    # For each clause (t-subset), track which literals satisfy it
    # clause_satisfiers[ti] = list of bi in x that are True and cover ti

    clause_sat_count = [0] * n_t
    for ti in range(n_t):
        clause_sat_count[ti] = sum(1 for bi in t_to_blocks[ti] if x[bi])

    unsat_clauses = [ti for ti in range(n_t) if clause_sat_count[ti] == 0]

    def num_unsat():
        return len(unsat_clauses)

    def num_true():
        return sum(1 for xi in x if xi)

    best_x = x[:]
    best_cost = num_true() + 1000 * num_unsat()  # heavy penalty for unsat

    def update_best():
        nonlocal best_x, best_cost
        cost = num_true() + 1000 * num_unsat()
        if cost < best_cost:
            best_cost = cost
            best_x = x[:]

    update_best()

    # WalkSAT loop
    # Use remaining time for WalkSAT
    remaining_time = deadline - 15.0 - time.time()
    final_selected = set(selected) # Fallback
    
    if remaining_time >= 5.0:
        # Estimate steps based on time
        max_steps = int(remaining_time * 10000)
        max_steps = max(10000, min(max_steps, 500000))

        for step in range(max_steps):
            if time.time() > deadline - 15.0:
                break
            
            if not unsat_clauses:
                # Try to remove a true variable that doesn't break coverage
                # To avoid O(n_k) scan every time, we could maintain a list of true vars, 
                # but for simplicity and given n_k is ~50k, random sampling is okay.
                # However, checking "removable" requires scanning block_covers.
                # We'll just pick a random true var.
                # Optimization: Keep a list of true vars? 
                # For now, random pick from range(n_k) and check if x[bi] is True.
                # This is inefficient if few are true.
                # Let's maintain a list of true vars for efficiency.
                pass # Placeholder for logic below
                
            # Efficient WalkSAT Step:
            # 1. If unsat, pick a clause from unsat_clauses.
            # 2. Pick a variable from that clause.
            # 3. Flip it.
            # 4. Update counts.
            # 5. If satisfied (no unsat), try to remove a random true var.
            
            if unsat_clauses:
                ti = _RNG.choice(unsat_clauses)
                candidates = t_to_blocks[ti]
                if candidates:
                    bi = _RNG.choice(candidates)
                    # Flip
                    old_val = x[bi]
                    x[bi] = not old_val
                    delta = 1 if x[bi] else -1
                    for ti2 in block_covers[bi]:
                        clause_sat_count[ti2] += delta
                    # Rebuild unsat_clauses list (expensive, but necessary for correctness)
                    # Optimization: We can update incrementally, but list reconstruction is simpler and less error-prone.
                    # Given time limit, let's check if we can afford it.
                    # n_t is ~3876. Rebuilding is O(n_t).
                    unsat_clauses = [ti2 for ti2 in range(n_t) if clause_sat_count[ti2] == 0]
                    update_best()
            else:
                # Satisfied. Try to reduce size.
                # Pick a random block. If it's selected, try to remove it.
                # If removal fails, keep it.
                # To be more effective, pick a random selected block.
                # We don't have a fast list of selected blocks here without scanning x.
                # Let's scan x to find a true var.
                bi = _RNG.randrange(n_k)
                if x[bi]:
                    # Try to remove
                    x[bi] = False
                    for ti2 in block_covers[bi]:
                        clause_sat_count[ti2] -= 1
                    # Check if any became 0
                    failed = False
                    for ti2 in block_covers[bi]:
                        if clause_sat_count[ti2] == 0:
                            failed = True
                            break
                    if failed:
                        # Restore
                        x[bi] = True
                        for ti2 in block_covers[bi]:
                            clause_sat_count[ti2] += 1
                    else:
                        # Successful removal
                        update_best()
                        # Update unsat_clauses (should still be empty)
                        # But we might have introduced unsat if our check was wrong? No, we checked.
                        # However, other clauses might have been at 1 and are now at 1? No, we decremented.
                        # If any was 1, it became 0? No, we checked if any became 0.
                        # So unsat_clauses should remain empty.
                        pass
                else:
                    # Picked a false var. Maybe add it? No, we want to minimize.
                    # Just continue.
                    pass

        # Use best found
        final_selected = set(bi for bi in range(n_k) if best_x[bi])

    # Final deletion reduction on the best solution
    # Ensure final_selected is a set of indices
    if not isinstance(final_selected, set):
        final_selected = set(final_selected)
        
    # Re-verify coverage and do a final strict reduction
    cover_count = [0] * n_t
    for bi in final_selected:
        for ti in block_covers[bi]:
            cover_count[ti] += 1
            
    # Check validity first
    if any(c == 0 for c in cover_count):
        # If invalid, fall back to the greedy solution which was verified
        # This should not happen if WalkSAT update_best was correct, but safety first.
        final_selected = set(selected)
        cover_count = [0] * n_t
        for bi in final_selected:
            for ti in block_covers[bi]:
                cover_count[ti] += 1

    # Reduction pass
    for bi in list(final_selected):
        if time.time() > deadline - 2.0:
            break
        # Try to remove bi
        for ti in block_covers[bi]:
            cover_count[ti] -= 1
        
        ok = True
        for ti in block_covers[bi]:
            if cover_count[ti] == 0:
                ok = False
                break
        
        if ok:
            final_selected.discard(bi)
        else:
            # Restore
            for ti in block_covers[bi]:
                cover_count[ti] += 1

    # Build output blocks
    blocks = [list(k_subs[bi]) for bi in sorted(final_selected)]

    # Final Validation Check (just in case)
    # This is expensive, so we only do it if we have time and it's small
    # But the task requires valid output.
    # The reduction logic above ensures validity if it started valid.
    # Greedy ensured validity.
    # WalkSAT update_best only updates best_x if cost is lower.
    # Cost includes penalty for unsat. If unsat > 0, cost is huge.
    # So best_x should always be valid if initial greedy was valid.
    
    return blocks
