import itertools
import numpy as np
import time
import random
import heapq
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin from 150s limit

    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)

    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    # Build sparse matrix M: M[i,j] = 1 if t-subset i is contained in k-subset j
    rows = []
    cols = []
    data = []

    subsets_of_block = [[] for _ in range(n_k)]
    blocks_of_subset = [[] for _ in range(n_t)]

    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            rows.append(i)
            cols.append(j)
            data.append(1.0)
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)

    M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))

    best_solution = None
    best_size = float('inf')

    # 2. Try LP Relaxation to get duals (weights for MILP objective)
    duals = None
    lp_solution = None
    try:
        if time.time() > deadline - 60:
            raise TimeoutError()

        c_lp = np.ones(n_k)
        A_ub = -M.T.tocsr()
        b_ub = -np.ones(n_t)
        bounds_lp = [(0, 1)] * n_k

        res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs',
                      options={"presolve": True, "time_limit": 20})

        if res.success:
            duals = res.ineqlin.marginals
            duals = np.abs(duals)
            if np.sum(duals) == 0:
                duals = None
            lp_solution = res.x
        else:
            duals = None
            lp_solution = None

    except Exception:
        duals = None
        lp_solution = None

    # 3. Try MILP with time limit (shorter, to leave time for greedy)
    try:
        if time.time() > deadline - 40:
            raise TimeoutError()

        # Use LP duals as objective weights (normalized)
        if duals is not None:
            obj_weights = duals.copy()
            # Normalize so that the objective is comparable
            obj_weights = obj_weights / (np.sum(obj_weights) + 1e-12) * n_k
        else:
            obj_weights = np.ones(n_k)

        # MILP: minimize sum(w_j * x_j)
        # subject to: M.T @ x >= 1 (every t-subset covered)
        #             0 <= x_j <= 1, x_j integer

        c_milp = obj_weights
        # M.T @ x >= 1  =>  -M.T @ x <= -1
        A_milp = -M.T.tocsr()
        b_milp = -np.ones(n_t)

        # Linear constraint: -M.T @ x <= -1
        lin_cons = LinearConstraint(A_milp, -np.inf, b_milp)

        # Integer bounds: 0 <= x_j <= 1
        integrality = np.ones(n_k)
        bounds = Bounds(lb=np.zeros(n_k), ub=np.ones(n_k))

        time_limit = max(1, min(25, int(deadline - time.time()) - 2))

        res_milp = milp(
            c=c_milp,
            constraints=lin_cons,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": time_limit, "mip_rel_gap": 0.0}
        )

        if res_milp.success or res_milp.status == 2:  # status 2 = local optimum
            x_sol = res_milp.x
            if x_sol is not None:
                # Round to integer
                x_int = np.round(x_sol).astype(int)
                # Ensure binary
                x_int = np.clip(x_int, 0, 1)

                # Verify coverage
                covered = M @ x_int
                if np.all(covered >= 1):
                    sol_indices = np.where(x_int == 1)[0]
                    if len(sol_indices) < best_size:
                        best_size = len(sol_indices)
                        best_solution = list(sol_indices)

        # If MILP didn't find a valid solution, try to repair
        if best_solution is None and res_milp.x is not None:
            x_sol = np.round(res_milp.x).astype(int)
            x_sol = np.clip(x_sol, 0, 1)
            covered = M @ x_sol
            if not np.all(covered >= 1):
                # Add missing blocks greedily
                for i in range(n_t):
                    if covered[i] < 1:
                        # Find a block covering this subset
                        for j in blocks_of_subset[i]:
                            if x_sol[j] == 0:
                                x_sol[j] = 1
                                covered = M @ x_sol
                                break
                if np.all(covered >= 1):
                    sol_indices = np.where(x_sol == 1)[0]
                    if len(sol_indices) < best_size:
                        best_size = len(sol_indices)
                        best_solution = list(sol_indices)

    except Exception:
        pass

    # 4. Fallback: Dynamic Greedy with LP duals
    if best_solution is None or time.time() < deadline - 3:
        if duals is None:
            duals = np.ones(n_t)

        # Use remaining time for multiple greedy restarts
        remaining_time = deadline - 5 - time.time()
        # Estimate time per greedy run (~2s typical)
        num_restarts = max(3, int(remaining_time / 2.5))
        num_restarts = min(num_restarts, 15)  # Cap restarts

        for restart in range(num_restarts):
            if time.time() > deadline - 2:
                break

            # Vary the duals slightly for diversity
            if restart == 0:
                block_weights = np.zeros(n_k)
                for j in range(n_k):
                    if subsets_of_block[j]:
                        block_weights[j] = np.sum(duals[subsets_of_block[j]])
            else:
                # Perturb duals for diversity
                perturbed_duals = duals * np.random.uniform(0.8, 1.2, n_t)
                block_weights = np.zeros(n_k)
                for j in range(n_k):
                    if subsets_of_block[j]:
                        block_weights[j] = np.sum(perturbed_duals[subsets_of_block[j]])
                # Add small random noise
                block_weights += np.random.uniform(0, 0.005, n_k)

            heap = [(-block_weights[j], j) for j in range(n_k)]
            heapq.heapify(heap)

            covered = np.zeros(n_t, dtype=bool)
            picked = np.zeros(n_k, dtype=bool)
            solution_blocks = []

            while heap:
                if time.time() > deadline - 1:
                    break

                neg_w, j = heapq.heappop(heap)
                w = -neg_w

                if picked[j]:
                    continue

                if w > block_weights[j] + 1e-9:
                    continue

                if w <= 1e-9:
                    break

                picked[j] = True
                solution_blocks.append(j)

                for s_idx in subsets_of_block[j]:
                    if not covered[s_idx]:
                        covered[s_idx] = True
                        y_s = duals[s_idx]
                        if y_s > 0:
                            for b_prime in blocks_of_subset[s_idx]:
                                if not picked[b_prime]:
                                    block_weights[b_prime] -= y_s
                                    heapq.heappush(heap, (-block_weights[b_prime], b_prime))

                if np.all(covered):
                    break

            if len(solution_blocks) < best_size and np.all(covered):
                best_size = len(solution_blocks)
                best_solution = solution_blocks.copy()

    # 5. New: LP-based rounding strategy
    if lp_solution is not None and time.time() < deadline - 2:
        try:
            # Sort blocks by LP solution value (descending)
            sorted_blocks = np.argsort(lp_solution)[::-1]
            
            covered = np.zeros(n_t, dtype=bool)
            solution_blocks = []
            
            for j in sorted_blocks:
                if time.time() > deadline - 1:
                    break
                if np.all(covered):
                    break
                solution_blocks.append(int(j))
                for s_idx in subsets_of_block[j]:
                    covered[s_idx] = True
            
            if np.all(covered) and len(solution_blocks) < best_size:
                best_size = len(solution_blocks)
                best_solution = solution_blocks.copy()
        except Exception:
            pass

    # 6. Fallback if still no solution
    if best_solution is None:
        uncovered = set(range(n_t))
        res_blocks = []
        while uncovered:
            best_b = -1
            max_cov = -1
            for j in range(n_k):
                cnt = 0
                for s in subsets_of_block[j]:
                    if s in uncovered:
                        cnt += 1
                if cnt > max_cov:
                    max_cov = cnt
                    best_b = j
            if best_b == -1 or max_cov == 0:
                break
            res_blocks.append(best_b)
            for s in subsets_of_block[best_b]:
                if s in uncovered:
                    uncovered.remove(s)
        best_solution = res_blocks
        best_size = len(res_blocks)

    # 7. Post-processing: Deletion Pass (multiple passes)
    if best_solution is not None and time.time() < deadline - 1:
        for pass_num in range(3):
            if time.time() > deadline - 1:
                break
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b_idx in best_solution:
                for s_idx in subsets_of_block[b_idx]:
                    cover_count[s_idx] += 1

            changed = True
            while changed and time.time() < deadline - 1:
                changed = False
                new_solution = []
                for b_idx in best_solution:
                    subs_b = subsets_of_block[b_idx]
                    can_remove = True
                    for s_idx in subs_b:
                        if cover_count[s_idx] == 1:
                            can_remove = False
                            break

                    if can_remove:
                        cover_count[subs_b] -= 1
                        changed = True
                    else:
                        new_solution.append(b_idx)

                best_solution = new_solution
                best_size = len(best_solution)

    return [list(k_subsets[idx]) for idx in best_solution]
