<<<<<<< SEARCH
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
=======
    # 3. Try MILP with time limit (shorter, to leave time for greedy)
    try:
        if time.time() > deadline - 40:
            raise TimeoutError()

        # Use natural objective: minimize number of blocks selected
        # LP duals inform branching but objective stays cardinality
        c_milp = np.ones(n_k)
        # M.T @ x >= 1  =>  -M.T @ x <= -1
        A_milp = -M.T.tocsr()
        b_milp = -np.ones(n_t)

        # Linear constraint: -M.T @ x <= -1
        lin_cons = LinearConstraint(A_milp, -np.inf, b_milp)

        # Integer bounds: 0 <= x_j <= 1
        integrality = np.ones(n_k)
        bounds = Bounds(lb=np.zeros(n_k), ub=np.ones(n_k))

        time_limit = max(1, min(30, int(deadline - time.time()) - 2))

        res_milp = milp(
            c=c_milp,
            constraints=lin_cons,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": time_limit, "mip_rel_gap": 0.01}
        )

        if res_milp.success or res_milp.status in (2, 3):  # status 2=local opt, 3=best found
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
                # Add missing blocks greedily, prioritizing blocks that cover most uncovered subsets
                uncovered_mask = covered < 1
                while np.any(uncovered_mask) and time.time() < deadline - 3:
                    # Find block covering most uncovered subsets
                    best_j = -1
                    best_cnt = 0
                    for j in range(n_k):
                        if x_sol[j] == 0:
                            cnt = sum(1 for s in subsets_of_block[j] if uncovered_mask[s])
                            if cnt > best_cnt:
                                best_cnt = cnt
                                best_j = j
                    if best_j == -1 or best_cnt == 0:
                        break
                    x_sol[best_j] = 1
                    for s in subsets_of_block[best_j]:
                        uncovered_mask[s] = False
                if not np.any(uncovered_mask):
                    sol_indices = np.where(x_sol == 1)[0]
                    if len(sol_indices) < best_size:
                        best_size = len(sol_indices)
                        best_solution = list(sol_indices)

    except Exception:
        pass
>>>>>>> REPLACE

