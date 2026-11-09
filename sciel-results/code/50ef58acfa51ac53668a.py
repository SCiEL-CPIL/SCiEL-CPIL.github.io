def construct():
    import time
    import numpy as np
    from scipy.optimize import linprog

    t0 = time.time()
    DEADLINE = 140.0

    def instance_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def reconstruct_paths(directed, n):
        """Given a list of directed edges (u,v) forming a collection of paths,
        reconstruct the vertex lists of the paths."""
        out_adj = {i: [] for i in range(n)}
        in_deg = {i: 0 for i in range(n)}
        for u, v in directed:
            out_adj[u].append(v)
            in_deg[v] += 1

        # starts: vertices with in_deg 0 and out_deg > 0
        starts = [v for v in range(n) if in_deg[v] == 0 and out_adj[v]]
        # Also handle vertices with in_deg==0 and out==0 (isolated, no edge) -> skip
        paths = []
        for s in starts:
            path = [s]
            cur = s
            while out_adj[cur]:
                nxt = out_adj[cur].pop(0)
                path.append(nxt)
                cur = nxt
            if len(path) > 1:
                paths.append(path)
        return paths

    def solve_instance(n, p, seed):
        edges = instance_graph(n, p, seed)
        if not edges:
            return []
        m = len(edges)

        # Build orientation LP.
        # Variable x_i in {0,1} for edge i=(u,v): 1 means orient u->v.
        # imbalance(v) = sum over incident edges of contribution.
        # For edge i=(u,v): if x_i=1 (u->v): outdeg(u)+=1, indeg(v)+=1
        #                    if x_i=0 (v->u): outdeg(v)+=1, indeg(u)+=1
        # imbalance(u) gets: +1 if x_i=1, -1 if x_i=0  =>  2*x_i - 1
        # imbalance(v) gets: -1 if x_i=1, +1 if x_i=0  =>  1 - 2*x_i
        #
        # Objective: min sum_v t_v, t_v >= imbalance(v), t_v >= -imbalance(v), t_v >= 0
        #
        # num_vars = m (x) + n (t)
        num_vars = m + n
        c = np.zeros(num_vars)
        for v in range(n):
            c[m + v] = 1.0

        # Precompute for each vertex v, the coefficient of each x_i in imbalance(v).
        # imbalance(v) = sum_i coef[v][i] * x_i + const[v]
        coef = np.zeros((n, m))
        const = np.zeros(n)
        for i, (u, v) in enumerate(edges):
            # edge i: x_i=1 => u->v
            # imbalance(u): 2*x_i - 1  => coef += 2, const += -1
            # imbalance(v): 1 - 2*x_i  => coef += -2, const += +1
            coef[u][i] += 2.0
            const[u] += -1.0
            coef[v][i] += -2.0
            const[v] += +1.0

        A_rows = []
        b_vals = []

        # t_v - imbalance(v) >= 0  =>  t_v - (coef[v]@x + const[v]) >= 0
        #  => -t_v + coef[v]@x + const[v] <= 0
        for v in range(n):
            row = np.zeros(num_vars)
            row[:m] = coef[v]
            row[m + v] = -1.0
            A_rows.append(row)
            b_vals.append(-const[v])

        # t_v + imbalance(v) >= 0  =>  t_v + coef[v]@x + const[v] >= 0
        #  => -t_v - coef[v]@x - const[v] <= 0
        for v in range(n):
            row = np.zeros(num_vars)
            row[:m] = -coef[v]
            row[m + v] = -1.0
            A_rows.append(row)
            b_vals.append(const[v])

        A_ub = np.array(A_rows)
        b_ub = np.array(b_vals)
        bounds = [(0, 1)] * m + [(0, None)] * n

        best_x = None
        best_obj = None
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                best_x = res.x[:m]
                best_obj = res.fun
        except Exception:
            pass

        # Round to integer orientation
        def round_to_orientation(x):
            orient = []
            for i in range(m):
                if x[i] > 0.5:
                    orient.append(1)
                else:
                    orient.append(0)
            return orient

        def build_directed(orient):
            directed = []
            for i, (u, v) in enumerate(edges):
                if orient[i] == 1:
                    directed.append((u, v))
                else:
                    directed.append((v, u))
            return directed

        def path_count(directed):
            # number of paths = sum_v max(0, outdeg - indeg)
            outd = {i: 0 for i in range(n)}
            ind = {i: 0 for i in range(n)}
            for u, v in directed:
                outd[u] += 1
                ind[v] += 1
            return sum(max(0, outd[v] - ind[v]) for v in range(n))

        if best_x is None:
            # fallback: all orient 1
            orient = [1] * m
        else:
            orient = round_to_orientation(best_x)

        directed = build_directed(orient)
        cur_count = path_count(directed)

        # Local search: try flipping individual edges, and 2-edge flips.
        def evaluate_orient(orient):
            directed = build_directed(orient)
            return path_count(directed)

        # Single flips
        improved = True
        while improved and (time.time() - t0) < DEADLINE - 5:
            improved = False
            for i in range(m):
                if (time.time() - t0) > DEADLINE - 5:
                    break
                new_orient = orient[:]
                new_orient[i] = 1 - new_orient[i]
                new_count = evaluate_orient(new_orient)
                if new_count < cur_count:
                    orient = new_orient
                    cur_count = new_count
                    improved = True

        # 2-edge flips
        improved = True
        while improved and (time.time() - t0) < DEADLINE - 3:
            improved = False
            for i in range(m):
                for j in range(i + 1, m):
                    if (time.time() - t0) > DEADLINE - 3:
                        break
                    new_orient = orient[:]
                    new_orient[i] = 1 - new_orient[i]
                    new_orient[j] = 1 - new_orient[j]
                    new_count = evaluate_orient(new_orient)
                    if new_count < cur_count:
                        orient = new_orient
                        cur_count = new_count
                        improved = True

        directed = build_directed(orient)
        return reconstruct_paths(directed, n)

    results = []
    for n, p, seed in [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]:
        if time.time() > DEADLINE:
            edges = instance_graph(n, p, seed)
            # trivial: each edge its own path
            results.append([[u, v] for u, v in edges])
        else:
            results.append(solve_instance(n, p, seed))
    return results
