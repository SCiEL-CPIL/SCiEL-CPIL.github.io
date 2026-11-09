def sign_edges(graph, budget):
    import numpy as np
    import time

    start_time = time.time()
    DEADLINE = 85.0

    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)

    u_arr = np.array([e[0] for e in edges], dtype=np.int32)
    v_arr = np.array([e[1] for e in edges], dtype=np.int32)

    def is_lift_connected(signs):
        lift_adj = [[] for _ in range(2 * n)]
        for i in range(num_edges):
            u = u_arr[i]
            v = v_arr[i]
            s = signs[i]
            if s == 1:
                lift_adj[2 * u].append(2 * v)
                lift_adj[2 * v].append(2 * u)
                lift_adj[2 * u + 1].append(2 * v + 1)
                lift_adj[2 * v + 1].append(2 * u + 1)
            else:
                lift_adj[2 * u].append(2 * v + 1)
                lift_adj[2 * v + 1].append(2 * u)
                lift_adj[2 * u + 1].append(2 * v)
                lift_adj[2 * v].append(2 * u + 1)
        visited = [False] * (2 * n)
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            node = queue.pop(0)
            for nb in lift_adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    queue.append(nb)
        return count == 2 * n

    def spectral_radius(signs):
        A = np.zeros((n, n), dtype=float)
        for i in range(num_edges):
            s = float(signs[i])
            A[u_arr[i], v_arr[i]] = s
            A[v_arr[i], u_arr[i]] = s
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))

    def evaluate(signs):
        if not is_lift_connected(signs):
            return float('inf')
        return spectral_radius(signs)

    rng = np.random.default_rng(3)

    best_signs = None
    best_radius = float('inf')

    def update_best(signs, radius):
        nonlocal best_signs, best_radius
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()

    def local_search(signs, max_flips=10000):
        current = signs.copy()
        current_r = evaluate(current)
        if current_r == float('inf'):
            return current, current_r
        update_best(current, current_r)
        flips = 0
        improved = True
        while improved and flips < max_flips:
            if time.time() > DEADLINE:
                break
            improved = False
            order = rng.permutation(num_edges)
            for i in order:
                if time.time() > DEADLINE:
                    break
                old = current[i]
                current[i] = -old
                new_r = evaluate(current)
                if new_r < current_r:
                    current_r = new_r
                    flips += 1
                    improved = True
                    update_best(current, current_r)
                else:
                    current[i] = old
        return current, current_r

    def sim_anneal(signs, num_steps=20000, T_start=0.5, T_end=0.01):
        current = signs.copy()
        current_r = evaluate(current)
        if current_r == float('inf'):
            return current, current_r
        update_best(current, current_r)
        best_local = current.copy()
        best_local_r = current_r
        for step in range(num_steps):
            if time.time() > DEADLINE:
                break
            T = T_start * (T_end / T_start) ** (step / max(num_steps, 1))
            i = int(rng.integers(num_edges))
            old = current[i]
            current[i] = -old
            new_r = evaluate(current)
            delta = new_r - current_r
            if delta < 0 or (T > 1e-10 and rng.random() < np.exp(-delta / T)):
                current_r = new_r
                if new_r < best_local_r:
                    best_local_r = new_r
                    best_local = current.copy()
                    update_best(best_local, best_local_r)
            else:
                current[i] = old
        return best_local, best_local_r

    def phase_init():
        thetas = rng.random(n) * 2 * np.pi
        signs = np.ones(num_edges, dtype=np.int8)
        for i in range(num_edges):
            diff = thetas[u_arr[i]] - thetas[v_arr[i]]
            if np.cos(diff) < 0:
                signs[i] = -1
        return signs

    # Phase 1: Random starts + local search
    for start in range(30):
        if time.time() > DEADLINE:
            break
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        local_search(signs, max_flips=3000)

    # Phase 2: Phase-based starts + local search
    for start in range(20):
        if time.time() > DEADLINE:
            break
        signs = phase_init()
        local_search(signs, max_flips=5000)

    # Phase 3: Simulated annealing from best
    if best_signs is not None:
        sim_anneal(best_signs, num_steps=30000)

    # Phase 4: More random starts
    for start in range(30):
        if time.time() > DEADLINE:
            break
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        local_search(signs, max_flips=5000)

    # Phase 5: Final polish
    if best_signs is not None:
        local_search(best_signs, max_flips=15000)

    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=np.int8)

    return best_signs
