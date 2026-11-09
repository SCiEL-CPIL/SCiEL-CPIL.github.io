import numpy as np
import time
import itertools

def heuristics(prizes, distance_matrix, max_len):
    start = time.time()
    DEADLINE = start + 300.0
    SAFE = 285.0  # seconds to stop spawning new search and return best

    n = len(prizes)
    d = np.asarray(distance_matrix, dtype=float)
    p = np.asarray(prizes, dtype=float)
    d0 = d[:, 0]

    # ---------- Greedy tour builder ----------
    def greedy_tour(p, d, d0, max_len, h_matrix, seed, alpha=1.0, beta=1.0):
        """Greedy with random tie-breaking using heuristic h_matrix."""
        rng = np.random.RandomState(seed)
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        cur = 0
        length = 0.0
        total = p[0]
        while True:
            best_j = -1
            best_score = -np.inf
            for j in range(1, n):
                if visited[j]:
                    continue
                if length + d[cur, j] + d0[j] > max_len:
                    continue
                # score = prize * heuristic^alpha / (dist^beta)
                hj = h_matrix[cur, j]
                if hj <= 0 or not np.isfinite(hj):
                    hj = 1e-9
                score = p[j] * (hj ** alpha) / ((d[cur, j] + 1e-9) ** beta)
                if best_score < 0:
                    best_score = score
                    best_j = j
                elif score > best_score:
                    best_score = score
                    best_j = j
            if best_j < 0:
                break
            length += d[cur, best_j]
            visited[best_j] = True
            total += p[best_j]
            cur = best_j
        return total

    # ---------- Stochastic beam search over tours ----------
    def beam_tour(p, d, d0, max_len, h_matrix, seed, beam_width=5, alpha=1.0, beta=1.0):
        """Beam search: keep top beam_width partial tours, expand each."""
        rng = np.random.RandomState(seed)
        # state: (node, length, prize_sum, visited_mask)
        states = [(0, 0.0, p[0], 1)]
        best_total = p[0]
        for _ in range(n - 1):
            new_states = []
            for (cur, length, total, mask) in states:
                for j in range(1, n):
                    if mask & (1 << j):
                        continue
                    if length + d[cur, j] + d0[j] > max_len:
                        continue
                    new_states.append((j, length + d[cur, j], total + p[j], mask | (1 << j)))
            if not new_states:
                break
            # rank by total prize (with small random perturbation)
            new_states.sort(key=lambda s: -(s[2] + rng.rand() * 1e-6))
            states = new_states[:beam_width]
            for s in states:
                if s[2] > best_total:
                    best_total = s[2]
        return best_total

    # ---------- Heuristic parameter space ----------
    # h[i,j] = p[j] * exp(-a * d[i,j]) * (1 + b * d0[j]) / (c + d[i,j])
    # plus a "return-aware" term: * (1 + e * d0[i])
    # Parameters: a (distance decay), b (return-cost bonus), c (distance floor), e (origin-distance bonus)
    def make_heuristic(params):
        a, b, c, e = params
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    H[i, j] = 1e-9
                    continue
                base = p[j] * np.exp(-a * d[i, j]) * (1.0 + b * d0[j]) / (c + d[i, j])
                base *= (1.0 + e * d0[i])
                H[i, j] = max(base, 1e-9)
        return H

    # ---------- Evaluate a heuristic via greedy + beam ----------
    def eval_heuristic(params, seed):
        H = make_heuristic(params)
        g = greedy_tour(p, d, d0, max_len, H, seed, alpha=1.0, beta=1.0)
        b = beam_tour(p, d, d0, max_len, H, seed, beam_width=3, alpha=1.0, beta=1.0)
        return max(g, b), H

    # ---------- Evolutionary search over parameters ----------
    # Initial population: diverse param vectors
    rng_main = np.random.RandomState(12345)
    pop_size = 8
    params_pop = []
    for _ in range(pop_size):
        a = rng_main.uniform(0.5, 5.0)
        b = rng_main.uniform(0.0, 2.0)
        c = rng_main.uniform(0.01, 0.5)
        e = rng_main.uniform(0.0, 1.5)
        params_pop.append(np.array([a, b, c, e]))
    # Add some structured initial points
    params_pop.append(np.array([2.0, 0.5, 0.1, 0.5]))
    params_pop.append(np.array([1.0, 1.0, 0.05, 1.0]))
    params_pop.append(np.array([3.0, 0.0, 0.2, 0.0]))

    best_params = params_pop[0].copy()
    best_score = -np.inf
    best_H = make_heuristic(best_params)

    # Evaluate initial population
    for idx, params in enumerate(params_pop):
        if time.time() > start + SAFE:
            break
        score, H = eval_heuristic(params, seed=idx + 1)
        if score > best_score:
            best_score = score
            best_params = params.copy()
            best_H = H

    # Evolutionary loop
    gen = 0
    while time.time() < start + SAFE:
        gen += 1
        new_pop = []
        # Elitism: keep best
        new_pop.append(best_params.copy())
        # Crossover + mutation
        for _ in range(pop_size - 1):
            # Pick two parents (tournament)
            i1 = rng_main.randint(0, len(params_pop))
            i2 = rng_main.randint(0, len(params_pop))
            p1 = params_pop[i1]
            p2 = params_pop[i2]
            # Uniform crossover
            child = np.where(rng_main.rand(4) < 0.5, p1, p2)
            # Mutation
            for k in range(4):
                if rng_main.rand() < 0.3:
                    child[k] *= (1.0 + rng_main.uniform(-0.3, 0.3))
                    child[k] = max(child[k], 1e-4)
            new_pop.append(child)

        params_pop = new_pop
        for idx, params in enumerate(params_pop):
            if time.time() > start + SAFE:
                break
            score, H = eval_heuristic(params, seed=gen * 100 + idx)
            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_H = H

    # Final polish: try a few more seeds on the best heuristic
    if time.time() < start + SAFE:
        for s in range(10):
            if time.time() > start + SAFE:
                break
            score, H = eval_heuristic(best_params, seed=1000 + s)
            if score > best_score:
                best_score = score
                best_H = H

    # Sanitize
    best_H = np.where(np.isfinite(best_H) & (best_H > 0), best_H, 1e-9)
    return best_H
