import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115

    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}

    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)

    # Precompute triple indices for each block and reverse mapping
    block_triple_indices = [[] for _ in range(num_blocks)]
    triple_to_blocks = [[] for _ in range(num_triples)]

    for i, b in enumerate(all_blocks):
        tris = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            tris.append(idx)
            triple_to_blocks[idx].append(i)
        block_triple_indices[i] = tris

    # Convert to numpy arrays for faster access in local search/SA
    btis = [np.array(bi, dtype=np.int32) for bi in block_triple_indices]

    def greedy_construct(seed):
        rng = random.Random(seed)
        uncovered = np.ones(num_triples, dtype=np.int8)

        selected = []
        selected_set = set()

        while time.time() < deadline - 10:
            best_gain = -1
            candidates = []

            for i in range(num_blocks):
                if i in selected_set:
                    continue
                gain = 0
                for tri_idx in block_triple_indices[i]:
                    if uncovered[tri_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)

            if best_gain <= 0:
                break

            chosen = rng.choice(candidates)
            selected.append(chosen)
            selected_set.add(chosen)

            for tri_idx in block_triple_indices[chosen]:
                uncovered[tri_idx] = 0

            if not uncovered.any():
                break

        return selected

    def local_search_minimize(selected_blocks, time_limit):
        sol = list(selected_blocks)
        sol_set = set(sol)

        counts = np.zeros(num_triples, dtype=np.int8)
        for idx in sol:
            counts[btis[idx]] += 1

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            i = 0
            while i < len(sol):
                idx = sol[i]
                if (counts[btis[idx]] <= 1).all():
                    sol.pop(i)
                    sol_set.discard(idx)
                    counts[btis[idx]] -= 1
                    improved = True
                else:
                    i += 1
        return sol

    def simulated_annealing(initial_sol, time_limit, seed):
        rng = random.Random(seed)
        sol = list(initial_sol)
        sol_set = set(sol)

        counts = np.zeros(num_triples, dtype=np.int8)
        for idx in sol:
            counts[btis[idx]] += 1

        def get_uncovered_count():
            return int(np.sum(counts == 0))

        current_energy = len(sol) + 100 * get_uncovered_count()
        best_sol = list(sol)
        best_energy = current_energy

        T = 5.0
        cooling = 0.99

        while time.time() < time_limit and T > 0.01:
            move_type = rng.random()

            if move_type < 0.7 and sol:
                to_remove = rng.choice(sol)
                redundant = (counts[btis[to_remove]] <= 1).all()
                if redundant:
                    sol.remove(to_remove)
                    sol_set.discard(to_remove)
                    counts[btis[to_remove]] -= 1
                    current_energy -= 1
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)
                else:
                    delta_uncovered = int(np.sum(counts[btis[to_remove]] == 1))
                    delta_e = 1 + 100 * delta_uncovered
                    if rng.random() < np.exp(-delta_e / T):
                        sol.remove(to_remove)
                        sol_set.discard(to_remove)
                        counts[btis[to_remove]] -= 1
                        current_energy += delta_e
            elif move_type < 0.9:
                if not sol:
                    continue
                to_remove = rng.choice(sol)
                uncovered_tris = btis[to_remove][counts[btis[to_remove]] == 1]

                if len(uncovered_tris) == 0:
                    continue

                candidates = []
                for tri_idx in uncovered_tris:
                    for b_idx in triple_to_blocks[tri_idx]:
                        if b_idx in sol_set:
                            continue
                        covers_all = np.all(counts[btis[b_idx][uncovered_tris]] > 0)
                        if covers_all:
                            candidates.append(b_idx)
                            break

                if candidates:
                    to_add = rng.choice(candidates)
                    sol.remove(to_remove)
                    sol_set.discard(to_remove)
                    counts[btis[to_remove]] -= 1

                    sol.append(to_add)
                    sol_set.add(to_add)
                    counts[btis[to_add]] += 1

                    current_energy = len(sol) + 100 * get_uncovered_count()
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)
            else:
                if time.time() > time_limit:
                    break
                to_add = rng.choice(range(num_blocks))
                if to_add in sol_set:
                    continue
                sol.append(to_add)
                sol_set.add(to_add)
                counts[btis[to_add]] += 1

                new_energy = len(sol) + 100 * get_uncovered_count()
                delta = new_energy - current_energy
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)
                else:
                    sol.remove(to_add)
                    sol_set.discard(to_add)
                    counts[btis[to_add]] -= 1

            T *= cooling

        return best_sol

    best_sol = None
    best_size = num_blocks

    num_restarts = 5

    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break

        print(f"Restart {i}", file=sys.stderr)

        g_sol = greedy_construct(1000 + i)
        ls_sol = local_search_minimize(g_sol, deadline - 30)

        if len(ls_sol) < best_size:
            covered = set()
            for idx in ls_sol:
                covered.update(block_triple_indices[idx])
            if len(covered) == num_triples:
                best_size = len(ls_sol)
                best_sol = ls_sol

        if time.time() < deadline - 10:
            sa_sol = simulated_annealing(ls_sol, deadline - 5, 2000 + i)
            if len(sa_sol) < best_size:
                covered = set()
                for idx in sa_sol:
                    covered.update(block_triple_indices[idx])
                if len(covered) == num_triples:
                    best_size = len(sa_sol)
                    best_sol = sa_sol

    if best_sol is None:
        best_sol = greedy_construct(1)
        best_sol = local_search_minimize(best_sol, deadline - 5)

    return [all_blocks[i] for i in best_sol]
