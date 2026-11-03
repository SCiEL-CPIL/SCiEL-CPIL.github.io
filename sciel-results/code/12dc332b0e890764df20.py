"""Kissing number in dimension 11: structural reformulation as MIS on support graph + GA."""
import numpy as np
import itertools
import time
import random


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    D = 11

    # Generate all 4-subsets (supports)
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)  # 330

    # Build conflict graph: edge if |intersection| == 3
    adj = [set() for _ in range(n_sup)]
    for i in range(n_sup):
        s_i = set(supports[i])
        for j in range(i + 1, n_sup):
            s_j = set(supports[j])
            if len(s_i & s_j) == 3:
                adj[i].add(j)
                adj[j].add(i)

    # Precompute adjacency as sorted lists and numpy bitsets for fast operations
    adj_list = [sorted(adj[i]) for i in range(n_sup)]
    # For each node, store its neighbors as a numpy boolean mask (length n_sup)
    adj_mask = np.zeros((n_sup, n_sup), dtype=np.uint8)
    for i in range(n_sup):
        for j in adj_list[i]:
            adj_mask[i, j] = 1
            adj_mask[j, i] = 1

    def greedy_is():
        remaining = set(range(n_sup))
        current_set = set()
        while remaining:
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            current_set.add(best_v)
            to_remove = {best_v} | (adj[best_v] & remaining)
            remaining -= to_remove
        return current_set

    def set_to_mask(s):
        mask = np.zeros(n_sup, dtype=np.uint8)
        for i in s:
            mask[i] = 1
        return mask

    def mask_to_set(mask):
        return set(np.where(mask)[0].tolist())

    def repair_local_search(mask, max_iters=500):
        """Remove violations from mask until it's a valid independent set."""
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            # Find all violations: pairs (i,j) both in mask with adj_mask[i,j]=1
            # Vectorized: count for each i in mask, how many neighbors are also in mask
            violation_counts = np.sum(adj_mask * mask[None, :], axis=1)
            # Nodes in mask that have at least one neighbor also in mask
            bad_nodes = np.where((mask > 0) & (violation_counts > 0))[0]
            if len(bad_nodes) == 0:
                break
            # Remove the node with the most violations (or random among worst)
            worst = bad_nodes[np.argmax(violation_counts[bad_nodes])]
            mask[worst] = 0
        return mask

    def local_search_improve(mask, max_iters=200):
        """Try to add nodes or swap to improve the independent set."""
        for _ in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            # Try adding a random node not in mask
            candidates = np.where(mask == 0)[0]
            if len(candidates) == 0:
                break
            v = candidates[random.randint(0, len(candidates) - 1)]
            # Check if v can be added: no neighbors in mask
            neighbors_in_mask = np.sum(adj_mask[v] * mask)
            if neighbors_in_mask == 0:
                mask[v] = 1
                continue
            # Try swap: remove some neighbors and add v
            # Only worthwhile if we remove fewer than 1 (i.e., 0) - already handled
            # Or if we can remove exactly 1 and it's a good swap (net 0 but different structure)
            # For now, skip swaps in this simple version
        return mask

    # Initial population via greedy + random perturbations
    np.random.seed(42)
    random.seed(42)

    greedy_set = greedy_is()
    greedy_mask = set_to_mask(greedy_set)
    greedy_mask = repair_local_search(greedy_mask)

    population = [greedy_mask.copy()]

    # Create diverse initial population
    for i in range(39):
        if time.time() > deadline - 60.0:
            break
        # Start from greedy, then randomly remove some nodes and repair
        m = greedy_mask.copy()
        # Remove 10-30% of nodes randomly
        n_remove = random.randint(5, int(0.3 * np.sum(m)))
        ones = np.where(m)[0]
        to_remove = random.sample(ones.tolist(), min(n_remove, len(ones)))
        for idx in to_remove:
            m[idx] = 0
        # Try to add back some nodes
        m = local_search_improve(m, max_iters=100)
        m = repair_local_search(m)
        population.append(m)

    # Genetic Algorithm
    pop_size = len(population)
    best_mask = None
    best_size = -1
    for m in population:
        s = int(np.sum(m))
        if s > best_size:
            best_size = s
            best_mask = m.copy()

    generation = 0
    max_generations = 500
    while generation < max_generations and time.time() < deadline - 30.0:
        generation += 1

        # Evaluate fitness (size of independent set)
        fitness = [int(np.sum(m)) for m in population]

        # Update best
        for i, f in enumerate(fitness):
            if f > best_size:
                best_size = f
                best_mask = population[i].copy()

        # Selection: tournament selection
        def tournament_select(k=3):
            idxs = random.sample(range(pop_size), k)
            return max(idxs, key=lambda i: fitness[i])

        # Create new population
        new_population = []

        # Elitism: keep top 2
        sorted_idx = sorted(range(pop_size), key=lambda i: fitness[i], reverse=True)
        for i in sorted_idx[:2]:
            new_population.append(population[i].copy())

        # Crossover and mutation
        while len(new_population) < pop_size:
            if time.time() > deadline - 30.0:
                break

            # Select parents
            p1_idx = tournament_select()
            p2_idx = tournament_select()
            p1 = population[p1_idx]
            p2 = population[p2_idx]

            # Uniform crossover
            child = np.zeros(n_sup, dtype=np.uint8)
            for j in range(n_sup):
                if random.random() < 0.5:
                    child[j] = p1[j]
                else:
                    child[j] = p2[j]

            # Mutation: flip random bits
            n_mutations = random.randint(1, 5)
            for _ in range(n_mutations):
                pos = random.randint(0, n_sup - 1)
                child[pos] = 1 - child[pos]

            # Repair: ensure valid independent set
            child = repair_local_search(child, max_iters=200)

            # Local improvement
            child = local_search_improve(child, max_iters=50)

            new_population.append(child)

        population = new_population

    # Final local search on best
    best_mask = local_search_improve(best_mask, max_iters=500)
    best_mask = repair_local_search(best_mask, max_iters=100)

    # Extract best set
    best_set = mask_to_set(best_mask)

    # Construct vectors from best_set: all 16 sign patterns for each chosen support
    vectors = []
    for idx in best_set:
        sup = supports[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in enumerate(sup):
                v[s] = signs[k]
            vectors.append(v)

    # Now try to add weight-3, weight-2, weight-1 vectors that are compatible with all
    # Generate weight-3 vectors
    weight3_candidates = []
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            weight3_candidates.append(v)

    # Generate weight-2 vectors
    weight2_candidates = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            weight2_candidates.append(v)

    # Generate weight-1 vectors
    weight1_candidates = []
    for pos in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[pos] = s
            weight1_candidates.append(v)

    # Check compatibility of each additional candidate with all existing vectors
    existing = np.array(vectors, dtype=np.int32)
    if len(existing) > 0:
        existing_norms_sq = np.sum(existing ** 2, axis=1)
        max_norm_sq = int(existing_norms_sq.max())

        # Try adding weight-3 vectors
        for w3 in weight3_candidates:
            if time.time() > deadline - 5.0:
                break
            w3_arr = np.array(w3, dtype=np.int32)
            w3_norm_sq = int(np.sum(w3_arr ** 2))
            dots = existing @ w3_arr
            min_dist_sq = w3_norm_sq + existing_norms_sq - 2 * dots
            if np.all(min_dist_sq >= max_norm_sq):
                vectors.append(w3)
                existing = np.vstack([existing, w3_arr])
                existing_norms_sq = np.sum(existing ** 2, axis=1)
                max_norm_sq = int(existing_norms_sq.max())

        # Try adding weight-2 vectors
        for w2 in weight2_candidates:
            if time.time() > deadline - 3.0:
                break
            w2_arr = np.array(w2, dtype=np.int32)
            w2_norm_sq = int(np.sum(w2_arr ** 2))
            dots = existing @ w2_arr
            min_dist_sq = w2_norm_sq + existing_norms_sq - 2 * dots
            if np.all(min_dist_sq >= max_norm_sq):
                vectors.append(w2)
                existing = np.vstack([existing, w2_arr])
                existing_norms_sq = np.sum(existing ** 2, axis=1)
                max_norm_sq = int(existing_norms_sq.max())

        # Try adding weight-1 vectors
        for w1 in weight1_candidates:
            if time.time() > deadline - 2.0:
                break
            w1_arr = np.array(w1, dtype=np.int32)
            w1_norm_sq = int(np.sum(w1_arr ** 2))
            dots = existing @ w1_arr
            min_dist_sq = w1_norm_sq + existing_norms_sq - 2 * dots
            if np.all(min_dist_sq >= max_norm_sq):
                vectors.append(w1)
                existing = np.vstack([existing, w1_arr])
                existing_norms_sq = np.sum(existing ** 2, axis=1)
                max_norm_sq = int(existing_norms_sq.max())

    if not vectors:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    result = np.array(vectors, dtype=np.int64)
    return result
