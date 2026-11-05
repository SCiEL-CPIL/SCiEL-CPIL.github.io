"""
Maximum Independent Set via Local Search on the Independent Set Itself.

Instead of optimizing a priority vector and evaluating it with a greedy decoder,
we directly optimize the set of vertices in the independent set.
We start with a greedy set (using min-degree heuristic) and then perform local
search moves (add, remove, swap) to maximize the set size while maintaining
the independent set property.

This is a structural change: the search space is the space of independent sets,
not the space of score vectors. The greedy decoder is only used for initialization.
"""
import numpy as np
import time
import random

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin below 150s limit

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric and zero diagonal
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5

    # Precompute neighbor lists and sets
    neighbors = [np.where(A_bool[i])[0] for i in range(N)]
    neighbor_sets = [set(n.tolist()) for n in neighbors]
    degrees = np.array([len(neighbors[i]) for i in range(N)])

    # --- Greedy Decoder (for initialization and fallback) ---
    def greedy_decode(scores):
        order = np.argsort(-scores, kind='stable')
        blocked = np.zeros(N, dtype=bool)
        picked = []
        for v in order:
            if not blocked[v]:
                picked.append(v)
                blocked[v] = True
                if neighbors[v].size > 0:
                    blocked[neighbors[v]] = True
        return picked

    # --- Fast Independent Set Size Evaluation ---
    def is_valid_independent_set(vertex_list):
        """Check if vertex_list is a valid independent set."""
        vset = set(vertex_list)
        for v in vertex_list:
            for u in neighbors[v]:
                if u in vset:
                    return False
        return True

    # --- Local Search on Independent Sets ---
    def local_search(initial_set, max_time=140.0):
        """
        Perform local search starting from initial_set.
        Moves:
        1. Add: Try adding a vertex not in the set and not adjacent to any vertex in the set.
        2. Remove: Try removing a vertex from the set (to enable other additions later).
        3. Swap: Remove one vertex and add another.
        """
        current_set = set(initial_set)
        current_size = len(current_set)
        best_set = set(current_set)
        best_size = current_size
        no_improve_count = 0
        max_no_improve = 5000

        # Precompute for each vertex: is it a candidate to add?
        # A vertex v can be added if it's not in current_set and no neighbor of v is in current_set.

        def can_add(v):
            if v in current_set:
                return False
            for u in neighbors[v]:
                if u in current_set:
                    return False
            return True

        def try_add():
            nonlocal current_set, current_size, best_set, best_size, no_improve_count
            # Find all vertices that can be added
            candidates = [v for v in range(N) if can_add(v)]
            if not candidates:
                return False
            # Try adding each candidate (or a random subset for speed)
            # For efficiency, try a random sample if too many candidates
            if len(candidates) > 20:
                sample = random.sample(candidates, 20)
            else:
                sample = candidates
            for v in sample:
                new_set = current_set | {v}
                new_size = current_size + 1
                if new_size > best_size:
                    current_set = new_set
                    current_size = new_size
                    best_set = set(current_set)
                    best_size = new_size
                    no_improve_count = 0
                    return True
            return False

        def try_swap():
            nonlocal current_set, current_size, best_set, best_size, no_improve_count
            # Remove a random vertex from the set
            if not current_set:
                return False
            v_remove = random.choice(list(current_set))
            new_set = current_set - {v_remove}
            # Now try to add one or more vertices that were blocked by v_remove
            # Vertices that are neighbors of v_remove and not in new_set might now be addable
            candidates = []
            for u in neighbors[v_remove]:
                if u not in new_set:
                    # Check if u can be added to new_set
                    ok = True
                    for w in neighbors[u]:
                        if w in new_set:
                            ok = False
                            break
                    if ok:
                        candidates.append(u)
            # Also try vertices that were already addable (not blocked by v_remove)
            # But that's expensive, so just try the newly unblocked ones
            if not candidates:
                # Try a random vertex not in new_set
                all_others = [v for v in range(N) if v not in new_set]
                if all_others:
                    candidates = random.sample(all_others, min(10, len(all_others)))
            
            for v_add in candidates:
                new_set2 = new_set | {v_add}
                new_size = len(new_set2)
                if new_size > best_size:
                    current_set = new_set2
                    current_size = new_size
                    best_set = set(current_set)
                    best_size = new_size
                    no_improve_count = 0
                    return True
            # Even if we don't improve, we might want to accept the swap to escape local optima
            # Accept with some probability (simulated annealing style)
            if random.random() < 0.1:  # 10% chance to accept non-improving swap
                current_set = new_set
                current_size = len(current_set)
                no_improve_count += 1
                return True
            return False

        def try_remove():
            nonlocal current_set, current_size, best_set, best_size, no_improve_count
            # Remove a random vertex (this decreases size, but might enable better swaps)
            if not current_set:
                return False
            v_remove = random.choice(list(current_set))
            new_set = current_set - {v_remove}
            current_set = new_set
            current_size = len(current_set)
            no_improve_count += 1
            return True

        while time.time() < max_time:
            r = random.random()
            if r < 0.5:
                improved = try_add()
            elif r < 0.9:
                improved = try_swap()
            else:
                improved = try_remove()
            
            if not improved:
                no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Restart from a slightly perturbed best_set
                current_set = set(best_set)
                # Remove a random vertex from best_set
                if current_set:
                    v_remove = random.choice(list(current_set))
                    current_set.discard(v_remove)
                current_size = len(current_set)
                no_improve_count = 0
                max_no_improve = int(max_no_improve * 0.8)
                if max_no_improve < 1000:
                    max_no_improve = 1000

        return best_set

    # --- Phase 1: Generate initial independent sets using various heuristics ---
    
    # Heuristic 1: Min degree
    scores1 = -degrees.astype(float)
    set1 = set(greedy_decode(scores1))

    # Heuristic 2: Min degree + low clustering
    A_sq = A @ A
    Tri = np.diag(A_sq) / 2.0
    D_safe = np.where(degrees > 0, degrees, 1.0)
    Clus = np.zeros(N)
    mask = degrees >= 2
    Clus[mask] = 2.0 * Tri[mask] / (degrees[mask] * (degrees[mask] - 1.0))
    Clus_n = (Clus - Clus.min()) / (Clus.max() - Clus.min() + 1e-10)
    scores2 = -degrees.astype(float) - 0.5 * Clus_n
    set2 = set(greedy_decode(scores2))

    # Heuristic 3: Random
    rng = np.random.default_rng(42)
    scores3 = rng.normal(0, 1, N)
    set3 = set(greedy_decode(scores3))

    # Heuristic 4: Min mean neighbor degree
    SND = A @ degrees
    MeanND = SND / D_safe
    MeanND_n = (MeanND - MeanND.min()) / (MeanND.max() - MeanND.min() + 1e-10)
    scores4 = -degrees.astype(float) - 0.5 * MeanND_n
    set4 = set(greedy_decode(scores4))

    initial_sets = [set1, set2, set3, set4]

    # --- Phase 2: Local search on each initial set ---
    best_overall_set = set()
    best_overall_size = 0

    for init_set in initial_sets:
        if time.time() > DEADLINE - 5:
            break
        result_set = local_search(init_set, max_time=DEADLINE - 5)
        if len(result_set) > best_overall_size:
            best_overall_size = len(result_set)
            best_overall_set = result_set

    # --- Phase 3: Final refinement with more aggressive local search ---
    if time.time() < DEADLINE - 2 and best_overall_set:
        # Run local search again from the best set with a tighter time budget
        final_set = local_search(best_overall_set, max_time=DEADLINE - 2)
        if len(final_set) > best_overall_size:
            best_overall_size = len(final_set)
            best_overall_set = final_set

    # --- Convert best independent set to priority scores ---
    # Vertices in the independent set get high scores, others get low scores.
    # The exact values don't matter as long as the greedy decoder picks the same set.
    scores = np.zeros(N)
    for v in best_overall_set:
        scores[v] = 1.0
    # Vertices not in the set get 0.0
    # But we need to ensure the greedy decoder picks exactly these vertices.
    # Since the set is an independent set, and we assign 1.0 to all of them,
    # the greedy decoder will pick all of them (in any order) before picking any 0.0 vertex.
    # So this works.

    return scores
