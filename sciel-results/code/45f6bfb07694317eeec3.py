"""Kissing number in dimension 11: construct maximal set of unit-norm-2 vectors with min distance >= max norm."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Generate all weight-4 vectors (entries in {-1,0,1}, exactly 4 non-zero entries)
    # Total: C(11,4) * 2^4 = 330 * 16 = 5280
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate weight-1 vectors (entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    N4 = len(vectors_v4)

    # Conflict: two vectors conflict if their dot product > 2 (i.e., distance^2 < 4)
    # For weight-4 vectors with entries in {-1,0,1}: dot product ranges from -4 to 4
    # Distance^2 = ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b = 4 + 4 - 2*a.b = 8 - 2*a.b
    # Need distance^2 >= 4 => 8 - 2*a.b >= 4 => a.b <= 2
    # So conflict when dot > 2, i.e., dot >= 3

    dots = vectors_v4 @ vectors_v4.T
    conflicts = (dots > 2).astype(np.int8)
    np.fill_diagonal(conflicts, 0)

    # Precompute neighbor lists as lists for fast access
    neighbors = [np.nonzero(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    def greedy_min_degree(seed_offset=0):
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy().astype(np.int32)
        local_rng = np.random.default_rng(42 + seed_offset)

        while np.any(alive) and time.time() < deadline - 30:
            alive_indices = np.nonzero(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] <= min_deg]

            if len(cands) > 1:
                # Tie-break: pick the one whose neighbors have lowest total degree
                scores = np.zeros(len(cands))
                for idx, c in enumerate(cands):
                    nbrs_c = neighbors[c]
                    alive_nbrs = nbrs_c[alive[nbrs_c]]
                    if len(alive_nbrs) > 0:
                        scores[idx] = np.sum(degrees[alive_nbrs])
                noise = local_rng.random(len(cands)) * 0.01
                scores += noise
                chosen = cands[np.argmin(scores)]
            else:
                chosen = cands[0]

            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.nonzero(alive)[0]
            if len(rem) > 0:
                # Update degrees
                for r in rem:
                    deg_change = np.sum(conflicts[r, to_remove])
                    degrees[r] -= deg_change

        return in_set

    def polish_add(final_set):
        """Add all vertices that can be added without conflict."""
        changed = True
        while changed:
            changed = False
            for i in range(N4):
                if not final_set[i]:
                    if np.sum(final_set[neighbors[i]]) == 0:
                        final_set[i] = True
                        changed = True
        return int(final_set.sum())

    def simulated_annealing(init_set, max_time=8.0):
        """SA with conflict-count energy to escape local optima."""
        x = init_set.copy()
        size = int(x.sum())

        # Compute initial conflict count
        conflict_count = 0
        in_indices = np.nonzero(x)[0]
        for i in in_indices:
            conflict_count += int(np.sum(x[neighbors[i]]))
        conflict_count //= 2

        current_energy = conflict_count - size
        best_energy = current_energy
        best_x = x.copy()

        T = 3.0
        min_T = 0.05
        cooling = 0.9995
        ls_deadline = time.time() + max_time
        local_rng = np.random.default_rng(123)
        stuck_counter = 0
        max_stuck = 2000

        while time.time() < ls_deadline:
            r = local_rng.random()
            move_executed = False

            if r < 0.35:
                # Add a random vertex
                v = local_rng.integers(N4)
                if not x[v]:
                    new_conflicts = int(np.sum(x[neighbors[v]]))
                    delta_E = new_conflicts - 1
                    if delta_E <= 0 or local_rng.random() < np.exp(-delta_E / T):
                        x[v] = True
                        size += 1
                        conflict_count += new_conflicts
                        current_energy = conflict_count - size
                        move_executed = True
            elif r < 0.55:
                # Remove a random vertex
                v = local_rng.integers(N4)
                if x[v]:
                    removed_conflicts = int(np.sum(x[neighbors[v]]))
                    delta_E = -removed_conflicts + 1
                    if delta_E <= 0 or local_rng.random() < np.exp(-delta_E / T):
                        x[v] = False
                        size -= 1
                        conflict_count -= removed_conflicts
                        current_energy = conflict_count - size
                        move_executed = True
            elif r < 0.85:
                # Swap: remove u (in set), add v (not in set)
                if size > 0:
                    u = local_rng.integers(N4)
                    v = local_rng.integers(N4)
                    if x[u] and not x[v]:
                        removed_edges = int(np.sum(x[neighbors[u]]))
                        added_edges = int(np.sum(x[neighbors[v]]))
                        if conflicts[u, v]:
                            added_edges -= 1
                        delta_C = added_edges - removed_edges
                        delta_E = delta_C
                        if delta_E <= 0 or local_rng.random() < np.exp(-delta_E / T):
                            x[u] = False
                            x[v] = True
                            conflict_count += delta_C
                            current_energy = conflict_count - size
                            move_executed = True
            else:
                # 2-for-1: remove two, add one
                if size >= 2:
                    u1 = local_rng.integers(N4)
                    u2 = local_rng.integers(N4)
                    v = local_rng.integers(N4)
                    if u1 != u2 and u1 != v and u2 != v:
                        if x[u1] and x[u2] and not x[v]:
                            deg_u1 = int(np.sum(x[neighbors[u1]]))
                            deg_u2 = int(np.sum(x[neighbors[u2]]))
                            deg_v = int(np.sum(x[neighbors[v]]))

                            R = deg_u1 + deg_u2
                            if conflicts[u1, u2]:
                                R -= 1

                            A = deg_v
                            if conflicts[v, u1]:
                                A -= 1
                            if conflicts[v, u2]:
                                A -= 1

                            delta_C = A - R
                            delta_S = -1
                            delta_E = delta_C - delta_S

                            if delta_E <= 0 or local_rng.random() < np.exp(-delta_E / T):
                                x[u1] = False
                                x[u2] = False
                                x[v] = True
                                size -= 1
                                conflict_count += delta_C
                                current_energy = conflict_count - size
                                move_executed = True

            if move_executed:
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x = x.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1

            T *= cooling
            if T < min_T:
                T = min_T

            if stuck_counter > max_stuck:
                # Perturbation: flip some random vertices
                n_flip = max(2, int(0.1 * size)) if size > 0 else 2
                flip_indices = local_rng.integers(N4, size=n_flip)
                for f in flip_indices:
                    x[f] = not x[f]

                size = int(x.sum())
                conflict_count = 0
                in_indices = np.nonzero(x)[0]
                for i in in_indices:
                    conflict_count += int(np.sum(x[neighbors[i]]))
                conflict_count //= 2
                current_energy = conflict_count - size

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x = x.copy()

                stuck_counter = 0
                T = max(T, 2.0)

        # Final polish: add all free vertices
        final_set = best_x.copy()
        final_size = polish_add(final_set)
        return final_set, final_size

    def branch_and_bound_mis(init_set, max_time=5.0):
        """Branch and bound for maximum independent set on the conflict graph."""
        x = init_set.copy()
        size = int(x.sum())

        # Compute conflict count
        conflict_count = 0
        in_indices = np.nonzero(x)[0]
        for i in in_indices:
            conflict_count += int(np.sum(x[neighbors[i]]))
        conflict_count //= 2

        if conflict_count == 0:
            final_size = polish_add(x)
            return x, final_size

        # Use a simple branch and bound with greedy coloring upper bound
        # This is a simplified version - just do more SA-like moves
        # Focus on removing conflicts
        local_rng = np.random.default_rng(77)
        bb_deadline = time.time() + max_time
        best_x = x.copy()
        best_size = size

        for iteration in range(20000):
            if time.time() > bb_deadline:
                break

            if conflict_count == 0:
                break

            # Find a conflict and resolve it
            # Pick a random vertex in the set that has conflicts
            in_idx = np.nonzero(x)[0]
            if len(in_idx) == 0:
                break

            # Find vertices with conflicts
            conflicted = []
            for i in in_idx:
                if np.sum(x[neighbors[i]]) > 0:
                    conflicted.append(i)

            if len(conflicted) == 0:
                break

            # Pick a random conflicted vertex and remove it or swap
            u = local_rng.choice(conflicted)
            nbrs_u = neighbors[u]
            x_nbrs = nbrs_u[x[nbrs_u]]

            if len(x_nbrs) == 0:
                continue

            # Option 1: remove u
            # Option 2: remove one of its neighbors
            # Option 3: try to find a vertex outside that can replace u
            if local_rng.random() < 0.5:
                # Remove u
                removed = int(np.sum(x[nbrs_u]))
                x[u] = False
                size -= 1
                conflict_count -= removed
            else:
                # Try to find a replacement
                # Look for a vertex not in set that conflicts with u's neighbors less
                # Just pick a random vertex outside
                v = local_rng.integers(N4)
                if not x[v]:
                    added = int(np.sum(x[neighbors[v]]))
                    if conflicts[u, v]:
                        added -= 1
                    # Remove u, add v
                    removed = int(np.sum(x[nbrs_u]))
                    x[u] = False
                    x[v] = True
                    conflict_count += added - removed

            if size > best_size and conflict_count == 0:
                best_size = size
                best_x = x.copy()

            if conflict_count == 0:
                break

        # Polish
        final_set = best_x.copy()
        final_size = polish_add(final_set)
        return final_set, final_size

    # Phase 1: Multiple greedy initializations
    greedy_inits = []
    for seed in range(30):
        if time.time() > deadline - 50:
            break
        in_set = greedy_min_degree(seed)
        greedy_inits.append(in_set)
        sz = int(in_set.sum())
        if sz > best_size:
            best_size = sz
            best_in_set = in_set.copy()

    # Phase 2: SA on each greedy init
    for init in greedy_inits:
        if time.time() > deadline - 15:
            break
        init_size = int(init.sum())
        remaining = deadline - 15 - time.time()
        if remaining < 2:
            break
        sa_time = min(remaining * 0.25, 8.0)
        if sa_time < 1.5:
            sa_time = 1.5

        sa_result, sa_size = simulated_annealing(init, max_time=sa_time)
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()

    # Phase 3: Additional SA rounds on best with perturbations
    remaining = deadline - 10 - time.time()
    if remaining > 5:
        for perturb_round in range(10):
            if time.time() > deadline - 10:
                break
            perturbed = best_in_set.copy()
            in_indices = np.nonzero(perturbed)[0]
            if len(in_indices) > 5:
                n_rem = max(2, int(0.1 * len(in_indices)))
                rem_idx = np.random.default_rng(42 + perturb_round).choice(
                    len(in_indices), size=n_rem, replace=False)
                for ri in rem_idx:
                    perturbed[in_indices[ri]] = False

            remaining = deadline - 10 - time.time()
            if remaining < 2:
                break
            sa_time = min(remaining * 0.3, 6.0)
            if sa_time < 1.5:
                sa_time = 1.5

            sa_result, sa_size = simulated_annealing(perturbed, max_time=sa_time)
            if sa_size > best_size:
                best_size = sa_size
                best_in_set = sa_result.copy()

    # Final polish
    final_size = polish_add(best_in_set)
    if final_size > best_size:
        best_size = final_size

    # Build final result: selected W4 vectors + all W1 vectors
    final_v4_indices = np.nonzero(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]

    # Verify that W1 vectors don't conflict with selected W4 vectors
    # W1 vector has one entry = +/-2, rest 0
    # W4 vector has four entries in {-1,0,1}
    # Dot product: if W1's non-zero position matches a non-zero position of W4, dot = +/-2
    # distance^2 = 4 + 4 - 2*(+/-2) = 8 - 4 or 8 + 4 = 4 or 12
    # So min distance^2 = 4, which equals the norm^2. This is valid (distance >= norm).
    # Actually we need distance >= max norm = 2, so distance^2 >= 4. Yes, 4 >= 4. OK.

    # Check W1-W1: two W1 vectors at different positions: dot=0, dist^2=8. OK.
    # Two W1 vectors at same position, opposite sign: dot=-4, dist^2=16. OK.
    # Two W1 vectors at same position, same sign: they're the same vector. We have both +2 and -2.
    # So W1 vectors are all mutually compatible.

    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors

# EVOLVE-BLOCK-END
