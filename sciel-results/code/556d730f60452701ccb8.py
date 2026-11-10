import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 590

    # Generate all integer vectors with squared norm 4 in Z^11
    # These are vectors with exactly 4 non-zero entries, each being +/-1
    candidates = []
    for comb in combinations(range(d), 4):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    for s4 in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[comb[0]] = s1
                        v[comb[1]] = s2
                        v[comb[2]] = s3
                        v[comb[3]] = s4
                        candidates.append(v)

    candidates = np.array(candidates, dtype=np.int64)
    N = len(candidates)

    # Compute dot product matrix
    dots = candidates @ candidates.T

    # Conflict: squared distance < 4, i.e., ||u-v||^2 < 4
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2*u.v = 4 + 4 - 2*u.v = 8 - 2*u.v
    # So conflict if 8 - 2*u.v < 4, i.e., u.v > 2
    # Also need to exclude self (u.v = 4)

    conflicts = (dots > 2)
    np.fill_diagonal(conflicts, False)

    # Convert to adjacency list
    adj = [np.where(conflicts[i])[0] for i in range(N)]

    # Greedy independent set with degree-based ordering
    def greedy_independent_set(order):
        selected = []
        selected_set = set()
        for v in order:
            if v in selected_set:
                continue
            can_add = True
            for u in selected:
                if v in adj[u]:
                    can_add = False
                    break
            if can_add:
                selected.append(v)
                selected_set.add(v)
        return selected

    # Multiple greedy restarts with different orderings
    best_set = []
    for restart in range(100):
        if time.time() > deadline - 100:
            break
        order = list(range(N))
        np.random.shuffle(order)
        sol = greedy_independent_set(order)
        if len(sol) > len(best_set):
            best_set = sol[:]

    # Local search / improvement
    def try_improve(current_set):
        current_set = set(current_set)
        improved = True
        while improved and time.time() < deadline - 100:
            improved = False
            for u in list(current_set):
                current_set.discard(u)
                # Find candidates that can be added
                candidates_to_add = []
                for v in range(N):
                    if v in current_set:
                        continue
                    # Check if v conflicts with any in current_set
                    can_add = True
                    for w in current_set:
                        if v in adj[w]:
                            can_add = False
                            break
                    if can_add:
                        candidates_to_add.append(v)
                if candidates_to_add:
                    # Try adding the best candidate (one that conflicts with fewest others)
                    best_v = None
                    min_conflicts = float('inf')
                    for v in candidates_to_add:
                        # Count how many of the remaining non-selected vertices this would conflict with
                        # Actually, just add a random one or the one with fewest adjacencies in the pool
                        # For simplicity, pick one at random
                        v = candidates_to_add[np.random.randint(len(candidates_to_add))]
                        current_set.add(v)
                        improved = True
                        break
                else:
                    # Couldn't add anything, put u back
                    current_set.add(u)
        return current_set

    improved_set = try_improve(set(best_set))
    if len(improved_set) > len(best_set):
        best_set = list(improved_set)

    # More greedy + local search iterations
    for restart in range(200):
        if time.time() > deadline - 50:
            break

        # Random restart with different perturbation
        order = list(range(N))
        np.random.shuffle(order)
        sol = set(greedy_independent_set(order))

        # Local search: try to add/remove/swap vertices
        for iteration in range(100):
            if time.time() > deadline - 50:
                break
            improved = False

            # Try adding a vertex
            addable = []
            for v in range(N):
                if v in sol:
                    continue
                can_add = True
                for u in sol:
                    if v in adj[u]:
                        can_add = False
                        break
                if can_add:
                    addable.append(v)

            if addable:
                v = addable[np.random.randint(len(addable))]
                sol.add(v)
                improved = True
            else:
                # Try swapping: remove one, add another
                if sol:
                    u = list(sol)[np.random.randint(len(sol))]
                    sol.discard(u)
                    addable2 = []
                    for v in range(N):
                        if v in sol:
                            continue
                        can_add = True
                        for w in sol:
                            if v in adj[w]:
                                can_add = False
                                break
                        if can_add:
                            addable2.append(v)
                    if addable2:
                        v = addable2[np.random.randint(len(addable2))]
                        sol.add(v)
                        improved = True
                    else:
                        sol.add(u)

            if not improved:
                break

        if len(sol) > len(best_set):
            best_set = list(sol)

    # Final check
    if not best_set:
        # Fallback to a small valid set
        return np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

    points = candidates[best_set]
    return np.around(points).astype(np.int64)
