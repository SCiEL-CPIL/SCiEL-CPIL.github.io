"""Strategy: Maximum Independent Set via ILP (scipy.optimize.milp) on a 3-uniform hypergraph,
with improved greedy initialization and multiple random-restart local search."""
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    """
    Returns a nonempty subset of [1, 180] with no three distinct elements a, b, c 
    satisfying 1/a = 1/b + 1/c. Maximizes cardinality.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    N = 180
    numbers = list(range(1, N + 1))

    # 1. Identify all forbidden triples (a, b, c) with distinct elements
    # 1/a = 1/b + 1/c  =>  bc = a(b+c)  =>  a = bc / (b+c)
    triples = set()
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            numerator = b * c
            denominator = b + c
            if numerator % denominator == 0:
                a = numerator // denominator
                if 1 <= a <= N and a != b and a != c:
                    triple = tuple(sorted([a, b, c]))
                    triples.add(triple)

    triples_list = list(triples)
    num_triples = len(triples_list)

    # Precompute for each element, which triples it participates in
    element_triples = {x: [] for x in numbers}
    for idx, t in enumerate(triples_list):
        for x in t:
            element_triples[x].append(idx)

    # Precompute for each element, its neighbors in the hypergraph (elements sharing a triple)
    # and the count of conflicts if we were to add it to a given set
    def count_conflicts(x, current_set):
        """Count how many triples would be violated if x is added to current_set."""
        conflicts = 0
        for t_idx in element_triples[x]:
            t = triples_list[t_idx]
            others = [v for v in t if v != x]
            if others[0] in current_set and others[1] in current_set:
                conflicts += 1
        return conflicts

    # 2. Greedy removal to get an initial feasible solution
    # Precompute degree of each element in the hypergraph
    degree = {x: len(element_triples[x]) for x in numbers}

    def greedy_initial():
        current_set = set(numbers)
        changed = True
        while changed:
            changed = False
            # Find a violating triple
            for t in triples_list:
                if t[0] in current_set and t[1] in current_set and t[2] in current_set:
                    # Remove the element with highest degree among the three
                    candidates = sorted([t[0], t[1], t[2]], key=lambda x: degree[x], reverse=True)
                    to_remove = candidates[0]
                    current_set.remove(to_remove)
                    changed = True
                    break
        return current_set

    best_set = greedy_initial()
    best_size = len(best_set)

    # 3. Local search: try to add back elements
    def local_search(initial_set, max_iters=50):
        current = set(initial_set)
        removed = set(numbers) - current
        improved = True
        while improved and len(removed) > 0:
            improved = False
            # Try to add back each removed element
            removed_list = list(removed)
            random.shuffle(removed_list)
            for x in removed_list:
                if time.time() > deadline - 5:
                    return current
                if x in current:
                    continue
                # Check if adding x creates any conflict
                conflict = False
                for t_idx in element_triples[x]:
                    t = triples_list[t_idx]
                    others = [v for v in t if v != x]
                    if others[0] in current and others[1] in current:
                        conflict = True
                        break
                if not conflict:
                    current.add(x)
                    removed.discard(x)
                    improved = True
        return current

    # Multiple random restarts of local search
    for restart in range(30):
        if time.time() > deadline - 10:
            break
        # Create a perturbed starting point
        current = set(best_set)
        # Randomly remove some elements
        num_remove = random.randint(1, min(5, len(current)))
        to_remove = random.sample(list(current), num_remove)
        for x in to_remove:
            current.discard(x)
        
        result = local_search(current)
        if len(result) > best_size:
            best_set = result
            best_size = len(result)

    # 4. Try ILP to find the optimal or near-optimal solution
    try:
        # c vector: objective coefficients (all 1s for maximization)
        c = np.ones(N)

        # Constraints: for each triple, x_a + x_b + x_c <= 2
        A = np.zeros((num_triples, N))
        for idx, t in enumerate(triples_list):
            A[idx, t[0] - 1] = 1
            A[idx, t[1] - 1] = 1
            A[idx, t[2] - 1] = 1

        # Bounds: x_i in [0, 1]
        integrality = np.ones(N)  # all variables are integer
        bounds = Bounds(lb=np.zeros(N), ub=np.ones(N))

        # Constraint: A @ x <= 2 for each row
        lin_cons = LinearConstraint(A, lb=-np.inf, ub=2)

        # Run MILP
        milp_result = milp(
            c=c,
            constraints=[lin_cons],
            integrality=integrality,
            bounds=bounds
        )

        if milp_result.success and milp_result.x is not None:
            # Extract solution
            x = milp_result.x
            x_rounded = np.round(x).astype(int)

            # Build the set
            ilp_set = set()
            for i in range(N):
                if x_rounded[i] == 1:
                    ilp_set.add(i + 1)

            # Verify no forbidden triples
            valid = True
            for t in triples_list:
                if t[0] in ilp_set and t[1] in ilp_set and t[2] in ilp_set:
                    valid = False
                    break

            if valid and len(ilp_set) >= best_size:
                best_set = ilp_set
                best_size = len(ilp_set)

    except Exception as e:
        # Fall back to greedy result
        pass

    # Final verification
    for t in triples_list:
        if t[0] in best_set and t[1] in best_set and t[2] in best_set:
            # Remove the element with highest degree
            candidates = sorted([t[0], t[1], t[2]], key=lambda x: degree[x], reverse=True)
            best_set.remove(candidates[0])

    if not best_set:
        best_set = {1}

    return list(best_set)
