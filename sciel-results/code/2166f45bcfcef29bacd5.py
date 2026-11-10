import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 590.0  # safety margin before 630s kill

    # 1. Precompute forbidden differences (non-zero squares mod 205)
    forbidden = set()
    for x in range(1, m):
        sq = (x * x) % m
        if sq != 0:
            forbidden.add(sq)

    # 2. Build neighbor bitsets: neighbors[u] is a bitmask of vertices conflicting with u
    neighbors = [0] * m
    for u in range(m):
        mask = 0
        for v in range(m):
            if u != v:
                d = (u - v) % m
                if d in forbidden:
                    mask |= (1 << v)
        neighbors[u] = mask

    # 3. Precompute degree for each vertex
    degrees = [bin(neighbors[u]).count('1') for u in range(m)]

    def greedy_construct():
        """Construct a maximal independent set using greedy with random ordering."""
        current_set = set()
        available = list(range(m))
        random.shuffle(available)
        for v in available:
            if v not in current_set:
                # Check if v can be added
                conflicts = 0
                for n in current_set:
                    if (neighbors[v] >> n) & 1:
                        conflicts += 1
                        break
                if conflicts == 0:
                    current_set.add(v)
        return current_set

    def local_search(initial_set, max_iters=5000):
        """
        Perform local search to improve the independent set.
        Uses 1-add, 1-swap, and 2-swap moves.
        """
        current_set = set(initial_set)
        current_size = len(current_set)
        best_set = set(current_set)
        best_size = current_size
        no_improvement_count = 0
        iterations = 0

        while iterations < max_iters and time.time() < deadline - 5:
            iterations += 1
            improved = False

            comp = set(range(m)) - current_set

            if not comp:
                break

            # 1. Try to add a valid vertex from complement
            comp_list = list(comp)
            random.shuffle(comp_list)
            added = False
            for u in comp_list:
                can_add = True
                for n in current_set:
                    if (neighbors[u] >> n) & 1:
                        can_add = False
                        break
                if can_add:
                    current_set.add(u)
                    current_size += 1
                    improved = True
                    added = True
                    break

            if added:
                no_improvement_count = 0
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
                continue

            # 2. Try 1-swap: Remove v, add u
            set_list = list(current_set)
            random.shuffle(set_list)
            swapped = False
            for v in set_list:
                # Candidates for u: in comp, conflicts with v, and v is the only conflict
                candidates = []
                for u in comp:
                    if (neighbors[u] >> v) & 1:
                        # Check if v is the only conflict for u in current_set
                        conflicts = []
                        for n in current_set:
                            if (neighbors[u] >> n) & 1:
                                conflicts.append(n)
                        if len(conflicts) == 1 and conflicts[0] == v:
                            candidates.append(u)

                if candidates:
                    u = random.choice(candidates)
                    current_set.remove(v)
                    current_set.add(u)
                    improved = True
                    swapped = True
                    break

            if swapped:
                no_improvement_count = 0
                continue

            # 3. Try 2-swap: Remove v1, v2, add u1, u2
            if no_improvement_count > 5:
                found_2swap = False
                if len(current_set) >= 2:
                    set_list2 = list(current_set)
                    random.shuffle(set_list2)
                    for trial in range(min(10, len(set_list2) - 1)):
                        v1 = set_list2[trial]
                        v2 = set_list2[trial + 1]

                        # Try to find two vertices in comp that can replace them
                        comp_list2 = list(comp)
                        random.shuffle(comp_list2)
                        found = False
                        for i in range(min(20, len(comp_list2))):
                            u1 = comp_list2[i]
                            for j in range(i + 1, min(20, len(comp_list2))):
                                u2 = comp_list2[j]
                                # Check u1, u2 independent
                                if (neighbors[u1] >> u2) & 1:
                                    continue
                                # Check u1, u2 against current_set without v1, v2
                                test_set = set(current_set)
                                test_set.discard(v1)
                                test_set.discard(v2)
                                can_add_u1 = True
                                can_add_u2 = True
                                for n in test_set:
                                    if (neighbors[u1] >> n) & 1:
                                        can_add_u1 = False
                                        break
                                    if (neighbors[u2] >> n) & 1:
                                        can_add_u2 = False
                                        break
                                if can_add_u1 and can_add_u2:
                                    current_set = test_set
                                    current_set.add(u1)
                                    current_set.add(u2)
                                    improved = True
                                    found = True
                                    found_2swap = True
                                    break
                            if found:
                                break
                        if found_2swap:
                            break

                if found_2swap:
                    no_improvement_count = 0
                    continue

            no_improvement_count += 1
            if no_improvement_count > 30:
                break

        return best_set, best_size

    def simulated_annealing(initial_set, max_iters=2000):
        """
        Simulated annealing to escape local optima.
        """
        current_set = set(initial_set)
        current_size = len(current_set)
        best_set = set(current_set)
        best_size = current_size

        T = 1.0
        cooling = 0.9995

        for it in range(max_iters):
            if time.time() >= deadline - 5:
                break

            comp = set(range(m)) - current_set
            set_list = list(current_set)
            comp_list = list(comp)

            if not comp_list or not set_list:
                break

            move_type = random.random()

            if move_type < 0.5:
                # Try to add a random vertex from comp
                if comp_list:
                    u = random.choice(comp_list)
                    can_add = True
                    for n in current_set:
                        if (neighbors[u] >> n) & 1:
                            can_add = False
                            break
                    if can_add:
                        current_set.add(u)
                        current_size += 1

            elif move_type < 0.85:
                # Try swap
                v = random.choice(set_list)
                u = random.choice(comp_list)
                test_set = set(current_set)
                test_set.discard(v)
                can_add = True
                for n in test_set:
                    if (neighbors[u] >> n) & 1:
                        can_add = False
                        break
                if can_add:
                    current_set = test_set
                    current_set.add(u)

            else:
                # Try to remove a random vertex (worsening move)
                v = random.choice(set_list)
                if random.random() < np.exp(-1.0 / T):
                    current_set.discard(v)
                    current_size -= 1

            T *= cooling

            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)

        return best_set, best_size

    best_set = set()
    best_size = 0

    # Run multiple rounds of Greedy + Local Search + Simulated Annealing
    while time.time() < deadline - 30:
        init_set = greedy_construct()
        ls_set, ls_size = local_search(init_set, max_iters=2000)
        sa_set, sa_size = simulated_annealing(ls_set, max_iters=1000)

        if sa_size > best_size:
            best_size = sa_size
            best_set = sa_set

    # Final polish
    if time.time() < deadline - 10:
        polished_set, polished_size = local_search(best_set, max_iters=5000)
        if polished_size > best_size:
            best_set = polished_set
            best_size = polished_size

    result = sorted(best_set)

    if not result:
        result = [0]

    return np.array(result, dtype=np.int64)
