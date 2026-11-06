import time
import math
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0  # Safety margin before 60s kill

    N = 256
    max_sum = 2 * N
    squares = set()
    k = 1
    while k * k <= max_sum:
        squares.add(k * k)
        k += 1

    # Identify valid nodes (those where 2*a is NOT a square)
    valid_nodes = []
    for i in range(1, N + 1):
        if 2 * i not in squares:
            valid_nodes.append(i)

    M = len(valid_nodes)
    if M == 0:
        return [1]

    # Build conflict graph using sets for fast intersection
    conflicts = {a: set() for a in valid_nodes}
    for idx, a in enumerate(valid_nodes):
        for b in valid_nodes[idx + 1:]:
            if (a + b) in squares:
                conflicts[a].add(b)
                conflicts[b].add(a)

    def try_add(s, v):
        return len(conflicts[v] & s) == 0

    def greedy_min_degree_random(candidates, adj):
        remaining = set(candidates)
        independent_set = set()
        while remaining:
            min_deg = float('inf')
            candidates_list = []
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    candidates_list = [v]
                elif deg == min_deg:
                    candidates_list.append(v)
            chosen = random.choice(candidates_list)
            independent_set.add(chosen)
            remaining.remove(chosen)
            for neighbor in adj[chosen]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
        return independent_set

    def local_search_improve(s, deadline):
        s = set(s)
        improved = True
        while improved and time.time() < deadline:
            improved = False

            # 1. Try to add any single node
            for v in valid_nodes:
                if v not in s and try_add(s, v):
                    s.add(v)
                    improved = True

            # 2. Try to remove one node to add >= 1 node (prefer >= 2 for size gain)
            removal_improved = False
            for u in list(s):
                s.remove(u)

                addable = []
                for v in valid_nodes:
                    if v not in s and v != u:
                        if try_add(s, v):
                            addable.append(v)

                if len(addable) >= 2:
                    for v in addable:
                        s.add(v)
                    removal_improved = True
                    improved = True
                    break
                elif len(addable) == 1:
                    # Swap: remove u, add v. Size same.
                    # Accept if new node has lower degree in current set (potential for future growth)
                    deg_u = len(conflicts[u] & s)
                    deg_v = len(conflicts[addable[0]] & s)
                    if deg_v < deg_u or random.random() < 0.1:
                        s.add(addable[0])
                        removal_improved = True
                        improved = True
                        break
                    else:
                        s.add(u)
                else:
                    s.add(u)

            # 3. Try 2-swap: remove two nodes, add up to three
            if not improved and time.time() < deadline:
                two_swap_improved = False
                s_list = list(s)
                for i in range(len(s_list)):
                    for j in range(i + 1, len(s_list)):
                        u1, u2 = s_list[i], s_list[j]
                        s.remove(u1)
                        s.remove(u2)

                        addable = []
                        for v in valid_nodes:
                            if v not in s and v != u1 and v != u2:
                                if try_add(s, v):
                                    addable.append(v)

                        if len(addable) >= 3:
                            for v in addable[:3]:
                                s.add(v)
                            two_swap_improved = True
                            improved = True
                            break
                        elif len(addable) == 2:
                            # Net zero change, but might unlock future moves
                            if random.random() < 0.05:
                                for v in addable:
                                    s.add(v)
                                two_swap_improved = True
                                improved = True
                                break
                        else:
                            s.add(u1)
                            s.add(u2)

                    if two_swap_improved:
                        break

        return s

    # Branch and bound for tight upper bounds and polishing
    def branch_and_bound_mis(vertices, adj, current_set, deadline):
        """
        Branch and bound to find a larger independent set.
        vertices: list of candidate vertices not in current_set
        Returns improved set or current_set if no improvement found
        """
        if time.time() >= deadline:
            return set(current_set)

        # Quick upper bound: current_size + len(vertices)
        # But we can do better with greedy coloring bound
        # For now, use a simple approach: try to extend greedily from current_set

        best = set(current_set)
        best_size = len(best)

        # Try each vertex in vertices as a forced inclusion
        for v in vertices:
            if time.time() >= deadline:
                break

            # Check if v can be added
            if not try_add(best, v):
                continue

            new_set = set(best)
            new_set.add(v)
            new_size = len(new_set)

            # Recursively try to extend (limited depth)
            remaining = [x for x in vertices if x != v and try_add(new_set, x)]
            if remaining:
                # Greedy extend
                for x in remaining:
                    if try_add(new_set, x):
                        new_set.add(x)

            if len(new_set) > best_size:
                best = new_set
                best_size = len(new_set)

        return best

    # Main search loop
    best_set = set()
    best_size = 0

    random.seed(42)

    # First, get a good initial solution with deterministic greedy
    initial = greedy_min_degree_random(valid_nodes, conflicts)
    initial = local_search_improve(initial, start_time + TIME_LIMIT)
    best_set = set(initial)
    best_size = len(best_set)

    # Multiple random restarts
    trial = 0
    while time.time() < start_time + TIME_LIMIT:
        trial += 1

        # Randomized greedy initialization
        shuffled = valid_nodes[:]
        random.shuffle(shuffled)
        current_set = set()
        for v in shuffled:
            if try_add(current_set, v):
                current_set.add(v)

        # Local search improvement
        current_set = local_search_improve(current_set, start_time + TIME_LIMIT)

        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)

        # Occasionally try branch-and-bound polish on the best set
        if trial % 10 == 0 and time.time() < start_time + TIME_LIMIT - 5:
            # Try to improve best_set using B&B on nodes not in best_set
            candidates = [v for v in valid_nodes if v not in best_set]
            if candidates:
                polished = branch_and_bound_mis(candidates, conflicts, best_set, start_time + TIME_LIMIT - 2)
                if len(polished) > best_size:
                    best_size = len(polished)
                    best_set = polished

    # Final verification
    result = sorted(best_set)
    for i, a in enumerate(result):
        for b in result[i:]:
            s = a + b
            root = math.isqrt(s)
            if root * root == s:
                raise ValueError(f"Conflict: {a} + {b} = {s}")

    if not result:
        result = [1]

    return result
