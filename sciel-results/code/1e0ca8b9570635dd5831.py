"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin

    N = 256
    # Build conflict: conflict[a] = set of b >= a where a+b is a perfect square
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1

    # conflict[a] contains all b in [1,N] such that a+b is a perfect square
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                conflict[a].add(b)
                conflict[b].add(a)

    # Self-conflicts: 2a is a perfect square
    self_conflict = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            self_conflict.add(a)

    # Start with all numbers ≡ 1 mod 3 (guaranteed independent set)
    base_set = set(range(1, N + 1, 3))

    # Try to greedily add elements from other residue classes
    # Sort candidates by degree (fewest conflicts first) to maximize additions
    candidates = [x for x in range(1, N + 1) if x % 3 != 1]

    # Precompute conflict count with current set for each candidate
    def conflicts_with_set(x, current_set):
        return len(conflict[x] & current_set)

    # Greedy: repeatedly add the candidate with fewest conflicts
    current_set = set(base_set)
    remaining = list(candidates)

    while remaining:
        # Check time
        if time.time() > deadline:
            break

        # Find candidate with minimum conflicts
        best_x = None
        best_conf = None
        for x in remaining:
            c = conflicts_with_set(x, current_set)
            if best_conf is None or c < best_conf:
                best_conf = c
                best_x = x

        if best_x is None:
            break

        if best_conf == 0:
            current_set.add(best_x)
            remaining.remove(best_x)
        else:
            # If minimum conflicts > 0, no more can be added greedily
            break

    # Try local search: remove one element, try to add two
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_list = list(current_set)
        for i in range(len(current_list)):
            if time.time() > deadline:
                break
            x = current_list[i]
            new_set = current_set - {x}
            # Try to add candidates
            for y in candidates:
                if y in new_set:
                    continue
                if len(conflict[y] & new_set) == 0:
                    new_set.add(y)
                    # Try to add one more
                    for z in candidates:
                        if z in new_set:
                            continue
                        if len(conflict[z] & new_set) == 0:
                            new_set.add(z)
                            if len(new_set) > len(current_set):
                                current_set = new_set
                                improved = True
                            break
                    break

    # Now use branch-and-bound to try to improve further
    # Convert to adjacency list for faster access
    adj = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        adj[a] = conflict[a] - {a}  # Exclude self for adjacency

    # Tomita-style maximum independent set with coloring upper bound
    # We work on the complement graph: independent set in G = clique in complement
    # But it's easier to work directly on G for independent set

    # Branch and bound for maximum independent set
    # State: (candidates, current_size, current_set)
    # Upper bound: current_size + len(candidates)
    # Better upper bound: use greedy coloring on candidates

    best_size = len(current_set)
    best_set = set(current_set)

    # Precompute degree for ordering
    def greedy_coloring(candidates, adj):
        """Color the subgraph induced by candidates. Returns list of colors."""
        colors = []
        uncolored = set(candidates)
        color = 1
        while uncolored:
            # Greedily assign this color to non-adjacent vertices
            color_set = set()
            for v in list(uncolored):
                if all(u not in color_set for u in adj[v] & uncolored):
                    color_set.add(v)
            for v in color_set:
                uncolored.remove(v)
            colors.append(color)
            color += 1
        return len(colors)

    # Branch and bound
    # Use a priority queue or recursive approach with pruning
    # For efficiency, use iterative deepening or simple recursion with time check

    # Let's use a simpler approach: local search with simulated annealing
    # Or: try to find a larger independent set via random restarts + local search

    # Actually, let's implement a proper branch and bound
    # But given the 60s limit, we need to be careful

    # Alternative: use a SAT-like approach or integer programming
    # Since we have limited time, let's try a more sophisticated local search

    # Simulated annealing
    def simulated_annealing(initial_set, adj, deadline):
        current = set(initial_set)
        best = set(initial_set)
        best_size = len(initial_set)
        temperature = 10.0
        cooling = 0.999
        min_temp = 0.01

        while temperature > min_temp and time.time() < deadline:
            # Pick a random element to remove
            if not current:
                break
            to_remove = random.choice(list(current))
            new_set = current - {to_remove}

            # Try to add random elements
            added = 0
            for _ in range(3):  # Try to add up to 3 elements
                candidates_to_add = [x for x in range(1, N + 1)
                                    if x not in new_set and
                                    len(adj[x] & new_set) == 0]
                if not candidates_to_add:
                    break
                to_add = random.choice(candidates_to_add)
                new_set.add(to_add)
                added += 1

            # Accept with probability based on temperature
            delta = len(new_set) - len(current)
            if delta >= 0:
                current = new_set
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(current)
            else:
                # Accept with probability exp(delta / temperature)
                if random.random() < math.exp(delta / temperature):
                    current = new_set

            temperature *= cooling

        return best, best_size

    # Run simulated annealing for a portion of the time
    sa_deadline = start_time + 40  # Use 40s for SA
    sa_best, sa_best_size = simulated_annealing(current_set, adj, sa_deadline)

    if sa_best_size > best_size:
        best_size = sa_best_size
        best_set = sa_best

    # Try a few random restarts with greedy
    for restart in range(10):
        if time.time() > deadline:
            break

        # Random initial set
        random_set = set()
        for x in range(1, N + 1):
            if random.random() < 0.3:
                random_set.add(x)

        # Greedy improvement
        improved = True
        while improved:
            improved = False
            for x in range(1, N + 1):
                if x in random_set:
                    continue
                if len(adj[x] & random_set) == 0:
                    random_set.add(x)
                    improved = True

        if len(random_set) > best_size:
            best_size = len(random_set)
            best_set = random_set

    result = sorted(best_set)
    return result if result else list(range(1, 257, 3))
# EVOLVE-BLOCK-END
