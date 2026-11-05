import random
import time

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    d = 7
    n = 1 << d
    total_edges = n * d  # 448

    # Generate all edges with their bit index
    # Edge stored as (u, v) with u < v, and we track the bit index
    edges = []
    edge_set = set()
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y, i))
                edge_set.add((x, y))

    # Precompute for each edge (as tuple (x,y)) the list of 3-tuples of other edges
    # that complete a C4 with it
    # For edge (x, y) where y = x ^ (1<<i), and for each j != i:
    #   The C4 is: x, x^(1<<j), y, y^(1<<j)
    #   Other edges: (x, x^(1<<j)), (y, y^(1<<j)), (x^(1<<j), y^(1<<j))
    edge_cycles = {}  # (x,y) -> list of (oe1, oe2, oe3)
    for (x, y, bit_i) in edges:
        cycles = []
        for j in range(d):
            if j == bit_i:
                continue
            c = x ^ (1 << j)
            e = y ^ (1 << j)
            oe1 = (min(x ^ (1 << j), y ^ (1 << j)), max(x ^ (1 << j), y ^ (1 << j)))
            oe2 = (min(x, c), max(x, c))
            oe3 = (min(y, e), max(y, e))
            cycles.append((oe1, oe2, oe3))
        edge_cycles[(x, y)] = cycles

    def is_valid_add(chosen, x, y, bit_i):
        """Check if adding edge (x,y) with bit_i to chosen creates a C4."""
        for (oe1, oe2, oe3) in edge_cycles[(x, y)]:
            if oe1 in chosen and oe2 in chosen and oe3 in chosen:
                return False
        return True

    def greedy_init(seed):
        """Run greedy with randomized edge order."""
        rng = random.Random(seed)
        edge_order = list(edges)
        rng.shuffle(edge_order)
        chosen = set()
        for (x, y, bit_i) in edge_order:
            if is_valid_add(chosen, x, y, bit_i):
                chosen.add((x, y))
        return chosen

    def systematic_add(chosen, rng, max_time):
        """Try to add all possible edges systematically, return count added."""
        added = 0
        # Shuffle for randomization
        candidates = [e for e in edges if (e[0], e[1]) not in chosen]
        rng.shuffle(candidates)
        for (x, y, bit_i) in candidates:
            if time.time() >= max_time:
                break
            if is_valid_add(chosen, x, y, bit_i):
                chosen.add((x, y))
                added += 1
        return added

    def local_search(chosen, rng, time_limit):
        """Improve chosen set via local search with swaps and 1-for-2 moves."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False

            # Phase 1: Try adding edges
            added = systematic_add(chosen, rng, time_limit)
            if added > 0:
                improved = True
                continue

            # Phase 2: Try 1-for-1 swaps
            edges_in = list(chosen)
            edges_out = [(x, y, i) for (x, y, i) in edges if (x, y) not in chosen]
            rng.shuffle(edges_in)
            rng.shuffle(edges_out)

            swap_found = False
            for (x_rm, y_rm) in edges_in:
                if time.time() >= time_limit:
                    break
                if swap_found:
                    break
                for (x_add, y_add, bit_add) in edges_out:
                    if time.time() >= time_limit:
                        break
                    # Try: remove (x_rm, y_rm), add (x_add, y_add)
                    # Check if (x_add, y_add) can be added after removing (x_rm, y_rm)
                    # We need to check all C4s containing (x_add, y_add)
                    # For each such C4, check if all other 3 edges are in chosen - {(x_rm, y_rm)}
                    valid = True
                    for (oe1, oe2, oe3) in edge_cycles[(x_add, y_add)]:
                        # Check if all three are in chosen, but allow one to be (x_rm, y_rm)
                        in_chosen = 0
                        is_removed = False
                        for oe in (oe1, oe2, oe3):
                            if oe == (x_rm, y_rm):
                                is_removed = True
                            elif oe in chosen:
                                in_chosen += 1
                        if in_chosen == 3 and not is_removed:
                            valid = False
                            break
                    if valid:
                        chosen.discard((x_rm, y_rm))
                        chosen.add((x_add, y_add))
                        improved = True
                        swap_found = True
                        break

            if swap_found:
                continue

            # Phase 3: Try 1-for-2 moves (remove one, try to add two)
            edges_in = list(chosen)
            edges_out = [(x, y, i) for (x, y, i) in edges if (x, y) not in chosen]
            rng.shuffle(edges_in)
            rng.shuffle(edges_out)

            for (x_rm, y_rm) in edges_in:
                if time.time() >= time_limit:
                    break
                # Tentatively remove
                reduced = chosen - {(x_rm, y_rm)}
                # Try to add as many as possible
                added = 0
                for (x_add, y_add, bit_add) in edges_out:
                    if time.time() >= time_limit:
                        break
                    if (x_add, y_add) not in reduced and is_valid_add(reduced, x_add, y_add, bit_add):
                        reduced.add((x_add, y_add))
                        added += 1
                if added > 1:
                    chosen = reduced
                    improved = True
                    break

        return chosen

    # Run multiple random restarts
    best_chosen = set()
    best_size = 0

    # Phase 1: Greedy initializations
    num_greedy = 50
    for restart in range(num_greedy):
        if time.time() >= deadline - 40:  # leave time for local search
            break

        seed = restart * 12345 + 67890
        chosen = greedy_init(seed)
        size = len(chosen)

        if size > best_size:
            best_size = size
            best_chosen = set(chosen)

    # Phase 2: Local search on best few solutions
    # We'll keep track of top solutions and do local search on them
    top_solutions = []
    for restart in range(min(num_greedy, 20)):
        seed = restart * 12345 + 67890
        chosen = greedy_init(seed)
        top_solutions.append((len(chosen), chosen, seed))

    top_solutions.sort(key=lambda x: -x[0])

    for idx, (size, chosen, seed) in enumerate(top_solutions[:5]):
        if time.time() >= deadline - 10:
            break
        rng = random.Random(seed + 99999)
        local_time = min(deadline - 5, time.time() + 15)
        chosen = local_search(chosen, rng, local_time)
        size = len(chosen)

        if size > best_size:
            best_size = size
            best_chosen = set(chosen)

    # Final polish on best solution
    if time.time() < deadline - 5 and len(best_chosen) > 0:
        rng = random.Random(99999)
        best_chosen = local_search(best_chosen, rng, deadline - 2)

    # Convert to list of [u, v]
    result = [[u, v] for u, v in best_chosen]
    return result

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Number of edges: {len(result)}")
