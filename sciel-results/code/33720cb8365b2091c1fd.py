import time
import random

N = 512

# Precompute forbidden triplets for each element
# For each element i, forbid_triples[i] contains tuples (a,b,c) such that {i,a,b,c} forms a 4-AP
forbid_triples = [[] for _ in range(N + 1)]

for d in range(1, N // 3 + 1):
    for start in range(1, N - 3 * d + 1):
        a = start
        b = start + d
        c = start + 2 * d
        e = start + 3 * d
        # 4-AP: a, b, c, e
        forbid_triples[a].append((b, c, e))
        forbid_triples[b].append((a, c, e))
        forbid_triples[c].append((a, b, e))
        forbid_triples[e].append((a, b, c))


def can_add_bitset(i, in_set_bits):
    """Check if element i can be added to the set represented by in_set_bits (int bitmask)."""
    for a, b, c in forbid_triples[i]:
        # Check if all three of a,b,c are in the set
        if (in_set_bits >> a) & 1 and (in_set_bits >> b) & 1 and (in_set_bits >> c) & 1:
            return False
    return True


def greedy_construction_bitset(order, deadline):
    in_set_bits = 0
    size = 0
    for i in order:
        if time.time() > deadline:
            break
        if can_add_bitset(i, in_set_bits):
            in_set_bits |= (1 << i)
            size += 1
    return in_set_bits, size


def local_search_bitset(in_set_bits, deadline, max_iters=100000):
    """
    Local search with simulated annealing on bitmask representation.
    Moves: add, remove, swap
    """
    size = bin(in_set_bits).count('1')
    best_bits = in_set_bits
    best_size = size
    current_temp = 100.0
    cooling_rate = 0.999

    for it in range(max_iters):
        if time.time() > deadline:
            break

        # Try to add a random element
        candidates_to_add = []
        for i in range(1, N + 1):
            if not ((in_set_bits >> i) & 1):
                candidates_to_add.append(i)

        if candidates_to_add:
            i = random.choice(candidates_to_add)
            if can_add_bitset(i, in_set_bits):
                in_set_bits |= (1 << i)
                size += 1
                if size > best_size:
                    best_size = size
                    best_bits = in_set_bits
                current_temp = max(current_temp * cooling_rate, 0.1)
                continue

        # Try a swap: remove x from S, add y not in S
        in_list = [i for i in range(1, N + 1) if (in_set_bits >> i) & 1]
        out_list = [i for i in range(1, N + 1) if not ((in_set_bits >> i) & 1)]

        if in_list and out_list:
            x = random.choice(in_list)
            y = random.choice(out_list)

            # Temporarily remove x
            in_set_bits ^= (1 << x)
            size -= 1

            if can_add_bitset(y, in_set_bits):
                in_set_bits |= (1 << y)
                size += 1
                # After swap, try to add another element
                for z in out_list:
                    if z == y:
                        continue
                    if can_add_bitset(z, in_set_bits):
                        in_set_bits |= (1 << z)
                        size += 1
                        if size > best_size:
                            best_size = size
                            best_bits = in_set_bits
                        current_temp = max(current_temp * cooling_rate, 0.1)
                        break

            # If we didn't improve, possibly revert
            if size <= best_size:
                # Revert: put x back, remove y
                if (in_set_bits >> y) & 1:
                    in_set_bits ^= (1 << y)
                    size -= 1
                in_set_bits |= (1 << x)
                size += 1
                # Annealing: accept with probability
                if random.random() < min(1.0, current_temp / 100.0):
                    # Keep the swap even if not improving
                    in_set_bits ^= (1 << x)
                    in_set_bits |= (1 << y)
                    # size stays same
                else:
                    pass  # Already reverted
        else:
            # No moves possible, try random restarts
            if random.random() < 0.01:
                # Slight perturbation: remove a random element
                if in_list:
                    x = random.choice(in_list)
                    in_set_bits ^= (1 << x)
                    size -= 1

        current_temp = max(current_temp * cooling_rate, 0.1)

    return best_bits, best_size


def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin

    best_bits = 0
    best_size = 0

    # Phase 1: Deterministic greedy constructions
    random.seed(42)

    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
    ]

    for order in orders:
        bits, size = greedy_construction_bitset(order, deadline)
        if size > best_size:
            best_size = size
            best_bits = bits

    # Phase 2: Polish with local search
    if best_bits:
        bits, size = local_search_bitset(best_bits, deadline, max_iters=50000)
        if size > best_size:
            best_size = size
            best_bits = bits

    # Phase 3: Randomized greedy + local search
    while time.time() < deadline - 5:
        order = list(range(1, N + 1))
        random.shuffle(order)
        bits, size = greedy_construction_bitset(order, deadline)

        bits, size = local_search_bitset(bits, deadline, max_iters=20000)

        if size > best_size:
            best_size = size
            best_bits = bits

    # Convert to sorted list
    if best_bits == 0:
        return [1]

    result = []
    for i in range(1, N + 1):
        if (best_bits >> i) & 1:
            result.append(i)

    return result


if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
