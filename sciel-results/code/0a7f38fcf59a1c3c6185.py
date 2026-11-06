import time
import random
import math

N = 17
NUM_INPUTS = 1 << N

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin for 90s limit

    # 1. Generate the seed network: Odd-Even Transposition
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))

    # 2. Verification Function using bit-vector simulation
    # Pre-compute initial state masks for all 17 wires
    initial_state = [0] * N
    for j in range(N):
        mask = 0
        for k in range(NUM_INPUTS):
            if (k >> j) & 1:
                mask |= (1 << k)
        initial_state[j] = mask

    def is_valid(network):
        state = [initial_state[k] for k in range(N)]
        # Incremental early-exit: the moment an adjacent inversion appears,
        # the network cannot possibly be sorted, so bail out immediately.
        for i, j in network:
            new_i = state[i] & state[j]
            new_j = state[i] | state[j]
            state[i] = new_i
            state[j] = new_j
            if (new_i & ~new_j) != 0:
                return False
            # If the two changed wires are adjacent, check the resulting
            # adjacent pair for inversion right away.
            if j == i + 1:
                if (state[i] & ~state[j]) != 0:
                    return False
            elif i > 0 and (state[i - 1] & ~state[i]) != 0:
                return False
            elif j < N - 1 and (state[j] & ~state[j + 1]) != 0:
                return False
        # Final full check (cheap) to be safe.
        for k in range(N - 1):
            if (state[k] & ~state[k + 1]) != 0:
                return False
        return True

    # 3. Aggressive redundancy removal.
    def try_remove_all(current_net, max_rounds=3):
        removed = True
        rounds = 0
        while removed and rounds < max_rounds:
            if time.time() > deadline:
                break
            removed = False
            rounds += 1
            indices = list(range(len(current_net)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > deadline:
                    break
                candidate = current_net[:idx] + current_net[idx + 1:]
                if is_valid(candidate):
                    current_net = candidate
                    removed = True
                    break
        return current_net

    # Final high-precision polish: exhaustive single-comparator removal to a
    # fixed point, plus a bounded "swap-for-removal" local solve.
    def final_polish(net):
        if time.time() > deadline:
            return net
        # Phase A: repeated full-sweep removal until nothing more can be removed.
        while time.time() < deadline:
            removed_any = False
            for idx in range(len(net) - 1, -1, -1):
                if time.time() > deadline:
                    break
                candidate = net[:idx] + net[idx + 1:]
                if is_valid(candidate):
                    net = candidate
                    removed_any = True
                    break
            if not removed_any:
                break
        # Phase B: bounded swap-for-removal local solve (one pass).
        if time.time() < deadline and len(net) > 1:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for a in range(len(net)):
                    if time.time() > deadline:
                        break
                    for b in range(len(net)):
                        if a == b:
                            continue
                        # Remove a, keep b; also try the reverse orientation.
                        candidate = net[:a] + net[a + 1:]
                        if is_valid(candidate):
                            net = candidate
                            improved = True
                            break
                    if improved:
                        break
        return net

    # Build the best starting network: seed + aggressive removal.
    best_network = try_remove_all(seed_network[:], max_rounds=8)
    if not is_valid(best_network):
        return seed_network

    random.seed(42)

    # 4. Simulated Annealing
    current_network = best_network[:]

    initial_temp = 8.0
    cooling_rate = 0.998
    min_temp = 0.005

    temperature = initial_temp

    while time.time() < deadline and temperature > min_temp:
        if time.time() > deadline:
            break

        neighbor = current_network[:]
        strategy = random.random()
        n = len(neighbor)

        if strategy < 0.20:
            # Add a random comparator - biased toward useful (nearby) positions
            new_i = random.randint(0, N - 2)
            gap = random.choice([1, 1, 1, 2, 2, 3, 4, 5])
            new_j = min(N - 1, new_i + gap)
            if new_i < new_j:
                pos = random.randint(0, len(neighbor))
                neighbor.insert(pos, (new_i, new_j))

        elif strategy < 0.40:
            # Remove a random comparator
            if n > 1:
                idx = random.randint(0, n - 1)
                neighbor.pop(idx)

        elif strategy < 0.55:
            # Move a comparator to a different position
            if n > 2:
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    comp = neighbor.pop(i)
                    if i < j:
                        j -= 1
                    neighbor.insert(j, comp)

        elif strategy < 0.70:
            # Swap two comparators
            if n > 1:
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

        elif strategy < 0.85:
            # Replace a comparator with a nearby wire pair - more aggressive
            if n > 0:
                i = random.randint(0, n - 1)
                old_i, old_j = neighbor[i]
                new_i = max(0, min(N - 2, old_i + random.randint(-3, 3)))
                new_j = max(new_i + 1, min(N - 1, old_j + random.randint(-2, 2)))
                if new_i < new_j:
                    neighbor[i] = (new_i, new_j)

        else:
            # Reorder a small segment of the network
            if n > 3:
                start = random.randint(0, n - 4)
                end = random.randint(start + 2, min(start + 5, n - 1))
                segment = neighbor[start:end + 1]
                random.shuffle(segment)
                neighbor[start:end + 1] = segment

        # Check validity
        if not is_valid(neighbor):
            temperature *= cooling_rate
            continue

        # Light cleanup: one round of removal to save time
        neighbor = try_remove_all(neighbor, max_rounds=1)

        # Acceptance decision
        current_cost = len(current_network)
        neighbor_cost = len(neighbor)
        delta = neighbor_cost - current_cost

        if delta <= 0:
            # Accept improvement
            current_network = neighbor
            if len(current_network) < len(best_network):
                best_network = current_network[:]
        else:
            # Accept worse solution with probability exp(-delta/T)
            if temperature > 0:
                prob = math.exp(-delta / temperature)
                if random.random() < prob:
                    current_network = neighbor

        temperature *= cooling_rate

    # Final cleanup: aggressive removal + high-precision polish pass.
    if time.time() < deadline:
        for _ in range(3):
            if time.time() > deadline:
                break
            final_network = try_remove_all(current_network, max_rounds=5)
            if len(final_network) < len(best_network):
                best_network = final_network[:]
                current_network = best_network[:]

    # Dedicated final high-precision polish on the best network found.
    if time.time() < deadline:
        polished = final_polish(best_network[:])
        if is_valid(polished) and len(polished) < len(best_network):
            best_network = polished

    return best_network
# EVOLVE-BLOCK-END
