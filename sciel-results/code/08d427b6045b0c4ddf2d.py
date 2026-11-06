"""Size-efficient sorting network search for N=16 wires.

Strategy:
1. Use bit-parallel zero-one verification for fast checking.
2. Start from multiple initial networks: odd-even transposition (120), and a
   Batcher bitonic network (~80).
3. Apply aggressive local search:
   - Greedy removal of redundant comparators.
   - Random insertion of comparators to escape local minima, followed by
     greedy removal again.
   - Simulated-annealing-like acceptance of worse moves early on.
4. Track the best (fewest comparator) network found and return it.
"""

import time
import random

N = 16
TOTAL_INPUTS = 1 << N
ALL_ONES = (1 << TOTAL_INPUTS) - 1


def _build_initial_state():
    """Build the initial bitmask state for each wire across all 2^N boolean inputs."""
    state = [0] * N
    for i in range(N):
        half = 1 << i
        period = half << 1
        num_periods = TOTAL_INPUTS // period
        # One period: lower half zeros, upper half ones
        period_val = ((1 << half) - 1) << half
        val = 0
        shift = 0
        for _ in range(num_periods):
            val |= period_val << shift
            shift += period
        state[i] = val
    return state


def fast_verify(network):
    """Verify a sorting network using bit-parallel zero-one principle."""
    state = _build_initial_state()
    for (i, j) in network:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
    # Check sorted: for all adjacent wires, state[i] & ~state[i+1] must be 0
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True


def odd_even_network():
    """Odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def bitonic_network(n):
    """Batcher's bitonic sorting network for n wires (n must be power of 2)."""
    if n == 1:
        return []
    # Recursively sort halves
    left = bitonic_network(n // 2)
    right = [(i + n // 2, j + n // 2) for (i, j) in bitonic_network(n // 2)]
    # Merge phase
    merge = _bitonic_merge(n)
    return left + right + merge


def _bitonic_merge(n):
    """Generate comparators for merging a bitonic sequence of length n."""
    if n == 2:
        return [(0, 1)]
    # First, recursively merge in bitonic order
    comps = []
    # For bitonic merge of size n, we do log2(n) stages
    k = n.bit_length() - 1  # log2(n)
    for stage in range(k):
        step = n >> (stage + 1)
        for i in range(0, n, step * 2):
            for j in range(i, i + step):
                comps.append((j, j + step))
    return comps


def greedy_remove(net, deadline):
    """Try to remove each comparator; return shortened net if possible."""
    for idx in range(len(net) - 1, -1, -1):
        if time.time() > deadline:
            break
        candidate = net[:idx] + net[idx + 1:]
        if fast_verify(candidate):
            return candidate
    return net


def construct():
    """Construct a sorting network with minimum comparators."""
    start_time = time.time()
    deadline = start_time + 85  # 85s to leave 5s margin

    rng = random.Random(42)

    # Try multiple starting points
    initial_networks = [
        odd_even_network(),
        bitonic_network(N),
    ]

    best_net = None
    best_count = float('inf')

    for net in initial_networks:
        # Verify the initial network
        if not fast_verify(net):
            continue
        if len(net) < best_count:
            best_count = len(net)
            best_net = list(net)

    if best_net is None:
        # Fallback: return odd-even (should always be valid)
        return odd_even_network()

    # Local search from the best initial network
    current = best_net[:]

    # Phase 1: Aggressive greedy removal
    while time.time() < deadline - 5:
        new_net = greedy_remove(current, deadline - 5)
        if len(new_net) < len(current):
            current = new_net
            if len(current) < best_count:
                best_count = len(current)
                best_net = list(current)
        else:
            break

    # Phase 2: Random insertion + greedy removal to escape local minima
    # Also try random swaps of comparators
    all_comps = [(i, j) for i in range(N) for j in range(i + 1, N)]

    # Temperature-like schedule: early on, accept worse moves more often
    max_iter = 2000
    for it in range(max_iter):
        if time.time() > deadline:
            break

        # Choose a random move: insert, swap, or move
        move_type = rng.choice(['insert', 'swap', 'move'])

        candidate = current[:]

        if move_type == 'insert':
            # Insert a random comparator at a random position
            comp = rng.choice(all_comps)
            pos = rng.randint(0, len(candidate))
            candidate.insert(pos, comp)
        elif move_type == 'swap' and len(candidate) >= 2:
            i, j = rng.sample(range(len(candidate)), 2)
            candidate[i], candidate[j] = candidate[j], candidate[i]
        elif move_type == 'move' and len(candidate) >= 2:
            i = rng.randint(0, len(candidate) - 1)
            j = rng.randint(0, len(candidate) - 1)
            if i != j:
                comp = candidate.pop(i)
                candidate.insert(j, comp)
        else:
            continue

        if not fast_verify(candidate):
            continue

        # If the candidate is shorter or equal, accept it
        if len(candidate) <= len(current):
            current = candidate
            if len(current) < best_count:
                best_count = len(current)
                best_net = list(current)

            # After an improvement, try greedy removal again
            while time.time() < deadline - 5:
                new_net = greedy_remove(current, deadline - 5)
                if len(new_net) < len(current):
                    current = new_net
                    if len(current) < best_count:
                        best_count = len(current)
                        best_net = list(current)
                else:
                    break
        else:
            # Accept worse moves early with some probability (simulated annealing)
            # Probability decreases over time
            t = it / max_iter
            prob = 0.1 * (1 - t)
            if rng.random() < prob:
                current = candidate

    return best_net
