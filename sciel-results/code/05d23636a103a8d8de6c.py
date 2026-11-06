"""Rewritten program: size-efficient sorting networks on 17 wires.

Strategy:
- Fast vectorized verification (numpy) over all 2^17 Boolean inputs.
- Start from the odd-even transposition seed (136 comparators).
- Aggressive greedy removal with randomized restarts and multiple passes.
- Local search: swap, insert, delete, and random-replace moves, followed by
  greedy reduction.
- Simulated-annealing-like exploration with a cooling schedule.
- All phases respect a hard wall-clock deadline with a safety margin.
"""

import time
import random
import numpy as np

N = 17
MAX_COMPARATORS = 4096

# Precompute sorted states: a 17-bit state is sorted (non-decreasing) iff
# all 0s precede all 1s, i.e. the state equals (1<<N) - (1<<(N-k)) for k in 0..N.
_SORTED_STATES = set()
for k in range(N + 1):
    if k == 0:
        _SORTED_STATES.add(0)
    else:
        _SORTED_STATES.add((1 << N) - (1 << (N - k)))

# Lookup table for fast membership: is_sorted_lookup[state] == True iff state is sorted.
_IS_SORTED = np.zeros(1 << N, dtype=bool)
for s in _SORTED_STATES:
    _IS_SORTED[s] = True

# Precompute all comparator masks for fast application
# For comparator (i, j) with i < j:
#   mask_swap = (1 << i) | (1 << j)
#   We need to apply: if bit i == 1 and bit j == 0, then XOR mask_swap.
# Vectorized: for a state array S, compute:
#   bi = (S >> i) & 1
#   bj = (S >> j) & 1
#   swap_mask = (bi & (1 - bj))  # 1 where swap needed
#   S[swap_mask] ^= mask_swap


def verify_network(network):
    """Vectorized verification: apply network to all 2^17 states, check sorted."""
    if not network:
        return True
    all_states = np.arange(1 << N, dtype=np.uint32)
    for i, j in network:
        bi = (all_states >> i) & 1
        bj = (all_states >> j) & 1
        swap_mask = bi & (1 - bj)
        all_states[swap_mask] ^= (1 << i) | (1 << j)
    return bool(np.all(_IS_SORTED[all_states]))


def seed_network():
    """Odd-even transposition network: N phases, alternating even/odd pairs."""
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net


def _greedy_remove(network, rng, deadline, max_passes=200):
    """Repeatedly try to remove a random comparator until none can be removed.
    Returns the reduced network."""
    current = list(network)
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        if len(current) <= 1:
            break
        # Try each position in random order
        indices = list(range(len(current)))
        rng.shuffle(indices)
        removed = False
        for idx in indices:
            if time.time() > deadline:
                return current
            candidate = current[:idx] + current[idx + 1:]
            if verify_network(candidate):
                current = candidate
                removed = True
                break
        if not removed:
            break
    return current


def _try_swap(network, idx, rng, deadline):
    """Try swapping adjacent comparators at idx and idx+1. Return new net or None."""
    if idx + 1 >= len(network):
        return None
    candidate = list(network)
    candidate[idx], candidate[idx + 1] = candidate[idx + 1], candidate[idx]
    if verify_network(candidate):
        return candidate
    return None


def _try_replace(network, idx, rng, deadline):
    """Try replacing comparator at idx with a random one. Return new net or None."""
    if idx < 0 or idx >= len(network):
        return None
    # Bias toward nearby wires for locality
    old_i, old_j = network[idx]
    # 50% chance: same i, random j; 50%: random i, same j
    if rng.random() < 0.5:
        new_i = old_i
        new_j = rng.randint(0, N - 1)
    else:
        new_j = old_j
        new_i = rng.randint(0, N - 1)
    if new_i == new_j:
        return None
    if new_i > new_j:
        new_i, new_j = new_j, new_i
    candidate = list(network)
    candidate[idx] = (new_i, new_j)
    if verify_network(candidate):
        return candidate
    return None


def _try_insert(network, rng, deadline):
    """Insert a random comparator at a random position. Return new net or None."""
    i = rng.randint(0, N - 1)
    j = rng.randint(0, N - 1)
    if i == j:
        return None
    if i > j:
        i, j = j, i
    pos = rng.randint(0, len(network))
    candidate = network[:pos] + [(i, j)] + network[pos:]
    if verify_network(candidate):
        return candidate
    return None


def _try_delete(network, idx, deadline):
    """Delete comparator at idx. Return new net or None."""
    if idx < 0 or idx >= len(network):
        return None
    candidate = network[:idx] + network[idx + 1:]
    if verify_network(candidate):
        return candidate
    return None


def construct():
    """Main entry point: search for a minimal sorting network on 17 wires."""
    start_time = time.time()
    # Safety margin: return well before 90s kill. Use 85s as hard deadline.
    deadline = start_time + 85.0

    rng = random.Random(12345)

    # Start from seed
    best_net = seed_network()
    best_count = len(best_net)
    # Verify seed
    if not verify_network(best_net):
        raise RuntimeError("Seed network is invalid!")

    # Phase 1: Aggressive greedy removal from seed
    current = list(best_net)
    current = _greedy_remove(current, rng, deadline, max_passes=500)
    if len(current) < best_count:
        best_net = current
        best_count = len(current)

    # Phase 2: Local search with multiple restarts
    # Each restart: take best_net, apply random perturbations, then greedy reduce.
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        current = list(best_net)

        # Random perturbation: do a few random moves
        num_moves = rng.randint(2, 8)
        for _ in range(num_moves):
            if time.time() > deadline:
                break
            move_type = rng.random()
            if move_type < 0.3 and len(current) > 1:
                # Swap
                idx = rng.randint(0, len(current) - 2)
                new_net = _try_swap(current, idx, rng, deadline)
                if new_net is not None:
                    current = new_net
            elif move_type < 0.6:
                # Replace
                idx = rng.randint(0, len(current) - 1)
                new_net = _try_replace(current, idx, rng, deadline)
                if new_net is not None:
                    current = new_net
            elif move_type < 0.8:
                # Insert
                new_net = _try_insert(current, rng, deadline)
                if new_net is not None:
                    current = new_net
            else:
                # Delete
                idx = rng.randint(0, len(current) - 1)
                new_net = _try_delete(current, idx, deadline)
                if new_net is not None:
                    current = new_net

        # Greedy reduce
        current = _greedy_remove(current, rng, deadline, max_passes=300)

        if len(current) < best_count:
            best_net = current
            best_count = len(current)

    # Phase 3: Simulated annealing-like search
    # Temperature starts high, cools down. Accept worse moves with probability exp(-delta/T).
    current = list(best_net)
    temperature = 50.0
    cooling = 0.995
    min_temp = 1.0

    for iteration in range(5000):
        if time.time() > deadline:
            break
        if temperature < min_temp:
            break

        # Generate a random move
        move_type = rng.random()
        if move_type < 0.35 and len(current) > 1:
            # Delete
            idx = rng.randint(0, len(current) - 1)
            candidate = _try_delete(current, idx, deadline)
            if candidate is None:
                continue
            delta = len(candidate) - len(current)  # negative if improvement
        elif move_type < 0.65:
            # Replace
            idx = rng.randint(0, len(current) - 1)
            candidate = _try_replace(current, idx, rng, deadline)
            if candidate is None:
                continue
            delta = len(candidate) - len(current)
        elif move_type < 0.85:
            # Insert
            candidate = _try_insert(current, rng, deadline)
            if candidate is None:
                continue
            delta = len(candidate) - len(current)
        else:
            # Swap
            if len(current) < 2:
                continue
            idx = rng.randint(0, len(current) - 2)
            candidate = _try_swap(current, idx, rng, deadline)
            if candidate is None:
                continue
            delta = len(candidate) - len(current)

        # Accept?
        if delta <= 0:
            current = candidate
        else:
            # Accept worse with probability exp(-delta / T)
            prob = np.exp(-delta / temperature)
            if rng.random() < prob:
                current = candidate

        # Greedy reduce after each accepted move
        if time.time() < deadline:
            current = _greedy_remove(current, rng, deadline, max_passes=50)

        if len(current) < best_count:
            best_net = current
            best_count = len(current)

        temperature *= cooling

    # Final verification
    if not verify_network(best_net):
        best_net = seed_network()

    return best_net
