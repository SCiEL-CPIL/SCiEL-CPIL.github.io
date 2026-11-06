"""
Size-efficient sorting networks on 17 wires.
Structural change: depth-aware local search with targeted moves.
1. Greedy deletion from odd-even transposition seed.
2. Depth-aware simulated annealing: cost = size + lambda * depth.
   Moves include deletion, insertion, wire-change, adjacent-swap, and
   depth-reducing reordering to escape size-only local minima.
"""
import time
import random
import math

N = 17
TOTAL_INPUTS = 1 << N  # 131072

# Precompute initial state bitmasks for verification.
# state[i] is a bitmask where bit k is 1 if input k has a 1 at wire i.
# We can construct these efficiently.
INITIAL_STATE = [0] * N
for i in range(N):
    mask = 0
    bit_val = 1 << i
    # Pattern for wire i: blocks of 2^i zeros, 2^i ones
    # We can use the iterative shift method or direct construction.
    # Direct construction is faster for precomputation.
    block = 1 << i
    val = 0
    for m in range(block, TOTAL_INPUTS, 2 * block):
        bits_to_set = (1 << block) - 1
        val |= bits_to_set << m
    INITIAL_STATE[i] = val

def _is_valid(network):
    """
    Check if network sorts all 0/1 inputs using zero-one principle.
    Uses bitmask simulation for speed.
    """
    if not network:
        return N <= 1

    st = list(INITIAL_STATE)
    n = N

    for i, j in network:
        si = st[i]
        sj = st[j]
        st[i] = si & sj
        st[j] = si | sj

    for i in range(n - 1):
        if st[i] & ~st[i + 1]:
            return False

    return True


def _compute_depth(network):
    """
    Compute the parallel depth of a comparator network.
    depth[i] = earliest stage where wire i can receive a new value.
    A comparator (i, j) at position p requires both wires to be free.
    """
    if not network:
        return 0
    wire_depth = [0] * N
    max_d = 0
    for i, j in network:
        d = max(wire_depth[i], wire_depth[j]) + 1
        wire_depth[i] = d
        wire_depth[j] = d
        if d > max_d:
            max_d = d
    return max_d

def _construct_odd_even_transposition():
    """Standard odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _greedy_delete(network, deadline):
    """
    Attempt to remove redundant comparators one by one.
    Returns the reduced network.
    """
    current = list(network)

    while time.time() < deadline:
        removed_any = False
        indices = list(range(len(current)))
        random.shuffle(indices)

        for idx in indices:
            if time.time() > deadline:
                break
            test_network = current[:idx] + current[idx+1:]
            if _is_valid(test_network):
                current = test_network
                removed_any = True
                break

        if not removed_any:
            break

    return current

def construct():
    """
    Construct a sorting network with as few comparators as possible.
    Structural change: depth-aware local search.
    Cost function: cost = size + lambda * depth.
    This creates pressure to find networks that are both small and shallow,
    which helps escape local minima where size-only optimization stalls.
    """
    start_time = time.time()
    deadline = start_time + 80.0

    current_network = _construct_odd_even_transposition()
    best_network = list(current_network)
    best_len = len(current_network)

    if not _is_valid(current_network):
        return best_network

    # Phase 1: Greedy Deletion
    greedy_deadline = start_time + 15.0
    current_network = _greedy_delete(current_network, greedy_deadline)

    if len(current_network) < best_len:
        best_network = list(current_network)
        best_len = len(current_network)

    current_len = len(current_network)
    current_depth = _compute_depth(current_network)

    # Phase 2: Depth-aware Simulated Annealing
    # Cost = size + lambda * depth
    # lambda is annealed: start high (favor depth reduction), decrease over time
    # to allow size-focused search later.
    T = 8.0
    cooling_rate = 0.997
    min_T = 0.05
    lambda_start = 2.0
    lambda_end = 0.3
    total_steps_estimate = int(math.log(min_T / T) / math.log(cooling_rate))
    step_count = 0

    while time.time() < deadline:
        if T < min_T:
            break

        # Anneal lambda from high to low
        progress = min(1.0, step_count / max(1, total_steps_estimate))
        lam = lambda_start + (lambda_end - lambda_start) * progress

        current_cost = current_len + lam * current_depth

        # Generate a candidate move
        move_type = random.random()

        if move_type < 0.35:
            # Delete a random comparator
            if len(current_network) > 15:
                idx = random.randint(0, len(current_network) - 1)
                candidate = current_network[:idx] + current_network[idx+1:]
            else:
                continue
        elif move_type < 0.50:
            # Add a random comparator near existing ones
            if len(current_network) > 0:
                # Pick a random existing comparator and add a nearby one
                ref_idx = random.randint(0, len(current_network) - 1)
                ref_i, ref_j = current_network[ref_idx]
                # Try wires close to the reference
                candidates_wires = []
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        ni = ref_i + di
                        nj = ref_j + dj
                        if 0 <= ni < N and 0 <= nj < N and ni < nj:
                            candidates_wires.append((ni, nj))
                if candidates_wires:
                    i, j = random.choice(candidates_wires)
                else:
                    i = random.randint(0, N - 2)
                    j = random.randint(i + 1, N - 1)
                idx = random.randint(0, len(current_network))
                candidate = current_network[:idx] + [(i, j)] + current_network[idx:]
            else:
                continue
        elif move_type < 0.65:
            # Swap adjacent comparators (reordering)
            if len(current_network) < 2:
                continue
            idx = random.randint(0, len(current_network) - 2)
            candidate = list(current_network)
            candidate[idx], candidate[idx + 1] = candidate[idx + 1], candidate[idx]
        elif move_type < 0.80:
            # Swap two comparators (non-adjacent reordering)
            if len(current_network) < 2:
                continue
            idx1 = random.randint(0, len(current_network) - 1)
            idx2 = random.randint(0, len(current_network) - 1)
            if idx1 == idx2:
                continue
            candidate = list(current_network)
            candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
        elif move_type < 0.90:
            # Modify a comparator's wires (small perturbation)
            if len(current_network) == 0:
                continue
            idx = random.randint(0, len(current_network) - 1)
            i, j = current_network[idx]
            # Perturb one wire by +/- 1 or +/- 2
            wire = random.choice(['i', 'j'])
            delta = random.choice([-2, -1, 1, 2])
            if wire == 'i':
                ni = i + delta
                if 0 <= ni < j:
                    candidate = list(current_network)
                    candidate[idx] = (ni, j)
                else:
                    continue
            else:
                nj = j + delta
                if i < nj < N:
                    candidate = list(current_network)
                    candidate[idx] = (i, nj)
                else:
                    continue
        else:
            # Depth-reducing move: try to move a comparator earlier
            # to reduce depth by reordering
            if len(current_network) < 3:
                continue
            idx = random.randint(1, len(current_network) - 1)
            # Try moving this comparator to an earlier position
            new_idx = random.randint(0, idx - 1)
            candidate = list(current_network)
            comp = candidate.pop(idx)
            candidate.insert(new_idx, comp)

        # Verify candidate
        if not _is_valid(candidate):
            continue

        new_len = len(candidate)
        new_depth = _compute_depth(candidate)
        new_cost = new_len + lam * new_depth

        delta_cost = new_cost - current_cost

        if delta_cost < 0:
            current_network = candidate
            current_len = new_len
            current_depth = new_depth
            if current_len < best_len:
                best_network = list(current_network)
                best_len = current_len
        else:
            if delta_cost == 0:
                if random.random() < 0.4:
                    current_network = candidate
                    current_len = new_len
                    current_depth = new_depth
            else:
                prob = math.exp(-delta_cost / T)
                if random.random() < prob:
                    current_network = candidate
                    current_len = new_len
                    current_depth = new_depth

        T *= cooling_rate
        step_count += 1

    return best_network

# EVOLVE-BLOCK-END
