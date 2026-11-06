"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17

# EVOLVE-BLOCK-START

def _seed_network():
    """Odd-even transposition sorting network for N wires."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net


def _apply_network(net, x):
    """Apply compare-exchange network to input vector x (numpy array)."""
    x = x.copy()
    for i, j in net:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]
    return x


def _verify_network(net):
    """Verify network sorts all 2^N Boolean inputs using zero-one principle."""
    for mask in range(1 << N):
        x = np.array([(mask >> k) & 1 for k in range(N)], dtype=np.int32)
        y = _apply_network(net, x)
        # Check sorted: non-decreasing
        if np.any(y[:-1] > y[1:]):
            return False
    return True


def _verify_network_fast(net):
    """Vectorized verification: apply all 2^N inputs at once."""
    n_inputs = 1 << N
    # Build input matrix: each row is a Boolean input
    # Use bit tricks
    idx = np.arange(n_inputs)
    # x[k, input] = (input >> k) & 1
    X = np.zeros((n_inputs, N), dtype=np.int32)
    for k in range(N):
        X[:, k] = (idx >> k) & 1

    for i, j in net:
        # Compare-exchange: if X[:,i] > X[:,j], swap
        swap_mask = X[:, i] > X[:, j]
        tmp = X[swap_mask, i].copy()
        X[swap_mask, i] = X[swap_mask, j]
        X[swap_mask, j] = tmp

    # Check all rows are sorted
    return not np.any(X[:, :-1] > X[:, 1:])


def _compute_depth(net):
    """Compute parallel depth of the network."""
    if not net:
        return 0
    wire_depth = [0] * N
    for i, j in net:
        d = max(wire_depth[i], wire_depth[j]) + 1
        wire_depth[i] = d
        wire_depth[j] = d
    return max(wire_depth)


def _try_remove(net, idx, deadline):
    """Try removing comparator at index idx. Return new net if valid, else None."""
    if time.time() > deadline:
        return None
    new_net = net[:idx] + net[idx+1:]
    if _verify_network_fast(new_net):
        return new_net
    return None


def _local_search_remove(net, deadline):
    """Local search: repeatedly try to remove comparators."""
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        n = len(net)
        # Randomize order of removal attempts
        order = list(range(n))
        random.shuffle(order)
        for idx in order:
            if time.time() > deadline:
                break
            new_net = _try_remove(net, idx, deadline)
            if new_net is not None:
                net = new_net
                improved = True
                break  # Restart scan after a successful removal
    return net


def _try_replace(net, idx, new_pair, deadline):
    """Try replacing comparator at idx with new_pair. Return new net if valid."""
    if time.time() > deadline:
        return None
    new_net = net[:idx] + [new_pair] + net[idx+1:]
    if _verify_network_fast(new_net):
        return new_net
    return None


def _local_search_replace(net, deadline):
    """Local search: try to replace comparators with different pairs to enable further removals."""
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        n = len(net)
        order = list(range(n))
        random.shuffle(order)
        for idx in order:
            if time.time() > deadline:
                break
            i, j = net[idx]
            # Try a few alternative pairs
            alternatives = []
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < nj < N and (ni, nj) != (i, j):
                        alternatives.append((ni, nj))
            random.shuffle(alternatives)
            for alt in alternatives:
                if time.time() > deadline:
                    break
                new_net = _try_replace(net, idx, alt, deadline)
                if new_net is not None:
                    net = new_net
                    improved = True
                    break
            if improved:
                break
    return net


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    # Start from seed network
    net = _seed_network()
    best_net = net
    best_len = len(net)

    # Phase 1: Aggressive removal local search
    if time.time() < deadline:
        net = _local_search_remove(net, deadline)
        if len(net) < best_len:
            best_net = net
            best_len = len(net)

    # Phase 2: Replace to unlock more removals, then remove again
    if time.time() < deadline:
        net = _local_search_replace(net, deadline)
        net = _local_search_remove(net, deadline)
        if len(net) < best_len:
            best_net = net
            best_len = len(net)

    # Phase 3: More removal passes
    if time.time() < deadline:
        for _ in range(5):
            if time.time() > deadline:
                break
            net = _local_search_remove(net, deadline)
            if len(net) < best_len:
                best_net = net
                best_len = len(net)

    # Phase 4: Try replacing and removing again
    if time.time() < deadline:
        net = _local_search_replace(net, deadline)
        net = _local_search_remove(net, deadline)
        if len(net) < best_len:
            best_net = net
            best_len = len(net)

    # Final safety: ensure we return a valid network
    # best_net should be valid by construction, but double-check
    # (skip final verify to save time; it was verified when accepted)

    return best_net

# EVOLVE-BLOCK-END
