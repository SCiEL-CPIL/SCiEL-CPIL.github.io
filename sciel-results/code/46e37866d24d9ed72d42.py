import time
import numpy as np

N = 18
NUM_MASKS = 1 << N

def _build_masks():
    """Build the 18 x 262144 matrix of all Boolean inputs."""
    masks = np.arange(NUM_MASKS, dtype=np.uint8)
    # Each column is one input; row i is bit i
    mat = np.zeros((N, NUM_MASKS), dtype=np.uint8)
    for i in range(N):
        mat[i] = (masks >> i) & 1
    return mat

def _verify_network(network, masks_mat):
    """Verify network sorts all 262144 Boolean inputs. Returns (bool, swap_counts).
    swap_counts[i] = number of times comparator i actually swaps (out of 262144)."""
    x = masks_mat.copy()
    swap_counts = np.zeros(len(network), dtype=np.int32)
    for idx, (i, j) in enumerate(network):
        # x[i] > x[j] means x[i]=1, x[j]=0
        swap = (x[i] > x[j])
        swap_counts[idx] = np.sum(swap)
        if np.any(swap):
            x[i] = np.where(swap, x[j], x[i])
            x[j] = np.where(swap, x[i], x[j])
    # Check sorted: for all adjacent pairs, x[k] <= x[k+1]
    for k in range(N - 1):
        if np.any(x[k] > x[k + 1]):
            return False, swap_counts
    return True, swap_counts

def _is_sorted_fast(network, masks_mat):
    """Quick check if network is valid."""
    ok, _ = _verify_network(network, masks_mat)
    return ok

def construct():
    start_time = time.time()
    deadline = start_time + 57  # 3s safety margin for 60s limit

    masks_mat = _build_masks()

    # Start with odd-even transposition network (153 comparators)
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]

    # Verify initial network
    ok, _ = _verify_network(network, masks_mat)
    assert ok, "Initial network is invalid"

    best_network = list(network)
    best_len = len(network)

    # Greedy removal: repeatedly try to remove each comparator
    # Process in order of increasing swap count (least active first)
    while (time.time() - start_time) < deadline - 3:
        if len(network) == 0:
            break

        # Verify and get swap counts
        ok, swap_counts = _verify_network(network, masks_mat)
        if not ok:
            break

        # Find the comparator with minimum swap count (most likely removable)
        # But we need to actually test removal
        # Try removing in order of increasing swap count
        order = np.argsort(swap_counts)

        found_removal = False
        for idx in order:
            if (time.time() - start_time) > deadline - 5:
                break
            idx = int(idx)

            # Skip if this comparator never swaps (removing it is safe)
            # But we still need to verify because the network structure matters
            test_network = network[:idx] + network[idx + 1:]

            if _is_sorted_fast(test_network, masks_mat):
                network = test_network
                if len(network) < best_len:
                    best_network = list(network)
                    best_len = len(network)
                found_removal = True
                break  # Restart the removal loop with the new network

        if not found_removal:
            # No single removal works; try add-then-remove
            if (time.time() - start_time) > deadline - 8:
                break

            # Try adding each possible comparator, then removing one
            existing = set(network)
            all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N) if (i, j) not in existing]

            # Limit the number of additions to try
            max_additions = min(len(all_pairs), 80)
            np.random.seed(42)
            if len(all_pairs) > max_additions:
                all_pairs = [all_pairs[i] for i in np.random.choice(len(all_pairs), max_additions, replace=False)]

            found_add_remove = False
            for (ai, aj) in all_pairs:
                if (time.time() - start_time) > deadline - 8:
                    break

                extended = network + [(ai, aj)]
                ok_ext, sc_ext = _verify_network(extended, masks_mat)
                if not ok_ext:
                    continue

                # Try removing each comparator from the extended network
                # (except the one we just added)
                order_ext = np.argsort(sc_ext)
                for ridx in order_ext:
                    if (time.time() - start_time) > deadline - 8:
                        break
                    ridx = int(ridx)
                    if ridx == len(extended) - 1:
                        continue  # Don't remove the one we just added

                    removed = extended[:ridx] + extended[ridx + 1:]
                    if len(removed) < len(network):
                        if _is_sorted_fast(removed, masks_mat):
                            network = removed
                            if len(network) < best_len:
                                best_network = list(network)
                                best_len = len(network)
                            found_add_remove = True
                            break
                if found_add_remove:
                    break

            if not found_add_remove:
                break

    return best_network
