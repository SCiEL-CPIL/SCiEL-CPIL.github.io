"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17

# EVOLVE-BLOCK-START
def _make_all_inputs():
    """Build the (2^N, N) uint8 array of all Boolean inputs."""
    n_bools = 1 << N
    indices = np.arange(n_bools, dtype=np.uint8)
    arr = np.zeros((n_bools, N), dtype=np.uint8)
    for w in range(N):
        arr[:, w] = (indices >> w) & 1
    return arr

def _validate_fast(comparators, arr=None):
    """Check if network sorts all Boolean inputs. Returns True if valid.
    If arr is None, builds it internally."""
    if arr is None:
        arr = _make_all_inputs()
    a = arr.copy()
    for i, j in comparators:
        mask = a[:, i] > a[:, j]
        if np.any(mask):
            a[mask, i] = 1
            a[mask, j] = 0
    for k in range(N - 1):
        if np.any(a[:, k] > a[:, k + 1]):
            return False
    return True

def _depth_of_network(comparators):
    """Compute the parallel depth of a network."""
    wire_depth = [0] * N
    max_depth = 0
    for i, j in comparators:
        d = 1 + max(wire_depth[i], wire_depth[j])
        wire_depth[i] = d
        wire_depth[j] = d
        max_depth = max(max_depth, d)
    return max_depth

def construct():
    """Construct a sorting network on 17 wires with as few comparators as possible."""
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    random.seed(42)

    # Start with the odd-even transposition network as seed
    seed_network = [(i, i + 1) for phase in range(N)
                    for i in range(phase % 2, N - 1, 2)]

    best_network = seed_network
    best_count = len(seed_network)

    # Pre-build the all-inputs array once for reuse
    all_inputs = _make_all_inputs()

    # Verify seed is correct
    if not _validate_fast(seed_network, all_inputs):
        return seed_network

    current = list(seed_network)
    current_count = len(current)

    # Phase 1: Greedy single-removal passes
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        indices = list(range(len(current)))
        random.shuffle(indices)

        for idx in indices:
            if time.time() > deadline - 10.0:
                break
            if idx >= len(current):
                continue

            candidate = current[:idx] + current[idx + 1:]
            if not candidate:
                break

            if _validate_fast(candidate, all_inputs):
                current = candidate
                current_count -= 1
                improved = True
                if current_count < best_count:
                    best_count = current_count
                    best_network = list(current)
                indices = list(range(len(current)))
                random.shuffle(indices)

    # Phase 2: Batch deletion search — try removing k comparators simultaneously
    # This explores a much larger search space per validation call.
    # Strategy: iteratively pick random subsets of size k to remove.
    # If valid, commit and continue with smaller k.
    # Also try "swap+remove" moves for structural improvement.
    batch_k = 3
    no_improve_count = 0
    max_no_improve = 8

    while time.time() < deadline - 5.0 and batch_k >= 1:
        if no_improve_count >= max_no_improve:
            batch_k -= 1
            no_improve_count = 0
            if batch_k < 1:
                break

        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            if len(current) <= batch_k:
                break

            # Pick batch_k random distinct indices to remove
            remove_set = set(random.sample(range(len(current)), batch_k))
            candidate = [c for idx, c in enumerate(current) if idx not in remove_set]

            if _validate_fast(candidate, all_inputs):
                current = candidate
                current_count -= batch_k
                no_improve_count = 0
                if current_count < best_count:
                    best_count = current_count
                    best_network = list(current)
            else:
                no_improve_count += 1

    # Phase 3: Final single-removal polish
    if time.time() < deadline - 3.0:
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            indices = list(range(len(current)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > deadline - 3.0:
                    break
                if idx >= len(current):
                    continue
                candidate = current[:idx] + current[idx + 1:]
                if candidate and _validate_fast(candidate, all_inputs):
                    current = candidate
                    current_count -= 1
                    improved = True
                    if current_count < best_count:
                        best_count = current_count
                        best_network = list(current)
                    indices = list(range(len(current)))
                    random.shuffle(indices)

    # Ensure we return a valid network
    if not best_network or not _validate_fast(best_network, all_inputs):
        return seed_network

    return best_network
# EVOLVE-BLOCK-END
