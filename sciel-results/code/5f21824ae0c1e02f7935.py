"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16
TOTAL_INPUTS = 1 << N

def _build_input_matrix():
    arange = np.arange(TOTAL_INPUTS, dtype=np.uint8)
    bits = np.zeros((TOTAL_INPUTS, N), dtype=np.uint8)
    for w in range(N):
        bits[:, w] = (arange >> w) & 1
    return bits

_INPUTS = _build_input_matrix()


def _verify_network(comparators):
    if len(comparators) == 0:
        return N <= 1
    state = _INPUTS.copy()
    for i, j in comparators:
        si = state[:, i]
        sj = state[:, j]
        state[:, i] = np.minimum(si, sj)
        state[:, j] = np.maximum(si, sj)
    diffs = np.diff(state, axis=1)
    return bool(np.all(diffs >= 0))


def _bitonic_network():
    comps = []
    k = 2
    while k <= N:
        j = k // 2
        while j >= 1:
            for i in range(0, N, k):
                for x in range(j):
                    if (i // k) % 2 == 0:
                        comps.append((i + x, i + x + j))
                    else:
                        comps.append((i + x + j, i + x))
            j //= 2
        k *= 2
    return comps


def _odd_even_transposition():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 84.0

    network = _bitonic_network()
    if not _verify_network(network):
        network = _odd_even_transposition()

    best = list(network)
    best_count = len(network)

    # Greedy removal
    changed = True
    while changed and (time.time() - start_time) < deadline - 20.0:
        changed = False
        for idx in range(len(best)):
            if (time.time() - start_time) > deadline - 20.0:
                break
            trial = best[:idx] + best[idx + 1:]
            if _verify_network(trial):
                best = trial
                best_count = len(best)
                changed = True
                break

    # Local search
    if (time.time() - start_time) < deadline - 10.0:
        rng = random.Random(12345)
        current = list(best)
        current_count = len(current)

        while (time.time() - start_time) < deadline - 5.0:
            op = rng.randint(0, 2)
            if op == 0 and current_count > 1:
                idx = rng.randint(0, current_count - 1)
                trial = current[:idx] + current[idx + 1:]
                if _verify_network(trial):
                    current = trial
                    current_count -= 1
            elif op == 1:
                if current_count >= 2:
                    idx1 = rng.randint(0, current_count - 2)
                    idx2 = rng.randint(idx1 + 1, current_count - 1)
                    trial = current[:]
                    trial[idx1], trial[idx2] = trial[idx2], trial[idx1]
                    if _verify_network(trial):
                        current = trial
            else:
                i = rng.randint(0, N - 2)
                j = rng.randint(i + 1, N - 1)
                if (i, j) not in current:
                    trial = current + [(i, j)]
                    if _verify_network(trial):
                        idx = rng.randint(0, len(trial) - 1)
                        trial2 = trial[:idx] + trial[idx + 1:]
                        if _verify_network(trial2):
                            current = trial2
                            current_count = len(current)

            if current_count < best_count:
                best = current
                best_count = current_count

    return best
# EVOLVE-BLOCK-END
