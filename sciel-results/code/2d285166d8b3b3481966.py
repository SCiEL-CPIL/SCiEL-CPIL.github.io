"""
Size-efficient sorting networks on 17 wires – fast local search with two-tier verification.
"""

import time
import random
import math
import numpy as np

N = 17
TOTAL_INPUTS = 1 << N  # 131072

# Precompute per-wire bitmask columns for full verification
WIRE_COLS = TOTAL_INPUTS // 64  # 2048

INITIAL_STATE = []
for i in range(N):
    byte_len = TOTAL_INPUTS // 8  # 16384
    ba = bytearray(byte_len)
    block = 1 << i
    for b in range(byte_len):
        byte_val = 0
        for bit in range(8):
            k = b * 8 + bit
            if (k // block) % 2 == 1:
                byte_val |= (1 << bit)
        ba[b] = byte_val
    arr = np.frombuffer(ba, dtype=np.uint64).copy()
    INITIAL_STATE.append(arr)


def _is_valid_full(network):
    """Full verification over all 131072 inputs."""
    if not network:
        return N <= 1

    state = [arr.copy() for arr in INITIAL_STATE]

    for i, j in network:
        si = state[i]
        sj = state[j]
        state[i] = si & sj
        state[j] = si | sj

    for i in range(N - 1):
        if np.any(state[i] & ~state[i + 1]):
            return False

    return True


# Precompute subset indices and their initial state for fast checking
SUBSET_SIZE = 2048
SUBSET_INDICES = np.array(sorted(random.sample(range(TOTAL_INPUTS), SUBSET_SIZE)), dtype=np.int32)
SUBSET_WORDS = SUBSET_SIZE // 64  # 32

SUBSET_INITIAL = []
for i in range(N):
    arr = np.zeros(SUBSET_WORDS, dtype=np.uint64)
    for w in range(SUBSET_WORDS):
        for b in range(64):
            idx = SUBSET_INDICES[w * 64 + b]
            if (idx >> i) & 1:
                arr[w] |= (1 << b)
    SUBSET_INITIAL.append(arr)


def _is_valid_subset(network):
    """Fast check on a subset of 2048 inputs."""
    if not network:
        return N <= 1

    state = [arr.copy() for arr in SUBSET_INITIAL]

    for i, j in network:
        si = state[i]
        sj = state[j]
        state[i] = si & sj
        state[j] = si | sj

    for i in range(N - 1):
        if np.any(state[i] & ~state[i + 1]):
            return False

    return True


def _construct_odd_even_transposition():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    current_network = _construct_odd_even_transposition()
    best_network = list(current_network)
    best_len = len(current_network)

    if not _is_valid_full(current_network):
        return best_network

    current_len = len(current_network)

    T = 5.0
    cooling_rate = 0.998
    min_T = 0.05

    while time.time() < deadline:
        if T < min_T:
            T = min_T * 0.5

        move_type = random.random()

        if move_type < 0.5:
            if len(current_network) > 5:
                idx = random.randint(0, len(current_network) - 1)
                candidate = current_network[:idx] + current_network[idx + 1:]
                new_len = current_len - 1
            else:
                continue
        elif move_type < 0.65:
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            idx = random.randint(0, len(current_network))
            candidate = current_network[:idx] + [(i, j)] + current_network[idx:]
            new_len = current_len + 1
        elif move_type < 0.8:
            if len(current_network) < 2:
                continue
            idx1 = random.randint(0, len(current_network) - 1)
            idx2 = random.randint(0, len(current_network) - 1)
            if idx1 == idx2:
                continue
            candidate = list(current_network)
            candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
            new_len = current_len
        else:
            if len(current_network) == 0:
                continue
            idx = random.randint(0, len(current_network) - 1)
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            candidate = list(current_network)
            candidate[idx] = (i, j)
            new_len = current_len

        if not _is_valid_subset(candidate):
            continue

        if not _is_valid_full(candidate):
            continue

        delta_cost = new_len - current_len
        if delta_cost < 0:
            current_network = candidate
            current_len = new_len
            if current_len < best_len:
                best_network = list(current_network)
                best_len = current_len
                T = max(T, 2.0)
        else:
            if delta_cost == 0:
                if random.random() < 0.3:
                    current_network = candidate
            else:
                prob = math.exp(-delta_cost / T)
                if random.random() < prob:
                    current_network = candidate
                    current_len = new_len

        T *= cooling_rate

    return best_network
