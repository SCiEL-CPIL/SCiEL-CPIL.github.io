"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
import itertools

N = 16

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85

    all_inputs = np.array(list(itertools.product([0, 1], repeat=N))).T

    def verify(network):
        x = all_inputs.copy()
        for i, j in network:
            x[i], x[j] = np.minimum(x[i], x[j]), np.maximum(x[i], x[j])
        return np.all(np.diff(x, axis=0) >= 0, axis=0).all()

    def odd_even_merge(n):
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        result = []
        for i in range(n // 2):
            result.append((i, i + n // 2))
        for (i, j) in odd_even_merge(n // 2):
            result.append((2 * i + 1, 2 * j + 1))
        for (i, j) in odd_even_merge(n // 2):
            result.append((2 * i, 2 * j))
        return result

    def batcher_oe(n):
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        result = []
        result.extend(batcher_oe(n // 2))
        result.extend([(i + n // 2, j + n // 2) for (i, j) in batcher_oe(n // 2)])
        result.extend(odd_even_merge(n))
        return result

    def odd_even_transposition():
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

    best = odd_even_transposition()

    for start_net in [batcher_oe(N), odd_even_transposition()]:
        if time.time() > deadline:
            break
        if not verify(start_net):
            continue
        if len(start_net) < len(best):
            best = list(start_net)

        net = list(start_net)
        while time.time() < deadline:
            improved = False
            indices = list(range(len(net)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > deadline:
                    break
                candidate = net[:idx] + net[idx+1:]
                if verify(candidate):
                    net = candidate
                    improved = True
                    if len(net) < len(best):
                        best = list(net)
            if not improved:
                break

        for _ in range(10):
            if time.time() > deadline:
                break
            if len(net) >= len(best):
                break
            i, j = random.sample(range(N), 2)
            if i > j:
                i, j = j, i
            candidate = net + [(i, j)]
            if verify(candidate):
                net = list(candidate)
                while time.time() < deadline:
                    improved = False
                    indices = list(range(len(net)))
                    random.shuffle(indices)
                    for idx in indices:
                        if time.time() > deadline:
                            break
                        candidate2 = net[:idx] + net[idx+1:]
                        if verify(candidate2):
                            net = candidate2
                            improved = True
                            if len(net) < len(best):
                                best = list(net)
                    if not improved:
                        break

    return best
# EVOLVE-BLOCK-END
