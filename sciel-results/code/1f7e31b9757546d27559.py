"""Size-efficient sorting network on 16 wires.

Combined strategy:
  * Foundation: Batcher's odd-even merge sort (63 comparators).
  * Refinement: Aggressive local search with "kicks" to escape local minima,
    targeting the known optimal 60-comparator network.
"""
import numpy as np
import time
import random

N = 16
_START = time.time()
_DEADLINE = _START + 85.0  # Safety margin


def _batcher_odd_even_merge(n):
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]

    cmps = []
    for a, b in _batcher_odd_even_merge(n // 2):
        cmps.append((2 * a + 1, 2 * b + 1))
    for a, b in _batcher_odd_even_merge(n // 2):
        cmps.append((2 * a, 2 * b))
    for i in range(n // 2):
        cmps.append((2 * i, 2 * i + 1))
    for i in range(n // 2 - 1):
        cmps.append((2 * i + 1, 2 * i + 2))
    return cmps


def _build_matrix():
    M = 1 << N
    idx = np.arange(M, dtype=np.int64)
    bits = np.zeros((M, N), dtype=np.int8)
    for k in range(N):
        bits[:, k] = ((idx >> k) & 1).astype(np.int8)
    return bits


def _is_sorted(bits):
    return not np.any(bits[:, :-1] > bits[:, 1:])


def _apply(bits, cmps, start):
    for i, j in cmps[start:]:
        a = bits[:, i]
        b = bits[:, j]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        bits[:, i] = lo
        bits[:, j] = hi


def _verify_zero_one(cmps):
    bits = _build_matrix()
    _apply(bits, cmps, 0)
    return _is_sorted(bits)


def _parallel_depth(cmps):
    end_time = [0] * N
    depth = 0
    for i, j in cmps:
        t = max(end_time[i], end_time[j]) + 1
        depth = max(depth, t)
        end_time[i] = t
        end_time[j] = t
    return depth


def _reduce_network(initial_cmps):
    current = list(initial_cmps)
    random.seed(42)
    
    # Pre-compute the full matrix for the initial state? No, it changes.
    
    while True:
        if time.time() > _DEADLINE:
            break
        n = len(current)
        if n == 60:
            break
        
        any_removed = False
        # Try to remove any comparator
        for idx in range(n):
            if time.time() > _DEADLINE:
                return current
            
            # Test removal
            cand = _build_matrix()
            for k in range(n):
                if k == idx:
                    continue
                i, j = current[k]
                a = cand[:, i]
                b = cand[:, j]
                lo = np.minimum(a, b)
                hi = np.maximum(a, b)
                cand[:, i] = lo
                cand[:, j] = hi
            
            if _is_sorted(cand):
                del current[idx]
                any_removed = True
                break # Restart scan from 0
        
        if not any_removed:
            # Local minimum. Try a kick.
            if n == 60:
                break
            
            # Kick: Swap two random comparators
            if n > 1:
                i = random.randint(0, n - 2)
                j = random.randint(i + 1, n - 1)
                current[i], current[j] = current[j], current[i]
                
                # Optional: If swap doesn't help after a few tries, try move
            else:
                break

    return current


def construct():
    net = _batcher_odd_even_merge(N)
    
    if not _verify_zero_one(net):
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

    best_net = _reduce_network(net)
    best_count = len(best_net)

    # If we didn't reach 60, try a few random restarts with different seeds
    if best_count > 60 and time.time() < _DEADLINE - 5:
        for trial in range(10):
            if time.time() > _DEADLINE - 5:
                break
            random.seed(100 + trial)
            # Perturb the initial network slightly
            perturbed = list(net)
            # Swap a few random pairs
            for _ in range(5):
                if len(perturbed) > 1:
                    i = random.randint(0, len(perturbed) - 2)
                    j = random.randint(i + 1, len(perturbed) - 1)
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            
            candidate = _reduce_network(perturbed)
            if len(candidate) < best_count:
                best_count = len(candidate)
                best_net = candidate
                if best_count == 60:
                    break

    if not _verify_zero_one(best_net):
        best_net = net

    return best_net


if __name__ == "__main__":
    net = construct()
    print("comparators:", len(net))
    print("depth:", _parallel_depth(net))
    print("zero-one verified:", _verify_zero_one(net))
