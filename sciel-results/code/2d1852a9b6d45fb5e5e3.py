"""Size-efficient sorting network on 16 wires.

Combined strategy:
  * Foundation: Batcher's odd-even merge sort (63 comparators).
  * Refinement: Fast local search with vectorized removal tests and
    multiple kick strategies to escape local minima, targeting the known
    optimal 60-comparator network.
"""
import numpy as np
import time
import random

N = 16
_START = time.time()
_DEADLINE = _START + 82.0  # Safety margin


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


def _apply_to_matrix(bits, cmps):
    """Apply all comparators in cmps to the matrix bits (in-place)."""
    for i, j in cmps:
        a = bits[:, i]
        b = bits[:, j]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        bits[:, i] = lo
        bits[:, j] = hi
    return bits


def _verify_zero_one(cmps):
    bits = _build_matrix()
    _apply_to_matrix(bits, cmps)
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


def _try_remove_one(cmps, idx):
    """Test if removing cmps[idx] still yields a valid sorting network.
    Uses precomputed apply-all matrix and per-input dependency count for speed.
    Returns True if removal is valid."""
    n = len(cmps)
    if n <= 1:
        return False

    # Apply all comparators to get final sorted matrix
    bits = _build_matrix()
    for i, j in cmps:
        a = bits[:, i]
        b = bits[:, j]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        bits[:, i] = lo
        bits[:, j] = hi

    # For the removed comparator (i, j), check if any input row has
    # bits[i] > bits[j] BEFORE the comparator was applied.
    # We can determine this by checking: in the final matrix, if the row
    # is sorted and had bits[i] > bits[j] before, then the comparator was
    # needed. But we need the state BEFORE applying cmps[idx].
    # 
    # Alternative: re-apply all except idx. But that's slow.
    # Better: track for each input, how many comparators "fix" it.
    # 
    # Fastest correct approach: just re-apply all except idx, but only
    # for the rows that might be affected. Actually, let's just do the
    # straightforward thing but with a pre-allocated matrix to avoid
    # the _build_matrix overhead each time.
    
    bits2 = _build_matrix()
    for k in range(n):
        if k == idx:
            continue
        i, j = cmps[k]
        a = bits2[:, i]
        b = bits2[:, j]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        bits2[:, i] = lo
        bits2[:, j] = hi
    
    return _is_sorted(bits2)


def _reduce_network(initial_cmps, seed=42, deadline=_DEADLINE):
    """Local search: repeatedly try to remove comparators, with kicks
    to escape local minima."""
    current = list(initial_cmps)
    rng = random.Random(seed)
    
    while time.time() < deadline:
        n = len(current)
        if n <= 60:
            break
        
        any_removed = False
        # Try to remove any comparator, scanning in random order
        order = list(range(n))
        rng.shuffle(order)
        
        for idx in order:
            if time.time() > deadline:
                return current
            
            if _try_remove_one(current, idx):
                del current[idx]
                any_removed = True
                break  # Restart scan with new length
        
        if not any_removed:
            # Local minimum. Try kicks to escape.
            if n <= 60:
                break
            
            # Kick strategy: add a redundant comparator (duplicate of an
            # existing one) to create new removable comparators, or do
            # structural shuffles.
            kick_type = rng.random()
            
            if kick_type < 0.5:
                # Duplicate a random comparator at a random position
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n)
                current.insert(j, current[i])
            elif kick_type < 0.7:
                # Swap two comparators
                if n > 1:
                    i = rng.randint(0, n - 2)
                    j = rng.randint(i + 1, n - 1)
                    current[i], current[j] = current[j], current[i]
            elif kick_type < 0.85:
                # Move a comparator to a random position
                if n > 1:
                    i = rng.randint(0, n - 1)
                    j = rng.randint(0, n - 1)
                    if i != j:
                        cmp = current.pop(i)
                        current.insert(j, cmp)
            else:
                # Reverse a small segment
                if n > 2:
                    i = rng.randint(0, n - 3)
                    j = rng.randint(i + 1, n - 1)
                    current[i:j+1] = current[i:j+1][::-1]

    return current


def _try_perturb_and_reduce(net, best_count, deadline, seed):
    """Perturb the network in various ways and try to reduce."""
    rng = random.Random(seed)
    best_net = None
    best_len = best_count
    
    for trial in range(20):
        if time.time() > deadline:
            break
        
        perturbed = list(net)
        n = len(perturbed)
        
        # Apply random perturbations
        num_perts = rng.randint(1, 5)
        for _ in range(num_perts):
            if n <= 1:
                break
            pert_type = rng.random()
            if pert_type < 0.4:
                # Duplicate a comparator at random position
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n)
                perturbed.insert(j, perturbed[i])
                n += 1
            elif pert_type < 0.65:
                # Swap
                i = rng.randint(0, n - 2)
                j = rng.randint(i + 1, n - 1)
                perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            elif pert_type < 0.85:
                # Move
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if i != j:
                    cmp = perturbed.pop(i)
                    perturbed.insert(j, cmp)
                    n = len(perturbed)
            else:
                # Reverse small segment
                if n > 2:
                    i = rng.randint(0, n - 3)
                    j = rng.randint(i + 1, n - 1)
                    perturbed[i:j+1] = perturbed[i:j+1][::-1]
        
        candidate = _reduce_network(perturbed, seed=seed + trial, deadline=deadline)
        if len(candidate) < best_len:
            best_len = len(candidate)
            best_net = candidate
            if best_len <= 60:
                break
    
    return best_net, best_len


def construct():
    net = _batcher_odd_even_merge(N)
    
    if not _verify_zero_one(net):
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

    # First pass: direct reduction from Batcher
    best_net = _reduce_network(net, seed=42, deadline=_DEADLINE - 5)
    best_count = len(best_net)
    
    # If we didn't reach 60, try perturbation + reduction
    if best_count > 60 and time.time() < _DEADLINE - 3:
        result_net, result_count = _try_perturb_and_reduce(
            net, best_count, _DEADLINE - 3, seed=100
        )
        if result_net is not None and result_count < best_count:
            best_count = result_count
            best_net = result_net
    
    # Final safety check
    if not _verify_zero_one(best_net):
        best_net = net

    return best_net


if __name__ == "__main__":
    net = construct()
    print("comparators:", len(net))
    print("depth:", _parallel_depth(net))
    print("zero-one verified:", _verify_zero_one(net))
