"""Size-efficient sorting network on 16 wires.

Strategy:
  * Start from Batcher's odd-even merge sort (63 comparators).
  * Aggressive local search with smart kicks to reach the known 60-comparator optimum.
  * Optimized verification using bit-packing for speed.
"""
import numpy as np
import time
import random

N = 16
_START = time.time()
_DEADLINE = _START + 80.0  # Safety margin (90s kill, 10s buffer)


def _batcher_odd_even_merge(n):
    """Build Batcher's odd-even merge sort network for n wires."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]

    cmps = []
    # Recursively sort odd and even positions
    for a, b in _batcher_odd_even_merge(n // 2):
        cmps.append((2 * a + 1, 2 * b + 1))
    for a, b in _batcher_odd_even_merge(n // 2):
        cmps.append((2 * a, 2 * b))
    # Odd-even merge: compare (0,1), (2,3), ..., then (1,2), (3,4), ...
    for i in range(n // 2):
        cmps.append((2 * i, 2 * i + 1))
    for i in range(n // 2 - 1):
        cmps.append((2 * i + 1, 2 * i + 2))
    return cmps


def _pack_bits(bits):
    """Pack a (M, N) int8 matrix into a (M,) uint64 array for faster ops."""
    M = bits.shape[0]
    packed = np.zeros(M, dtype=np.uint64)
    for k in range(N):
        packed |= (bits[:, k].astype(np.uint64) << k)
    return packed


def _unpack_packed(packed):
    """Unpack a (M,) uint64 array into a (M, N) int8 matrix."""
    M = packed.shape[0]
    bits = np.zeros((M, N), dtype=np.int8)
    for k in range(N):
        bits[:, k] = ((packed >> k) & 1).astype(np.int8)
    return bits


def _apply_cmp_packed(packed, i, j):
    """Apply compare-exchange to wires i and j on packed data (in-place)."""
    mask_i = (1 << i)
    mask_j = (1 << j)
    val_i = (packed & mask_i)
    val_j = (packed & mask_j)
    # min goes to i, max goes to j
    # If val_i > val_j, swap them
    # For single bits: min = val_i & val_j, max = val_i | val_j
    lo = val_i & val_j
    hi = val_i | val_j
    # Clear both positions
    packed = packed & ~(mask_i | mask_j)
    # Set lo at i, hi at j
    packed = packed | (lo << i) | (hi << j)
    return packed


def _is_sorted_packed(packed):
    """Check if all rows are sorted (non-decreasing left to right).
    
    A row is sorted iff for every adjacent pair (k, k+1), bit k <= bit k+1.
    This means no pattern 10 in adjacent bits.
    """
    # Check for any adjacent 1,0 pattern
    # For each adjacent pair (k, k+1): violation if bit k=1 and bit k+1=0
    # mask for violations: for each k from 0 to N-2, (bit k AND NOT bit k+1)
    for k in range(N - 1):
        bit_k = (packed >> k) & 1
        bit_k1 = (packed >> (k + 1)) & 1
        if np.any(bit_k == 1) and np.any(bit_k1 == 0):
            # More precise: check if there's any row where bit_k=1 AND bit_k1=0
            violation = ((packed >> k) & 1) & (((packed >> (k + 1)) & 1) ^ 1)
            if np.any(violation):
                return False
    return True


def _verify_zero_one(cmps):
    """Verify sorting network using zero-one principle with packed representation."""
    M = 1 << N
    idx = np.arange(M, dtype=np.uint64)
    # Pack the identity: wire k gets bit k of the index
    packed = idx.copy()
    for i, j in cmps:
        packed = _apply_cmp_packed(packed, i, j)
    return _is_sorted_packed(packed)


def _parallel_depth(cmps):
    end_time = [0] * N
    depth = 0
    for i, j in cmps:
        t = max(end_time[i], end_time[j]) + 1
        depth = max(depth, t)
        end_time[i] = t
        end_time[j] = t
    return depth


def _fast_test_removal(cmps, idx_to_remove):
    """Test if removing cmps[idx_to_remove] still yields a sorting network.
    
    Uses packed representation for speed.
    """
    M = 1 << N
    idx = np.arange(M, dtype=np.uint64)
    packed = idx.copy()
    n = len(cmps)
    
    for k in range(n):
        if k == idx_to_remove:
            continue
        i, j = cmps[k]
        packed = _apply_cmp_packed(packed, i, j)
        # Early exit check every 8 comparators
        if k % 8 == 7 and k < n - 1:
            if not _is_sorted_packed(packed):
                return False
    
    return _is_sorted_packed(packed)


def _reduce_network(initial_cmps, seed=42):
    """Local search to minimize comparator count."""
    current = list(initial_cmps)
    random.seed(seed)
    
    def _kick(cmps):
        """Apply a random kick to escape local minimum."""
        n = len(cmps)
        if n < 2:
            return cmps
        kick_type = random.random()
        if kick_type < 0.20:
            # Swap two comparators
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            cmps[i], cmps[j] = cmps[j], cmps[i]
        elif kick_type < 0.40:
            # Move a comparator to a new position
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            cmps.insert(j, cmps.pop(i))
        elif kick_type < 0.55:
            # Insert a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, n)
            cmps.insert(pos, (i, j))
        elif kick_type < 0.70:
            # Swap a comparator with a nearby one
            i = random.randint(0, n - 1)
            offset = random.choice([-2, -1, 1, 2])
            j = i + offset
            if 0 <= j < n:
                cmps[i], cmps[j] = cmps[j], cmps[i]
        elif kick_type < 0.85:
            # Duplicate a random comparator at a different position
            i = random.randint(0, n - 1)
            pos = random.randint(0, n)
            cmps.insert(pos, cmps[i])
        else:
            # Remove a random comparator and add a different one (replacement)
            i = random.randint(0, n - 1)
            cmps.pop(i)
            a = random.randint(0, N - 2)
            b = random.randint(a + 1, N - 1)
            pos = random.randint(0, len(cmps))
            cmps.insert(pos, (a, b))
        return cmps
    
    best = list(current)
    best_len = len(current)
    kick_count = 0
    
    while True:
        if time.time() > _DEADLINE:
            break
        n = len(current)
        if n == 60:
            break
        
        any_removed = False
        # Try to remove any comparator, starting from random positions
        order = list(range(n))
        random.shuffle(order)
        
        for idx in order:
            if time.time() > _DEADLINE:
                return best
            
            if _fast_test_removal(current, idx):
                del current[idx]
                any_removed = True
                if len(current) < best_len:
                    best_len = len(current)
                    best = list(current)
                break  # Restart scan
        
        if not any_removed:
            # Local minimum. Try a kick.
            if n == 60:
                break
            kick_count += 1
            current = _kick(current)
            # After a kick that grows the network, try to shrink back
            if len(current) > best_len:
                for idx in range(len(current)):
                    if time.time() > _DEADLINE:
                        return best
                    if _fast_test_removal(current, idx):
                        del current[idx]
                        if len(current) < best_len:
                            best_len = len(current)
                            best = list(current)
                        break
            # Limit kicks to avoid infinite loops
            if kick_count > 200:
                break

    return best


def construct():
    net = _batcher_odd_even_merge(N)
    
    if not _verify_zero_one(net):
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

    best_net = net
    best_count = len(net)
    
    # Try multiple seeds with different perturbations
    seeds = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]
    for seed in seeds:
        if time.time() > _DEADLINE - 5:
            break
        if best_count == 60:
            break
        
        # Create perturbed initial network
        random.seed(seed)
        perturbed = list(net)
        
        # Apply several random perturbations for diversity
        n_perturb = 3 + (seed % 5)
        for _ in range(n_perturb):
            if len(perturbed) > 1:
                choice = random.random()
                if choice < 0.4:
                    # Swap
                    i = random.randint(0, len(perturbed) - 2)
                    j = random.randint(i + 1, len(perturbed) - 1)
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
                elif choice < 0.7:
                    # Move
                    i = random.randint(0, len(perturbed) - 1)
                    j = random.randint(0, len(perturbed) - 1)
                    perturbed.insert(j, perturbed.pop(i))
                else:
                    # Replace a comparator with a random one
                    i = random.randint(0, len(perturbed) - 1)
                    a = random.randint(0, N - 2)
                    b = random.randint(a + 1, N - 1)
                    perturbed[i] = (a, b)
        
        candidate = _reduce_network(perturbed, seed=seed)
        if len(candidate) < best_count:
            best_count = len(candidate)
            best_net = candidate

    # Final verification
    if not _verify_zero_one(best_net):
        best_net = net

    return best_net


if __name__ == "__main__":
    net = construct()
    print("comparators:", len(net))
    print("depth:", _parallel_depth(net))
    print("zero-one verified:", _verify_zero_one(net))
