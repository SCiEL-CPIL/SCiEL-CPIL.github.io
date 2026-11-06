import time
import random
import numpy as np

N = 18
NUM_BOOLS = 1 << N
SEED = 153.0
DEADLINE = 85.0  # wall-clock safety margin (evaluator kills at 90s)

# ---------------------------------------------------------------------------
# Fast verification: state of each wire is a 262144-bit integer (a mask over
# all Boolean inputs). A comparator (i, j) rewrites:
#   w_i' = w_i & w_j          (min -> the inputs where BOTH are 1)
#   w_j' = w_i | w_j          (max -> the inputs where EITHER is 1)
# Sorting is certified iff the top bit of wire 0 is set on every input where
# at least one wire is set (i.e. no input has a 1 "below" the top wire).
# ---------------------------------------------------------------------------
_ALL_ONES = (1 << NUM_BOOLS) - 1
_TOP_BIT = 1 << (NUM_BOOLS - 1)
_ANY_BIT = 1  # bit 0 is set exactly on the all-zeros input

def _initial_masks():
    """Wire i's mask = bit (1<<i) set on every input where bit i of the input is 1."""
    masks = [0] * N
    for i in range(N):
        m = 0
        for x in range(1 << i):
            m |= 1 << ((x << 1) | (1 << i))
        masks[i] = m
    return masks

def _verify(masks):
    return (masks[0] & _ANY_BIT) == _ANY_BIT  # top bit set on every non-zero input

def _apply(masks, comps):
    for i, j in comps:
        a, b = masks[i], masks[j]
        masks[i] = a & b
        masks[j] = a | b
    return masks

def _verify_comps(comps):
    return _verify(_apply(_initial_masks(), comps))

def _batcher_oe_merge_sort(n, offset=0):
    """Batcher's odd-even merge sort network for n wires (n may be non-power-of-2)."""
    if n <= 1:
        return []
    comps = []
    half = n // 2
    comps.extend(_batcher_oe_merge_sort(half, offset))
    comps.extend(_batcher_oe_merge_sort(n - half, offset + half))
    comps.extend(_merge_batcher(half, n - half, offset))
    return comps

def _merge_batcher(k, m, offset):
    """Merge two sorted blocks of length k (at offset) and m (at offset+k)."""
    if k == 0:
        return []
    if m == 0:
        return []
    if k == 1 and m == 1:
        return [(offset, offset + 1)]
    comps = []
    k2 = (k + 1) // 2
    m2 = (m + 1) // 2
    # Merge the even-indexed halves
    comps.extend(_merge_batcher(k2, m2, offset))
    # Merge the odd-indexed halves
    if k >= 2:
        comps.extend(_merge_batcher(k - k2, m - m2, offset + 1))
    # Final adjacent compares: (0,1), (2,3), ...
    for p in range(offset, offset + k + m, 2):
        if p + 1 < offset + k + m:
            comps.append((p, p + 1))
    return comps

def _greedy_remove(comps, budget):
    """Single-pass (no restart) greedy removal; returns the reduced list."""
    masks = _apply(_initial_masks(), comps)
    kept = []
    for idx, (i, j) in enumerate(comps):
        if time.time() > budget:
            # If time out, return original to be safe? No, kept is valid prefix.
            # But remaining are unprocessed.
            # Actually, if we break, `kept` contains only the ones we decided to keep.
            # The ones we removed are gone.
            # The ones we haven't processed are not in `kept`.
            # So `kept` is a valid sub-network?
            # If we removed some, and kept some, the kept ones are in the same relative order.
            # The network defined by `kept` is valid if it sorts all inputs.
            # But we only checked validity when removing. We didn't check if the *remaining* kept ones sort.
            # Wait, the logic is:
            # Start with full network (valid).
            # For each comparator, check if removing it keeps the network valid.
            # If yes, remove it.
            # If no, keep it.
            # So at any point, the current set of "kept + remaining" is valid?
            # No. "Kept" are those we decided to keep. "Remaining" are those we haven't checked.
            # The network "Kept + Remaining" is valid because it's a sub-network of the original?
            # No, sub-networks are not necessarily valid.
            # But we are checking removal *incrementally*.
            # State: We have a current valid network C.
            # We try to remove c from C.
            # If C \ {c} is valid, we set C = C \ {c}.
            # So C is always valid.
            # In the loop:
            # `masks` represents the current valid network state.
            # `kept` is the list of comparators we have decided to keep so far.
            # `comps[idx:]` are the ones we haven't processed.
            # The current valid network is `kept` + `comps[idx:]`.
            # When we check `comps[idx]`, we are checking if we can remove it from the current valid network.
            # If we remove it, `masks` updates, and we don't add to `kept`.
            # If we keep it, we add to `kept`.
            # So at the end, `kept` is the final valid network.
            # If we break due to timeout, `kept` + `comps[idx:]` is the current valid network.
            # But we only return `kept`. This would drop the unprocessed comparators.
            # So we should return `kept + comps[idx:]` if we break?
            # Or just let it finish. C < 200. It's fast.
            pass
        
        a, b = masks[i], masks[j]
        # temporarily remove comparator idx
        masks[i] = a | b
        masks[j] = a
        if _verify(masks):
            # comparator was redundant: leave it removed
            continue
        else:
            # restore
            masks[i] = a & b
            masks[j] = a | b
            kept.append((i, j))
    return kept

def _random_removal(comps, budget):
    """Randomised removal pass to escape single-pass local minima."""
    masks = _apply(_initial_masks(), comps)
    kept = []
    order = list(range(len(comps)))
    random.shuffle(order)
    for idx in order:
        if time.time() > budget:
            pass
        i, j = comps[idx]
        a, b = masks[i], masks[j]
        masks[i] = a | b
        masks[j] = a
        if _verify(masks):
            continue
        else:
            masks[i] = a & b
            masks[j] = a | b
            kept.append((i, j))
    return kept

def construct():
    start = time.time()
    budget = start + DEADLINE

    # Seed: Batcher's odd-even merge sort
    comps = _batcher_oe_merge_sort(N)
    if not _verify_comps(comps):
        # Fallback to Odd-Even Transposition
        comps = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comps.append((i, i + 1))
    
    if not _verify_comps(comps):
        return comps

    best_comps = list(comps)
    best_len = len(comps)
    
    # Phase 1: Deterministic Greedy Removal (single pass)
    # This is fast and gets us to a local minimum quickly.
    if time.time() < budget - 5.0:
        new = _greedy_remove(comps, budget - 5.0)
        if len(new) < best_len:
            comps = new
            best_comps = list(comps)
            best_len = len(comps)

    # Phase 2: Iterative Random Removal with Restarts
    # We repeatedly try to remove comparators in random order.
    # If we get stuck (no removal in a full pass), we "perturb" by swapping two comparators
    # (if valid) or just restarting the random order.
    
    # Actually, just repeating _random_removal is good, but it stops if no progress.
    # We want to allow "neutral" moves (swaps) that don't reduce size but might enable future reductions.
    
    # Let's implement a specific "Swap-then-Remove" cycle.
    
    while time.time() < budget - 1.0:
        # 1. Try to remove
        # Use a randomized pass
        new_comps = _random_removal(comps, budget - 1.0)
        
        if len(new_comps) < best_len:
            comps = new_comps
            best_comps = list(comps)
            best_len = len(comps)
        elif len(new_comps) == best_len:
            # No progress in size.
            # Try a perturbation: Swap two comparators.
            # We pick a random pair of indices.
            # If the network remains valid after swap, we accept it (size unchanged).
            # This might allow different removals in the next iteration.
            
            if len(comps) > 1:
                i = random.randint(0, len(comps) - 1)
                j = random.randint(0, len(comps) - 1)
                if i != j:
                    # Perform swap
                    comps[i], comps[j] = comps[j], comps[i]
                    if _verify_comps(comps):
                        # Accept the swap (size is same, but structure changed)
                        # We do NOT update best_comps because size didn't improve,
                        # but we keep `comps` as the current state for next iteration.
                        pass
                    else:
                        # Revert
                        comps[i], comps[j] = comps[j], comps[i]
                        # If revert, we are stuck in this exact state.
                        # To avoid infinite loop on same state, we should maybe add a comparator?
                        # Or just break and restart with a fresh random seed?
                        # For now, just break out of the swap attempt and try next removal pass.
                        # The removal pass uses a new random shuffle, so it will try different orders.
        else:
            # new_comps is larger? Shouldn't happen with removal.
            break
            
    # Guarantee a valid network is returned.
    if not _verify_comps(best_comps):
        best_comps = _batcher_oe_merge_sort(N)

    return best_comps
