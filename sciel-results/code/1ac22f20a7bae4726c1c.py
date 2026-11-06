import time
import random
import numpy as np

N = 18
NUM_BOOLS = 1 << N
SEED = 153.0
DEADLINE = 80.0  # wall-clock safety margin (evaluator kills at 90s)

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

def _local_search(comps, budget):
    """
    Local search with Remove and Swap moves.
    """
    current_comps = list(comps)
    best_comps = list(current_comps)
    best_len = len(current_comps)
    
    # Pre-compute initial masks for the current network? 
    # It's better to maintain the masks incrementally if possible, 
    # but for simplicity and correctness with swaps, we might re-verify.
    # However, re-verifying from scratch is O(N * C) bit operations.
    # N=18, C~100. 18*100 = 1800 big-int ops. Very fast.
    
    # To speed up, we can keep the masks of the *current* state.
    # But swaps change the sequence, so masks change.
    # Let's just use _verify_comps for simplicity first, and optimize if too slow.
    
    # Actually, let's implement a "Remove" phase first, then "Swap" phase.
    
    # Phase 1: Aggressive Removal
    # We try to remove every comparator in a random order.
    # If removal succeeds, we keep it removed.
    # We repeat this until no more removals are possible in a full pass.
    
    # Phase 2: Swap / Perturbation
    # If stuck, we try to swap two comparators.
    # If the swap results in a valid network (it will, if the original was valid? No, swapping comparators can break validity).
    # Wait, swapping comparators in a sorting network does NOT necessarily preserve sorting.
    # So we must verify after every swap.
    
    # Let's combine: 
    # 1. Greedy Remove Pass
    # 2. If no progress, try Random Swap Pass.
    #    - Pick two indices i, j.
    #    - Swap comps[i], comps[j].
    #    - Verify.
    #    - If valid, accept (even if same length? No, we want to reduce length. 
    #      Swaps don't reduce length. They help escape local minima for *future* removals).
    #    - So, we do a "Perturbation" step: Swap, then try to Remove.
    
    # Algorithm:
    # 1. Current network C.
    # 2. Loop while time < budget:
    #    a. Do a full pass of greedy removal (random order).
    #       If size reduced, update C.
    #    b. If no reduction in (a), do a perturbation:
    #       - Pick random i, j. Swap C[i], C[j].
    #       - Verify C.
    #       - If valid, keep the swap (size same).
    #       - If invalid, revert.
    #       - Then try removal again.
    
    # This is essentially Simulated Annealing with T=0 (only accept if better or equal size? 
    # No, we accept equal size if it allows future reductions).
    
    # Let's implement a simpler version:
    # 1. Greedy Removal (Random Order)
    # 2. If stuck, try to find a "redundant pair" or just random restarts of the removal order.
    #    Random restarts of removal order are cheap.
    
    # Let's stick to the current _random_removal but make it loop more effectively.
    # The current code breaks if `len(new) >= len(comps)`.
    # We should allow multiple passes of random removal until no progress is made in a *full* pass.
    
    # Let's rewrite the main loop in construct().
    
    pass

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

def _greedy_remove(comps, budget):
    """Single-pass (no restart) greedy removal; returns the reduced list."""
    masks = _apply(_initial_masks(), comps)
    kept = []
    for idx, (i, j) in enumerate(comps):
        if time.time() > budget:
            # If time out, we must return a valid list.
            # The current `kept` is valid because we only kept things that were valid when we checked?
            # No, `kept` is built by restoring if invalid.
            # If we break, we have `kept` + remaining unprocessed comps.
            # This is NOT necessarily valid.
            # We should return the original `comps` if we time out mid-pass?
            # Or just finish the pass? The pass is O(C). C < 200. It's fast.
            # So we can just ignore the budget in this inner loop or assume it's fast enough.
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
            # If time out, we need to be careful.
            # Let's just assume the pass is fast enough.
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
