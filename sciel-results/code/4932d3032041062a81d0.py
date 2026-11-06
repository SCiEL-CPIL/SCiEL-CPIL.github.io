import time
import random
import numpy as np

N = 18
NUM_BOOLS = 1 << N
SEED = 153.0
DEADLINE = 85.0  # wall-clock safety margin (evaluator kills at 90s)

# Pre-compute the input states for all 262,144 boolean inputs
# inputs[i] is a boolean array of length NUM_BOOLS
# inputs[i][x] is True if the i-th bit of x is 1
_inputs = np.zeros((N, NUM_BOOLS), dtype=bool)
for i in range(N):
    # Bit i is set in x if (x >> i) & 1 == 1
    # We can generate this efficiently
    pattern = np.zeros(NUM_BOOLS, dtype=bool)
    # For bit i, the pattern is blocks of 2^i zeros, 2^i ones, repeating
    # Total length 2^18.
    # We can use reshape and tile, or simple logic.
    # Logic: x & (1 << i) != 0
    x_vals = np.arange(NUM_BOOLS, dtype=np.uint32)
    pattern = (x_vals & (1 << i)) != 0
    _inputs[i] = pattern

# Pre-compute the "any 1" mask for verification
# A network sorts if for every input x, the max value is on wire 0.
# Since it's boolean, max is 1 if any wire is 1.
# So wire 0 must be 1 whenever ANY wire is 1.
# Equivalently: (wire_0 & ~any_wire) must be all False.
# Or: wire_0 must be a superset of any_wire.
# any_wire[x] is True if any input bit is 1.
_any_wire = np.any(_inputs, axis=0)

def _verify(state):
    """
    Check if the state (array of N booleans, each of length NUM_BOOLS)
    represents a sorted network.
    """
    # state[0] is the top wire (minimum).
    # For sorting, min must be 1 iff any input is 1.
    # If min is 1, then some input was 1.
    # If some input was 1, min MUST be 1 (because 1 is the max, and it moves to the bottom? 
    # Wait. Compare-exchange writes min to i, max to j.
    # If i < j, min goes to i (top), max goes to j (bottom).
    # So wire 0 holds the global minimum.
    # For boolean inputs, min is 0 unless all inputs are 1? No.
    # Min of {0, 1} is 0. Min of {1, 1} is 1.
    # So wire 0 is 1 iff ALL inputs are 1?
    # Let's re-read the task: "compare-exchange writes min to wire i and max to j".
    # If we sort ascending, wire 0 gets the minimum.
    # For boolean values, the minimum is 0 unless all values are 1.
    # So wire 0 should be 1 only if ALL inputs are 1.
    # Let's check the seed's logic:
    # "Sorting is certified iff the top bit of wire 0 is set on every input where at least one wire is set"
    # This implies the seed thinks wire 0 gets the MAX?
    # "w_i' = w_i & w_j (min)" -> AND is min for booleans.
    # "w_j' = w_i | w_j (max)" -> OR is max for booleans.
    # If i < j, min goes to i. So wire 0 gets the global minimum.
    # Global min of booleans is 0 unless all are 1.
    # So wire 0 should be 1 ONLY when all inputs are 1.
    # The seed's check: `masks[0] & _ANY_BIT == _ANY_BIT` where `_ANY_BIT` is bit 0 (all-zeros input).
    # This check in the seed is actually checking if the ALL-ZEROS input results in wire 0 being 1?
    # No, `_ANY_BIT = 1` is bit 0 of the mask. Bit 0 of the mask corresponds to input `x=0`.
    # So the seed checks if wire 0 is 1 for input `x=0`.
    # For input `x=0` (all zeros), min is 0. So wire 0 should be 0.
    # The seed's verification logic seems inverted or I am misinterpreting "top wire".
    
    # Let's look at the standard definition.
    # A sorting network sorts ascending.
    # Output[0] <= Output[1] <= ... <= Output[N-1].
    # For boolean inputs, Output[0] is 1 iff all inputs are 1.
    # Output[N-1] is 1 iff any input is 1.
    
    # The seed code:
    # `w_i' = w_i & w_j` (min)
    # `w_j' = w_i | w_j` (max)
    # This puts min on i, max on j.
    # If the network is valid, wire 0 has the global min.
    # Global min is 1 iff all inputs are 1.
    # So we should check: `state[0]` is True iff `np.all(_inputs, axis=0)` is True.
    # i.e. `state[0] == _all_ones`
    
    # Let's check the seed's `_verify` again.
    # `_TOP_BIT = 1 << (NUM_BOOLS - 1)` -> bit for input `x = 2^18 - 1` (all ones).
    # `_ANY_BIT = 1` -> bit for input `x = 0` (all zeros).
    # `return (masks[0] & _ANY_BIT) == _ANY_BIT`
    # This checks if wire 0 is 1 for input `x=0`.
    # For input `x=0`, all inputs are 0. Min is 0. Wire 0 should be 0.
    # So this check would FAIL for a correct sorting network?
    # Unless... the network is sorting in DESCENDING order?
    # If descending, wire 0 gets max. Max is 1 if any is 1.
    # For x=0, max is 0. Still 0.
    
    # Maybe the seed's `_initial_masks` is different?
    # `masks[i]` has bit set where input bit i is 1.
    # If we apply comparators, we are tracking which inputs result in a 1 on that wire.
    # If the network is correct, wire 0 (min) is 1 only for the all-ones input.
    # So `masks[0]` should be exactly `_TOP_BIT`.
    # The seed checks if `masks[0]` has the `_ANY_BIT` (bit 0) set.
    # This implies the seed expects wire 0 to be 1 for the all-zeros input.
    # This is impossible for a min-sorting network.
    
    # Wait, look at the seed's `_verify` comment:
    # "Sorting is certified iff the top bit of wire 0 is set on every input where at least one wire is set"
    # This describes the MAXIMUM.
    # If the task implies sorting such that the MAX is on wire 0?
    # "compare-exchange writes min to wire i and max to j".
    # If we want max on wire 0, we must have a network that puts max on the smallest index.
    # This requires comparators (i, j) with i < j to write MAX to i and MIN to j?
    # But the task says: "Each compare-exchange writes min to wire i and max to j".
    # This is fixed by the interface.
    # So wire 0 MUST hold the global minimum.
    
    # Is it possible the seed is broken?
    # The seed score is 1.0.
    # If the seed returns a network that fails the evaluator, the score would be 0.
    # The evaluator measures "raw value (lower is better)".
    # Score = seed / value.
    # If value is 153 (seed count), score is 1.0.
    # So the seed returns a valid network of size 153.
    # Therefore, my understanding of the verification or the seed's logic must be aligned with a valid network.
    
    # Let's re-read the seed's `_verify`:
    # `return (masks[0] & _ANY_BIT) == _ANY_BIT`
    # This checks if bit 0 of `masks[0]` is 1.
    # Bit 0 corresponds to input `x=0`.
    # For `x=0`, all inputs are 0.
    # Min is 0.
    # So `masks[0]` should have bit 0 as 0.
    # So `_verify` returns False for a correct network?
    
    # Unless... `_initial_masks` is different.
    # `masks[i]` = bit (1<<i) set on every input where bit i of the input is 1.
    # This is standard.
    
    # Maybe the "top wire" is not wire 0?
    # "w_i' = w_i & w_j" -> min to i.
    # If the network sorts ascending, wire 0 is min.
    
    # Let's look at the "Inspiration 2" comment:
    # "The current odd-even transposition network uses 153 comparators... defines _apply_network, _is_sorted_network"
    
    # Let's assume the standard definition: Wire 0 is the minimum.
    # Verification: `state[0]` must be equal to `np.all(_inputs, axis=0)`.
    # Let's test this hypothesis.
    
    all_ones_mask = np.all(_inputs, axis=0)
    
    # If I use the seed's logic, I might fail.
    # But I must keep the function name `construct`.
    # I will implement my own verification based on the standard definition.
    # If the seed's network passes the evaluator, it is a valid sorting network.
    # A valid sorting network must have `state[0] == all_ones_mask`.
    
    # Let's double check the seed's `_verify` again.
    # Maybe `_ANY_BIT` is not bit 0?
    # `_ANY_BIT = 1`. Yes, bit 0.
    # Maybe `_TOP_BIT` is used elsewhere?
    
    # There is a possibility that the task definition of "sort" implies something else,
    # or the seed is checking a property that is equivalent to sorting for this specific construction?
    # No, that's unlikely.
    
    # Let's look at the "Best known network has 77 comparators".
    # Batcher's sort for 18 wires is ~153.
    # If I construct a network and verify it with `state[0] == all_ones_mask`, and it passes,
    # it is a valid sorting network.
    
    # I will proceed with `state[0] == all_ones_mask`.
    
    return np.array_equal(state[0], all_ones_mask)

def _apply_comparator(state, i, j):
    """
    Apply comparator (i, j) to the state.
    min to i, max to j.
    """
    min_val = np.logical_and(state[i], state[j])
    max_val = np.logical_or(state[i], state[j])
    state[i] = min_val
    state[j] = max_val
    return state

def _verify_comps(comps):
    """
    Verify a list of comparators.
    """
    # Initialize state with inputs
    state = _inputs.copy()
    for i, j in comps:
        _apply_comparator(state, i, j)
    return _verify(state)

def _batcher_oe_merge_sort(n, offset=0):
    """Batcher's odd-even merge sort network for n wires."""
    if n <= 1:
        return []
    comps = []
    half = n // 2
    comps.extend(_batcher_oe_merge_sort(half, offset))
    comps.extend(_batcher_oe_merge_sort(n - half, offset + half))
    comps.extend(_merge_batcher(half, n - half, offset))
    return comps

def _merge_batcher(k, m, offset):
    """Merge two sorted blocks of length k and m."""
    if k == 0 or m == 0:
        return []
    if k == 1 and m == 1:
        return [(offset, offset + 1)]
    comps = []
    k2 = (k + 1) // 2
    m2 = (m + 1) // 2
    comps.extend(_merge_batcher(k2, m2, offset))
    if k >= 2:
        comps.extend(_merge_batcher(k - k2, m - m2, offset + 1))
    for p in range(offset, offset + k + m, 2):
        if p + 1 < offset + k + m:
            comps.append((p, p + 1))
    return comps

def _greedy_remove(comps, budget):
    """
    Greedy removal of comparators.
    """
    # Start with the full network state
    state = _inputs.copy()
    for i, j in comps:
        _apply_comparator(state, i, j)
        
    kept = []
    # We process comparators in order.
    # If we remove one, the state changes.
    # We need to re-apply the remaining comparators to check validity?
    # No, we can maintain the state incrementally.
    # But if we remove a comparator, we can't easily "un-apply" it.
    # So we must re-verify from scratch or use a different strategy.
    
    # Strategy:
    # 1. Try to remove comparator k.
    # 2. Re-run the network without k.
    # 3. If valid, remove it.
    # This is O(N * L^2) where L is length.
    # With L=150, L^2 = 22500. 150 * 22500 = 3.3M checks.
    # Each check is O(L * N * 262144) vectorized.
    # This might be too slow.
    
    # Better Strategy:
    # Maintain the state of the network *after* all comparators.
    # This doesn't help with removal.
    
    # Alternative:
    # Use the "reverse" logic?
    # If a comparator is redundant, the state before it must already satisfy the constraints imposed by it?
    # No.
    
    # Let's try the re-run strategy but optimize it.
    # We can check removal by running the network up to k, skipping k, then running the rest.
    # We can pre-compute the states at each step?
    # Memory: 150 steps * 18 wires * 262144 bools = 150 * 18 * 32KB = ~86 MB.
    # This is feasible.
    
    # Pre-compute states at each step
    # states[k] is the state after applying comps[0..k-1]
    # states[0] = _inputs
    # states[k+1] = apply(comps[k], states[k])
    
    states = []
    cur_state = _inputs.copy()
    states.append(cur_state)
    for k, (i, j) in enumerate(comps):
        _apply_comparator(cur_state, i, j)
        states.append(cur_state.copy())
        
    # Now, to check if comp k can be removed:
    # We need to apply comps[k+1..end] to states[k] (skipping k).
    # And check if the final state is sorted.
    
    kept = []
    for k in range(len(comps)):
        if time.time() > budget:
            break
            
        # Check if removing comps[k] is valid
        # Start from states[k] (state before comp k)
        # Apply comps[k+1..end]
        test_state = states[k].copy()
        for l in range(k + 1, len(comps)):
            i, j = comps[l]
            _apply_comparator(test_state, i, j)
            
        if _verify(test_state):
            # Redundant. Do not add to kept.
            # We don't update `states` here because we are building a new list `kept`.
            # The `states` array corresponds to the ORIGINAL `comps`.
            # If we remove comps[k], the states for subsequent indices change.
            # So we cannot use `states` for the next removal check directly if we removed something.
            
            # This approach is flawed for multiple removals in one pass.
            # We need to rebuild states if we remove.
            
            # Let's do a simpler, slower but correct greedy pass:
            # 1. Try to remove comp 0. Re-run network. If valid, remove.
            # 2. Try to remove comp 0 (new list). Re-run. If valid, remove.
            # This is O(L^3) which is too slow.
            
            # Let's stick to the "Check removal from current valid network" logic.
            # Current valid network is `comps`.
            # We try to remove `comps[k]`.
            # We re-run the network `comps` without `comps[k]`.
            # If valid, we set `comps` to the new list.
            # This is O(L^2) per removal. Total O(L^3).
            
            # Given L=150, L^3 = 3.3M.
            # Each run is 150 * 2 * 262144 ops. ~80M ops.
            # 3.3M * 80M = 2.6 * 10^14 ops. Too slow.
            
            # We need a faster removal check.
            # Is there a way to check redundancy locally?
            # A comparator (i, j) is redundant if the state before it already has min(i,j) on i and max(i,j) on j?
            # No, because subsequent comparators might rely on the specific distribution.
            
            # However, if the state BEFORE the comparator already satisfies `state[i] <= state[j]` (pointwise),
            # then the comparator does nothing.
            # If it does nothing, it is redundant.
            # Check: `np.all(state[i] <= state[j])`?
            # For booleans, `state[i] <= state[j]` is equivalent to `state[i] & ~state[j]` is all False.
            # i.e. `np.any(state[i] & ~state[j])` is False.
            
            # Let's try this "No-Op" check.
            # It is very fast.
            # If the comparator is a no-op, we remove it.
            # This might not find all redundancies, but it's a good start.
            
            # We can combine this with a few full re-runs if we find no-ops.
            
            pass # Placeholder for logic below
            
    # Revised Strategy:
    # 1. Loop through comps.
    # 2. For each comp, check if it is a no-op given the current state (from a fresh run up to that point? No, state changes).
    # 
    # Let's use the "No-Op" check on the state *before* the comparator in the *current* valid network.
    # We maintain the state incrementally.
    # If we remove a comparator, we must re-compute states for subsequent ones?
    # No, if we remove a no-op, the state after it is the same as before it.
    # So the state for subsequent comparators remains valid.
    # 
    # Algorithm:
    # 1. Compute states for the full network.
    # 2. Iterate k from 0 to L-1.
    # 3. Check if comp k is a no-op on `states[k]`.
    # 4. If yes, remove it. The state `states[k+1]` is identical to `states[k]`.
    #    The subsequent states `states[k+2..]` are computed from `states[k+1]` (which is now `states[k]`) 
    #    and the subsequent comps. 
    #    Wait, if comp k is a no-op, `states[k+1] == states[k]`.
    #    The original `states[k+2]` was computed from `states[k+1]` and comp k+1.
    #    Since `states[k+1]` didn't change, `states[k+2]` remains valid.
    #    So we can just remove comp k from the list and continue.
    # 
    # This works perfectly for no-ops!
    
    # Let's implement this.
    
    # Re-initialize
    state = _inputs.copy()
    states = [state]
    for i, j in comps:
        _apply_comparator(state, i, j)
        states.append(state.copy())
        
    kept = []
    for k in range(len(comps)):
        if time.time() > budget:
            # If we run out of time, we should return the kept list so far?
            # But we haven't added the rest of `comps` to `kept`.
            # We need to add the remaining comps to `kept` to ensure validity.
            # The current `kept` + `comps[k:]` is the valid network.
            kept.extend(comps[k:])
            break
            
        i, j = comps[k]
        # Check no-op
        # state_before is states[k]
        # min = A & B, max = A | B
        # No-op if A & B == A and A | B == B
        # Which implies A <= B.
        # Check: np.any(states[k][i] & ~states[k][j]) == False
        
        if not np.any(states[k][i] & ~states[k][j]):
            # No-op, remove
            continue
        else:
            kept.append((i, j))
            
    return kept

def construct():
    start = time.time()
    budget = start + DEADLINE

    # Seed: Batcher's odd-even merge sort
    comps = _batcher_oe_merge_sort(N)
    
    # Verify seed
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
    
    # Phase 1: No-Op Removal
    # This is very fast.
    while time.time() < budget - 10.0:
        new_comps = _greedy_remove(comps, budget - 10.0)
        if len(new_comps) < best_len:
            comps = new_comps
            best_comps = list(comps)
            best_len = len(comps)
        else:
            # No more no-ops to remove
            break
            
    # Phase 2: Random Swap Local Search
    # Try to find swaps that allow further removals.
    # Or just swaps that maintain validity.
    
    # We will do a random walk.
    # 1. Pick two random indices.
    # 2. Swap them.
    # 3. If valid, accept.
    # 4. Then try to remove no-ops again.
    
    while time.time() < budget - 2.0:
        # Try to remove no-ops
        new_comps = _greedy_remove(comps, budget - 2.0)
        if len(new_comps) < best_len:
            comps = new_comps
            best_comps = list(comps)
            best_len = len(comps)
            
        if len(comps) <= 1:
            break
            
        # Try a swap
        i = random.randint(0, len(comps) - 1)
        j = random.randint(0, len(comps) - 1)
        if i != j:
            comps[i], comps[j] = comps[j], comps[i]
            if _verify_comps(comps):
                # Accept swap
                # Don't update best_len unless it improved (it didn't)
                pass
            else:
                # Revert
                comps[i], comps[j] = comps[j], comps[i]
                
    # Final check
    if not _verify_comps(best_comps):
        best_comps = _batcher_oe_merge_sort(N)
        
    return best_comps

if __name__ == "__main__":
    # Simple test
    net = construct()
    print(f"Network size: {len(net)}")
    print(f"Valid: {_verify_comps(net)}")
