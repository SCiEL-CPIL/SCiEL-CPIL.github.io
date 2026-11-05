"""Search for a size-efficient sorting network on 17 wires using
zero-one-principle verification (all 2^17 Boolean inputs) and a
greedy redundant-comparator removal post-processor."""
import time
import numpy as np

N = 17
M = 1 << N  # 131072 Boolean inputs

# Precompute the Boolean input matrix: row r is the bit pattern of r.
# We store it as a uint16 array of shape (M, N), wire k = bit k of r.
# wire 0 is the least significant bit.
_INPUTS = None

def _get_inputs():
    global _INPUTS
    if _INPUTS is None:
        r = np.arange(M, dtype=np.uint32)
        # We need a uint16 array; N=17 so we need 17 bits.
        # Build by extracting each bit.
        arr = np.empty((M, N), dtype=np.uint16)
        for k in range(N):
            arr[:, k] = ((r >> k) & 1).astype(np.uint16)
        _INPUTS = arr
    return _INPUTS


def _apply_comparators(inputs, comps):
    """Apply a list of (i,j) comparators to the (M, N) uint16 array.
    Returns a new array (or modifies in place on a copy)."""
    a = inputs.copy()
    for (i, j) in comps:
        ai = a[:, i]
        aj = a[:, j]
        # min goes to i, max goes to j
        # min = ai & aj ; max = ai | aj  (for bits 0/1)
        a[:, i] = ai & aj
        a[:, j] = ai | aj
    return a


def _is_sorted(a):
    """Check if every row is non-increasing in wire index 0..N-1? 
    Wait: compare-exchange writes min to i and max to j with i<j.
    So wire 0 gets the smallest, wire N-1 gets the largest.
    Sorted means a[:,0] <= a[:,1] <= ... <= a[:,N-1] for every row.
    Since values are 0/1, sorted means once a 1 appears, all later are 1.
    Equivalently, for each row, the sequence is 0..0 1..1.
    """
    # Check that for each adjacent pair, a[:,k] <= a[:,k+1]
    # i.e. not (a[:,k]==1 and a[:,k+1]==0)
    for k in range(N - 1):
        bad = (a[:, k] == 1) & (a[:, k + 1] == 0)
        if np.any(bad):
            return False
    return True


def _verify(comps):
    """Return True if the network sorts all Boolean inputs."""
    a = _apply_comparators(_get_inputs(), comps)
    return _is_sorted(a)


def _seed_network():
    # Odd-even transposition: N phases.
    # phase p: compare (i, i+1) for i = p%2, p%2+2, ... < N-1
    comps = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    best = _seed_network()
    if not _verify(best):
        # Should never happen; odd-even transposition is correct.
        # Fallback: return as-is.
        return best

    best_len = len(best)

    # Greedy removal: repeatedly scan and remove any redundant comparator.
    # A comparator is redundant if removing it still yields a correct network.
    # To be efficient, we test removals in an order likely to find redundancies.
    # We'll do multiple passes; in each pass, try removing each comparator (in
    # a fixed order) and keep the removal if it stays correct.
    # Stop when a full pass removes nothing, or when we hit the deadline.

    cur = list(best)
    cur_len = len(cur)

    # We'll cache the result of _apply_comparators for the current network
    # to speed up the "is it sorted" check after a single removal? 
    # Actually, removing one comparator and re-checking requires re-running
    # the remaining comparators. That's O(cur_len * M * N) per test, which is
    # expensive. 
    # 
    # Optimization: Instead of re-running the whole network for each candidate
    # removal, we can note that removing comparator at position k means the
    # network is comps[0:k] + comps[k+1:]. 
    # We can precompute the state after comps[0:k] for all k? That's O(N * M * N)
    # memory if we store all intermediate states, which is 17 * 131072 * 17 * 2 bytes
    # = ~76 MB. That's feasible but let's think of a better way.
    #
    # Alternative: For each candidate removal, run the network with that comparator
    # skipped. The cost is O(cur_len) comparator applications, each O(M).
    # cur_len ~ 136, M = 131072. So one test is ~136 * 131072 * (few ops) ~ 18M ops.
    # In NumPy, each comparator is a few vectorized ops on (M,) arrays.
    # 136 comparators * ~5 vector ops * 131072 = ~89M element ops. That's ~0.1-0.3s.
    # Doing this for 136 candidates per pass = ~20-40s per pass. Too slow for many passes.
    #
    # Better approach: Precompute the "state before each comparator" and "state after
    # each comparator" for the current network. Then, to test removing comparator k,
    # we need to apply comparators 0..k-1 to get state S_{k-1}, then skip k, then
    # apply k+1..end. But the state after k+1..end depends on the input to k+1, which
    # is S_k (the state after applying k). If we skip k, the input to k+1 is S_{k-1}
    # instead of S_k. So we can't just reuse the suffix.
    #
    # However, we CAN precompute the prefix states. And for the suffix, we'd need to
    # recompute. 
    #
    # Let's try a different tactic: only test removals near the end first, where the
    # suffix is short. Or, test removals in random order and hope.
    #
    # Given the time budget, let's do a simpler approach: 
    # 1. Precompute prefix states for the current network.
    # 2. For each candidate k (from last to first, or in some order), 
    #    reconstruct the network without k and verify.
    #    To speed up, we can apply prefix 0..k-1 (already have state), then apply
    #    suffix k+1..end. The suffix length is cur_len - k - 1.
    #    If we test k from the end, the suffix is short.
    #
    # Let's precompute prefix states: state_before[k] = state after applying comps[0:k]
    # state_before[0] = inputs
    # state_before[k+1] = apply comps[k] to state_before[k]
    #
    # To test removing k: 
    #   state = state_before[k]  (after comps 0..k-1)
    #   for m in range(k+1, cur_len):
    #       apply comps[m] to state
    #   check sorted
    #
    # Cost per test: (cur_len - k - 1) comparator applications.
    # If we test k from cur_len-1 down to 0, the total cost is sum_{k=0}^{cur_len-1} (cur_len-k-1)
    # = cur_len*(cur_len-1)/2 comparator applications. For cur_len=136, that's ~9200
    # comparator applications. Each is ~5 vector ops on (M,) = 131072 elements.
    # 9200 * 5 * 131072 = ~6e9 element ops. That's ~6-20 seconds in NumPy. Acceptable
    # for one pass. We can do a few passes.

    # Precompute prefix states for current network
    def _prefix_states(comps):
        states = [None] * (len(comps) + 1)
        states[0] = _get_inputs()
        for k in range(len(comps)):
            states[k + 1] = _apply_comparators(states[k], [comps[k]])
        return states

    # In each pass, try to remove comparators. We'll do it in one pass, testing
    # from the last comparator to the first, and removing any that are redundant.
    # After a removal, the prefix states change, so we need to recompute.
    # To keep it simple, we'll do: in each outer iteration, recompute prefix states,
    # then scan from last to first, and if a removal is valid, remove it and restart
    # the scan (recompute prefix states). This is O(cur_len) recomputations per
    # removal, each O(cur_len * M). Total per removal: O(cur_len^2 * M). 
    # With ~136 removals possible, that's 136 * 136^2 * 131072 ≈ 3.3e11. Too slow.
    #
    # Better: In one pass, scan from last to first, and for each k, test if removing
    # k is valid *given the current network*. If valid, remove it and continue with
    # the next k (now shifted). But the prefix states are stale after a removal.
    #
    # Compromise: Do a pass where we test each k using the prefix states computed
    # at the start of the pass. If multiple removals are found, apply them all at
    # the end (if they're independent). But removals aren't independent.
    #
    # Let's just do: repeatedly, compute prefix states, then find the last removable
    # comparator, remove it, and repeat. Each iteration costs O(cur_len) for prefix
    # states + O(cur_len^2/2) for scanning (worst case). But in practice, we'll find
    # a removal quickly if we scan from the end.
    #
    # Actually, let's scan from the end and test each k. The first removable k from
    # the end will have a short suffix. Once we find one, remove it and restart.
    # This is O(cur_len) prefix computation + O(cur_len) scanning (on average) per
    # removal. Total: O(num_removals * cur_len^2 * M). With num_removals ~ 50,
    # cur_len ~ 100, M = 131072: 50 * 10000 * 131072 = 6.5e10. Still ~10-30s.
    #
    # Let's just do it and see if it fits in the budget.

    while time.time() < deadline:
        cur_len = len(cur)
        if cur_len <= best_len:
            break
        # Compute prefix states
        states = _prefix_states(cur)
        # Scan from last to first
        removed_any = False
        for k in range(cur_len - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
            # Test removing k
            # State before k: states[k]
            # Apply comps k+1..end
            state = states[k].copy()
            for m in range(k + 1, cur_len):
                i, j = cur[m]
                ai = state[:, i]
                aj = state[:, j]
                state[:, i] = ai & aj
                state[:, j] = ai | aj
            if _is_sorted(state):
                # Remove comparator k
                cur = cur[:k] + cur[k + 1:]
                removed_any = True
                break  # Restart scan with new network
        if not removed_any:
            # No more removals possible
            break

    if len(cur) < best_len:
        best = cur
        best_len = len(cur)

    return best
