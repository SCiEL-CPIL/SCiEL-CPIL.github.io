import time
import numpy as np

N = 16
ALL_INPUTS = 1 << N  # 65536
TIME_LIMIT = 80.0    # safety margin under 90s kill
_start = time.time()

# Precompute all 2^16 boolean inputs as a (65536, 16) uint8 matrix
# Row r corresponds to integer r, bit i is (r >> (N-1-i)) & 1
# This matches the standard interpretation where wire 0 is MSB or LSB?
# The task says: "compare-exchange writes min to wire i and max to j".
# The order of wires is fixed.
# Let's use: wire i corresponds to bit (N-1-i) of the input mask.
# So wire 0 is the most significant bit? Or least?
# Usually, index 0 is the first element.
# If mask = 1 (binary ...0001), and N=16.
# If wire 0 is MSB: mask=1 -> wire 15 is 1, others 0.
# If wire 0 is LSB: mask=1 -> wire 0 is 1, others 0.
# The zero-one principle holds regardless of bit ordering as long as it's consistent.
# Let's stick to: wire i gets bit (mask >> i) & 1. So wire 0 is LSB.
# This is a permutation of the input space, so validity is invariant.

bits = np.arange(ALL_INPUTS, dtype=np.uint32)[:, None]
shifts = np.arange(N, dtype=np.uint32)[None, :]  # 0 to 15
INPUTS = ((bits >> shifts) & 1).astype(np.uint8)


def _elapsed():
    return time.time() - _start


def _apply_comparators(inputs, comps):
    """Apply a list of (i,j) comparators to the input matrix.
    Returns the matrix after all comparators."""
    x = inputs.copy()
    for i, j in comps:
        a = x[:, i]
        b = x[:, j]
        # min to i, max to j
        mn = np.minimum(a, b)
        mx = np.maximum(a, b)
        x[:, i] = mn
        x[:, j] = mx
    return x


def _is_sorted(x):
    """True if every row is non-decreasing (sorted ascending)."""
    return bool(np.all(x[:, :-1] <= x[:, 1:]))


def _verify(comps):
    """Return True if the comparator list sorts all 65536 Boolean inputs."""
    x = _apply_comparators(INPUTS, comps)
    return _is_sorted(x)


def _baseline_network():
    """Odd-even transposition network: 120 comparators."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _all_pairs():
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


def construct():
    best = _baseline_network()
    best_len = len(best)
    
    # Sanity check baseline
    if not _verify(best):
        # This should never happen, but if it does, we are in trouble.
        # Return a known valid network if possible, but baseline is standard.
        pass

    pairs = _all_pairs()
    n_pairs = len(pairs)

    # 1. Greedy Deletion
    # Start from baseline, try to remove each comparator.
    cur = list(best)
    changed = True
    while changed:
        changed = False
        for idx in range(len(cur) - 1, -1, -1):
            if _elapsed() > TIME_LIMIT:
                break
            cand = cur[:idx] + cur[idx + 1:]
            if _verify(cand):
                cur = cand
                changed = True
        if _elapsed() > TIME_LIMIT:
            break

    if len(cur) < best_len:
        best = cur
        best_len = len(cur)

    # 2. Local Search / Swap & Delete
    # Try to swap two comparators and then delete one, or just delete.
    # We use a random restart approach within the time budget.
    import random
    rng = random.Random(42)
    
    # We will perform a series of local moves.
    # Move types:
    # a) Delete a comparator if valid.
    # b) Swap two comparators if valid (no size change, but might enable future deletions).
    # c) Replace one comparator with another if valid (size change? No, usually we want to reduce size).
    # Actually, the most effective move is "Swap and Delete".
    # Pick i, j (indices in network), swap them, then pick k and delete.
    # If valid, accept.
    
    cur = list(best)
    
    # Optimization: Cache verification results? 
    # The state space is large, but we only verify specific candidates.
    # _verify is O(N * 2^N * N) roughly? No, _apply is O(C * 2^N).
    # C ~ 100, 2^N = 65536. 100 * 65536 = 6.5M ops.
    # NumPy vectorization makes this very fast. ~1ms per verify?
    # Let's check. 65536 * 16 bytes = 1MB. 
    # 100 comparators * 16 wires * 65536 inputs.
    # It's fast enough.
    
    def try_local_search(start_net, budget=2.0):
        nonlocal best, best_len
        cur_net = list(start_net)
        start_time = time.time()
        
        while time.time() - start_time < budget and _elapsed() < TIME_LIMIT:
            n = len(cur_net)
            if n == 0: break
            
            # Try random deletion
            if rng.random() < 0.5:
                idx = rng.randrange(n)
                cand = cur_net[:idx] + cur_net[idx+1:]
                if _verify(cand):
                    cur_net = cand
                    if len(cur_net) < best_len:
                        best_len = len(cur_net)
                        best = list(cur_net)
            
            # Try swap + delete
            else:
                if n < 2: continue
                i = rng.randrange(n)
                j = rng.randrange(n)
                if i == j: continue
                cand = list(cur_net)
                cand[i], cand[j] = cand[j], cand[i]
                
                # Try deleting a random element from the swapped list
                # To be more efficient, we can just check if the swap itself is valid?
                # No, swap keeps size same. We need to delete to improve score.
                # So we must delete one.
                k = rng.randrange(len(cand))
                cand2 = cand[:k] + cand[k+1:]
                if _verify(cand2):
                    cur_net = cand2
                    if len(cur_net) < best_len:
                        best_len = len(cur_net)
                        best = list(cur_net)
                    
                # Occasionally accept valid swaps (same size) to explore landscape
                elif rng.random() < 0.1:
                    if _verify(cand):
                        cur_net = cand

    # Run local search for a while
    # We can run multiple restarts or just one long one.
    # Given the time limit, one long run from the best found so far is good.
    # But random restarts might escape local minima.
    # Let's do a few short runs.
    
    remaining_time = TIME_LIMIT - _elapsed()
    if remaining_time > 1.0:
        # Split time into chunks
        num_chunks = 5
        chunk_time = remaining_time / num_chunks
        for _ in range(num_chunks):
            if _elapsed() > TIME_LIMIT: break
            # Restart from current best or a perturbed version?
            # Restarting from best is safe.
            # Perturbing might help.
            start_net = list(best)
            # Add a few random comparators to perturb?
            # Or just use best.
            try_local_search(start_net, budget=chunk_time)
            
    # Final polish: pure deletion pass on best
    cur = list(best)
    changed = True
    while changed and _elapsed() < TIME_LIMIT:
        changed = False
        for idx in range(len(cur) - 1, -1, -1):
            if _elapsed() > TIME_LIMIT: break
            cand = cur[:idx] + cur[idx + 1:]
            if _verify(cand):
                cur = cand
                changed = True
                if len(cur) < best_len:
                    best_len = len(cur)
                    best = list(cur)
                break

    return best
