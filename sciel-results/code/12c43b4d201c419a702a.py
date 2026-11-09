"""Search for a size-efficient 16-wire sorting network (fewer than 120 comparators)."""
import time
import numpy as np
import random

N = 16
ALL_INPUTS = 1 << N  # 65536
TIME_LIMIT = 55.0    # safety margin under 60s kill
_start = time.time()

# Precompute the (65536, 16) uint8 matrix of all Boolean inputs once.
_bits = np.arange(ALL_INPUTS, dtype=np.uint32)[:, None]
_shifts = (np.arange(N - 1, -1, -1, dtype=np.uint32))[None, :]
_INPUT_MATRIX = ((_bits >> _shifts) & 1).astype(np.uint8)

# Precompute a random subset for fast verification (e.g., 1024 samples)
# This is statistically robust for catching errors in small networks.
_SAMPLE_SIZE = 1024
_rng_seed = 42
_np_rng = np.random.RandomState(_rng_seed)
_sample_indices = _np_rng.choice(ALL_INPUTS, _SAMPLE_SIZE, replace=False)
_SAMPLE_MATRIX = _INPUT_MATRIX[_sample_indices]


def _elapsed():
    return time.time() - _start


def _apply_comparators(inputs, comps):
    """Apply a list of (i,j) comparators to a (M, 16) uint8 matrix.
    Returns the matrix after all comparators (min->i, max->j)."""
    x = inputs.copy()
    for i, j in comps:
        a = x[:, i]
        b = x[:, j]
        mn = np.minimum(a, b)
        mx = np.maximum(a, b)
        x[:, i] = mn
        x[:, j] = mx
    return x


def _is_sorted(x):
    """True if every row is non-decreasing (sorted)."""
    return bool(np.all(x[:, :-1] <= x[:, 1:]))


def _verify_fast(comps):
    """Fast verification using a subset of inputs."""
    x = _apply_comparators(_SAMPLE_MATRIX, comps)
    return _is_sorted(x)


def _verify_full(comps):
    """Full verification using all 65536 inputs."""
    x = _apply_comparators(_INPUT_MATRIX, comps)
    return _is_sorted(x)


def _batcher_network(n):
    """Generate Batcher's odd-even merge sort network for n wires.
    Returns list of (i, j) comparators."""
    
    def _merge_indices(a, b):
        """Odd-even merge two sorted lists of wire indices.
        Returns list of comparators (i, j) where i < j is not guaranteed, 
        but min->i, max->j logic is applied later? 
        Actually, standard comparator (i, j) assumes i < j in the interface 
        "0 <= i < j < N". 
        Wait, the interface says: "Each compare-exchange writes min to wire i and max to j."
        And "0 <= i < j < N".
        So the comparator MUST be (smaller_index, larger_index).
        
        In Batcher's OES, we compare elements at positions p and q.
        If p < q, we add (p, q).
        If p > q, we add (q, p).
        """
        if len(a) == 1 and len(b) == 1:
            p, q = a[0], b[0]
            if p < q:
                return [(p, q)]
            else:
                return [(q, p)]
        
        # Split into even and odd indexed elements
        a_even = a[0::2]
        a_odd = a[1::2]
        b_even = b[0::2]
        b_odd = b[1::2]
        
        comps = []
        comps.extend(_merge_indices(a_even, b_even))
        comps.extend(_merge_indices(a_odd, b_odd))
        
        # Now merge the results of the even and odd merges?
        # No, Batcher's OES works like this:
        # 1. Merge even-indexed elements of A with even-indexed elements of B.
        # 2. Merge odd-indexed elements of A with odd-indexed elements of B.
        # 3. Interleave the results and compare adjacent elements.
        
        # Let's re-implement this correctly.
        # The recursive definition of Odd-Even Merge Sort (OES):
        # OES(A, B) where A and B are sorted lists of length 2^k.
        # If 2^k == 1: compare A[0] and B[0].
        # Else:
        #   1. OES(A_even, B_even) -> M_even (sorted list of length 2^(k-1))
        #   2. OES(A_odd, B_odd) -> M_odd (sorted list of length 2^(k-1))
        #   3. For i in 0 to 2^(k-1)-1:
        #      Compare M_odd[i] and M_even[i+1] (if i < 2^(k-1)-1)
        #      Wait, the final step is to merge M_even and M_odd by comparing
        #      adjacent elements in the interleaved sequence.
        #      The interleaved sequence is: M_even[0], M_odd[0], M_even[1], M_odd[1], ...
        #      We compare (M_even[i], M_odd[i])? No.
        #      
        # Let's look up the exact comparator pattern for OES.
        # From literature:
        # To merge two sorted sequences of length n (n power of 2):
        # 1. Split into even and odd.
        # 2. Recursively merge evens and odds.
        # 3. Compare adjacent elements in the resulting sequence where the indices differ by 1.
        #    Specifically, if the merged even sequence is E and merged odd is O,
        #    the final sequence is E[0], O[0], E[1], O[1], ...
        #    We compare (E[i], O[i])? No, that's not right.
        #    We compare (O[i], E[i+1])?
        
        # Let's use a simpler, known-correct implementation of Batcher's OES.
        # Reference: "Sorting Networks" by Knuth or similar.
        # For n=16, we can just hardcode the structure or use a reliable generator.
        
        # Alternative: Use Batcher's Bitonic Sort which is easier to verify?
        # Bitonic Sort for N=16 uses 32 comparators? 
        # No, Bitonic Sort uses (N/2) * log2(N) comparators.
        # For N=16: 8 * 4 = 32. 
        # Is 32 enough to sort 16 elements? 
        # The minimum number of comparators to sort 16 elements is 53 (Batcher OES) or 60 (Optimal).
        # 32 is definitely not enough. My previous reasoning was wrong.
        # Bitonic sort depth is log2(N), but size is N/2 * log2(N) * (log2(N)+1)/2?
        # Let's check: Bitonic sort size for N=2^k is (N/2) * k * (k+1) / 2?
        # For N=16 (k=4): 8 * 4 * 5 / 2 = 80 comparators.
        # So Bitonic Sort uses 80 comparators.
        # Batcher OES uses 53? Or 72?
        # Actually, Batcher OES for N=16 is known to use 72 comparators?
        # Let's check: 
        # N=2: 1
        # N=4: 3 (OES) vs 5 (Bitonic)
        # N=8: 19 (OES) vs 28 (Bitonic)
        # N=16: 72 (OES) vs 80 (Bitonic)?
        # Wait, 19 for N=8 seems high. 
        # Let's calculate OES size recursively.
        # C(n) = 2*C(n/2) + n/2 - 1?
        # C(2) = 1
        # C(4) = 2*1 + 2 - 1 = 3
        # C(8) = 2*3 + 4 - 1 = 9? No, that's too low.
        
        # Let's stick to a robust method: 
        # 1. Start with Odd-Even Transposition (120 comps).
        # 2. Use Fast Verification to aggressively delete redundant comparators.
        # This is simpler and less prone to implementation errors.
        
        return []

    # Since implementing OES correctly is error-prone, we will rely on 
    # the baseline + fast greedy deletion strategy.
    return _baseline_network()


def _baseline_network():
    """Odd-even transposition network: 120 comparators."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def construct():
    best = _baseline_network()
    best_len = len(best)
    
    # Verify baseline with full check (once)
    if not _verify_full(best):
        return best

    # Phase 1: Fast Greedy Deletion
    # Use _verify_fast for speed.
    cur = list(best)
    changed = True
    while changed and _elapsed() < TIME_LIMIT * 0.5: # Use first 50% of time for greedy
        changed = False
        for idx in range(len(cur) - 1, -1, -1):
            if _elapsed() > TIME_LIMIT * 0.5:
                break
            cand = cur[:idx] + cur[idx + 1:]
            if _verify_fast(cand):
                cur = cand
                changed = True
                break  # Restart scan from beginning after a deletion
    
    if len(cur) < best_len:
        best = cur
        best_len = len(cur)
        # Check if this fast-verified result is actually valid
        if _verify_full(best):
            pass
        else:
            # If fast verification failed (false positive), revert to last known good
            # But we don't track last known good easily. 
            # Let's assume fast verification is reliable enough for small deletions.
            # To be safe, we will only update 'best' if full verify passes.
            # But that defeats the purpose. 
            # Let's do this: Keep a 'candidate' that is fast-verified.
            # Only promote to 'best' if full verify passes.
            
            # Re-verify the current 'cur' fully.
            if not _verify_full(cur):
                # Find the last valid state? 
                # Simpler: Just keep the initial best if this fails.
                # But we want to improve. 
                # Let's use a different strategy: 
                # We will perform deletions, and every K deletions, we do a full check.
                # If full check fails, we revert to the last full-check state.
                
                # This is getting complicated. 
                # Let's simplify: 
                # 1. Fast verify allows us to test 1000x more candidates.
                # 2. We will run a local search using fast verify.
                # 3. We will periodically (or at the end) run full verify on the best candidate.
                # 4. If the best candidate fails full verify, we return the baseline.
                
                pass

    # Phase 2: Local Search (Swap + Delete) using Fast Verify
    rng = random.Random(12345)
    cur = list(best)
    
    # We will keep track of the best candidate that passes fast verify.
    # We will also keep the last candidate that passed FULL verify.
    last_full_valid = list(best)
    
    for iteration in range(5000):
        if _elapsed() > TIME_LIMIT * 0.8:
            break
        n = len(cur)
        if n == 0:
            break
        
        # Strategy 1: Try to delete a random comparator
        d = rng.randrange(n)
        cand = cur[:d] + cur[d + 1:]
        if _verify_fast(cand):
            cur = cand
            # Every 50 successful moves, check full validity
            if iteration % 50 == 0:
                if _verify_full(cur):
                    last_full_valid = list(cur)
                    if len(cur) < best_len:
                        best_len = len(cur)
                        best = list(cur)
            continue
        
        # Strategy 2: Try swapping two comparators
        a = rng.randrange(n)
        b = rng.randrange(n)
        if a == b:
            continue
        cand = list(cur)
        cand[a], cand[b] = cand[b], cand[a]
        if _verify_fast(cand):
            cur = cand
            if iteration % 50 == 0:
                if _verify_full(cur):
                    last_full_valid = list(cur)
                    if len(cur) < best_len:
                        best_len = len(cur)
                        best = list(cur)

    # Final Phase: Try to delete more from the best fast-verified candidate
    # But ensure it is fully valid.
    candidate = list(best)
    
    # Attempt final deletions on the best candidate
    for idx in range(len(candidate) - 1, -1, -1):
        if _elapsed() > TIME_LIMIT * 0.95:
            break
        cand = candidate[:idx] + candidate[idx + 1:]
        if _verify_fast(cand):
            if _verify_full(cand):
                candidate = cand
                best = cand
                best_len = len(best)

    # Ensure we return a valid network
    if _verify_full(best):
        return best
    else:
        return _baseline_network()
