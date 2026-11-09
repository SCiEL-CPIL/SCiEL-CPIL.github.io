import time
import random
import sys

_N = 128
_LO = 128
_HI = 256
_RANGE = _HI - _LO + 1

def _get_counts(A_set):
    """
    Returns a list of representation counts for sums in [_LO, _HI].
    Returns None if A is empty.
    """
    if not A_set:
        return None
    A_sorted = sorted(A_set)
    n = len(A_sorted)
    counts = [0] * _RANGE
    
    for i in range(n):
        a = A_sorted[i]
        for j in range(i, n):
            b = A_sorted[j]
            s = a + b
            if _LO <= s <= _HI:
                counts[s - _LO] += 1
    return counts

def _is_valid(counts):
    """Check if all sums in the range are covered."""
    if counts is None:
        return False
    return all(c >= 1 for c in counts)

def _get_peak(counts):
    """Get the maximum representation count."""
    if counts is None:
        return float('inf')
    return max(counts)

def _greedy_peel(A_init):
    """
    Greedily remove elements from A_init to minimize peak while maintaining validity.
    """
    A = set(A_init)
    counts = _get_counts(A)
    
    if not _is_valid(counts):
        return A, _get_peak(counts)

    current_peak = _get_peak(counts)
    best_A = A.copy()
    best_peak = current_peak
    
    # Map from sum index to list of pairs (i, j) in sorted A?
    # It's easier to just recompute or maintain a structure.
    # Given N=128, recomputing is O(128^2) ~ 16k ops.
    # We might do 128 removals. 128 * 16k = 2M ops. Very fast.
    
    while True:
        # Find the sum with max count
        max_idx = 0
        max_val = counts[0]
        for i in range(1, _RANGE):
            if counts[i] > max_val:
                max_val = counts[i]
                max_idx = i
        
        if max_val <= 1:
            break # Cannot reduce further without breaking validity (if all are 1)
            
        # Find all elements involved in sums with count == max_val?
        # Or just elements involved in the specific max_idx sum?
        # Removing an element affects many sums.
        # Let's try removing each element in A and see if validity holds.
        # To be efficient, we can prioritize elements that are "central" or frequent.
        
        A_list = sorted(A)
        improved = False
        
        # Try removing each element
        # Sort candidates by how many times they appear in the "hot" sums?
        # Simple approach: try all, pick the one that reduces peak the most or maintains validity with smallest size.
        # Since we want to minimize peak, we want to reduce the max count.
        
        best_candidate = None
        best_candidate_peak = current_peak
        
        for x in A_list:
            new_A = A.copy()
            new_A.discard(x)
            if not new_A:
                continue
            
            new_counts = _get_counts(new_A)
            if not _is_valid(new_counts):
                continue
            
            new_peak = _get_peak(new_counts)
            
            # We accept if peak is lower, or if peak is same but size is smaller?
            # Actually, we just want to minimize peak.
            # If peak is same, removing an element is still good for "sparsity" but doesn't help score.
            # However, it might allow further removals.
            # Let's accept if new_peak < current_peak OR (new_peak == current_peak AND len(new_A) < len(A))
            # Actually, just accepting any valid removal helps explore the space of smaller sets.
            
            if new_peak < best_candidate_peak:
                best_candidate_peak = new_peak
                best_candidate = new_A
            elif new_peak == best_candidate_peak and best_candidate is not None and len(new_A) < len(best_candidate):
                best_candidate = new_A
            elif best_candidate is None and new_peak == current_peak:
                # If no peak reduction, take a removal that keeps peak same (to enable future reductions)
                best_candidate = new_A
                best_candidate_peak = new_peak

        if best_candidate is not None:
            A = best_candidate
            counts = _get_counts(A)
            current_peak = _get_peak(counts)
            
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = A.copy()
            
            # If we didn't reduce peak, we might loop. 
            # But we reduced size.
            # To avoid infinite loop if peak doesn't drop but size does, we rely on size reduction.
            # Max size 129, so max 129 iterations.
        else:
            break
            
    return best_A, best_peak

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85
    
    # Start with the full interval
    A_init = set(range(0, _N + 1))
    
    # Run greedy peeling
    A_res, peak_res = _greedy_peel(A_init)
    
    best_A = A_res
    best_peak = peak_res
    
    # Try a few random perturbations of the initial set to escape local minima of the greedy peeling
    rng = random.Random(42)
    
    # Generate a few random valid starting sets?
    # Or just perturb the result of the first peel?
    # Let's try peeling from slightly different intervals.
    
    # Interval [1, 128] is not valid (0 missing, 128+128=256 ok, but 128 needs 0+128 or 1+127 etc).
    # If we use [1, 128], 128 can be 1+127... 64+64.
    # But 0 is not in A.
    # Let's try [0, 127] U {128}? That's [0, 128].
    
    # Try removing random elements from [0, 128] first, then peeling?
    # No, greedy peeling is deterministic.
    
    # Let's try multiple random restarts of the "peeling" process where we randomly skip some removals?
    # Or just run local search on the result of greedy peeling.
    
    # Let's implement a simple local search on the result of greedy peeling to see if we can improve.
    
    if time.time() < DEADLINE - 5:
        A_curr = best_A.copy()
        counts_curr = _get_counts(A_curr)
        peak_curr = best_peak
        
        # Local search: try swaps
        # Since we are already "peeled", size is small.
        # Swaps might allow better distribution.
        
        in_A = list(A_curr)
        not_in_A = [x for x in range(_N + 1) if x not in A_curr]
        
        no_improve = 0
        max_no_improve = 20
        
        while time.time() < DEADLINE - 5 and no_improve < max_no_improve:
            improved = False
            # Try random swaps
            for _ in range(50):
                if not in_A or not not_in_A:
                    break
                x = rng.choice(in_A)
                y = rng.choice(not_in_A)
                
                new_A = A_curr.copy()
                new_A.discard(x)
                new_A.add(y)
                
                new_counts = _get_counts(new_A)
                if not _is_valid(new_counts):
                    continue
                
                new_peak = _get_peak(new_counts)
                
                if new_peak < peak_curr:
                    A_curr = new_A
                    counts_curr = new_counts
                    peak_curr = new_peak
                    in_A = list(A_curr)
                    not_in_A = [x for x in range(_N + 1) if x not in A_curr]
                    improved = True
                    no_improve = 0
                    
                    if peak_curr < best_peak:
                        best_peak = peak_curr
                        best_A = A_curr.copy()
                    break # Restart loop with new state
            if not improved:
                no_improve += 1
                
    return sorted(best_A)
