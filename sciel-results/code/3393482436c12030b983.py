"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit
    
    MOD = 1009
    MAX_VAL = MOD - 1  # 1008

    # Precompute modular inverses for 1 to 1008
    inv = [0] * (MAX_VAL + 1)
    for x in range(1, MAX_VAL + 1):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Convert to numpy array for vectorized operations
    inv_arr = np.array(inv, dtype=np.int32)
    
    best_start = 2
    best_length = 0
    
    # The interval must be within [2, 1007] because 1 and 1008 are self-inverse
    # (1*1=1, 1008*1008=1 mod 1009), so any interval containing 1 or 1008 is invalid.
    # Thus s >= 2 and e <= 1007.
    
    # For each start s, greedily extend e to the right.
    # An interval [s, e] is valid iff for all x in [s, e], inv[x] is NOT in [s, e].
    # Greedy extension: when adding next_e = e+1, the only new constraint is that
    # inv[next_e] is not in [s, next_e]. Since inv is an involution, this is
    # equivalent to inv[next_e] not in [s, e] (as inv[next_e] != next_e for
    # next_e in [2, 1007]).
    
    # Use a sliding window / two-pointer approach with early termination
    # For each start s, find the maximum valid end e.
    # Key insight: if [s, e] is valid, then [s+1, e] may or may not be valid,
    # but we can use the fact that the constraint is monotonic in a certain way.
    
    # Alternative: for each possible end e, find the minimum start s such that [s, e] is valid.
    # This can be done by tracking the minimum "forbidden start" as we extend e.
    
    # For a fixed end e, [s, e] is valid iff for all x in [s, e], inv[x] < s or inv[x] > e.
    # Equivalently, for all x in [s, e] with inv[x] > x (so inv[x] could be in [x+1, e]),
    # we need inv[x] > e. And for all x in [s, e] with inv[x] < x, we need inv[x] < s.
    
    # Let's define for each x:
    #   if inv[x] > x: the constraint is that e < inv[x], i.e., e <= inv[x] - 1
    #   if inv[x] < x: the constraint is that s > inv[x], i.e., s >= inv[x] + 1
    #   if inv[x] == x: x cannot be in any valid interval (self-inverse)
    
    # So for a valid interval [s, e]:
    #   For all x in [s, e]: 
    #     if inv[x] > x, then e <= inv[x] - 1
    #     if inv[x] < x, then s >= inv[x] + 1
    #     inv[x] != x
    
    # Let R[x] = inv[x] - 1 if inv[x] > x, else INF (no constraint on e)
    # Let L[x] = inv[x] + 1 if inv[x] < x, else 0 (no constraint on s)
    # Let valid[x] = (inv[x] != x)
    
    # For interval [s, e] to be valid:
    #   e <= min(R[x] for x in [s, e])
    #   s >= max(L[x] for x in [s, e])
    #   all x in [s, e] must be valid
    
    # We can precompute prefix/suffix minima and maxima, but the constraint is circular
    # (s and e appear on both sides). Let's use a two-pointer approach.
    
    # Two-pointer: start with s=2, e=2. Extend e as far as possible.
    # Then increment s and adjust e if needed.
    
    # Actually, let's just do the greedy approach but with numpy for speed.
    # The greedy approach is O(n^2) worst case but n=1006, so it's fast enough.
    
    # Let's use a more efficient approach:
    # For each s from 2 to 1007:
    #   Find the maximum e such that [s, e] is valid.
    #   This can be done by binary search if we can check validity in O(1) or O(log n).
    
    # Precompute for each position x:
    #   right_constraint[x] = the maximum e such that x doesn't violate the right constraint
    #   left_constraint[x] = the minimum s such that x doesn't violate the left constraint
    
    # For x in [2, 1007]:
    #   if inv[x] > x: right_constraint[x] = inv[x] - 1, left_constraint[x] = 2
    #   if inv[x] < x: right_constraint[x] = 1007, left_constraint[x] = inv[x] + 1
    #   if inv[x] == x: x is invalid, right_constraint[x] = 1, left_constraint[x] = 1008
    
    right_constraint = np.full(MAX_VAL + 1, 1007, dtype=np.int32)
    left_constraint = np.zeros(MAX_VAL + 1, dtype=np.int32)
    
    for x in range(2, MAX_VAL):  # x from 2 to 1007
        ix = inv[x]
        if ix == x:
            right_constraint[x] = 1  # invalid
            left_constraint[x] = 1008  # invalid
        elif ix > x:
            right_constraint[x] = ix - 1
            left_constraint[x] = 2
        else:  # ix < x
            right_constraint[x] = 1007
            left_constraint[x] = ix + 1
    
    # Now for each start s, find the maximum e such that:
    #   e <= min(right_constraint[s:e+1])
    #   s >= max(left_constraint[s:e+1])
    
    # This is a sliding window problem. Let's use a two-pointer approach.
    # Maintain a window [s, e] and track the min of right_constraint and max of left_constraint.
    
    # Use a deque-based approach for sliding window min/max, or just recompute since n is small.
    
    # Given n=1006, let's just do O(n^2) with early termination, which is ~1M operations.
    
    best_start = 2
    best_length = 0
    
    for s in range(2, MAX_VAL):
        if time.time() > deadline:
            break
        
        # Find maximum e such that [s, e] is valid
        # Start with e = s, extend as far as possible
        e = s
        
        # Track constraints as we extend
        min_rc = right_constraint[s]
        max_lc = left_constraint[s]
        
        if min_rc < s or max_lc > s:
            # [s, s] is not valid
            continue
        
        while e < MAX_VAL - 1:  # e can go up to 1007
            next_e = e + 1
            new_min_rc = min(min_rc, right_constraint[next_e])
            new_max_lc = max(max_lc, left_constraint[next_e])
            
            if new_min_rc >= next_e and new_max_lc <= s:
                e = next_e
                min_rc = new_min_rc
                max_lc = new_max_lc
            else:
                break
        
        length = e - s + 1
        if length > best_length:
            best_length = length
            best_start = s

    # Fallback if no valid interval found (should not happen as [2,2] is valid)
    if best_length == 0:
        best_start = 2
        best_length = 1

    return [best_start, best_length]
# EVOLVE-BLOCK-END
