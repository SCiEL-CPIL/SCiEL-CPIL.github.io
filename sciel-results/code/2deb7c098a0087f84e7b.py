# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a large 4-term-progression-free subset of [1, 512].
    Strategy:
      1. Base set: numbers in [1,512] whose base-3 representation uses only digits 0 and 1
         (Salem-Spencer type construction, guaranteed 3-AP-free, hence 4-AP-free).
      2. Greedy extension: try to add remaining numbers in ascending order if they don't
         create a nonconstant 4-term AP with already-selected elements.
      3. Local improvement: try to swap out a selected element for an unselected one if
         it increases the set size (net gain) while preserving 4-AP-freeness.
    """
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    n = 512

    def time_up():
        return time.time() - start_time > deadline

    def has_4ap(S, a, d):
        """Check if a, a+d, a+2d, a+3d are all in S (nonconstant 4-AP)."""
        return a in S and (a+d) in S and (a+2*d) in S and (a+3*d) in S

    def creates_4ap(S, x):
        """
        Check if adding x to S creates any nonconstant 4-term AP.
        x can be the 1st, 2nd, 3rd, or 4th term of the AP.
        """
        # x is the 1st term: x, x+d, x+2d, x+3d
        for d in range(1, (n - x) // 3 + 1):
            if (x+d) in S and (x+2*d) in S and (x+3*d) in S:
                return True
        # x is the 2nd term: x-d, x, x+d, x+2d
        for d in range(1, x):
            if (x-d) in S and (x+d) in S and (x+2*d) in S:
                return True
        # x is the 3rd term: x-2d, x-d, x, x+d
        for d in range(1, x // 2 + 1):
            if (x-2*d) in S and (x-d) in S and (x+d) in S:
                return True
        # x is the 4th term: x-3d, x-2d, x-d, x
        for d in range(1, x // 3 + 1):
            if (x-3*d) in S and (x-2*d) in S and (x-d) in S:
                return True
        return False

    def is_4ap_free(S):
        """Verify S contains no nonconstant 4-term AP."""
        S_set = set(S)
        for a in S_set:
            for d in range(1, (n - a) // 3 + 1):
                if (a+d) in S_set and (a+2*d) in S_set and (a+3*d) in S_set:
                    return False
        return True

    # Step 1: Base-3 construction (digits 0 and 1 only in base 3)
    def base3_01(n):
        result = set()
        for x in range(1, n + 1):
            temp = x
            valid = True
            while temp > 0:
                if temp % 3 == 2:
                    valid = False
                    break
                temp //= 3
            if valid:
                result.add(x)
        return result

    S = base3_01(n)

    # Step 2: Greedy extension - try adding each non-selected number in ascending order
    for x in range(1, n + 1):
        if time_up():
            break
        if x in S:
            continue
        if not creates_4ap(S, x):
            S.add(x)

    # Step 3: Local improvement via swaps
    # Try to remove one element and add two (or net gain) to increase size
    # Simpler: try swapping - remove a selected element, add an unselected one if it fits
    # and the resulting set is still 4-AP-free (size stays same, but may enable further additions)
    improved = True
    while improved and not time_up():
        improved = False
        selected_list = sorted(S)
        unselected = [x for x in range(1, n + 1) if x not in S]
        
        for out_x in selected_list:
            if time_up():
                break
            for in_x in unselected:
                if time_up():
                    break
                # Try: remove out_x, add in_x
                S.remove(out_x)
                if not creates_4ap(S, in_x):
                    S.add(in_x)
                    # After swap, try to add more elements
                    added_any = False
                    for x in range(1, n + 1):
                        if time_up():
                            break
                        if x in S:
                            continue
                        if not creates_4ap(S, x):
                            S.add(x)
                            added_any = True
                    if len(S) > len(selected_list):
                        improved = True
                        break
                    else:
                        # Revert swap if no net gain
                        S.remove(in_x)
                        S.add(out_x)
                else:
                    S.add(out_x)
            if improved:
                break

    # Final verification
    if not is_4ap_free(S):
        # Fallback: pure greedy
        S = set()
        for x in range(1, n + 1):
            if not creates_4ap(S, x):
                S.add(x)

    return sorted(S)
# EVOLVE-BLOCK-END
