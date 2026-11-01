import time
import numpy as np
from collections import defaultdict
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = 135.0  # Safety margin below 150s

    # 1. Generate initial superpermutation using standard recursive construction
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)

        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0:
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if n <= 4:
        return "".join(map(str, word))

    # 2. Local Search with vectorized deletion checks
    current = np.array(word, dtype=np.int32)
    L = len(current)
    target_perms = 1
    for i in range(2, n + 1):
        target_perms *= i

    # Use a base that fits in int64 safely for n=7 (7^7 ~ 823543, 7^6 ~ 117649, etc.)
    # Max code: 7 * (7^6 + 7^5 + ... + 1) < 7 * 100000 = 700000, well within int64
    # Actually max value is sum_{i=0}^{n-1} (n-1) * n^i = n * (n^n - 1) / (n-1)
    # For n=7: 7 * (7^7 - 1) / 6 = 7 * 823542 / 6 = 960799.
    base = np.array([n ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    def compute_all_codes(arr, L_val):
        """Compute codes for all windows of length n."""
        if L_val < n:
            return np.array([], dtype=np.int64)
        windows = np.lib.stride_tricks.sliding_window_view(arr, n)
        return windows.astype(np.int64) @ base

    def compute_counts_from_codes(codes):
        if len(codes) == 0:
            return {}
        unique, counts = np.unique(codes, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    codes = compute_all_codes(current, L)
    perm_count = compute_counts_from_codes(codes)

    if len(perm_count) < target_perms:
        return "".join(map(str, current.tolist()))

    best_len = L

    # Precompute weighted prefix sums for fast window code computation
    # weighted[i] = arr[i] * base_shifted... actually let's use a different approach
    # For try_deletion, we need codes of windows that are concatenations of two parts.
    # Let's precompute: for each position i, the "prefix code" of arr[i:i+k] for k=1..n
    # This is: pref_code[i][k] = sum_{j=0}^{k-1} arr[i+j] * base[j]
    # We can compute this incrementally.
    
    # Actually, a simpler optimization: precompute the base-weighted array
    # warr[i] = arr[i] (as int64)
    # Then window code for arr[a:a+n] = dot(arr[a:a+n], base)
    # For the split window in try_deletion, we just need two dot products.
    # The current approach already does this. The main cost is the Python loop over j.
    # Let's vectorize: for all j simultaneously, compute the codes.

    def try_deletion(arr, L_val, codes_arr, pcount, pos):
        """Check if position pos can be deleted. Returns (can_delete, new_codes_set) or (False, None)."""
        n_val = n
        # Old windows containing position pos: start from max(0, pos-n+1) to min(pos, L_val-n)
        w_start = max(0, pos - n_val + 1)
        w_end = min(pos, L_val - n_val)

        if w_start > w_end:
            return False, None

        old_codes = codes_arr[w_start:w_end + 1]

        # Build new windows using vectorized computation
        # New windows start at j in [max(0, pos-n_val+1), pos-1]
        # For window starting at j:
        #   l1 = pos - j (length of prefix part arr[j:pos])
        #   l2 = n_val - l1 (length of suffix part arr[pos+1:pos+1+l2])
        #   code = dot(arr[j:pos], base[:l1]) + dot(arr[pos+1:pos+1+l2], base[l1:l1+l2])
        
        new_codes_set = set()
        j_min = max(0, pos - n_val + 1)
        j_max = pos  # exclusive
        
        # Check if we have enough elements after pos
        # For j = j_min, l2 = n_val - (pos - j_min) = j_min + n_val - pos
        # We need pos+1+l2-1 < L_val, i.e., pos + l2 < L_val
        # l2 = n_val - pos + j, so pos + n_val - pos + j = n_val + j < L_val
        # j < L_val - n_val, which is always true since j < pos and pos <= L_val - 1
        # But we need pos+1+l2 <= L_val, i.e., l2 <= L_val - pos - 1
        # l2 = n_val - pos + j <= L_val - pos - 1 => j <= L_val - n_val - 1
        # Since j < pos and pos can be up to L_val-1, we need to check
        
        # Vectorize: compute all new window codes at once
        # For each j, l1 = pos - j, l2 = n_val - l1
        # We can build a 2D array where row j has the window elements
        
        # Simpler: loop but use precomputed suffix dot products
        # Precompute: for k = 0..n_val, suffix_dot[k] = dot(arr[pos+1:pos+1+k], base[0:k])
        # Then for window at j with l1 = pos-j:
        #   part2 = dot(arr[pos+1:pos+1+l2], base[l1:l1+l2])
        #   = dot(arr[pos+1:pos+1+l2], base[l1:l1+l2])
        #   This is not directly suffix_dot since base indices are shifted.
        
        # Let's just do the loop but make it fast with local variables
        base_local = base
        arr_local = arr
        L_local = L_val
        n_local = n_val
        pos_local = pos
        
        for j in range(j_min, j_max):
            l1 = pos_local - j
            l2 = n_local - l1
            
            if l2 > L_local - pos_local - 1:
                continue
            
            # Compute code using two dot products
            if l1 > 0:
                part1 = int(arr_local[j:pos_local].astype(np.int64) @ base_local[:l1])
            else:
                part1 = 0
            if l2 > 0:
                part2 = int(arr_local[pos_local+1:pos_local+1+l2].astype(np.int64) @ base_local[l1:l1+l2])
            else:
                part2 = 0
            
            new_codes_set.add(part1 + part2)

        # Check each old code with count 1
        for oc in old_codes:
            oc_int = int(oc)
            if pcount.get(oc_int, 0) == 1:
                if oc_int not in new_codes_set:
                    return False, None

        return True, new_codes_set

    # Phase 1: Repeated deletion passes
    while time.time() < deadline:
        deleted_this_pass = False
        # Try deleting from right to left (more effective)
        for pos in range(L - 1, -1, -1):
            if time.time() > deadline - 2.0:
                break

            can_del, new_codes_set = try_deletion(current, L, codes, perm_count, pos)

            if can_del:
                # Update counts
                w_start = max(0, pos - n + 1)
                w_end = min(pos, L - n)
                old_codes = codes[w_start:w_end + 1]
                for oc in old_codes:
                    oc_int = int(oc)
                    perm_count[oc_int] -= 1
                    if perm_count[oc_int] == 0:
                        del perm_count[oc_int]

                for nc in new_codes_set:
                    perm_count[nc] = perm_count.get(nc, 0) + 1

                # Delete position pos
                current = np.concatenate([current[:pos], current[pos + 1:]])
                L -= 1
                best_len = min(best_len, L)

                # Recompute codes (expensive but necessary for correctness)
                codes = compute_all_codes(current, L)
                deleted_this_pass = True
                break  # Restart scan from right

        if not deleted_this_pass:
            # Try left to right as well
            for pos in range(L):
                if time.time() > deadline - 2.0:
                    break

                can_del, new_codes_set = try_deletion(current, L, codes, perm_count, pos)

                if can_del:
                    w_start = max(0, pos - n + 1)
                    w_end = min(pos, L - n)
                    old_codes = codes[w_start:w_end + 1]
                    for oc in old_codes:
                        oc_int = int(oc)
                        perm_count[oc_int] -= 1
                        if perm_count[oc_int] == 0:
                            del perm_count[oc_int]

                    for nc in new_codes_set:
                        perm_count[nc] = perm_count.get(nc, 0) + 1

                    current = np.concatenate([current[:pos], current[pos + 1:]])
                    L -= 1
                    best_len = min(best_len, L)
                    codes = compute_all_codes(current, L)
                    deleted_this_pass = True
                    break

        if not deleted_this_pass:
            break

    # Phase 2: Try adjacent swaps to unlock further deletions
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        # Try swapping adjacent elements and see if it enables more deletions
        swap_done = False
        for round_num in range(10):
            if time.time() > deadline - 2.0:
                break
            
            found_improvement = False
            for pos in range(L - 1):
                if time.time() > deadline - 2.0:
                    break
                
                # Try swapping current[pos] and current[pos+1]
                saved = current.copy()
                current[pos], current[pos+1] = current[pos+1], current[pos]
                
                # Check if this swap maintains validity (all perms still present)
                new_codes = compute_all_codes(current, L)
                new_pcount = compute_counts_from_codes(new_codes)
                
                if len(new_pcount) >= target_perms:
                    # Check all perms are present
                    # Quick check: compare with old perm_count
                    # For each perm in old perm_count with count 1, check it's still in new_pcount
                    valid = True
                    for code, cnt in perm_count.items():
                        if cnt == 1 and new_pcount.get(code, 0) == 0:
                            valid = False
                            break
                    
                    if valid:
                        # Accept swap, recompute codes
                        codes = new_codes
                        perm_count = new_pcount
                        found_improvement = True
                        break
                    else:
                        current = saved
                else:
                    current = saved
            
            if not found_improvement:
                break
            
            # After a successful swap, try more deletions
            for pos in range(L - 1, -1, -1):
                if time.time() > deadline - 2.0:
                    break
                can_del, new_codes_set = try_deletion(current, L, codes, perm_count, pos)
                if can_del:
                    w_start = max(0, pos - n + 1)
                    w_end = min(pos, L - n)
                    old_codes = codes[w_start:w_end + 1]
                    for oc in old_codes:
                        oc_int = int(oc)
                        perm_count[oc_int] -= 1
                        if perm_count[oc_int] == 0:
                            del perm_count[oc_int]
                    for nc in new_codes_set:
                        perm_count[nc] = perm_count.get(nc, 0) + 1
                    current = np.concatenate([current[:pos], current[pos + 1:]])
                    L -= 1
                    best_len = min(best_len, L)
                    codes = compute_all_codes(current, L)
                    break
    
    # Phase 3: Try 2-opt style moves (delete and re-insert)
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        # Try random 2-opt moves
        rng = np.random.default_rng(42)
        max_2opt_rounds = int(remaining_time / 3.0)  # ~3s per round estimate

        for round_num in range(max_2opt_rounds):
            if time.time() > deadline - 1.0:
                break

            # Pick a random position to delete
            pos = int(rng.integers(0, L))
            symbol = current[pos]

            # Check if we can delete pos
            can_del, new_codes_set = try_deletion(current, L, codes, perm_count, pos)
            if not can_del:
                continue

            # Save state
            saved_current = current.copy()
            saved_L = L
            saved_codes = codes.copy()
            saved_perm_count = dict(perm_count)

            # Perform deletion
            w_start = max(0, pos - n + 1)
            w_end = min(pos, L - n)
            old_codes = codes[w_start:w_end + 1]
            for oc in old_codes:
                oc_int = int(oc)
                perm_count[oc_int] -= 1
                if perm_count[oc_int] == 0:
                    del perm_count[oc_int]

            for nc in new_codes_set:
                perm_count[nc] = perm_count.get(nc, 0) + 1

            current = np.concatenate([current[:pos], current[pos + 1:]])
            L -= 1
            codes = compute_all_codes(current, L)

            # Now try to re-insert symbol at various positions
            best_insert_pos = -1
            best_insert_valid = False

            # Try inserting at each position
            for insert_pos in range(L + 1):
                if time.time() > deadline - 1.0:
                    break

                # Check if inserting symbol at insert_pos creates a valid superpermutation
                # We need to check that all windows formed are valid permutations
                # and that we don't lose any permutation

                # The new windows are those that include the inserted symbol
                # Window starting at j in [max(0, insert_pos-n+1), insert_pos]
                # But we only need to check windows that cross the insertion point

                valid_insert = True
                new_window_codes = set()

                for j in range(max(0, insert_pos - n + 1), min(insert_pos + 1, L - n + 1)):
                    # Build window in the new string
                    # Positions before insert_pos map to old positions
                    # Positions >= insert_pos map to old positions shifted by 1
                    window = []
                    for k in range(n):
                        new_pos = j + k
                        if new_pos < insert_pos:
                            window.append(current[new_pos])
                        else:
                            if new_pos == insert_pos:
                                window.append(symbol)
                            elif new_pos - 1 < L:
                                window.append(current[new_pos - 1])
                            else:
                                valid_insert = False
                                break
                    if not valid_insert:
                        break

                    if len(set(window)) < n:
                        # Not a valid permutation (has duplicates)
                        # This is actually fine - we just can't use this window
                        # But wait, we need ALL permutations to be present
                        # A window with duplicates doesn't contribute to any permutation
                        # So it's okay to have such windows
                        pass

                    window_arr = np.array(window, dtype=np.int64)
                    wc = int(window_arr @ base)
                    new_window_codes.add(wc)

                if not valid_insert:
                    continue

                # Check: for each old window (in the deleted string) with count 1,
                # it must still be present after re-insertion
                # Old windows are in `codes` (the deleted string)
                # New windows are: old windows not affected + new windows from insertion
                # Old windows not affected: those not crossing insert_pos in new string
                # which correspond to old windows not crossing the gap

                # Actually, let's just verify all 5040 perms are present
                # This is expensive, so let's do a quick check first
                # Quick check: the new windows must cover all perms that were only in affected old windows

                # For simplicity, let's just try it and verify
                # Build the full new string
                new_string = np.concatenate([current[:insert_pos], np.array([symbol], dtype=np.int32), current[insert_pos:]])
                new_L = L + 1
                new_codes_arr = compute_all_codes(new_string, new_L)
                new_pcount = compute_counts_from_codes(new_codes_arr)

                if len(new_pcount) >= target_perms:
                    # Check all target perms are present
                    # We need to check that every permutation code is in new_pcount
                    # Generate all permutation codes
                    all_perm_codes = set()
                    for perm in permutations(range(1, n + 1)):
                        code = int(np.array(perm, dtype=np.int64) @ base)
                        all_perm_codes.add(code)

                    if all(c in new_pcount for c in all_perm_codes):
                        best_insert_pos = insert_pos
                        best_insert_valid = True
                        break

            if best_insert_valid:
                # Perform the re-insertion
                current = np.concatenate([current[:best_insert_pos], np.array([symbol], dtype=np.int32), current[best_insert_pos:]])
                L = L + 1
                best_len = min(best_len, L)
                codes = compute_all_codes(current, L)
                perm_count = compute_counts_from_codes(codes)
            else:
                # Revert
                current = saved_current
                L = saved_L
                codes = saved_codes
                perm_count = saved_perm_count

    return "".join(map(str, current.tolist()))
