import itertools
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 140
    
    # 1. Generate standard superpermutation
    # L_n = 1! + 2! + ... + n!
    # Construction:
    # S_1 = [1]
    # S_m = S_{m-1} + m + S_{m-1} ... wait, standard is:
    # Let P_m be the list of permutations of 1..m in standard order.
    # S_m is formed by taking S_{m-1}, and for each permutation p in P_{m-1},
    # we insert m after p? 
    # Actually, the recursive construction is:
    # Superperm(n) = Superperm(n-1) + n + ... 
    # Let's use the explicit loop from the seed but ensure it's correct.
    
    word = [1]
    for m in range(2, n + 1):
        # Extract permutations of 1..m-1 from current word
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Overlap optimization
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    # 2. Optimization: Local Deletion Search
    # We want to minimize length such that all n! permutations are present.
    
    # Precompute all valid permutations
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Function to check if word is valid and return counts
    def get_counts(w):
        counts = {}
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            if p in all_perms: # Optimization: only count valid perms? 
                # Actually, if p is not a permutation, it doesn't contribute to coverage.
                # But we need to ensure we don't lose valid ones.
                counts[p] = counts.get(p, 0) + 1
        return counts

    # Initial counts
    counts = get_counts(word)
    
    # Check validity
    def is_valid(counts):
        for p in all_perms:
            if counts.get(p, 0) == 0:
                return False
        return True
        
    # If not valid, return standard (shouldn't happen)
    if not is_valid(counts):
        return "".join(map(str, word))
        
    # Iterative deletion
    # We scan from left to right.
    # If we delete, the indices shift. So it's safer to scan right to left?
    # Or just restart the scan after a deletion.
    # Given L is small, restarting is fine.
    
    improved = True
    while improved:
        if time.time() > deadline - 10:
            break
        improved = False
        # Scan from left to right
        for i in range(1, len(word) - 1):
            if time.time() > deadline - 5:
                return "".join(map(str, word))
            
            # Tentatively delete word[i]
            # Identify windows covering i
            # Window j covers i if j <= i < j+n
            # So j is in [i-n+1, i]
            # And j must be a valid window start: 0 <= j <= len(word)-n
            
            start = max(0, i - n + 1)
            end = min(i, len(word) - n)
            
            affected_windows = []
            old_counts = {}
            
            valid_deletion = True
            
            for j in range(start, end + 1):
                p = tuple(word[j:j+n])
                if p in all_perms:
                    affected_windows.append((j, p))
                    old_counts[p] = counts.get(p, 0)
                    counts[p] -= 1
                    if counts[p] == 0:
                        valid_deletion = False
                        break
            
            if valid_deletion:
                # Check if any other count became 0? 
                # No, only affected windows changed.
                # However, we must ensure that the *new* windows formed by the deletion
                # don't somehow invalidate things? 
                # Wait, deletion removes windows. It doesn't create new permutations 
                # that weren't there before in a way that helps, but it *removes* coverage.
                # The only risk is losing coverage of a permutation that was ONLY 
                # present in the deleted windows.
                # Our check `counts[p] == 0` handles this for the affected windows.
                # Are there any other windows that become invalid? No.
                # Are there any new windows that appear?
                # When we remove word[i], the word shrinks.
                # The windows that *survive* are those not covering i.
                # The windows that *cover* i are removed.
                # New windows are formed by the concatenation of the left and right parts.
                # Do these new windows provide coverage for permutations that might 
                # have been lost?
                # No, the new windows are subsets of the original word's content 
                # (shifted).
                # Actually, the new windows are *different* tuples.
                # Example: 1 2 3 4 5, n=3. Remove 3. Word: 1 2 4 5.
                # Old windows: (1,2,3), (2,3,4), (3,4,5).
                # New windows: (1,2,4), (2,4,5).
                # (1,2,4) is a new permutation not in the old set.
                # So deletion can *add* coverage of new permutations?
                # But it *removes* coverage of old ones.
                # If we remove a window that was the *only* instance of permutation P, 
                # we lose P.
                # If we form a new window Q that was not present before, we gain Q.
                # But we only care about *keeping* all permutations.
                # So we must ensure we don't lose any.
                # Our check `counts[p] == 0` for affected windows ensures we don't 
                # lose any permutation that was covered *only* by the removed windows.
                # What if a permutation was covered by a removed window AND a new window?
                # The new window is formed by shifting.
                # It is not possible for a new window to be the "same" permutation 
                # as a removed one in a way that compensates, because the symbols 
                # are distinct in a permutation.
                # Actually, if we remove a symbol, the relative order of others stays.
                # But the specific n-tuple changes.
                # So, the check on the *removed* windows is sufficient to ensure 
                # we don't drop below 1 count for any permutation that was previously 
                # covered *only* by those removed windows.
                # Wait, what if a permutation P was covered by window W1 (removed) 
                # and window W2 (surviving)?
                # Then count[P] was 2. We decrement to 1. It's still > 0. Valid.
                # What if P was covered by W1 (removed) and W3 (removed)?
                # Count was 2. Decrement twice?
                # My loop iterates over all affected windows.
                # So if two windows covered P, we decrement twice.
                # If count was 2, it becomes 0. We flag invalid. Correct.
                
                # So the logic holds.
                
                # Perform deletion
                word.pop(i)
                improved = True
                
                # Recalculate counts for the new word?
                # It is safer to recalculate counts from scratch to avoid 
                # bookkeeping errors with the "new windows" that appear.
                # The "new windows" are not accounted for in `counts` 
                # because we only decremented the old ones.
                # We need to add the counts for the new windows formed.
                # This is complex.
                # Given L is small, just recalculate `counts` from scratch.
                counts = get_counts(word)
                
                # Restart scan? Or continue?
                # Continuing might be okay, but indices shift.
                # Let's break and restart the outer loop for safety.
                break
            else:
                # Revert counts
                for j, p in affected_windows:
                    counts[p] = old_counts[p]
                # If we broke early in the inner loop, we might have 
                # decremented some but not all?
                # The `break` in `if counts[p] == 0` exits the `for j` loop.
                # We need to revert the decrements that happened.
                # My code above: `old_counts` stores the value *before* decrement.
                # So reverting is just setting them back.
                # But we only stored `old_counts` for windows we processed.
                # If we broke, we processed some. We revert those.
                # The ones we didn't process weren't decremented.
                # So this is correct.
                pass

    return "".join(map(str, word))
