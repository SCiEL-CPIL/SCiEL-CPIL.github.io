import time
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Classic Recursive Construction
    # Length is 1! + 2! + ... + n!
    word = [1]
    for m in range(2, n + 1):
        # Extract unique permutations of length m-1 from word
        # in order of appearance
        perms = []
        seen = set()
        # word length is L. windows of size m-1
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            # Check if it's a permutation (distinct elements)
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            # Block: p[0]...p[m-2], m, p[0]...p[m-2]
            block = list(p) + [m] + list(p)
            
            # Find max overlap with new_word
            max_overlap = 0
            limit = min(len(new_word), len(block))
            for o in range(limit, 0, -1):
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            
            new_word.extend(block[max_overlap:])
        
        word = new_word
        
        # Time check
        if time.time() > deadline:
            break

    # For n <= 6, the recursive construction is often optimal or close enough
    # and the overhead of complex local search might not be worth it or might risk timeout.
    # However, for n=7, we have plenty of time (150s) and the recursive construction
    # is 5905, while optimal is ~5884+. We can try to shave off characters.
    
    if n >= 5 and time.time() < deadline - 20.0:
        
        # Map permutations to IDs for O(1) count access
        # n! for n=7 is 5040. This is small.
        # We only care about valid permutations of 1..n
        # Generate all permutations? 
        # For n=7, 5040 is fine. For n=8, 40320 is fine.
        
        # We don't strictly need to map ALL permutations to IDs if we use a dict,
        # but an array is faster.
        # However, generating all permutations takes time.
        # Alternative: Use a dict `counts` mapping tuple -> int.
        # Tuple hashing is fast enough for 5040 items.
        
        def get_perm_id(p):
            # Simple mapping: treat permutation as base-n number?
            # Or just use the tuple itself as key in a dict.
            pass

        # Let's use a dict for simplicity and robustness against n changes,
        # but optimize the inner loop.
        
        # Build initial counts
        counts = {}
        for i in range(len(word) - n + 1):
            p = tuple(word[i:i+n])
            if len(set(p)) == n:
                counts[p] = counts.get(p, 0) + 1
        
        # Verify validity
        # Every permutation of 1..n must be present.
        # We can check if len(counts) == n!
        # Factorial check
        fact = 1
        for i in range(1, n+1):
            fact *= i
        if len(counts) < fact:
             # Should not happen for recursive construction
             return "".join(map(str, word))

        # Local Search: Greedy Deletion with efficient update
        # We scan from left to right.
        # If we delete a character, the word shrinks.
        # We restart the scan from the beginning after each successful deletion
        # to ensure we don't miss opportunities created by the shift.
        # However, restarting from 0 is expensive.
        # Optimization: After deleting at index i, the indices < i are unchanged.
        # Indices > i are shifted.
        # We can resume from index i (in the new word, which corresponds to old i+1).
        # But the "context" for index i-1 in the new word has changed (it now ends with word[i+1]).
        # So we should restart from i-1 (or 0 if i=0) to be safe.
        
        current_word = word
        # To avoid modifying the list while iterating in a complex way, 
        # we will use a while loop and manual index management.
        
        # Actually, the previous implementation used `for i in range(L)` and `break` after deletion.
        # This effectively restarts the outer loop (the `for _ in range(max_iterations)`).
        # This is $O(D \cdot L)$ passes.
        
        # Let's implement the efficient deletion test.
        
        def try_delete(i):
            """
            Attempts to delete current_word[i].
            Returns True if successful.
            Modifies current_word and counts if successful.
            """
            L = len(current_word)
            if L <= n:
                return False
            
            # 1. Identify old windows containing i
            # k in [max(0, i-n+1), min(i, L-n)]
            k_start_old = max(0, i - n + 1)
            k_end_old = min(i, L - n)
            
            old_perms = []
            for k in range(k_start_old, k_end_old + 1):
                p = tuple(current_word[k:k+n])
                if len(set(p)) == n:
                    old_perms.append(p)
                    counts[p] -= 1
                    if counts[p] == 0:
                        # This permutation is lost if we don't form it in the new word
                        # We can't decide yet, need to check new windows
                        pass
            
            # 2. Identify new windows formed
            # New word length L-1.
            # Junction is at i-1 and i in new word.
            # Windows in new word starting at k that span the junction:
            # k <= i-1 and k+n-1 >= i
            # k in [i-n+1, i-1]
            # Also k >= 0 and k+n-1 < L-1 => k < L-n
            k_start_new = max(0, i - n + 1)
            k_end_new = min(i - 1, L - n - 1)
            
            new_perms = []
            for k in range(k_start_new, k_end_new + 1):
                # New window is current_word[k:i] + current_word[i+1:k+n+1]
                # Note: current_word is still the OLD word.
                # current_word[k:i] is length i-k.
                # current_word[i+1:k+n+1] is length (k+n+1) - (i+1) = n - i + k.
                # Total length n.
                w1 = current_word[k:i]
                w2 = current_word[i+1:k+n+1]
                p = tuple(w1 + w2)
                if len(set(p)) == n:
                    new_perms.append(p)
                    counts[p] = counts.get(p, 0) + 1
            
            # 3. Check validity
            # We need to ensure no count is 0.
            # We only modified counts for old_perms and new_perms.
            # We need to check if any of these is now 0.
            # Actually, we decremented old_perms. If any old_perm is NOT in new_perms (or not restored), it might be 0.
            # We incremented new_perms.
            
            valid = True
            # Check all affected perms
            # Union of old_perms and new_perms
            # To avoid creating a set, we can just check the list.
            # But we must ensure we don't double count if a perm is in both.
            # It's safer to just check all keys in the `counts` dict? No, that's O(n!).
            # We only changed specific keys.
            
            # Let's collect unique affected perms
            affected = set(old_perms)
            affected.update(new_perms)
            
            for p in affected:
                if counts.get(p, 0) == 0:
                    valid = False
                    break
            
            if valid:
                # Commit deletion
                del current_word[i]
                return True
            else:
                # Revert
                for p in old_perms:
                    counts[p] += 1
                for p in new_perms:
                    counts[p] -= 1
                    if counts[p] == 0:
                        del counts[p] # Optional cleanup
                return False

        # Main deletion loop
        # We want to remove as many chars as possible.
        # We scan from 0 to L-1.
        # If we delete at i, we restart from max(0, i-1) to catch interactions.
        # But restarting from 0 is simpler and safer.
        # Given L ~ 6000, and we expect maybe 10-50 deletions, restarting is fine.
        
        # Optimization: 
        # Instead of `for i in range(L)`, we use a while loop.
        
        i = 0
        while i < len(current_word):
            if time.time() > deadline - 5.0:
                break
            
            if try_delete(i):
                # Deletion successful, word is shorter.
                # The character that was at i+1 is now at i.
                # We should check index i again (which is the new character at i).
                # So we do NOT increment i.
                # Also, the character at i-1 might now form new windows with i.
                # So we should restart from max(0, i-1).
                i = max(0, i - 1)
            else:
                i += 1
                
            # Periodic check
            if time.time() > deadline - 5.0:
                break

    return "".join(map(str, word))
