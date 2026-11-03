import time
from collections import defaultdict
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Standard Recursive Construction
    word = [1]
    for m in range(2, n + 1):
        # Collect all unique (m-1)-permutations in the current word
        # In the standard construction, the word is built such that 
        # we iterate through the permutations of the previous step.
        # A simpler way to generate the standard superpermutation:
        pass
        
    # Let's use the explicit permutation generation for clarity and correctness
    # The standard construction:
    # S_1 = "1"
    # S_n is formed by taking S_{n-1}, and for every permutation p of 1..n-1 
    # that appears in S_{n-1}, we insert p + n + p.
    # Actually, the standard algorithm is:
    # Start with [1]
    # For m from 2 to n:
    #   new_word = []
    #   for each permutation p of 1..m-1 in the order they appear in the current word:
    #       append p
    #       append m
    #       append p
    #   # Wait, this is not quite right. 
    #   # The standard construction is:
    #   # S_n = S_{n-1} with n inserted.
    #   # Specifically, if S_{n-1} contains permutations p_1, p_2, ..., p_{(n-1)!}
    #   # then S_n is formed by:
    #   # p_1, n, p_1, p_2, n, p_2, ..., p_{(n-1)!}, n, p_{(n-1)!}
    #   # But we need to handle overlaps.
    
    # Let's implement the standard construction correctly.
    word = [1]
    for m in range(2, n + 1):
        # Extract all unique (m-1)-length permutations from word
        # The standard construction relies on the order of permutations.
        # We can just generate the standard superpermutation directly.
        
        # Standard algorithm:
        # S_1 = "1"
        # S_n: 
        #   Take S_{n-1}.
        #   For each permutation p of {1,...,n-1} in the order they appear in S_{n-1}:
        #       Append p
        #       Append n
        #       Append p
        #   But we merge overlaps.
        
        # Let's just use the known property:
        # The standard superpermutation of length L_n = n! + (n-1)! + ... + 1
        # can be generated recursively.
        
        # Here is a robust way to generate the standard one:
        # Start with list of perms of 1..m-1
        # This is getting complicated to implement from scratch perfectly in a short time.
        # Let's use the iterative expansion from the seed, which was correct.
        
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap
            max_overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    max_overlap = o
                    break
            new += block[max_overlap:]
        word = new

    # 2. Local Search to Shorten
    # Convert to list of ints for manipulation
    w = word
    
    def get_perm_counts(w, n):
        counts = defaultdict(int)
        for i in range(len(w) - n + 1):
            counts[tuple(w[i:i+n])] += 1
        return counts

    perm_counts = get_perm_counts(w, n)
    
    # We try to delete characters
    # We do multiple passes
    # To avoid infinite loops or slow progress, we can use a "simulated annealing" like approach
    # or just greedy deletion with restarts.
    
    # Since the gap is small (5913 -> 5905 is 8 chars), greedy should find them.
    
    for _ in range(100): # Max 100 passes
        if time.time() > deadline:
            break
        improved = False
        # Try deleting from right to left to avoid index shifting issues
        # Or just rebuild the list
        i = len(w) - 1
        while i >= 0:
            if time.time() > deadline:
                break
            
            # Check if deleting w[i] is safe
            # Affected windows start at k in [i-n+1, i]
            safe = True
            for k in range(max(0, i - n + 1), min(i, len(w) - n) + 1):
                p = tuple(w[k:k+n])
                if perm_counts[p] == 1:
                    safe = False
                    break
            
            if safe:
                # Update counts
                # Old windows affected: k in [i-n+1, i]
                for k in range(max(0, i - n + 1), min(i, len(w) - n) + 1):
                    p = tuple(w[k:k+n])
                    perm_counts[p] -= 1
                    if perm_counts[p] == 0:
                        del perm_counts[p]
                
                # Delete
                del w[i]
                
                # New windows affected: k in [i-n+1, i] in the NEW word
                # Note: indices shift for k > i? No, we deleted i.
                # Windows starting at k < i are unchanged in content?
                # If we delete index i, windows starting at k < i-n+1 are unchanged.
                # Windows starting at k in [i-n+1, i-1] are changed.
                # Windows starting at k >= i in new word correspond to k+1 in old word?
                # Actually, simpler: just recalculate counts for the affected region.
                # The affected region in the new word is around index i.
                # Windows starting at k in [max(0, i-n+1), min(i, len(w)-n)]
                # Wait, the new word is shorter.
                # Let's just update the counts for the new windows that overlap the deletion site.
                # The deletion site is at i.
                # New windows that include index i (or were shifted) need updating.
                # Actually, windows starting at k in [i-n+1, i] in the NEW word are the ones changed.
                # Note: The new word has length len(w).
                # The windows that were valid before and are still valid are those not overlapping i.
                # We already decremented the old ones.
                # Now we need to increment the new ones.
                # The new windows are formed by w[k:k+n] for k in [max(0, i-n+1), min(i, len(w)-n)]
                # BUT, we must be careful. The windows that did NOT overlap the deletion are unchanged.
                # The windows that DID overlap the deletion in the old word are gone.
                # The windows that are NEW (formed by the shift) are those starting at k in the new word
                # that correspond to the region around the deletion.
                
                # Let's just re-calculate the counts for the affected indices in the NEW word.
                # Affected indices in new word: [max(0, i-n+1), min(i, len(w)-n)]
                # We need to add these back to perm_counts.
                for k in range(max(0, i - n + 1), min(i, len(w) - n) + 1):
                    p = tuple(w[k:k+n])
                    perm_counts[p] += 1
                    
                improved = True
                # Do not decrement i, because the element at i is now the one that was at i+1
                # And we need to check it again (or the one before it)
                # Actually, deleting i means the new element at i is old i+1.
                # We should check i again.
            else:
                i -= 1
        
        if not improved:
            break

    return "".join(map(str, w))
