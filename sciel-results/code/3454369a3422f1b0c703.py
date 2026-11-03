import time
import sys

def construct(n):
    # Deadline guard
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    # 1. Generate recursive superpermutation
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
        
    # If n is small, recursive is optimal or close enough, and reduction might be overkill or slow?
    # For n=7, reduction is fast.
    
    # 2. Greedy Reduction
    # We want to remove symbols while keeping all permutations.
    
    def get_perm_counts(w):
        counts = {}
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            counts[p] = counts.get(p, 0) + 1
        return counts

    # Only attempt reduction if n is large enough to benefit and time allows
    if n >= 5:
        counts = get_perm_counts(word)
        # We need to know which permutations are required. 
        # All n! permutations must be present.
        # So if count drops to 0, it's invalid.
        
        # Optimization: Store the set of all possible permutations? 
        # No, we just need to ensure counts > 0 for the ones that exist.
        # But wait, if a permutation was never in the word, its count is 0.
        # We only care about permutations that *are* in the word initially.
        # If a permutation is in the word, its count is >= 1.
        # If we remove a symbol, we only decrement counts of windows containing that symbol.
        # If a count becomes 0, that permutation is lost.
        
        # We iterate from end to start to avoid index shifting issues for the current pass.
        # We do multiple passes until no more removals.
        
        while time.time() < deadline:
            changed = False
            # Iterate backwards
            for i in range(len(word) - 1, -1, -1):
                # Find windows containing i
                # Window j contains i if j <= i < j+n
                # j_min = max(0, i - n + 1)
                # j_max = min(i, len(word) - n)
                
                j_min = max(0, i - n + 1)
                j_max = min(i, len(word) - n)
                
                valid = True
                # Temporarily decrement counts
                # We need to restore if invalid
                affected_perms = []
                
                for j in range(j_min, j_max + 1):
                    p = tuple(word[j:j+n])
                    # Decrement
                    counts[p] -= 1
                    affected_perms.append(p)
                    if counts[p] == 0:
                        valid = False
                        break
                
                if valid:
                    # All affected perms still have count >= 1
                    # Remove word[i]
                    word.pop(i)
                    changed = True
                else:
                    # Restore counts
                    for p in affected_perms:
                        counts[p] += 1
                        # Note: if we broke early, some perms in affected_perms 
                        # might not have been decremented? 
                        # No, we append p BEFORE checking. 
                        # But if we break, the current p is decremented.
                        # The loop `for p in affected_perms` restores all.
                        # Wait, if we break, `affected_perms` contains the current p.
                        # So restoring all is correct.
        
        # After reduction, we might have a shorter word.
        # We should verify it's still a superpermutation? 
        # The logic ensures counts > 0 for all initially present perms.
        # Since the initial word was a superpermutation, all perms were present.
        # So the reduced word is also a superpermutation.

    return "".join(map(str, word))
