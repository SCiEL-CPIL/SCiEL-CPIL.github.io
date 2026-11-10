import time
import random
from itertools import permutations

def standard_superperm(n):
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
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial solution
    best_word = standard_superperm(n)
    best_len = len(best_word)
    
    # Helper to check if a removal is valid
    def is_valid_removal(word, idx, n):
        L = len(word)
        if L - 1 < n:
            return False
        # Windows affected: start s such that s <= idx < s+n
        # s_min = max(0, idx - n + 1)
        # s_max = min(L - n, idx)
        
        # We need to check if any permutation count drops to 0.
        # Instead of maintaining counts globally, we can just check if the removed windows are unique.
        # But "unique" depends on the rest of the word.
        # So we must maintain counts.
        
        # This function is called inside the loop, so it's better to have a class or closure with state.
        # But for simplicity in the restart loop, we can just rebuild counts if needed?
        # No, rebuilding counts is O(L). Removal is O(1) if we have counts.
        
        # Let's restructure: maintain counts in the outer loop.
        pass

    # Better approach: 
    # Maintain word and perm_counts.
    # Try removals.
    
    def try_improve(word, n):
        L = len(word)
        # Build counts
        counts = {}
        for i in range(L - n + 1):
            w = tuple(word[i:i+n])
            counts[w] = counts.get(w, 0) + 1
        
        improved = True
        while improved:
            improved = False
            # Try removing from left to right
            for i in range(L):
                # Check if removal of i is valid
                # Affected windows: s in [max(0, i-n+1), min(L-n, i)]
                s_min = max(0, i - n + 1)
                s_max = min(L - n, i)
                
                valid = True
                removed_windows = []
                for s in range(s_min, s_max + 1):
                    w = tuple(word[s:s+n])
                    if counts[w] == 1:
                        valid = False
                        break
                    counts[w] -= 1
                    removed_windows.append(w)
                
                if valid:
                    # Perform removal
                    word = word[:i] + word[i+1:]
                    L -= 1
                    # Restore counts for the windows that are now "new" or affected?
                    # Actually, removing i shifts indices.
                    # The windows that remain are the same, except the ones removed are gone.
                    # But the windows that *were* affected and *are* still present might have changed?
                    # No, the content of windows not containing i is unchanged.
                    # The windows containing i are removed.
                    # So we just need to ensure we don't decrement counts for windows that are now out of bounds or shifted.
                    # Wait, if we remove i, the windows starting at s > i shift left by 1.
                    # Their content is unchanged.
                    # The windows starting at s <= i are either removed or their content changes?
                    # If s <= i, the window includes index i. So it is removed.
                    # So all affected windows are removed.
                    # No new windows are created.
                    # So the counts for remaining windows are correct.
                    
                    # We need to update L.
                    improved = True
                    break # Restart scan
            if time.time() > deadline:
                break
        
        return word

    # Run improvement
    current_word = standard_superperm(n)
    current_word = try_improve(current_word, n)
    
    if len(current_word) < best_len:
        best_len = len(current_word)
        best_word = current_word

    # If time permits, try random restarts
    while time.time() < deadline:
        # Perturb or just restart standard?
        # Standard is deterministic.
        # Maybe randomize the permutation order in the construction?
        # But standard construction relies on the specific order.
        
        # Let's just try the removal heuristic again with a different random seed for tie-breaking?
        # Or just return best found so far.
        
        # Given the small gap (8 chars), removal might be sufficient.
        # If not, we might need a more complex search.
        
        # Let's add a simple random swap local search?
        # Swap two adjacent chars. If valid and length same, no gain.
        # We need to reduce length.
        
        # So only removal helps.
        
        # If we are stuck, we might just return the best.
        break

    return "".join(map(str, best_word))
