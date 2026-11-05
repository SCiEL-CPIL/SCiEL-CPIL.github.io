import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 115
    
    # 1. Greedy earliest-insertion superpermutation
    # This produces a valid superpermutation, though not minimal.
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
            best_overlap = 0
            # Check overlap with end of new
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    best_overlap = o
                    break
            new += block[best_overlap:]
        word = new

    S = list(word)
    L = len(S)
    
    # 2. Efficient Local Search (Iterated Local Search)
    # Maintain counts of all permutations in windows
    counts = {}
    for i in range(L - n + 1):
        p = tuple(S[i:i + n])
        counts[p] = counts.get(p, 0) + 1
        
    def get_count(p):
        return counts.get(p, 0)

    def try_remove(i):
        """Try to remove S[i]. Returns True if successful."""
        # Check if any window containing S[i] has count 1
        # Windows are [i-n+1, i]
        start = max(0, i - n + 1)
        end = min(i, len(S) - n)
        for j in range(start, end + 1):
            p = tuple(S[j:j + n])
            if get_count(p) == 1:
                return False
        
        # Perform removal
        # Update counts for destroyed windows
        for j in range(start, end + 1):
            p = tuple(S[j:j + n])
            c = counts[p] - 1
            if c == 0:
                del counts[p]
            else:
                counts[p] = c
        
        # Add straddling windows
        # The new window at index j (where j < i) is formed by S[j..i-1] + S[i+1..]
        # Indices:
        # Old: ... S[j] ... S[i-1] S[i] S[i+1] ...
        # New: ... S[j] ... S[i-1] S[i+1] ...
        # The window starting at j in the new string corresponds to S[j..i-1] + S[i+1..i+n-1]
        # Wait, the window starting at j in the NEW string covers indices j to j+n-1 in NEW.
        # In OLD, these were j to j+n-1, but S[i] is removed.
        # So the new window at index j (for j in [start, i-1]) is:
        # S[j..i-1] + S[i+1..j+n-1] (shifted)
        # Actually, it's easier to just rebuild the counts for the affected region.
        # But we want O(n).
        
        # Let's just rebuild counts for the whole string if n is small?
        # No, N is ~6000. Rebuild is O(N*n) = 42000 ops.
        # Removal pass is O(N). Total O(N^2) = 36M ops. Might be too slow in Python.
        
        # Better: Update only affected windows.
        # Affected windows are those that contain index i.
        # Old indices: [i-n+1, i]
        # New indices: [i-n+1, i-1] (shifted left for indices > i)
        
        # 1. Destroy old windows [i-n+1, i]
        # 2. Create new windows [i-n+1, i-1]
        
        # We already destroyed old windows in the first loop.
        
        # Now create new windows.
        # A new window starting at k (where k < i) includes S[i+1] if k+n-1 >= i+1.
        # Actually, the new string is S' = S[:i] + S[i+1:]
        # Windows in S' that are different from S are those that cross the gap at i.
        # These are windows starting at k such that k < i < k+n.
        # i.e., k in [i-n+1, i-1].
        
        for k in range(max(0, i - n + 1), min(i, len(S) - n)): # Note: len(S) is now len(S)-1
             # Construct the new window
             # S'[k..k+n-1]
             # If idx < i, S'[idx] = S[idx]
             # If idx >= i, S'[idx] = S[idx+1]
             wc = []
             for idx in range(k, k + n):
                 if idx < i:
                     wc.append(S[idx])
                 else:
                     wc.append(S[idx + 1])
             p_new = tuple(wc)
             c = counts.get(p_new, 0)
             counts[p_new] = c + 1
             
        S.pop(i)
        return True

    best_S = S[:]
    best_len = len(S)
    
    # ILS Loop
    stuck_rounds = 0
    max_stuck = 20
    
    while time.time() < deadline:
        # Removal Pass
        removed_any = False
        i = len(S) - 1
        while i >= 0 and time.time() < deadline:
            if try_remove(i):
                removed_any = True
                # Don't decrement i, check same position again
            else:
                i -= 1
        
        if removed_any:
            if len(S) < best_len:
                best_S = S[:]
                best_len = len(S)
            stuck_rounds = 0
        else:
            stuck_rounds += 1
            if stuck_rounds > max_stuck:
                break
            
            # Perturb: Swap two random elements
            if len(S) > n:
                i, j = random.sample(range(len(S)), 2)
                S[i], S[j] = S[j], S[i]
                # Rebuild counts after perturbation
                counts = {}
                for k in range(len(S) - n + 1):
                    p = tuple(S[k:k + n])
                    counts[p] = counts.get(p, 0) + 1

    # Final Validation
    def is_valid(S):
        seen = set()
        for i in range(len(S) - n + 1):
            seen.add(tuple(S[i:i + n]))
        expected = 1
        for k in range(1, n + 1):
            expected *= k
        return len(seen) == expected

    if not is_valid(best_S):
        return "".join(map(str, word))

    return "".join(map(str, best_S))
