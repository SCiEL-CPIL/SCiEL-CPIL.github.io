import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234123142312431214321"

    # Phase 1: Greedy Construction
    # Start with "1"
    # We want to build a string such that every permutation appears.
    # Greedy strategy: At each step, append the smallest digit d in 1..n
    # such that the last n-1 digits + d form a new permutation.
    # If no such digit exists, we are stuck (should not happen for small n if we are careful, 
    # but for n>=4 simple greedy might get stuck or produce non-optimal length).
    # Actually, simple greedy "smallest new perm" is known to work and produce short superperms.
    
    seen = set()
    # Initialize with the first permutation? 
    # Let's start with empty string and build up.
    # But we need a window of size n.
    # A common start is "12...n".
    initial = list(range(1, n+1))
    word = initial[:]
    seen.add(tuple(initial))
    
    # We need to add more permutations.
    # The current suffix is word[-(n-1):]
    # We try to append a digit.
    
    # This simple greedy might get stuck. 
    # A more robust greedy is to use the recursive structure but with a twist, 
    # or just use the standard recursive construction as a fallback if greedy fails.
    # However, for n=7, the standard recursive is 5913.
    # Let's try the "Lexicographic Greedy" which is known to be better.
    
    # Reset
    word = []
    seen = set()
    
    # Start with 1
    word.append(1)
    
    # We need to form n-1 more digits to get the first permutation?
    # No, we build the string.
    # Let's use a known effective greedy:
    # Start with 1.
    # While we haven't seen all perms:
    #   For d in 1..n:
    #      If len(word) >= n-1:
    #         candidate_window = tuple(word[-(n-1):] + [d])
    #         If len(set(candidate_window)) == n and candidate_window not in seen:
    #             word.append(d)
    #             seen.add(candidate_window)
    #             break
    #   If no d found: 
    #       # Stuck. This can happen.
    #       # Fallback: Use recursive construction.
    #       pass

    # The simple greedy above often gets stuck for n>=4 because it doesn't plan ahead.
    # A better constructive method is the "Standard Recursive" but optimized.
    # Or, we can use the recursive construction as the base, and then heavily optimize.
    # The recursive construction is deterministic and fast.
    # Length: 5913 for n=7.
    # Optimization via deletion can easily remove ~10-20 chars.
    
    # Let's stick to the Recursive Construction for Phase 1 because it's reliable and fast.
    # Then Phase 2 does the heavy lifting with Local Search.
    
    word = "1"
    for m in range(2, n + 1):
        if time.time() > deadline - 50:
            break
        # Extract all (m-1)-permutations from the current word
        perms = []
        seen_p = set()
        for i in range(len(word) - (m - 1) + 1):
            window = word[i:i + m - 1]
            if len(set(window)) == m - 1:
                if window not in seen_p:
                    seen_p.add(window)
                    perms.append(window)
        
        new_word = ""
        for p in perms:
            # Block is p + str(m) + p
            # But we can overlap.
            block = p + str(m) + p
            max_overlap = 0
            limit = min(len(new_word), len(block))
            for o in range(limit, 0, -1):
                if new_word.endswith(block[:o]):
                    max_overlap = o
                    break
            new_word += block[max_overlap:]
        word = new_word

    # Convert to list of integers for faster manipulation
    arr = [int(c) for c in word]
    
    if time.time() > deadline - 20:
        return word

    # Phase 2: Optimization via Local Search (Deletion)
    # We maintain a count of each permutation.
    # If we remove a character, only n permutations are affected.
    
    all_perms_set = set(permutations(range(1, n + 1)))
    target_count = len(all_perms_set)
    
    # Initial counts
    perm_count = {}
    for i in range(len(arr) - n + 1):
        p = tuple(arr[i:i + n])
        if p in all_perms_set:
            perm_count[p] = perm_count.get(p, 0) + 1
    
    # If we don't have all perms, something went wrong in construction (unlikely)
    if len(perm_count) < target_count:
         # Fallback: return what we have
         return word

    best_arr = arr[:]
    best_len = len(arr)
    
    # Local Search Loop
    # We try to delete characters.
    # Strategy: 
    # 1. Pass from left to right, deleting if possible.
    # 2. Pass from right to left.
    # 3. Random passes.
    
    # To speed up, we can check validity in O(n).
    # To delete arr[i], we need to check the n windows that cover i.
    # Windows start at j such that j <= i <= j+n-1 => i-n+1 <= j <= i.
    # Also j >= 0 and j <= len(arr)-n.
    
    def try_delete(arr, i, perm_count, n, all_perms_set):
        # Returns True if deletion is valid, and updates perm_count if so.
        # We must revert if invalid.
        
        start_j = max(0, i - n + 1)
        end_j = min(i, len(arr) - n)
        
        affected = []
        valid = True
        
        # Check if any affected perm has count 1
        # Actually, we need to see if the count becomes 0.
        # So we check current counts.
        for j in range(start_j, end_j + 1):
            p = tuple(arr[j:j + n])
            if p in all_perms_set:
                c = perm_count.get(p, 0)
                if c == 1:
                    valid = False
                    break
                affected.append(p)
        
        if not valid:
            return False
        
        # Perform deletion
        # Decrement counts
        for p in affected:
            perm_count[p] -= 1
            if perm_count[p] == 0:
                del perm_count[p]
        
        # Remove from arr
        arr.pop(i)
        
        return True

    # Optimization Loop
    # We use a "greedy" pass first.
    # Then random.
    
    # To avoid O(N) passes taking too long if N is large (5913 is small), 
    # we can just do many random deletions.
    # But greedy passes are very effective at the start.
    
    # Greedy Pass 1: Left to Right
    i = 0
    while i < len(arr):
        if time.time() > deadline - 5:
            break
        if try_delete(arr, i, perm_count, n, all_perms_set):
            # Length decreased, don't increment i (check same index again)
            if len(arr) < best_len:
                best_len = len(arr)
                best_arr = arr[:]
        else:
            i += 1
            
    # Greedy Pass 2: Right to Left
    i = len(arr) - 1
    while i >= 0:
        if time.time() > deadline - 5:
            break
        if try_delete(arr, i, perm_count, n, all_perms_set):
            i -= 1
            if len(arr) < best_len:
                best_len = len(arr)
                best_arr = arr[:]
        else:
            i -= 1

    # Random Search
    # Pick random indices to try deleting.
    # This helps escape local minima where greedy gets stuck.
    
    # We can also try "swaps" if deletions stop, but deletions are the primary goal.
    # Let's stick to deletions for simplicity and speed.
    
    while time.time() < deadline - 5:
        # Try a batch of random deletions
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            if len(arr) <= n:
                break
            i = random.randint(0, len(arr) - 1)
            if try_delete(arr, i, perm_count, n, all_perms_set):
                if len(arr) < best_len:
                    best_len = len(arr)
                    best_arr = arr[:]
                    # Optional: Restart greedy passes if we made significant progress?
                    # For now, just continue.

    return ''.join(str(x) for x in best_arr)
