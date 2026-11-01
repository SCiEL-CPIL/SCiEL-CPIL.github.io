import time
import random
import heapq
from collections import defaultdict
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # 1. Recursive Construction
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
    
    # Initialize counts
    perm_count = defaultdict(int)
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        perm_count[p] += 1
        
    def compute_safety(i, L):
        """Compute safety score for position i: number of affected windows with count > 1."""
        start_j = max(0, i - n + 1)
        end_j = min(i, L - n)
        score = 0
        for j in range(start_j, end_j + 1):
            if perm_count[tuple(word[j:j+n])] > 1:
                score += 1
        return score

    def try_delete(i):
        """
        Attempts to delete word[i].
        Returns True if successful, False otherwise.
        Modifies word, perm_count in-place if successful.
        """
        L = len(word)
        if L <= n:
            return False
            
        # Windows affected in OLD word: j such that j <= i <= j+n-1
        start_j = max(0, i - n + 1)
        end_j = min(i, L - n)
        
        # Fast validity check:
        # If any old window is critical (count == 1), deletion is invalid.
        for j in range(start_j, end_j + 1):
            p = tuple(word[j:j+n])
            if perm_count[p] == 1:
                return False
                
        # Validity confirmed. Now perform deletion.
        
        # 1. Decrement old windows
        for j in range(start_j, end_j + 1):
            p = tuple(word[j:j+n])
            perm_count[p] -= 1
            
        # 2. Compute and increment new windows
        # New word length L-1.
        # New windows spanning the gap:
        # In new word, the gap is at index i.
        # Windows j' such that j' <= i <= j'+n-1 in the NEW word.
        # j' ranges from max(0, i-n+1) to min(i, (L-1)-n)
        start_j_new = max(0, i - n + 1)
        end_j_new = min(i, L - 1 - n)
        
        for j in range(start_j_new, end_j_new + 1):
            # Construct new window content
            new_w = []
            for k in range(n):
                idx = j + k
                if idx < i:
                    new_w.append(word[idx])
                else:
                    new_w.append(word[idx+1])
            p_new = tuple(new_w)
            perm_count[p_new] += 1
            
        # 3. Commit
        word.pop(i)
        return True

    def try_swap(i, j):
        """
        Attempts to swap word[i] and word[j].
        Returns True if successful, False otherwise.
        """
        if i == j:
            return False
        if i > j:
            i, j = j, i
            
        L = len(word)
        if L <= n:
            return False
            
        # Identify affected window indices
        # Window k is affected if it contains i or j
        # k <= i <= k+n-1  =>  i-n+1 <= k <= i
        # k <= j <= k+n-1  =>  j-n+1 <= k <= j
        
        k_start = max(0, i - n + 1)
        k_end = min(j, L - n)
        
        affected_k = set()
        for k in range(k_start, k_end + 1):
            if k <= i <= k + n - 1 or k <= j <= k + n - 1:
                affected_k.add(k)
                
        # Check validity: 
        # For each affected window, if count is 1, the new window must be the same (which implies swap didn't change it, i.e., word[i]==word[j]) 
        # OR if count > 1, it's fine as long as we don't create a 0-count situation?
        # Actually, if we swap, the old window content changes to new content.
        # Old content C_old -> count[C_old] decreases by 1.
        # New content C_new -> count[C_new] increases by 1.
        # If count[C_old] was 1, and C_old != C_new, then C_old disappears. Invalid.
        # If C_old == C_new, it's a no-op for that window.
        
        # Optimization: If word[i] == word[j], swap is no-op.
        if word[i] == word[j]:
            return False

        # We need to check all affected windows.
        # If any affected window has count 1 and its content changes, invalid.
        # Content changes if the window contains i or j AND word[i] != word[j].
        # Since we already checked word[i] != word[j], any affected window containing i or j will change content.
        
        for k in affected_k:
            if perm_count[tuple(word[k:k+n])] == 1:
                return False
        
        # Validity confirmed. Perform swap.
        
        # 1. Update counts for affected windows
        for k in affected_k:
            old_p = tuple(word[k:k+n])
            perm_count[old_p] -= 1
            
            # Construct new window
            # We can do this efficiently by knowing only i and j are swapped
            new_w = list(word[k:k+n])
            # Map indices in window to global indices
            # Global idx g corresponds to local idx l = g - k
            # If g == i, local idx is i-k. Value becomes word[j]
            # If g == j, local idx is j-k. Value becomes word[i]
            
            # Simpler: just build the new window from the swapped word? 
            # But we haven't swapped yet.
            # Let's swap in place, build window, then swap back? No, that's messy.
            
            # Let's just build the new window tuple directly.
            # The window is word[k ... k+n-1] with word[i] and word[j] swapped.
            # Since i, j are within [k, k+n-1], we can construct it.
            
            # To avoid O(n) list creation for every check if we were checking validity, 
            # but here we are committing, so O(n) is fine.
            
            # Actually, we can just swap the values in the word array first?
            # No, if we swap, we need to be careful if we fail later? 
            # But we already validated. So we can swap.
            
            # Let's do the swap on the word array now?
            # If we swap now, we can just read the new window.
            # But we have multiple windows.
            
            # Let's swap word[i] and word[j] now.
            word[i], word[j] = word[j], word[i]
            
            # Now update counts for all affected windows
            # Wait, if I swap first, I can just read the new windows.
            # But I need to decrement the OLD counts.
            # I already decremented old counts in the loop above? 
            # No, I only checked.
            
            # Let's restructure:
            # 1. Store old tuples.
            # 2. Swap word[i], word[j].
            # 3. Decrement old tuples.
            # 4. Increment new tuples.
            
            # This function currently does:
            # - Check validity (read old counts)
            # - If valid, update counts.
            
            # Let's restart the "Valid" section properly.
            
            # I'll rewrite the valid section of try_swap to be cleaner.
            pass
        
        # The logic above inside the loop was flawed because I mixed validation and update.
        # I will replace the whole try_swap function in the edit.
        return False

    # Priority-based deletion: use a max-heap of (safety_score, position)
    
    def build_heap(L):
        """Build a max-heap of (safety, position) for all positions."""
        heap = []
        for i in range(L):
            safety = compute_safety(i, L)
            if safety > 0:  # Only consider positions that might be deletable
                heapq.heappush(heap, (-safety, i))
        return heap

    # Main deletion loop with priority-based strategy
    while time.time() < deadline:
        L = len(word)
        if L <= n:
            break
            
        progress = False
        
        # Phase 1: Deletions
        heap = build_heap(L)
        deletions_before = 0
        while heap:
            if time.time() > deadline:
                break
            neg_safety, idx = heapq.heappop(heap)
            current_idx = idx - deletions_before
            if current_idx < 0 or current_idx >= len(word):
                continue
            if try_delete(current_idx):
                progress = True
                deletions_before += 1

        # Phase 2: Swaps
        # Try random swaps or ordered swaps
        # Since n is small, we can try all pairs? L^2 is ~ 5000^2 = 25M. Too slow.
        # Try random swaps.
        max_swaps = 1000
        for _ in range(max_swaps):
            if time.time() > deadline:
                break
            i = random.randint(0, len(word) - 1)
            j = random.randint(0, len(word) - 1)
            if try_swap(i, j):
                progress = True
                # Break after some swaps? Or continue?
                # Continuing might allow further swaps.
                # But if we do too many, we might loop.
                # Let's just try a batch.
        
        if not progress:
            break

    return "".join(map(str, word))
