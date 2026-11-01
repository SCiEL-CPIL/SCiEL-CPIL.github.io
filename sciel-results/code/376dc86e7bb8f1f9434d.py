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
                
        # Optimization: If word[i] == word[j], swap is no-op.
        if word[i] == word[j]:
            return False

        # Check validity: 
        # If any affected window has count 1 and its content changes, invalid.
        for k in affected_k:
            if perm_count[tuple(word[k:k+n])] == 1:
                return False
        
        # Validity confirmed. Perform swap.
        
        # Store old tuples
        old_tuples = {}
        for k in affected_k:
            old_tuples[k] = tuple(word[k:k+n])
        
        # Swap word[i] and word[j]
        word[i], word[j] = word[j], word[i]
        
        # Update counts
        for k in affected_k:
            old_p = old_tuples[k]
            perm_count[old_p] -= 1
            
            new_p = tuple(word[k:k+n])
            perm_count[new_p] += 1
            
        return True

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
        max_swaps = 1000
        for _ in range(max_swaps):
            if time.time() > deadline:
                break
            i = random.randint(0, len(word) - 1)
            j = random.randint(0, len(word) - 1)
            if try_swap(i, j):
                progress = True
        
        if not progress:
            break

    return "".join(map(str, word))
