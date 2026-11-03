import time
import random
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # 1. Generate standard superpermutation
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
            # Find max overlap
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if n < 3:
        return "".join(map(str, word))

    # Optimization: Deterministic Greedy Deletion
    best_word = word
    best_len = len(word)
    
    def get_perm_indices(w):
        perm_indices = defaultdict(list)
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                perm_indices[window].append(i)
        return perm_indices

    def try_greedy_deletion(w, order="right"):
        """
        Attempts to shorten w by greedily removing safe characters.
        Returns the shortened word.
        """
        w = list(w)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            perm_indices = get_perm_indices(w)
            L = len(w)
            
            # Determine order of indices to check
            if order == "right":
                indices_to_check = range(L - 1, -1, -1)
            elif order == "left":
                indices_to_check = range(L)
            else: # random
                indices_to_check = list(range(L))
                random.shuffle(indices_to_check)
            
            for i in indices_to_check:
                if time.time() >= deadline:
                    break
                
                # Check if index i is safe to remove
                # Affected windows start at k such that max(0, i-n+1) <= k <= min(i, L-n)
                start_k = max(0, i - n + 1)
                end_k = min(i, L - n)
                
                if start_k > end_k:
                    # No windows affected (shouldn't happen for valid i in range)
                    # But if it does, it's safe?
                    # Actually if i is outside any window, it's not part of any permutation, so safe.
                    # But in a superpermutation, every char is part of at least one window.
                    pass

                at_risk = False
                for k in range(start_k, end_k + 1):
                    p = tuple(w[k:k + n])
                    if len(set(p)) == n:
                        # Check if p appears anywhere else
                        indices = perm_indices[p]
                        # If all indices of p are in [start_k, end_k], it is at risk
                        has_safe_index = False
                        for idx in indices:
                            if idx < start_k or idx > end_k:
                                has_safe_index = True
                                break
                        if not has_safe_index:
                            at_risk = True
                            break
                
                if not at_risk:
                    # Perform deletion
                    w = w[:i] + w[i+1:]
                    improved = True
                    # Restart the loop to re-evaluate safety of other indices
                    # This is safe because we recompute perm_indices
                    break
            
        return w

    # Try different orders to escape local minima
    # The greedy deletion is order-dependent.
    orders = ["right", "left", "random", "random", "random"]
    
    for order in orders:
        if time.time() >= deadline:
            break
        candidate = try_greedy_deletion(best_word, order)
        if len(candidate) < best_len:
            best_len = len(candidate)
            best_word = candidate
    
    # If we still have time, we can try to polish best_word with more random restarts
    # but the deterministic passes usually find the global minimum for this specific gap.
    
    return "".join(map(str, best_word))
