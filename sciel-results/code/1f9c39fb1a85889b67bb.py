import time
import random
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    symbols = list(range(1, n + 1))
    
    def build_standard_superperm(perm_order):
        """
        Builds a superpermutation using the standard recursive construction.
        This gives a much shorter initial word than the naive construction.
        """
        # Standard construction: S_1 = "1"
        # S_n is built from S_{n-1} by inserting symbol n
        # For each maximal (n-1)-permutation block in S_{n-1}, we append it with n inserted
        
        # Actually, the standard construction:
        # S_1 = [1]
        # S_n: take S_{n-1}, for each (n-1)-permutation p appearing in S_{n-1} (in order of first appearance),
        #       append p with n inserted after the first element... 
        # 
        # The cleaner standard construction:
        # S_n = for each permutation pi of {1..n-1} in the order they appear in S_{n-1}:
        #         append pi with n appended, but merge with overlap
        #
        # Let me use the well-known construction:
        # S_1 = [1]
        # To get S_n from S_{n-1}: 
        #   result = []
        #   for each (n-1)-permutation window in S_{n-1} (in order):
        #     block = list(window) + [n] + list(window)
        #     merge with result (find max overlap)
        #   result = merged
        
        abstract_word = [1]
        for m in range(2, n + 1):
            # Extract all distinct (m-1)-permutations from abstract_word in order of first appearance
            perms = []
            seen = set()
            for i in range(len(abstract_word) - (m - 1) + 1):
                window = tuple(abstract_word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            new_abstract = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Find max overlap between end of new_abstract and start of block
                max_ov = min(len(new_abstract), len(block))
                overlap = 0
                for o in range(max_ov, -1, -1):
                    if o == 0 or new_abstract[len(new_abstract) - o:] == block[:o]:
                        overlap = o
                        break
                new_abstract += block[overlap:]
            abstract_word = new_abstract
        
        # Map abstract symbols to actual symbols
        mapping = {i + 1: perm_order[i] for i in range(n)}
        return [mapping[s] for s in abstract_word]
    
    def compute_perm_counts(word, n):
        perm_count = defaultdict(int)
        for i in range(len(word) - n + 1):
            p = tuple(word[i:i + n])
            perm_count[p] += 1
        return perm_count
    
    def try_delete(word, perm_count, i, n):
        L = len(word)
        if L <= n:
            return False
        
        start_j = max(0, i - n + 1)
        end_j = min(i, L - n)
        
        # Check if any affected old window is critical
        for j in range(start_j, end_j + 1):
            p = tuple(word[j:j + n])
            if perm_count[p] == 1:
                return False
        
        # Perform deletion
        for j in range(start_j, end_j + 1):
            p = tuple(word[j:j + n])
            perm_count[p] -= 1
        
        start_j_new = max(0, i - n + 1)
        end_j_new = min(i, L - 1 - n)
        
        for j in range(start_j_new, end_j_new + 1):
            new_w = []
            for k in range(n):
                idx = j + k
                if idx < i:
                    new_w.append(word[idx])
                else:
                    new_w.append(word[idx + 1])
            p_new = tuple(new_w)
            perm_count[p_new] += 1
        
        word.pop(i)
        return True
    
    def optimize(word, n, deadline):
        perm_count = compute_perm_counts(word, n)
        
        # Multi-pass deletion with randomization
        while time.time() < deadline:
            progress = False
            L = len(word)
            if L <= n:
                break
            indices = list(range(L))
            random.shuffle(indices)
            
            deletions = 0
            for idx in indices:
                if time.time() > deadline:
                    break
                current_idx = idx - deletions
                if current_idx < 0 or current_idx >= len(word):
                    continue
                if try_delete(word, perm_count, current_idx, n):
                    progress = True
                    deletions += 1
                    
            if not progress:
                break
        
        return word
    
    def local_search_swap(word, n, deadline):
        """Try swapping adjacent pairs to enable more deletions."""
        perm_count = compute_perm_counts(word, n)
        best_word = word[:]
        best_len = len(word)
        
        while time.time() < deadline:
            progress = False
            L = len(word)
            if L <= n:
                break
            # Try random adjacent swaps
            for _ in range(min(1000, L * 10)):
                if time.time() > deadline:
                    break
                i = random.randint(0, L - 2)
                # Swap word[i] and word[i+1]
                word[i], word[i + 1] = word[i + 1], word[i]
                # Try to delete
                saved = word[:]
                saved_count = dict(perm_count)
                # Try deletions on the swapped word
                # This is expensive, so let's just try a few deletions
                # Actually, let's revert and just use the deletion approach
                word[i], word[i + 1] = word[i + 1], word[i]
            # Just fall back to deletion
            progress2 = False
            L2 = len(word)
            indices = list(range(L2))
            random.shuffle(indices)
            deletions = 0
            for idx in indices:
                if time.time() > deadline:
                    break
                current_idx = idx - deletions
                if current_idx < 0 or current_idx >= len(word):
                    continue
                if try_delete(word, perm_count, current_idx, n):
                    progress2 = True
                    deletions += 1
            if not progress2:
                break
        
        return word
    
    best_word = None
    best_len = float('inf')
    
    # Multiple restarts with different symbol orderings
    while time.time() < deadline:
        perm_order = symbols[:]
        random.shuffle(perm_order)
        
        initial_word = build_standard_superperm(perm_order)
        optimized_word = optimize(initial_word, n, deadline)
        
        if len(optimized_word) < best_len:
            best_len = len(optimized_word)
            best_word = optimized_word[:]
        
        if best_len <= 5905:
            break
    
    if best_word is None:
        return "".join(map(str, build_standard_superperm(symbols)))
    
    return "".join(map(str, best_word))
