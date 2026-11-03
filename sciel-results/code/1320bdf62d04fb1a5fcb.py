import itertools
import time
import random

def construct(n):
    if n == 1:
        return "1"
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    perm_set = set(perms)
    N = len(perms)
    
    # Build classic recursive superpermutation
    def build_classic(n):
        if n == 1:
            return [1]
        word = build_classic(n - 1)
        new_word = []
        for i in range(len(word)):
            if i == 0:
                new_word.extend(word[:i+1])
                new_word.append(n)
            else:
                new_word.extend(word[i-1:i+1])
                new_word.append(n)
        new_word.extend(word)
        return new_word
    
    # Alternative: use the standard construction
    def build_standard(n):
        if n == 1:
            return [1]
        word = build_standard(n - 1)
        new_word = []
        for i in range(len(word) - n + 2):
            window = tuple(word[i:i + n - 1])
            if len(set(window)) == n - 1:
                new_word.extend(window)
                new_word.append(n)
        new_word.extend(word)
        return new_word
    
    # Use the greedy approach as starting point
    def build_greedy(n):
        perms_sorted = sorted(itertools.permutations(range(1, n + 1)))
        N_local = len(perms_sorted)
        
        prefix_to_indices = {}
        for i, p in enumerate(perms_sorted):
            for k in range(1, n):
                prefix = p[:k]
                if prefix not in prefix_to_indices:
                    prefix_to_indices[prefix] = []
                prefix_to_indices[prefix].append(i)
        
        visited = [False] * N_local
        current_word = list(perms_sorted[0])
        visited[0] = True
        
        for _ in range(N_local - 1):
            found = False
            for k in range(n - 1, 0, -1):
                suffix = tuple(current_word[-k:])
                candidates = prefix_to_indices.get(suffix, [])
                
                best_idx = -1
                best_next_symbol = n + 1
                best_perm = None
                for idx in candidates:
                    if not visited[idx]:
                        next_perm = perms_sorted[idx]
                        next_symbol = next_perm[k]
                        if next_symbol < best_next_symbol or (next_symbol == best_next_symbol and (best_perm is None or next_perm < best_perm)):
                            best_next_symbol = next_symbol
                            best_idx = idx
                            best_perm = next_perm
                
                if best_idx != -1:
                    visited[best_idx] = True
                    current_word.extend(best_perm[k:])
                    found = True
                    break
            
            if not found:
                for i in range(N_local):
                    if not visited[i]:
                        visited[i] = True
                        current_word.extend(perms_sorted[i])
                        found = True
                        break
        
        return current_word
    
    # Start with greedy construction
    word = build_greedy(n)
    best_word = list(word)
    best_len = len(word)
    
    # Function to check if a word is valid and count covered permutations
    def count_covered(word, n):
        covered = set()
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i+n])
            if len(set(window)) == n:
                covered.add(window)
        return len(covered)
    
    # Local search: try to remove redundant symbols
    def try_local_improvement(word, n, max_iterations=1000):
        word = list(word)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try removing each position
            for i in range(1, len(word) - 1):
                # Only try to remove if it doesn't break a permutation
                old_word = word
                word.pop(i)
                
                # Check if all permutations are still covered
                if count_covered(word, n) == N:
                    improved = True
                    break
                else:
                    word = old_word
            
            # Try swapping adjacent elements
            if not improved:
                for i in range(len(word) - 1):
                    old_word = list(word)
                    word[i], word[i+1] = word[i+1], word[i]
                    
                    if count_covered(word, n) == N:
                        improved = True
                        break
                    else:
                        word = old_word
        
        return word
    
    # Try local improvement
    if time.time() < deadline - 30:
        improved_word = try_local_improvement(word, n, max_iterations=500)
        if len(improved_word) < best_len and count_covered(improved_word, n) == N:
            best_word = improved_word
            best_len = len(improved_word)
    
    # Return the best word found
    return "".join(map(str, best_word))
