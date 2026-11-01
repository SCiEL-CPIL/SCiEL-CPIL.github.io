"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import Counter

def construct(n):
    if n == 1:
        return "1"
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Classic recursive construction of superpermutation
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # For n <= 4, the classic construction is optimal, return immediately
    if n <= 4:
        return "".join(map(str, word))
    
    # For n >= 5, attempt greedy deletion to shorten the word
    # Precompute all permutations of 1..n
    all_perms = set(itertools.permutations(range(1, n + 1)))
    all_perms_set = set(all_perms)  # For fast lookup
    
    # Count how many times each permutation appears as a window
    perm_counts = Counter()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        perm_counts[window] += 1
    
    # Helper to check if a word is valid
    def is_valid(word_list):
        if len(word_list) < n:
            return False
        seen = set()
        for i in range(len(word_list) - n + 1):
            seen.add(tuple(word_list[i:i + n]))
        return all_perms_set.issubset(seen)
    
    # Efficient greedy deletion with batch processing
    def greedy_delete(word, perm_counts, max_iterations=1000):
        """Aggressively delete positions that don't uncover any permutation."""
        deleted_count = 0
        for _ in range(max_iterations):
            if time.time() - start_time > deadline - 5:
                break
            
            found_deletion = False
            # Scan from left to right, try to delete each position
            for i in range(len(word)):
                if time.time() - start_time > deadline - 5:
                    break
                
                # Determine which windows are affected by removing position i
                s_min = max(0, i - n + 1)
                s_max = min(i, len(word) - n)
                
                # Check if removing position i would uncover any permutation
                safe = True
                for s in range(s_min, s_max + 1):
                    window = tuple(word[s:s + n])
                    if window in all_perms_set and perm_counts[window] == 1:
                        safe = False
                        break
                
                if safe:
                    # Remove position i and update counts
                    word.pop(i)
                    deleted_count += 1
                    found_deletion = True
                    
                    # Update perm_counts: remove windows that contained position i
                    # Windows starting from s_min to s_max are affected
                    for s in range(s_min, s_max + 1):
                        window = tuple(word[s:s + n])
                        if window in perm_counts:
                            perm_counts[window] -= 1
                            if perm_counts[window] <= 0:
                                del perm_counts[window]
                    
                    # Add new windows that formed after deletion
                    # Windows starting from max(0, i-n+1) to min(i, len(word)-n)
                    new_s_min = max(0, i - n + 1)
                    new_s_max = min(i, len(word) - n)
                    for s in range(new_s_min, new_s_max + 1):
                        window = tuple(word[s:s + n])
                        perm_counts[window] += 1
                    
                    break  # Restart scan as indices changed
            
            if not found_deletion:
                break
        
        return deleted_count
    
    # Apply greedy deletion
    greedy_delete(word, perm_counts, max_iterations=5000)
    
    # Local search: try to swap adjacent elements to create more deletion opportunities
    def local_search_swap(word, perm_counts, max_swaps=100):
        """Try swapping adjacent elements to enable more deletions."""
        improved = 0
        for _ in range(max_swaps):
            if time.time() - start_time > deadline - 5:
                break
            
            best_swap = None
            best_deletions = 0
            
            # Try a sample of adjacent swaps
            sample_positions = list(range(len(word) - 1))
            import random
            random.shuffle(sample_positions)
            sample_positions = sample_positions[:min(50, len(sample_positions))]
            
            for i in sample_positions:
                if time.time() - start_time > deadline - 5:
                    break
                
                # Try swapping word[i] and word[i+1]
                word[i], word[i+1] = word[i+1], word[i]
                
                # Check if this swap enables more deletions
                # Count how many positions can be deleted after this swap
                deletable = 0
                for j in range(len(word)):
                    s_min = max(0, j - n + 1)
                    s_max = min(j, len(word) - n)
                    
                    safe = True
                    for s in range(s_min, s_max + 1):
                        window = tuple(word[s:s + n])
                        if window in all_perms_set and perm_counts[window] == 1:
                            safe = False
                            break
                    
                    if safe:
                        deletable += 1
                
                if deletable > best_deletions:
                    best_deletions = deletable
                    best_swap = i
                else:
                    # Revert the swap
                    word[i], word[i+1] = word[i+1], word[i]
            
            # Apply the best swap if it's beneficial
            if best_swap is not None and best_deletions > 0:
                word[best_swap], word[best_swap+1] = word[best_swap+1], word[best_swap]
                improved += 1
                
                # Update perm_counts after swap
                # Recompute counts for affected region
                affected_start = max(0, best_swap - n + 1)
                affected_end = min(best_swap + n, len(word) - 1)
                
                # Remove old counts
                for s in range(affected_start, affected_end + 1):
                    if s < len(word) - n + 1:
                        window = tuple(word[s:s + n])
                        if window in perm_counts:
                            perm_counts[window] -= 1
                            if perm_counts[window] <= 0:
                                del perm_counts[window]
                
                # Add new counts
                for s in range(affected_start, affected_end + 1):
                    if s < len(word) - n + 1:
                        window = tuple(word[s:s + n])
                        perm_counts[window] += 1
                
                # Run greedy deletion again
                greedy_delete(word, perm_counts, max_iterations=1000)
            else:
                break
        
        return improved
    
    # Apply local search
    if time.time() - start_time < deadline - 10:
        local_search_swap(word, perm_counts, max_swaps=50)
    
    # Final greedy deletion pass
    if time.time() - start_time < deadline - 5:
        greedy_delete(word, perm_counts, max_iterations=2000)
    
    # Verify validity before returning
    if not is_valid(word):
        # Fallback: return the original classic construction
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
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
