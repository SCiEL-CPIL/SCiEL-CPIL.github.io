import time
import math
import random

def construct(n):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Generate classic superpermutation
    word = generate_classic(n)
    
    target_count = math.factorial(n)
    
    # Local search with bitset tracking
    word = local_search_optimize(word, n, target_count, start_time, deadline)
    
    return "".join(map(str, word))

def generate_classic(n):
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1)
                          if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word

def local_search_optimize(word, n, target_count, start_time, deadline):
    """
    Optimized local search using efficient window tracking.
    Tries single deletions, block removals, and 2-opt moves to reduce length.
    """
    L = len(word)
    if L < target_count + n - 1:
        return word
    
    # Precompute factorial for ranking
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i-1] * i
    
    def perm_rank_fast(p):
        """Compute rank of permutation p (0-indexed) using Lehmer code - optimized"""
        rank = 0
        used_mask = 0
        for i in range(n):
            val = p[i]
            # Count unused values less than val using bit manipulation
            lower_mask = (1 << (val - 1)) - 1
            unused_lower = (~used_mask) & lower_mask & ((1 << n) - 1)
            count = bin(unused_lower).count('1')
            rank += count * fact[n - 1 - i]
            used_mask |= (1 << (val - 1))
        return rank
    
    def rebuild_tracking(w):
        """Rebuild window_ranks and rank_count for word w"""
        Lw = len(w)
        num_windows = Lw - n + 1
        window_ranks = [0] * num_windows
        rank_count = [0] * target_count
        
        for i in range(num_windows):
            r = perm_rank_fast(w[i:i+n])
            window_ranks[i] = r
            rank_count[r] += 1
        
        return window_ranks, rank_count
    
    def check_deletion_valid(i, window_ranks, rank_count, L):
        """Check if deleting position i is valid (conservative)"""
        j_min = max(0, i - n + 1)
        j_max = min(i, L - n)
        
        for j in range(j_min, j_max + 1):
            r = window_ranks[j]
            if rank_count[r] == 1:
                return False
        return True
    
    def check_block_removal_valid(s, k, window_ranks, rank_count, L):
        """Check if removing positions s..s+k-1 is valid (conservative)"""
        j_min = max(0, s - n + 1)
        j_max = min(s + k - 1, L - n)
        
        for j in range(j_min, j_max + 1):
            if rank_count[window_ranks[j]] == 1:
                return False
        return True
    
    best_word = word[:]
    best_len = L
    
    # Phase 1: Greedy single deletions
    for round_num in range(200):
        if time.time() > deadline - 30:  # Leave time for later phases
            break
        
        L = len(best_word)
        if L <= n:
            break
        
        window_ranks, rank_count = rebuild_tracking(best_word)
        
        # Find all valid deletion positions, try them in random order for diversity
        valid_deletions = []
        for i in range(1, L - 1):
            if check_deletion_valid(i, window_ranks, rank_count, L):
                valid_deletions.append(i)
        
        if not valid_deletions:
            break
        
        # Pick a random valid deletion (or first for determinism in early rounds)
        if round_num < 10:
            i = valid_deletions[0]
        else:
            i = valid_deletions[random.randint(0, len(valid_deletions) - 1)]
        
        best_word = best_word[:i] + best_word[i+1:]
        best_len = len(best_word)
    
    # Phase 2: Block removals
    for round_num in range(100):
        if time.time() > deadline - 15:
            break
        
        L = len(best_word)
        if L <= n + 1:
            break
        
        window_ranks, rank_count = rebuild_tracking(best_word)
        
        improved = False
        # Try larger blocks first (more impactful)
        for k in range(min(n, L - 2), 1, -1):
            if time.time() > deadline - 15:
                break
            
            for s in range(1, L - k + 1):
                if time.time() > deadline - 15:
                    break
                
                if check_block_removal_valid(s, k, window_ranks, rank_count, L):
                    best_word = best_word[:s] + best_word[s+k:]
                    improved = True
                    best_len = len(best_word)
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    # Phase 3: 2-opt style moves - move a block to a different position
    for round_num in range(50):
        if time.time() > deadline - 5:
            break
        
        L = len(best_word)
        if L <= n + 2:
            break
        
        window_ranks, rank_count = rebuild_tracking(best_word)
        
        improved = False
        # Try moving small blocks (1-3 elements) to nearby positions
        for block_size in [1, 2, 3]:
            if time.time() > deadline - 5:
                break
            
            for s in range(1, L - block_size - 1):
                if time.time() > deadline - 5:
                    break
                
                # Try moving block from s to s+block_size (swap with next element)
                # This is a local 2-opt move
                e = s + block_size
                if e >= L - 1:
                    continue
                
                # Extract block
                block = best_word[s:e]
                # New word: remove block, insert after position e
                new_word = best_word[:s] + best_word[e:]
                # Insert block at position e (which is now s + (L - block_size) - e + 1... complex)
                # Simpler: just try adjacent swaps as a form of 2-opt
                if s + 1 < L:
                    # Swap element at s with element at s+1
                    if best_word[s] != best_word[s+1]:
                        new_word = best_word[:]
                        new_word[s], new_word[s+1] = new_word[s+1], new_word[s]
                        
                        # Check validity
                        new_window_ranks, new_rank_count = rebuild_tracking(new_word)
                        all_covered = all(new_rank_count[r] > 0 for r in range(target_count))
                        
                        if all_covered:
                            best_word = new_word
                            improved = True
                            best_len = len(best_word)
                            break
            
            if improved:
                break
        
        if not improved:
            break
    
    # Final verification
    final_window_ranks, final_rank_count = rebuild_tracking(best_word)
    if all(final_rank_count[r] > 0 for r in range(target_count)):
        return best_word
    else:
        return word  # Return original if optimization broke validity
    
    return best_word
