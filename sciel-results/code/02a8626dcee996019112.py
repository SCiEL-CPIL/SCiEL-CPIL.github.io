"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import Counter

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    if n == 1:
        return "1"

    # Step 1: Build the classic superpermutation for n-1
    def build_superperm_recursive(n_val):
        if n_val == 1:
            return [1]
        word = [1]
        for m in range(2, n_val + 1):
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
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word

    base_word = build_superperm_recursive(n - 1)
    
    # Step 2: Extract all unique permutations of 1..n-1 from base_word
    perms, seen = [], set()
    for i in range(len(base_word) - (n - 2)):
        window = tuple(base_word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    m = len(perms)
    
    # Step 3: Create blocks for adding symbol n
    blocks = [list(p) + [n] + list(p) for p in perms]
    block_len = 2 * (n - 1) + 1

    # Step 4: Precompute overlap matrix
    overlap_mat = [[0] * m for _ in range(m)]
    for i in range(m):
        bi_str = "".join(map(str, blocks[i]))
        for j in range(m):
            bj_str = "".join(map(str, blocks[j]))
            max_ov = block_len - 1
            ov = 0
            for o in range(max_ov, -1, -1):
                if bi_str[-o:] == bj_str[:o]:
                    ov = o
                    break
            overlap_mat[i][j] = ov

    # Also compute overlap with the end of base_word
    base_str = "".join(map(str, base_word))
    first_ov = [0] * m
    for i in range(m):
        bi_str = "".join(map(str, blocks[i]))
        max_ov = min(len(base_str), block_len - 1)
        for o in range(max_ov, -1, -1):
            if base_str[-o:] == bi_str[:o]:
                first_ov[i] = o
                break

    # Step 5: Local Search to maximize total overlap
    order = list(range(m))
    
    def calc_total_overlap(ord):
        total = first_ov[ord[0]]
        for k in range(1, len(ord)):
            total += overlap_mat[ord[k-1]][ord[k]]
        return total

    current_overlap = calc_total_overlap(order)
    
    # 2-Opt Local Search
    improved = True
    while improved and (time.time() < deadline - 10):
        improved = False
        for i in range(m - 1):
            if time.time() > deadline - 10:
                break
            for j in range(i + 1, m):
                idx_a = i - 1
                idx_b = i
                idx_c = j
                idx_d = j + 1
                
                old_ov = 0
                new_ov = 0
                
                if idx_a >= 0:
                    old_ov += overlap_mat[order[idx_a]][order[idx_b]]
                    new_ov += overlap_mat[order[idx_a]][order[idx_c]]
                else:
                    old_ov += first_ov[order[idx_b]]
                    new_ov += first_ov[order[idx_c]]
                    
                if i < j:
                    for k in range(i, j):
                        old_ov += overlap_mat[order[k]][order[k+1]]
                    for k in range(i, j):
                        new_ov += overlap_mat[order[k+1]][order[k]]
                        
                if idx_d < m:
                    old_ov += overlap_mat[order[idx_c]][order[idx_d]]
                    new_ov += overlap_mat[order[idx_b]][order[idx_d]]
                    
                if new_ov > old_ov:
                    order[i:j+1] = reversed(order[i:j+1])
                    current_overlap = new_ov + (current_overlap - old_ov)
                    improved = True

    # Step 6: Construct the final word
    final_word = list(base_word)
    first_block_idx = order[0]
    first_block = blocks[first_block_idx]
    ov = first_ov[first_block_idx]
    final_word += first_block[ov:]
    
    for k in range(1, m):
        prev_idx = order[k-1]
        curr_idx = order[k]
        curr_block = blocks[curr_idx]
        ov = overlap_mat[prev_idx][curr_idx]
        final_word += curr_block[ov:]
        
    # Step 7: Fast validity check and greedy deletion with efficient counting
    all_perms = set(itertools.permutations(range(1, n+1)))
    
    def is_valid(w):
        if len(w) < n:
            return False
        present = set()
        for i in range(len(w) - n + 1):
            present.add(tuple(w[i:i+n]))
        return all_perms.issubset(present)
        
    if not is_valid(final_word):
        return "".join(map(str, final_word))
        
    # Efficient greedy deletion using permutation counts
    perm_counts = Counter()
    for i in range(len(final_word) - n + 1):
        window = tuple(final_word[i:i+n])
        if window in all_perms:
            perm_counts[window] += 1
    
    changed = True
    while changed and (time.time() < deadline - 2):
        changed = False
        # Scan from right to left for deletions
        for i in range(len(final_word) - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            
            # Determine which windows are affected by removing position i
            s_min = max(0, i - n + 1)
            s_max = min(i, len(final_word) - n)
            
            # Check if removing position i would lose any permutation
            safe = True
            for s in range(s_min, s_max + 1):
                window = tuple(final_word[s:s+n])
                if window in all_perms and perm_counts[window] == 1:
                    safe = False
                    break
            
            if safe:
                # Remove position i and update counts
                final_word.pop(i)
                changed = True
                
                # Update perm_counts: remove windows that contained position i
                for s in range(s_min, s_max + 1):
                    if s < len(final_word) - n + 1:
                        window = tuple(final_word[s:s+n])
                        if window in perm_counts:
                            perm_counts[window] -= 1
                            if perm_counts[window] <= 0:
                                del perm_counts[window]
                
                # Add new windows that formed after deletion
                new_s_min = max(0, i - n + 1)
                new_s_max = min(i, len(final_word) - n)
                for s in range(new_s_min, new_s_max + 1):
                    if s < len(final_word) - n + 1:
                        window = tuple(final_word[s:s+n])
                        if window in all_perms:
                            perm_counts[window] += 1
                
                break  # Restart scan as indices changed

    # Final validity check
    if not is_valid(final_word):
        # Fallback: rebuild classic construction
        word = [1]
        for m_val in range(2, n + 1):
            perms2, seen2 = [], set()
            for i in range(len(word) - m_val + 2):
                window = tuple(word[i:i + m_val - 1])
                if len(set(window)) == m_val - 1 and window not in seen2:
                    seen2.add(window)
                    perms2.append(window)
            new2 = []
            for p in perms2:
                block = list(p) + [m_val] + list(p)
                max_overlap = min(len(new2), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if new2[len(new2) - o:] == block[:o]:
                        overlap = o
                        break
                new2 += block[overlap:]
            word = new2
        return "".join(map(str, word))
    
    return "".join(map(str, final_word))

# EVOLVE-BLOCK-END
