import numpy as np
import time
import random

def construct(q, n, radius):
    # Parameters specific to the task instance are likely q=3, n=8, radius=1
    # But we write generally for radius=1 as per the problem statement "at most one position"
    # The problem statement says "differs ... in at most one position", so radius=1.
    
    total = q ** n
    cov_size = 1 + n * (q - 1)
    
    # Precompute neighbors for all indices
    # This is expensive for large total, but 3^8 = 6561 is small.
    # neighbors[i] is a list of indices covered by codeword i.
    powers = [q ** (n - 1 - pos) for pos in range(n)]
    
    neighbors = []
    for i in range(total):
        digits = [0] * n
        tmp = i
        for pos in range(n - 1, -1, -1):
            digits[pos] = tmp % q
            tmp //= q
            
        nbrs = [i]
        for pos in range(n):
            orig_val = digits[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_idx = i + (new_val - orig_val) * powers[pos]
                nbrs.append(new_idx)
        neighbors.append(nbrs)
        
    # 1. Initial Greedy Solution
    uncovered = np.ones(total, dtype=bool)
    selected = []
    
    # Precompute coverage matrix for greedy speed? 
    # Or just use the neighbors list.
    # Greedy: pick codeword covering most uncovered.
    
    # To speed up greedy, we can maintain counts.
    # But for 6561 items, simple numpy is fine.
    
    # We will use the neighbors list.
    # For each candidate, count how many uncovered neighbors it has.
    
    # Optimization: Use a numpy array for uncovered status.
    # We need to efficiently find the best candidate.
    
    # Let's build the coverage matrix once, it's small (6561 x 25 for q=3,n=8).
    # cov_size = 1 + 8*2 = 17.
    # Matrix size 6561 * 17 ~ 111k entries. Very small.
    
    coverage_matrix = np.zeros((total, cov_size), dtype=np.int32)
    for i in range(total):
        coverage_matrix[i] = neighbors[i]
        
    # Greedy Pass
    while uncovered.any():
        # Count uncovered covered by each candidate
        # uncovered[coverage_matrix] gives boolean matrix
        # Sum along axis 1
        counts = uncovered[coverage_matrix].sum(axis=1)
        
        max_count = np.max(counts)
        if max_count == 0:
            break
            
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[random.randint(0, len(candidates) - 1)]
        
        selected.append(best_idx)
        uncovered[coverage_matrix[best_idx]] = False
        
    # 2. Local Search: Redundancy Removal and Swaps
    # Maintain coverage counts
    cov_counts = np.zeros(total, dtype=np.int32)
    selected_set = set(selected)
    
    for idx in selected:
        for nb in neighbors[idx]:
            cov_counts[nb] += 1
            
    start_time = time.time()
    time_limit = 115.0
    
    while time.time() - start_time < time_limit:
        # Step A: Remove redundant codewords
        removed_any = False
        # Find codewords that can be removed
        # A codeword c can be removed if for all nb in neighbors[c], cov_counts[nb] > 1
        
        # We can check this efficiently.
        # For each c in selected, check min(cov_counts[neighbors[c]])
        
        # To make this fast, we can iterate.
        # Since selected is small (~400), and neighbors are ~17, this is fast.
        
        # We'll collect removable codewords
        removable = []
        for c in selected:
            # Check if all neighbors have count > 1
            # If any neighbor has count == 1, c is essential (cannot be removed)
            # If all neighbors have count > 1, c is redundant
            min_count = np.min(cov_counts[neighbors[c]])
            if min_count > 1:
                removable.append(c)
                
        if removable:
            # Remove one random removable codeword
            c_to_remove = random.choice(removable)
            selected.remove(c_to_remove)
            selected_set.remove(c_to_remove)
            for nb in neighbors[c_to_remove]:
                cov_counts[nb] -= 1
            removed_any = True
            continue # Restart the loop to check for more removals
            
        # Step B: Try Swaps
        # If no redundant codewords, try to swap one codeword with a random non-codeword
        # to potentially create redundancy.
        
        if not selected:
            break
            
        c_old = random.choice(selected)
        # Pick a random non-codeword
        # We can just pick a random index from 0..total-1 and check if it's in selected_set
        # To avoid infinite loop if selected is large, we can sample.
        
        # A better swap strategy: 
        # Pick c_old.
        # Pick c_new randomly from 0..total-1.
        # If c_new == c_old or c_new in selected_set, retry.
        
        # For efficiency, we can just pick a random index.
        c_new = random.randint(0, total - 1)
        if c_new == c_old or c_new in selected_set:
            continue
            
        # Check if swapping c_old with c_new is feasible
        # Temporarily remove c_old, add c_new.
        # Check if any word becomes uncovered.
        
        # Words affected:
        # 1. Words covered by c_old but not c_new: if count becomes 0, infeasible.
        # 2. Words covered by c_new but not c_old: count increases, always ok.
        # 3. Words covered by both: count unchanged.
        
        # So we only need to check:
        # For nb in neighbors[c_old]:
        #   if nb not in neighbors[c_new] and cov_counts[nb] == 1:
        #       infeasible
        # Also, if c_new covers a word that was ONLY covered by c_old? 
        # Wait, if c_new covers it, count becomes 1 (from 0? no, from 1 if c_old covered it and c_new covers it).
        
        # Let's just compute the new counts for the affected set.
        # Affected set = set(neighbors[c_old]) U set(neighbors[c_new])
        
        # This is small (at most 34 elements).
        
        feasible = True
        # We need to know if any word in N(c_old) \ N(c_new) has count 1.
        # Because those words will lose coverage.
        # Words in N(c_new) \ N(c_old) gain coverage.
        # Words in N(c_old) \cap N(c_new) keep coverage.
        
        # So check:
        for nb in neighbors[c_old]:
            if cov_counts[nb] == 1:
                # This word is covered only by c_old.
                # If c_new does NOT cover it, it becomes uncovered.
                if nb not in neighbors[c_new]:
                    feasible = False
                    break
                    
        if feasible:
            # Perform swap
            selected.remove(c_old)
            selected_set.remove(c_old)
            selected.append(c_new)
            selected_set.add(c_new)
            
            # Update counts
            for nb in neighbors[c_old]:
                cov_counts[nb] -= 1
            for nb in neighbors[c_new]:
                cov_counts[nb] += 1
            
            # After swap, try to remove redundancies again (loop continues)
            continue
            
        # If not feasible, try another swap (loop continues)
        
    # Convert indices to words
    codewords = []
    for idx in selected:
        digits = [0] * n
        tmp = idx
        for pos in range(n - 1, -1, -1):
            digits[pos] = tmp % q
            tmp //= q
        codewords.append(digits)
        
    return codewords
