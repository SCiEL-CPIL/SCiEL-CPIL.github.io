import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    best_seq = []
    
    # Initial greedy to get a baseline
    curr_seq = []
    curr_sums = set()
    for x in range(1, 257):
        tails = [x]
        total = x
        for a in reversed(curr_seq):
            total += a
            tails.append(total)
        if not curr_sums.intersection(tails):
            curr_seq.append(x)
            curr_sums.update(tails)
    best_seq = curr_seq[:]
    
    # Iterative DFS
    # Stack items: (seq, sums, last_val)
    # To save memory and time, we might pass lists and sets, but we need to backtrack.
    # Iterative DFS with explicit state copying is safer than recursion for deep searches if we manage the stack well.
    # However, copying sets is O(N^2).
    # Let's stick to recursion but with the strong pruning. Recursion is cleaner for backtracking.
    # The previous DFS was slow because it didn't prune enough.
    
    def max_sum_of_k_largest(k, last_val):
        # Returns sum of the largest k integers in [last_val+1, 256]
        if k <= 0:
            return 0
        if k > 256 - last_val:
            return -1 # Impossible
        # Largest are 256, 255, ..., 256 - k + 1
        # Sum = k * (first + last) / 2 = k * (256 + 256 - k + 1) / 2
        return k * (512 - k + 1) // 2
        
    def dfs(seq, sums, last_val):
        nonlocal best_seq
        if time.time() > deadline:
            return
        
        n = len(seq)
        if n > len(best_seq):
            best_seq = seq[:]
            
        # Pruning: Can we extend to beat best_seq?
        # Target length: len(best_seq)
        # If n == len(best_seq), we can't beat it by extending, so we can stop?
        # No, we might find a different sequence of the same length, but that doesn't improve score.
        # We only care about length > len(best_seq).
        if n == len(best_seq):
            return
            
        # Check if we can reach length len(best_seq) + 1
        # Let target_len = len(best_seq) + 1
        # k = target_len - n
        k_needed = len(best_seq) + 1 - n
        if k_needed > 256 - last_val:
            return
            
        # Calculate max possible sum for the remaining k_needed elements
        # These elements must be > last_val
        max_possible_sum_new = max_sum_of_k_largest(k_needed, last_val)
        if max_possible_sum_new == -1:
            return
            
        total_sum_max = sum(seq) + max_possible_sum_new
        min_sum = seq[0] if n > 0 else 1 # If n==0, min sum is at least 1
        
        # Total number of sums for length L is L(L+1)/2
        target_len = len(best_seq) + 1
        required_sums = target_len * (target_len + 1) // 2
        
        # Range of sums is [min_sum, total_sum_max]
        # Size of range = total_sum_max - min_sum + 1
        if total_sum_max - min_sum + 1 < required_sums:
            return
            
        # Try next values
        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                return
                
            # Calculate new sums ending at x
            new_sums = []
            total = x
            new_sums.append(total)
            for a in reversed(seq):
                total += a
                new_sums.append(total)
                
            # Check for conflicts
            conflict = False
            for s in new_sums:
                if s in sums:
                    conflict = True
                    break
                    
            if not conflict:
                # Update sums
                sums.update(new_sums)
                seq.append(x)
                
                dfs(seq, sums, x)
                
                # Backtrack
                seq.pop()
                sums.difference_update(new_sums)
                
    # Start DFS
    dfs([], set(), 0)
    
    return best_seq
