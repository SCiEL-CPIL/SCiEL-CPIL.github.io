import time
import sys

def construct():
        start_time = time.time()
        time_limit = 55.0
        
        # 1. Greedy Baseline
        def greedy_asc():
            vals = []
            sums_set = set()
            for x in range(1, 257):
                new_sums = set()
                total = x
                new_sums.add(total)
                # Suffix sums of vals
                # Actually, new sums are x + suffix_sums_of_vals
                # Let's compute suffix sums of vals
                suffix_sums = [0]
                s = 0
                for a in reversed(vals):
                    s += a
                    suffix_sums.append(s)
                
                for ss in suffix_sums:
                    ns = x + ss
                    new_sums.add(ns)
                
                if not sums_set.intersection(new_sums):
                    vals.append(x)
                    sums_set.update(new_sums)
            return vals

        seq_greedy = greedy_asc()
        best_seq = seq_greedy
        best_len = len(seq_greedy)
        
        # 2. DFS to extend
        # We try to find sequences of length L > best_len
        # Start from L = best_len + 1
        
        def dfs_search(target_len):
            """
            Returns a list of length target_len if found, else None.
            Uses global time limit.
            """
            # State
            seq = []
            sums = set()
            
            # Helper to check validity of adding x
            def can_add(x, seq, sums):
                # Compute new sums
                # Suffix sums of seq
                suffix_sums = [0]
                s = 0
                for a in reversed(seq):
                    s += a
                    suffix_sums.append(s)
                
                for ss in suffix_sums:
                    ns = x + ss
                    if ns in sums:
                        return False
                return True
            
            def recurse(depth, last_val):
                if time.time() - start_time > time_limit:
                    return None
                
                if depth == target_len:
                    return seq.copy()
                
                # Pruning: Not enough numbers left
                if 256 - last_val < target_len - depth:
                    return None
                
                # Candidates
                # Try x from last_val + 1 to 256
                # Heuristic: Try smaller x first?
                for x in range(last_val + 1, 257):
                    if can_add(x, seq, sums):
                        # Add x
                        seq.append(x)
                        # Update sums
                        suffix_sums = [0]
                        s = 0
                        for a in reversed(seq[:-1]): # Exclude x for now? No, we need suffix sums of seq BEFORE x
                            # Wait, can_add already computed them or we can recompute.
                            # Let's recompute for update
                            pass
                        
                        # Efficient update:
                        # New sums are x + suffix_sums_of_old_seq
                        # We need suffix sums of old seq.
                        # Let's store suffix sums in the state or recompute.
                        # Recomputing is O(N). N is small.
                        
                        old_suffix_sums = [0]
                        s = 0
                        for a in reversed(seq[:-1]):
                            s += a
                            old_suffix_sums.append(s)
                        
                        new_sums = set()
                        for ss in old_suffix_sums:
                            new_sums.add(x + ss)
                        
                        sums.update(new_sums)
                        
                        res = recurse(depth + 1, x)
                        if res is not None:
                            return res
                        
                        # Backtrack
                        seq.pop()
                        for ns in new_sums:
                            sums.remove(ns)
                
                return None
            
            return recurse(0, 0)

        # Try to improve
        # Start from best_len + 1
        # But DFS from scratch might be slow.
        # We can seed the DFS with the greedy sequence?
        # No, DFS builds from scratch.
        
        # To make it faster, we can try different starting strategies.
        # But let's try simple DFS with time limit.
        
        # We will try L from best_len + 1 down to best_len + 5 (max 5 attempts)
        # Or just try best_len + 1 with a time limit.
        
        # Actually, let's try a few lengths.
        # If best_len is 20, try 21, 22.
        
        for L in range(best_len + 1, best_len + 6):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Run DFS for L
            # To avoid getting stuck in one branch, we can randomize candidate order?
            # Or just rely on the time limit.
            
            # Optimization: Use a smaller time limit per DFS call?
            # No, let's give it a chance.
            
            # We need to pass time limit to dfs_search.
            # The dfs_search uses global start_time and time_limit.
            
            result = dfs_search(L)
            if result is not None:
                best_seq = result
                best_len = L
                # Found one, try to go higher?
                # The loop continues, so it will try L+1.
        
        return best_seq
