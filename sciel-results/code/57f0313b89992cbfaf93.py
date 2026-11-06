import time

def construct():
        start_time = time.time()
        deadline = start_time + 110  # 110s limit
        
        # Work in 0-indexed space [0, 283]
        # Preset: 0, 1, 4, 10
        initial = [0, 1, 4, 10]
        
        # Sums of initial
        sums = set()
        for i in range(len(initial)):
            for j in range(i, len(initial)):
                sums.add(initial[i] + initial[j])
                
        best_set = initial[:]
        
        # Candidates are 0..283
        # We need to add 16 more numbers to reach size 20.
        
        # To speed up, we can pre-filter candidates that don't conflict with initial.
        # But conflicts depend on the current set.
        
        def dfs(current_set, current_sums, depth):
            nonlocal best_set, deadline
            
            if time.time() > deadline:
                return False
            
            if len(current_set) > len(best_set):
                best_set = current_set[:]
                
            if len(current_set) == 20:
                return True
            
            # Heuristic: Try candidates in a specific order.
            # Maybe try to fill gaps?
            # Or just iterate 0..283?
            # To avoid re-trying failed branches, we can just iterate all available.
            
            # Optimization: Only consider numbers > max(current_set)?
            # Sidon sets can be unsorted, but we can assume sorted order for uniqueness.
            # If we assume sorted, we only pick x > max(current_set).
            # This reduces branching significantly.
            
            last = current_set[-1]
            
            # Try next numbers
            for x in range(last + 1, 284):
                # Check collision
                # New sums: x + y for y in current_set, and 2x
                conflict = False
                for y in current_set:
                    if x + y in current_sums:
                        conflict = True
                        break
                if not conflict and 2*x in current_sums:
                    conflict = True
                
                if conflict:
                    continue
                
                # Add x
                new_set = current_set + [x]
                new_sums = set(current_sums)
                for y in current_set:
                    new_sums.add(x + y)
                new_sums.add(2*x)
                
                # Recurse
                if dfs(new_set, new_sums, depth + 1):
                    return True
            
            return False
        
        # Initial call
        # To help the search, we can use a greedy initialization to get a larger starting set
        # and then try to swap/expand?
        # Or just DFS from the 4 elements.
        # DFS from 4 elements to 20 is depth 16.
        # Branching factor might be high at first, but pruning is effective.
        
        # Let's try a greedy first to get a good baseline and maybe a hint for ordering.
        greedy_set = initial[:]
        greedy_sums = set(sums)
        for x in range(11, 284):
            conflict = False
            for y in greedy_set:
                if x + y in greedy_sums:
                    conflict = True
                    break
            if not conflict and 2*x in greedy_sums:
                conflict = True
            
            if not conflict:
                greedy_set.append(x)
                for y in greedy_set[:-1]:
                    greedy_sums.add(x + y)
                greedy_sums.add(2*x)
                
        # Use greedy_set as a starting point for a local search or just as a lower bound?
        # Actually, if greedy finds 20, we are done.
        if len(greedy_set) == 20:
             return [x + 1 for x in greedy_set]
        
        # If greedy didn't find 20, use DFS.
        # To make DFS faster, we can order candidates by how "good" they are (e.g., greedy order).
        # But standard DFS with sorted order (increasing x) is usually sufficient for these small N.
        
        # One issue: DFS might get stuck in a bad branch.
        # We can add a timeout and restart with a different heuristic (e.g., random order, or reverse order).
        
        # Let's stick to a single robust DFS with time check.
        # If it times out, it returns the best found.
        
        # To ensure we don't miss the optimal, we rely on the fact that 20 is the max.
        # If a size 20 solution exists, DFS should find it.
        
        # Let's run the DFS.
        # Note: The initial set is sorted. We only add x > last.
        
        # To speed up collision check, use a boolean array for sums?
        # Max sum is 283+283 = 566.
        # Boolean array of size 570 is very fast.
        
        sums_arr = [False] * 570
        for s in sums:
            if s < 570:
                sums_arr[s] = True
        
        def dfs_fast(current_set, sums_arr, depth):
            nonlocal best_set, deadline
            
            if time.time() > deadline:
                return False
            
            if len(current_set) > len(best_set):
                best_set = current_set[:]
                
            if len(current_set) == 20:
                return True
            
            last = current_set[-1]
            
            for x in range(last + 1, 284):
                # Check collision
                conflict = False
                for y in current_set:
                    s = x + y
                    if sums_arr[s]:
                        conflict = True
                        break
                if not conflict and sums_arr[2*x]:
                    conflict = True
                
                if conflict:
                    continue
                
                # Update sums_arr
                # We need to restore it after recursion
                added_sums = []
                for y in current_set:
                    s = x + y
                    sums_arr[s] = True
                    added_sums.append(s)
                sums_arr[2*x] = True
                added_sums.append(2*x)
                
                current_set.append(x)
                
                if dfs_fast(current_set, sums_arr, depth + 1):
                    return True
                
                current_set.pop()
                
                # Restore sums_arr
                for s in added_sums:
                    sums_arr[s] = False
            
            return False
        
        # Run DFS
        dfs_fast(initial[:], sums_arr, 0)
        
        # Convert back to 1..284
        result = [x + 1 for x in best_set]
        return result
