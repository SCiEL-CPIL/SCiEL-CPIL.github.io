# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    N = 256
    candidates = list(range(1, N + 1))
    n_cand = N
    
    # Precompute conflicts as sets of indices
    # conflicts[i] is a set of j such that gcd(candidates[i], candidates[j]) > 1
    conflicts = [set() for _ in range(n_cand)]
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if gcd(candidates[i], candidates[j]) > 1:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # Sort indices by value descending for B&B ordering
    # We want to consider high values first
    order = sorted(range(n_cand), key=lambda x: candidates[x], reverse=True)
    
    # Greedy Initialization
    # Start with empty set, add items in order if compatible
    def greedy_init():
        selected = []
        blocked = set()
        for idx in order:
            if idx not in blocked:
                selected.append(idx)
                blocked.add(idx)
                blocked.update(conflicts[idx])
        return selected

    best_set = greedy_init()
    best_sum = sum(candidates[i] for i in best_set)
    
    # Local Search to improve greedy solution
    # Try to swap or add items
    def local_search(initial_set, max_iter=100):
        current_set = set(initial_set)
        current_sum = sum(candidates[i] for i in current_set)
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            iters += 1
            # Try adding a single item
            for idx in range(n_cand):
                if idx in current_set:
                    continue
                # Check if idx is compatible with all in current_set
                # i.e., no conflict between idx and any item in current_set
                # This is equivalent to: current_set.isdisjoint(conflicts[idx])
                if current_set.isdisjoint(conflicts[idx]):
                    new_sum = current_sum + candidates[idx]
                    if new_sum > current_sum: # Always true since values > 0
                        current_set.add(idx)
                        current_sum = new_sum
                        improved = True
            # Try swapping: remove one, add one
            # This is O(N^2) per iteration, might be slow. 
            # But N=256, so 65k checks. 
            # We can optimize by only trying to swap if it looks promising.
            # For now, let's stick to single additions and maybe simple swaps if time permits.
            # Actually, single additions are the most impactful.
            # Let's try simple 1-for-1 swaps.
            for idx_out in list(current_set):
                for idx_in in range(n_cand):
                    if idx_in in current_set:
                        continue
                    # Check if idx_in is compatible with (current_set - {idx_out})
                    # i.e., current_set - {idx_out} is disjoint with conflicts[idx_in]
                    # And idx_in is not idx_out
                    if idx_in == idx_out:
                        continue
                    # Optimization: Check if idx_in conflicts with idx_out? No, they are swapped.
                    # We need idx_in to be compatible with all OTHER selected items.
                    # other_selected = current_set - {idx_out}
                    # if other_selected.isdisjoint(conflicts[idx_in]):
                    #    if candidates[idx_in] > candidates[idx_out]:
                    #       swap
                    # This is expensive. Let's skip complex swaps for now and rely on B&B.
                    pass
        return list(current_set), current_sum

    best_set, best_sum = local_search(best_set)
    
    # Branch and Bound
    # We will use the sorted order.
    # available: set of indices that are still candidates.
    # We pass the set of available indices.
    
    # To speed up B&B, we can precompute the sum of values for any subset? No, too big.
    # We just sum the available set.
    
    def dfs(pos, current_sum, available):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
        
        # Bound: current_sum + sum of all available values
        # If this is <= best_sum, prune
        # Calculating sum(available) is O(|available|). 
        # We can maintain a running sum of available? 
        # Let's pass available_sum.
        
        if current_sum + available_sum <= best_sum:
            return
        
        if not available:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = list(available) # Wait, available is empty. We need to track selected.
                # We didn't track selected in this dfs signature. 
                # Let's add selected to state or reconstruct.
                # Reconstructing is hard. Let's pass selected list.
            return
        
        # Get the next candidate to consider
        # We should consider candidates in 'order'.
        # But 'available' is a set. We need to find the next one in 'order' that is in 'available'.
        # This is O(N) per step.
        
        next_idx = -1
        for idx in order:
            if idx in available:
                next_idx = idx
                break
        
        if next_idx == -1:
             # No more candidates in order? 
             # This shouldn't happen if available is not empty, unless order doesn't cover all?
             # Order covers all 0..255.
             pass
             
        # Option 1: Exclude next_idx
        # available remains same (minus next_idx? No, we just skip it for now? 
        # No, in B&B over a fixed order, if we skip it, it's gone.
        # So new_available = available - {next_idx}
        # But we must ensure we don't pick it later.
        
        # Option 2: Include next_idx
        # new_available = available - {next_idx} - conflicts[next_idx]
        
        # Let's refine the dfs signature to include selected set for reconstruction.
        # But passing a set by value (copy) is O(N). 
        # We can use a global stack for reconstruction? 
        # Or just store the best sum and reconstruct at the end? 
        # Reconstruction from scratch is hard.
        # Let's pass a list of selected indices.
        
    # Refactored DFS with selected list
    def dfs2(pos, current_sum, available, selected):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
        
        # Bound
        # Sum of values in available
        # To avoid O(N) sum calculation at every node, we can pass available_sum
        # But available changes.
        # Let's just calculate it. N=256.
        
        avail_sum = 0
        for i in available:
            avail_sum += candidates[i]
            
        if current_sum + avail_sum <= best_sum:
            return
        
        if not available:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = selected[:]
            return
        
        # Find next candidate in 'order' that is in 'available'
        next_idx = -1
        for idx in order:
            if idx in available:
                next_idx = idx
                break
                
        if next_idx == -1:
            # This implies available is empty? 
            # Or order is exhausted? 
            # If available is not empty, but none of its elements are in order? Impossible.
            # So this block is likely dead code if available is not empty.
            return

        # Exclude next_idx
        new_avail_excl = available - {next_idx}
        dfs2(pos + 1, current_sum, new_avail_excl, selected)
        
        # Include next_idx
        if time.time() > deadline:
            return
            
        # Check if next_idx is actually in available (it is by definition)
        # Remove next_idx and its conflicts
        new_avail_incl = available - {next_idx}
        new_avail_incl -= conflicts[next_idx]
        
        selected.append(next_idx)
        dfs2(pos + 1, current_sum + candidates[next_idx], new_avail_incl, selected)
        selected.pop()
        
    # Initial call
    # We start with all candidates available.
    # But we already have a good solution from greedy.
    # We can start B&B from the root.
    
    initial_available = set(range(n_cand))
    dfs2(0, 0, initial_available, [])
    
    # Convert indices back to values
    result = [candidates[i] for i in best_set]
    
    # Ensure it's a list of ints
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
