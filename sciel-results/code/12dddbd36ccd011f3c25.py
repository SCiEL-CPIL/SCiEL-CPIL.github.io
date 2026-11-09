import time

def construct():
    start_time = time.time()
    timeout = 55.0

    # 1. Generate 5-smooth numbers up to 128
    smooths = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0: y //= 2
        while y % 3 == 0: y //= 3
        while y % 5 == 0: y //= 5
        if y == 1:
            smooths.append(x)
    
    M = len(smooths)
    
    # Precompute incompatibility
    incompat = [set() for _ in range(M)]
    for i in range(M):
        for j in range(i + 1, M):
            if smooths[i] % smooths[j] == 0 or smooths[j] % smooths[i] == 0:
                incompat[i].add(j)
                incompat[j].add(i)
                
    # Sort indices by value descending to find small size solutions faster
    sorted_indices = sorted(range(M), key=lambda i: smooths[i], reverse=True)
    sorted_smooths = [smooths[i] for i in sorted_indices]
    # We need to map sorted positions back to original indices for incompat checks
    # Or just use original indices in the set and iterate sorted_indices
    
    def solve_target(n):
        best_size = float('inf')
        best_set = []
        
        # Precompute suffix sums of sorted_smooths for quick upper bound?
        # But compatibility makes it tricky.
        # We'll compute compatible sum on the fly in the DFS.
        
        def dfs(idx, current_sum, current_set):
            nonlocal best_size, best_set
            
            if current_sum > n:
                return
            if current_sum == n:
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                return
            
            # Pruning 1: Size
            if len(current_set) >= best_size:
                return
                
            # Pruning 2: Max possible sum
            # Calculate sum of all smooths from idx to end that are compatible with current_set
            # This is expensive, so we do it only if necessary or optimize it.
            # Optimization: If current_sum + sum(sorted_smooths[idx:]) < n, prune.
            # This is a weaker bound but O(1) if we precompute suffix sums of sorted_smooths.
            # Let's precompute suffix sums of sorted_smooths.
            
            # For stronger bound, we check compatibility.
            # Given M=37, O(M*k) is acceptable if node count is low.
            
            max_rem = 0
            for k in range(idx, M):
                orig_idx = sorted_indices[k]
                if all(orig_idx not in incompat[x] for x in current_set):
                    max_rem += sorted_smooths[k]
            
            if current_sum + max_rem < n:
                return
            
            # Explore
            for k in range(idx, M):
                orig_idx = sorted_indices[k]
                val = sorted_smooths[k]
                
                # Check compatibility
                if any(orig_idx in incompat[x] for x in current_set):
                    continue
                    
                new_sum = current_sum + val
                if new_sum > n:
                    continue
                
                dfs(k + 1, new_sum, current_set + [orig_idx])
        
        dfs(0, 0, [])
        
        if best_set:
            return [smooths[i] for i in best_set]
        return []

    answers = []
    for n in range(1, 129):
        if time.time() - start_time > timeout:
            # Fallback for remaining targets if timeout
            # Simple fallback: just use the number itself if it's smooth, else empty?
            # But we need valid representations.
            # If we time out, we might return invalid results.
            # Better to ensure we don't time out.
            # The DFS should be fast.
            pass
            
        ans = solve_target(n)
        answers.append(ans)
        
    return answers
