def construct():
    scale = 5354228880

    def greedy():
        sums = {0}
        chosen = []
        for i in range(24, 0, -1):
            w = scale // i
            if w not in sums:
                chosen.append(i)
                sums = sums | {s + w for s in sums}
        return sorted(chosen)

    best_set = greedy()
    
    # Backtracking search
    # To avoid deep recursion issues and memory blowup, we limit the search
    # or use a more efficient representation.
    # Given N=24, and greedy likely finding a large set, 
    # we can try to improve by swapping or adding.
    
    # Let's try a simple local search / hill climbing
    # Start with best_set
    # Try to add any element not in best_set
    # If valid, add it.
    # If not, try to remove one element and add the new one?
    
    # Actually, the backtracking above is a full search.
    # Let's try to implement it with a time limit or depth limit if necessary.
    # But for N=24, it might be okay.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def backtrack(i, current_set, subset_sums):
        nonlocal best_set
        
        # Pruning
        if len(current_set) + (24 - i + 1) <= len(best_set):
            return
            
        if i == 0:
            if len(current_set) > len(best_set):
                best_set = list(current_set)
            return
        
        w = scale // i
        
        # Try include
        if w not in subset_sums:
            new_sums = subset_sums | {s + w for s in subset_sums}
            backtrack(i - 1, current_set + [i], new_sums)
            
        # Try exclude
        backtrack(i - 1, current_set, subset_sums)
        
    # To prevent timeout, we can run backtrack only if greedy size is small
    # Or we can just run it. 
    # If it times out, we return best_set found so far.
    # But we can't catch timeout easily in a single function without threads.
    
    # Let's assume the search is fast enough due to pruning.
    backtrack(24, [], {0})
    
    return sorted(best_set)
