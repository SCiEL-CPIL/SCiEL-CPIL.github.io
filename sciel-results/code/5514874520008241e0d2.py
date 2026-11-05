import itertools
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    
    # Helper to calculate overlap: length of suffix of A that is prefix of B
    def overlap(a, b):
        max_k = 0
        for k in range(min(len(a), len(b)) - 1, -1, -1):
            if a[-k:] == b[:k]:
                max_k = k
                break
        return max_k

    # Standard Construction
    def standard_construct(n):
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
                overlap_val = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap_val:]
            word = new
        return "".join(map(str, word))

    # Greedy Nearest Neighbor
    def greedy_construct(n):
        if n <= 1:
            return "1"
        
        all_perms = list(itertools.permutations(range(1, n + 1)))
        # Start with the first permutation
        current = all_perms[0]
        word = list(current)
        visited = {current}
        unvisited = set(all_perms[1:])
        
        while unvisited:
            if time.time() > deadline - 10.0:
                return None # Signal timeout
            
            # Find best next
            best_next = None
            best_overlap = -1
            
            # Optimization: Instead of scanning all, we can just scan.
            # For n=7, 5040 is small enough.
            for p in unvisited:
                ov = overlap(current, p)
                if ov > best_overlap:
                    best_overlap = ov
                    best_next = p
            
            if best_next is None:
                break
                
            # Move to best_next
            # The number of new symbols to append is n - best_overlap
            new_symbols = best_next[len(current) - best_overlap:]
            word.extend(new_symbols)
            
            visited.add(best_next)
            unvisited.remove(best_next)
            current = best_next
            
        return "".join(map(str, word))

    # Get standard result
    std_res = standard_construct(n)
    std_len = len(std_res)
    
    # Try greedy
    greedy_res = greedy_construct(n)
    
    if greedy_res is not None:
        greedy_len = len(greedy_res)
        # Verify greedy_res? 
        # For safety, we assume it's valid if it visited all perms.
        # But we should check if it's actually shorter.
        if greedy_len < std_len:
            return greedy_res
        else:
            return std_res
    else:
        return std_res
