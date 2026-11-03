import time
import sys
import random

def _overlap(a, b):
    """Calculate the length of the longest suffix of a that is a prefix of b."""
    max_len = min(len(a), len(b))
    for i in range(max_len, -1, -1):
        if a[-i:] == b[:i]:
            return i
    return 0

def construct(n):
    start_time = time.time()
    word = [1]
    
    # Build up to n-1 using standard recursive construction
    for m in range(2, n):
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
        
        if time.time() - start_time > 100:
            # If we run out of time, we can't optimize the last level
            # Just finish the standard construction for n
            break

    # If we broke early, we need to finish n normally
    if m < n:
        # Standard step for n
        perms, seen = [], set()
        for i in range(len(word) - n + 2):
            window = tuple(word[i:i + n - 1])
            if len(set(window)) == n - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [n] + list(p)
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        return "".join(map(str, word))

    # Now m == n-1, we need to construct level n with optimization
    # Extract perms of size n-1
    perms, seen = [], set()
    for i in range(len(word) - n + 2):
        window = tuple(word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    K = len(perms)
    if K == 0:
        # Fallback
        return "".join(map(str, word))

    # Precompute blocks
    blocks = [list(p) + [n] + list(p) for p in perms]
    
    # Compute overlap matrix
    # overlap[i][j] = overlap(perms[i], perms[j])
    # Note: overlap of suffix of block i and prefix of block j
    # Block i ends with perms[i]. Block j starts with perms[j].
    # So we want overlap(perms[i], perms[j])
    
    overlap_matrix = [[0] * K for _ in range(K)]
    for i in range(K):
        for j in range(K):
            overlap_matrix[i][j] = _overlap(perms[i], perms[j])
            
    # Compute start overlaps
    # Suffix of current word
    suffix = word
    start_overlap = [0] * K
    for i in range(K):
        start_overlap[i] = _overlap(suffix, perms[i])
        
    # Initial order: order of appearance (which is the order in perms list)
    order = list(range(K))
    
    # 2-Opt Optimization
    # We want to maximize: start_overlap[order[0]] + sum(overlap_matrix[order[i]][order[i+1]])
    # Equivalent to minimizing negative sum.
    
    def get_cost(order):
        cost = -start_overlap[order[0]]
        for i in range(K - 1):
            cost -= overlap_matrix[order[i]][order[i+1]]
        return cost
        
    current_cost = get_cost(order)
    
    improved = True
    while improved:
        improved = False
        for i in range(K - 1):
            for j in range(i + 1, K):
                # Reverse order[i:j+1]
                # Check cost change
                # Old links: (i-1, i) and (j, j+1)
                # New links: (i-1, j) and (i, j+1)
                
                old_cost = 0
                new_cost = 0
                
                # Handle boundaries
                if i > 0:
                    old_cost -= overlap_matrix[order[i-1]][order[i]]
                    new_cost -= overlap_matrix[order[i-1]][order[j]]
                else:
                    old_cost -= start_overlap[order[i]]
                    new_cost -= start_overlap[order[j]]
                    
                if j < K - 1:
                    old_cost -= overlap_matrix[order[j]][order[j+1]]
                    new_cost -= overlap_matrix[order[i]][order[j+1]]
                else:
                    # No term for j+1 in sum, but the last element doesn't have an outgoing edge in the sum?
                    # Wait, the sum is over i=0 to K-2 of overlap(order[i], order[i+1]).
                    # If j is the last element (K-1), there is no overlap_matrix[order[j]][order[j+1]].
                    # So the term was 0.
                    # And new link is (i, j+1) which doesn't exist.
                    # So no change for the right side.
                    pass
                
                # Note: The cost function defined in get_cost uses negative overlaps.
                # So we are minimizing cost.
                # old_cost is the sum of NEGATIVE overlaps for the affected edges.
                # If new_cost < old_cost, we improved.
                
                if new_cost < old_cost:
                    order[i:j+1] = reversed(order[i:j+1])
                    current_cost = new_cost + (current_cost - old_cost) # Update current cost? 
                    # Actually, just recalculate or update delta.
                    # Delta = new_cost - old_cost
                    current_cost += new_cost - old_cost
                    improved = True
                    # Break and restart? Or continue?
                    # Restarting is safer for 2-opt.
                    break
            if improved:
                break
                
        # Time check
        if time.time() - start_time > 140:
            break

    # Construct final word
    new = list(word)
    for idx in order:
        block = blocks[idx]
        max_overlap = min(len(new), len(block))
        overlap = 0
        for o in range(max_overlap, -1, -1):
            if new[len(new) - o:] == block[:o]:
                overlap = o
                break
        new += block[overlap:]
        
    return "".join(map(str, new))
