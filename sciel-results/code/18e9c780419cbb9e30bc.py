"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    N = 256
    
    # Precompute all 3-term APs and 4-term APs
    # 3-AP: (a, b, c) where a < b < c and a+c = 2b
    # 4-AP: (a, b, c, d) where a < b < c < d and a+d = b+c = 2*(a+d)/2
    # i.e., a, a+d, a+2d, a+3d
    
    # For each element x, precompute which 3-APs it participates in
    # and which 4-APs it participates in
    
    # 3-APs: (a, a+d, a+2d) for a+d <= N, a+2d <= N
    # 4-APs: (a, a+d, a+2d, a+3d) for a+3d <= N
    
    # Build lists: for each x, list of 3-AP indices it's in, and 4-AP indices it's in
    # But we need to track counts efficiently
    
    # Let's precompute for each pair (x, d) whether adding x would create a 4-AP
    # A 4-AP ending at x with difference d: x-3d, x-2d, x-d, x all in set
    # A 4-AP with x as second term: x, x+d, x+2d, x+3d all in set
    # A 4-AP with x as third term: x-2d, x-d, x, x+d all in set
    # A 4-AP with x as first term: x, x+d, x+2d, x+3d all in set
    
    # For local search, we need to quickly check if adding/removing an element creates/destroys a 4-AP
    
    # Strategy:
    # 1. Greedy: add elements in order that maximizes 3-AP gain while avoiding 4-APs
    # 2. Local search: try swaps and additions/removals to improve 3-AP count
    
    # Precompute 3-AP contributions
    # For each x, count how many 3-APs it's part of with current set
    # 3-AP containing x:
    #   x as middle: (x-d, x, x+d) for d>=1, x-d>=1, x+d<=N
    #   x as first: (x, x+d, x+2d) for d>=1, x+2d<=N  
    #   x as last: (x-2d, x-d, x) for d>=1, x-2d>=1
    
    # Let's use a bitset approach for speed
    # Represent set as a boolean array of size N+1 (index 1..N)
    
    def count_3aps_in_set(S):
        """Count number of 3-term APs in set S (as a set of ints)."""
        count = 0
        S_set = set(S)
        for a in S:
            for d in range(1, N - a + 1):
                b = a + d
                c = a + 2*d
                if c > N:
                    break
                if b in S_set and c in S_set:
                    count += 1
        return count
    
    def has_4ap(S):
        """Check if set S contains a 4-term AP."""
        S_set = set(S)
        for a in S:
            max_d = (N - a) // 3
            for d in range(1, max_d + 1):
                if (a + d in S_set and a + 2*d in S_set and a + 3*d in S_set):
                    return True
        return False
    
    # Greedy construction: add elements one by one, choosing the one that
    # maximizes 3-AP gain while keeping no 4-APs
    
    def greedy_construct():
        S = set()
        remaining = list(range(1, N + 1))
        
        while remaining:
            best_x = None
            best_gain = -1
            for x in remaining:
                # Check if adding x creates a 4-AP
                creates_4ap = False
                for d in range(1, N):
                    # x as first term of 4-AP: x, x+d, x+2d, x+3d
                    if x + 3*d <= N and (x+d in S and x+2*d in S and x+3*d in S):
                        creates_4ap = True
                        break
                    # x as second term: x-d, x, x+d, x+2d
                    if x - d >= 1 and (x-d in S and x+d in S and x+2*d in S):
                        creates_4ap = True
                        break
                    # x as third term: x-2d, x-d, x, x+d
                    if x - 2*d >= 1 and (x-2*d in S and x-d in S and x+d in S):
                        creates_4ap = True
                        break
                    # x as fourth term: x-3d, x-2d, x-d, x
                    if x - 3*d >= 1 and (x-3*d in S and x-2*d in S and x-d in S):
                        creates_4ap = True
                        break
                    if d > (N-1)//3 and x - 3*d < 1:
                        # No more possible 4-APs with this x as any position
                        pass
                if creates_4ap:
                    continue
                
                # Count 3-AP gain
                gain = 0
                for d in range(1, N):
                    # x as middle: (x-d, x, x+d)
                    if x-d >= 1 and x+d <= N and (x-d in S and x+d in S):
                        gain += 1
                    # x as first: (x, x+d, x+2d)
                    if x+2*d <= N and (x+d in S and x+2*d in S):
                        gain += 1
                    # x as last: (x-2d, x-d, x)
                    if x-2*d >= 1 and (x-2*d in S and x-d in S):
                        gain += 1
                    if d > (N-1)//2 and x+2*d > N and x-2*d < 1:
                        pass
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            
            S.add(best_x)
            remaining.remove(best_x)
            
            if time.time() > deadline:
                break
        
        return S
    
    # Local search: try to improve by adding, removing, or swapping elements
    def local_search(S, max_iter=5000):
        S = set(S)
        current_3ap = count_3aps_in_set(S)
        best_S = set(S)
        best_3ap = current_3ap
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try adding a random element
            candidates_to_add = [x for x in range(1, N+1) if x not in S]
            if candidates_to_add:
                x = random.choice(candidates_to_add)
                # Check if adding x creates a 4-AP
                creates_4ap = False
                for d in range(1, N):
                    if x + 3*d <= N and (x+d in S and x+2*d in S and x+3*d in S):
                        creates_4ap = True
                        break
                    if x - d >= 1 and (x-d in S and x+d in S and x+2*d in S):
                        creates_4ap = True
                        break
                    if x - 2*d >= 1 and (x-2*d in S and x-d in S and x+d in S):
                        creates_4ap = True
                        break
                    if x - 3*d >= 1 and (x-3*d in S and x-2*d in S and x-d in S):
                        creates_4ap = True
                        break
                    if d > (N-1)//3 and x - 3*d < 1 and x + 3*d > N:
                        pass
                
                if not creates_4ap:
                    new_3ap = current_3ap
                    # Count new 3-APs involving x
                    for d in range(1, N):
                        if x-d >= 1 and x+d <= N and (x-d in S and x+d in S):
                            new_3ap += 1
                        if x+2*d <= N and (x+d in S and x+2*d in S):
                            new_3ap += 1
                        if x-2*d >= 1 and (x-2*d in S and x-d in S):
                            new_3ap += 1
                        if d > (N-1)//2 and x+2*d > N and x-2*d < 1:
                            pass
                    
                    if new_3ap > best_3ap:
                        S.add(x)
                        current_3ap = new_3ap
                        best_S = set(S)
                        best_3ap = new_3ap
                    elif new_3ap == best_3ap and random.random() < 0.1:
                        S.add(x)
                        current_3ap = new_3ap
                        best_S = set(S)
            
            # Try removing a random element and adding another
            if S:
                x_remove = random.choice(list(S))
                S.discard(x_remove)
                current_3ap = count_3aps_in_set(S)
                
                candidates_to_add = [x for x in range(1, N+1) if x not in S]
                if candidates_to_add:
                    x_add = random.choice(candidates_to_add)
                    creates_4ap = False
                    for d in range(1, N):
                        if x_add + 3*d <= N and (x_add+d in S and x_add+2*d in S and x_add+3*d in S):
                            creates_4ap = True
                            break
                        if x_add - d >= 1 and (x_add-d in S and x_add+d in S and x_add+2*d in S):
                            creates_4ap = True
                            break
                        if x_add - 2*d >= 1 and (x_add-2*d in S and x_add-d in S and x_add+d in S):
                            creates_4ap = True
                            break
                        if x_add - 3*d >= 1 and (x_add-3*d in S and x_add-2*d in S and x_add-d in S):
                            creates_4ap = True
                            break
                        if d > (N-1)//3 and x_add - 3*d < 1 and x_add + 3*d > N:
                            pass
                    
                    if not creates_4ap:
                        new_3ap = count_3aps_in_set(S | {x_add})
                        if new_3ap >= best_3ap:
                            S.add(x_add)
                            current_3ap = new_3ap
                            if new_3ap > best_3ap:
                                best_S = set(S)
                                best_3ap = new_3ap
                        else:
                            # Revert
                            pass
                    else:
                        # Revert removal
                        S.add(x_remove)
                        current_3ap = count_3aps_in_set(S)
                else:
                    S.add(x_remove)
                    current_3ap = count_3aps_in_set(S)
        
        return best_S, best_3ap
    
    # Run greedy
    S = greedy_construct()
    if not S:
        S = {1}
    
    # Run local search multiple times
    best_S = S
    best_3ap = count_3aps_in_set(S)
    
    for trial in range(10):
        if time.time() > deadline:
            break
        
        # Perturb and local search
        perturbed = set(S)
        # Remove a few random elements
        for _ in range(random.randint(1, 3)):
            if perturbed:
                perturbed.discard(random.choice(list(perturbed)))
        
        result_S, result_3ap = local_search(perturbed, max_iter=2000)
        if result_3ap > best_3ap:
            best_S = result_S
            best_3ap = result_3ap
    
    return sorted(best_S)
# EVOLVE-BLOCK-END
