import random
import time

N = 40

def construct():
    start_time = time.time()
    timeout = 85.0
    
    # Target differences 1..38
    # Bitmask for differences 1..38
    # Bit i represents diff i
    target_mask = 0
    for d in range(1, 39):
        target_mask |= (1 << d)
    
    best_S = None
    found = False
    
    def get_mask(S):
        mask = 0
        for i in range(len(S)):
            si = S[i]
            for j in range(i+1, len(S)):
                d = abs(si - S[j])
                if d < 39:
                    mask |= (1 << d)
        return mask

    # Search for Size 10
    # If not found, Size 11
    
    # Pre-compute some random candidates or use greedy
    
    for size in [10, 11]:
        for restart in range(200):
            if time.time() - start_time > timeout - 5:
                break
                
            # Greedy Initialization
            S = [1, 39]
            current_mask = 1 << 38 # Diff 38
            
            for _ in range(size - 2):
                best_candidate = -1
                best_new_mask = 0
                best_score = -1
                
                for c in range(2, 39):
                    if c in S:
                        continue
                    # Calculate new mask
                    new_mask = current_mask
                    for s in S:
                        d = abs(c - s)
                        if d < 39:
                            new_mask |= (1 << d)
                    score = bin(new_mask).count('1')
                    if score > best_score:
                        best_score = score
                        best_new_mask = new_mask
                        best_candidate = c
                
                if best_candidate == -1:
                    break
                S.append(best_candidate)
                current_mask = best_new_mask
            
            # If greedy didn't finish, pad randomly
            while len(S) < size:
                c = random.randint(1, 39)
                if c not in S:
                    S.append(c)
                    # Update mask
                    for s in S[:-1]:
                        d = abs(c - s)
                        if d < 39:
                            current_mask |= (1 << d)
            
            # Local Search
            T = 5.0
            cooling = 0.999
            # Limit iterations per restart to ensure we finish
            max_iters = 20000 if size == 10 else 10000
            
            for iter in range(max_iters):
                if time.time() - start_time > timeout - 2:
                    break
                    
                if current_mask == target_mask:
                    best_S = S
                    found = True
                    break
                    
                # Move: Pick random index, change value
                idx = random.randint(0, size-1)
                old_val = S[idx]
                
                # Try small perturbations first, then random
                if random.random() < 0.5:
                    delta = random.choice([-2, -1, 1, 2])
                    new_val = old_val + delta
                    if new_val < 1 or new_val > 39:
                        continue
                    if new_val in S and new_val != old_val:
                        continue
                else:
                    new_val = random.randint(1, 39)
                    if new_val in S and new_val != old_val:
                        continue
                
                S[idx] = new_val
                new_mask = get_mask(S)
                
                # Score: number of bits set
                old_score = bin(current_mask).count('1')
                new_score = bin(new_mask).count('1')
                
                if new_score >= old_score:
                    current_mask = new_mask
                else:
                    delta_score = old_score - new_score
                    if random.random() < (2.718281828 ** (-delta_score / T)):
                        current_mask = new_mask
                    else:
                        S[idx] = old_val
                
                T *= cooling
            
            if found:
                break
        
        if found:
            break
    
    if not found:
        # Fallback to a known valid size 11 set if we can construct one, or just return a valid graph
        # A simple valid set for size 11:
        # We can try to hardcode a set if search fails, but search should succeed.
        # Let's try one more specific heuristic:
        # Use a known sparse ruler structure.
        # For N=40, a ruler of size 11 exists.
        # Example: 0, 1, 4, 10, 12, 17, 20, 24, 28, 33, 38 (shifted by 1 for 1..39?)
        # Diffs:
        # 1, 3, 9, 11, 16, 19, 23, 27, 32, 37
        # 3, 9, 11, 16, 19, 23, 27, 32, 37
        # ...
        # Let's just rely on the search. If it fails, we return a larger graph.
        # But size 11 is very likely to be found.
        
        # Fallback: Return a graph with more edges to ensure validity
        # Use all spokes? No, that's 39 edges + 38 path = 77 edges.
        # Objective 37.
        # Better fallback: Use a known valid set.
        # Let's try to construct one deterministically.
        S = [1, 2, 5, 11, 13, 18, 21, 25, 29, 33, 39]
        # Check coverage
        mask = get_mask(S)
        if mask == target_mask:
            best_S = S
        else:
            # If this specific set fails, just add more spokes until it works
            S = list(range(1, 39)) # All of them, size 38.
            # Reduce?
            # Just use the search result if it found size 11.
            # If search failed completely, use a dense graph.
            edges = [(0, i) for i in range(1, N)] + [(i, i+1) for i in range(1, N-1)]
            cycles = [list(range(k)) for k in range(3, N+1)]
            return {'edges': edges, 'cycles': cycles}
            
    # Construct graph with best_S
    path_edges = [(i, i+1) for i in range(1, N-1)]
    spoke_edges = [(0, s) for s in best_S]
    
    edges = path_edges + spoke_edges
    
    S_set = set(best_S)
    cycles = []
    for L in range(3, N+1):
        diff_needed = L - 2
        found_cycle = False
        for u in best_S:
            v = u + diff_needed
            if v in S_set:
                cycle_verts = [0] + list(range(u, v+1))
                cycles.append(cycle_verts)
                found_cycle = True
                break
        if not found_cycle:
            raise Exception("Cycle not found for length " + str(L))
            
    return {'edges': edges, 'cycles': cycles}
