import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # Step 1: Find all bad triples (a,b,c) where lcm(a,b)==lcm(a,c)==lcm(b,c)
    # Use efficient approach: group by LCM value
    # For each pair (a,b), compute lab = lcm(a,b)
    # For a triple to be bad, we need c such that lcm(a,c)==lab and lcm(b,c)==lab
    
    # Precompute: for each L value, which pairs have lcm == L?
    # Actually, let's just enumerate triples directly but with pruning.
    
    bad_triples = []
    
    # For efficiency, precompute lcm for all pairs
    # lcm_table[i][j] for 1<=i<=j<=128
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples
    # a < b < c, lcm(a,b) == lcm(a,c) == lcm(b,c)
    for a in range(1, n + 1):
        if time.time() > deadline:
            break
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if time.time() > deadline:
                    break
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    
    if time.time() > deadline:
        # Fallback: return all numbers (might be invalid but better than nothing)
        return numbers
    
    T = len(bad_triples)
    if T == 0:
        return numbers
    
    # Step 2: Greedy vertex cover on hypergraph
    # We want to remove minimum elements to break all triples.
    # Greedy: repeatedly pick element in most active triples.
    
    # counts[x] = number of active triples containing x
    counts = [0] * (n + 1)
    for (a, b, c) in bad_triples:
        counts[a] += 1
        counts[b] += 1
        counts[c] += 1
    
    active = [True] * T
    active_count = T
    removed = [False] * (n + 1)
    
    while active_count > 0:
        if time.time() > deadline:
            break
        
        # Find element with max count among non-removed
        best_x = -1
        best_cnt = -1
        for x in range(1, n + 1):
            if not removed[x] and counts[x] > best_cnt:
                best_cnt = counts[x]
                best_x = x
        
        if best_x == -1 or best_cnt <= 0:
            break
        
        removed[best_x] = True
        
        # Update: break all active triples containing best_x
        for i in range(T):
            if not active[i]:
                continue
            a, b, c = bad_triples[i]
            if a == best_x or b == best_x or c == best_x:
                active[i] = False
                active_count -= 1
                # Decrement counts for the other two elements
                for x in (a, b, c):
                    if x != best_x:
                        counts[x] -= 1
    
    # Step 3: Local search polish
    # Try to add back removed elements if possible
    # Try 1-for-2 swaps: remove one kept element to add two removed elements
    
    result_set = set(x for x in range(1, n + 1) if not removed[x])
    
    def is_valid_set(s):
        """Check if set s has no bad triple."""
        s_list = sorted(s)
        len_s = len(s_list)
        for i in range(len_s):
            for j in range(i + 1, len_s):
                a = s_list[i]
                b = s_list[j]
                lab = lcm_table[a][b]
                for k in range(j + 1, len_s):
                    c = s_list[k]
                    if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                        return False
        return True
    
    # Try adding back each removed element
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if removed[x]:
            # Check if adding x to result_set creates a bad triple
            # A bad triple with x would be (x, y, z) where y,z in result_set
            # and lcm(x,y)==lcm(x,z)==lcm(y,z)
            can_add = True
            for y in result_set:
                lxy = lcm_table[x][y]
                for z in result_set:
                    if z <= y:
                        continue
                    if lcm_table[x][z] == lxy and lcm_table[y][z] == lxy:
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                result_set.add(x)
                removed[x] = False
    
    # Try 1-for-2 swaps: remove one element from result_set, add two from removed
    # This is expensive, so do limited attempts
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if not removed[x]:
            # Try removing x and adding two removed elements
            result_set.remove(x)
            
            # Find pairs of removed elements that can be added
            added_any = False
            for y in range(1, n + 1):
                if time.time() > deadline:
                    break
                if not removed[y]:
                    continue
                # Check if y can be added to current result_set (without x)
                can_add_y = True
                for z in result_set:
                    lzy = lcm_table[z][y]
                    for w in result_set:
                        if w <= z:
                            continue
                        if lcm_table[z][w] == lzy and lcm_table[y][w] == lzy:
                            can_add_y = False
                            break
                    if not can_add_y:
                        break
                if not can_add_y:
                    continue
                
                # Try to find another removed element w that can be added with y
                for w in range(y + 1, n + 1):
                    if time.time() > deadline:
                        break
                    if not removed[w]:
                        continue
                    # Check if w can be added to result_set ∪ {y}
                    can_add_w = True
                    for z in result_set:
                        lzw = lcm_table[z][w]
                        for v in result_set:
                            if v <= z:
                                continue
                            if lcm_table[z][v] == lzw and lcm_table[w][v] == lzw:
                                can_add_w = False
                                break
                        if not can_add_w:
                            break
                    # Also check triple (y, w, z) for z in result_set
                    if can_add_w:
                        lwy = lcm_table[y][w]
                        for z in result_set:
                            if lcm_table[y][z] == lwy and lcm_table[w][z] == lwy:
                                can_add_w = False
                                break
                    
                    if can_add_w:
                        result_set.add(y)
                        result_set.add(w)
                        removed[y] = False
                        removed[w] = False
                        added_any = True
                        break
                if added_any or time.time() > deadline:
                    break
            
            if not added_any:
                # Put x back
                result_set.add(x)
                removed[x] = False
    
    # Final validation
    result_list = sorted(result_set)
    if not result_list:
        return [1]
    
    # Validate the result
    if not is_valid_set(result_set):
        # Fallback to greedy-only result
        result_list = sorted(x for x in range(1, n + 1) if not removed[x])
        if not result_list:
            return [1]
    
    return result_list
