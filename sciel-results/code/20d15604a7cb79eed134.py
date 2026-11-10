"""
Strategy: Exact Maximum Independent Set via Bitmask Backtracking.
Problem: Maximize subset size such that no three distinct elements form a bad LCM triple.
This is a Maximum Independent Set in a 3-uniform hypergraph.
We use a recursive branch-and-bound algorithm with bitset optimizations for speed and exactness.
"""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # 1. Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l

    # 2. Identify all bad triples (a, b, c) with a < b < c
    # such that lcm(a,b) == lcm(a,c) == lcm(b,c)
    # We store these in a structure useful for backtracking.
    # bad_triples: list of (a, b, c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))

    if not bad_triples:
        return numbers

    # 3. Precompute conflict structures
    # For each pair (u, v), find all w such that (u, v, w) is a bad triple.
    # We use 0-indexed internally for bitmasks, but keep 1-indexed for logic.
    # Map number -> index (0 to 127)
    # idx = num - 1
    
    # conflict_pairs[u][v] = bitmask of w (0-indexed) such that (u+1, v+1, w+1) is bad.
    # Note: The triple is unordered, so if (a,b,c) is bad, then:
    # conflict_pairs[a][b] has bit c
    # conflict_pairs[a][c] has bit b
    # conflict_pairs[b][c] has bit a
    # (using 0-indexed indices)
    
    # We also need to know, for a candidate x and a selected z, which candidates y are forbidden.
    # If x and z are in the set, we cannot add any y such that (x, y, z) is bad.
    # This is exactly conflict_pairs[x][z].
    
    # Initialize conflict matrix
    # n=128, so 128x128 matrix of 128-bit integers (Python ints).
    conflict_pairs = [[0] * n for _ in range(n)]
    
    for a, b, c in bad_triples:
        ia, ib, ic = a - 1, b - 1, c - 1
        conflict_pairs[ia][ib] |= (1 << ic)
        conflict_pairs[ib][ia] |= (1 << ic)
        
        conflict_pairs[ia][ic] |= (1 << ib)
        conflict_pairs[ic][ia] |= (1 << ib)
        
        conflict_pairs[ib][ic] |= (1 << ia)
        conflict_pairs[ic][ib] |= (1 << ia)

    # Precompute degree for heuristic
    # Degree of a node in the hypergraph context:
    # How many "choices" does it restrict?
    # Simpler heuristic: Count how many bad triples it is part of.
    degree = [0] * n
    for a, b, c in bad_triples:
        degree[a-1] += 1
        degree[b-1] += 1
        degree[c-1] += 1

    # 4. Backtracking Search
    # State: 
    #   candidates: bitmask of available numbers
    #   selected: bitmask of selected numbers
    #   current_size: int
    #   best_size: global max found
    #   best_mask: global best solution
    
    best_size = 0
    best_mask = 0
    full_mask = (1 << n) - 1
    
    # To speed up, we can order the candidates.
    # But the bitmask approach allows us to pick any candidate.
    # Heuristic: Pick the candidate with the highest degree among remaining candidates.
    # This requires scanning the candidates bitmask.
    
    # Optimization: 
    # If time is running out, return best found.
    
    def get_highest_degree_candidate(cand_mask):
        max_deg = -1
        best_cand = -1
        # Iterate through bits set in cand_mask
        m = cand_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if degree[idx] > max_deg:
                max_deg = degree[idx]
                best_cand = idx
            m ^= lsb
        return best_cand

    # Recursive function
    # Using a list to store best_state to avoid closure overhead in deep recursion?
    # Python recursion limit might be an issue if depth is 128, but we prune heavily.
    # However, B&B can be deep.
    # Let's use an iterative stack or increase recursion limit.
    import sys
    sys.setrecursionlimit(3000)
    
    def backtrack(cand_mask, selected_mask, current_size):
        nonlocal best_size, best_mask
        
        # Pruning: Upper bound
        # Max possible size = current_size + popcount(cand_mask)
        if current_size + cand_mask.bit_count() <= best_size:
            return
        
        if cand_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = selected_mask
            return
        
        # Check time
        if time.time() > deadline:
            return

        # Select candidate
        x_idx = get_highest_degree_candidate(cand_mask)
        if x_idx == -1:
            return
            
        x_bit = 1 << x_idx
        
        # Option 1: Include x_idx
        # Check if x_idx conflicts with selected_mask
        # We need to ensure no bad triple is formed.
        # A bad triple is formed if there are two elements u, v in selected_mask
        # such that (x_idx, u, v) is a bad triple.
        # This is equivalent to: 
        # For each u in selected_mask, if v is in selected_mask and v is in conflict_pairs[x_idx][u], then conflict.
        
        # Efficient check:
        # Let S = selected_mask.
        # We want to know if there exists u, v in S such that v is in conflict_pairs[x_idx][u].
        # This is: Does (conflict_pairs[x_idx][u] & S) contain any u' != u?
        # Actually, simpler:
        # For a fixed x, the set of pairs (u,v) in S that form a bad triple with x is determined by the conflict matrix.
        # If we include x, we must remove from candidates any y such that (x, y, z) is bad for some z in S.
        # This is: new_cand = cand_mask & ~(union of conflict_pairs[x][z] for z in S)
        
        # But first, is x valid to add?
        # x is valid if no two elements in S form a bad triple with x.
        # i.e., for all u in S, (conflict_pairs[x][u] & S) == 0? 
        # No. conflict_pairs[x][u] contains v such that (x,u,v) is bad.
        # If there is a v in S such that v is in conflict_pairs[x][u], then (x,u,v) is a bad triple in S U {x}.
        # So we must check:
        # invalid = False
        # temp_s = selected_mask
        # while temp_s:
        #    lsb = temp_s & -temp_s
        #    u_idx = lsb.bit_length() - 1
        #    if conflict_pairs[x_idx][u_idx] & selected_mask:
        #         # There is some v in selected_mask that conflicts with x and u
        #         invalid = True
        #         break
        #    temp_s ^= lsb
        # 
        # If not invalid:
        #    # Calculate forbidden candidates
        #    forbidden = 0
        #    temp_s = selected_mask
        #    while temp_s:
        #       lsb = temp_s & -temp_s
        #       u_idx = lsb.bit_length() - 1
        #       forbidden |= conflict_pairs[x_idx][u_idx]
        #       temp_s ^= lsb
        #    # Remove x and forbidden from candidates
        #    new_cand = (cand_mask ^ x_bit) & ~forbidden
        #    backtrack(new_cand, selected_mask | x_bit, current_size + 1)
        
        # Option 2: Exclude x_idx
        # backtrack(cand_mask ^ x_bit, selected_mask, current_size)
        
        # Implementation with optimization:
        # We can precompute the "forbidden mask" update incrementally?
        # No, it depends on S.
        
        # Let's implement the check.
        # To speed up, we can order the bits in selected_mask? No.
        
        invalid = False
        temp_s = selected_mask
        # Optimization: If selected_mask is empty, it's always valid.
        if selected_mask:
            while temp_s:
                lsb = temp_s & -temp_s
                u_idx = lsb.bit_length() - 1
                # If any v in selected_mask is in conflict_pairs[x_idx][u_idx], invalid
                if conflict_pairs[x_idx][u_idx] & selected_mask:
                    invalid = True
                    break
                temp_s ^= lsb
        
        if not invalid:
            # Calculate forbidden candidates
            forbidden = 0
            temp_s = selected_mask
            while temp_s:
                lsb = temp_s & -temp_s
                u_idx = lsb.bit_length() - 1
                forbidden |= conflict_pairs[x_idx][u_idx]
                temp_s ^= lsb
            
            new_cand = (cand_mask ^ x_bit) & ~forbidden
            backtrack(new_cand, selected_mask | x_bit, current_size + 1)
        
        # Exclude x
        backtrack(cand_mask ^ x_bit, selected_mask, current_size)

    # Start search
    # Initial candidates: all numbers
    backtrack(full_mask, 0, 0)
    
    if time.time() > deadline:
         # If we timed out, we might have a partial best.
         # But backtrack returns early on timeout.
         pass
         
    if best_mask == 0:
        return [1]
        
    # Convert mask to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        m ^= lsb
        
    return sorted(result)

# EVOLVE-BLOCK-END
