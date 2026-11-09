import sys
from itertools import combinations

def construct():
    N = 128
    # 1. Precompute LCMs
    # lcm[i][j] for 1 <= i, j <= N
    # Using math.lcm is fast enough for 128*128
    from math import lcm
    
    # 2. Identify Bad Triples
    # bad_triples is a set of frozensets {a, b, c}
    # Or better, for each pair (a, b), store the set of c such that {a,b,c} is bad.
    # pair_bad[a][b] = set of c
    
    # Indices 1..128
    # We'll use 0..127 for bitsets, so value v corresponds to bit v-1.
    
    pair_bad = [[set() for _ in range(N)] for _ in range(N)]
    
    # Precompute LCM matrix
    lcm_mat = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            lcm_mat[i][j] = lcm(i+1, j+1)
            lcm_mat[j][i] = lcm_mat[i][j]
            
    # Find bad triples
    # Iterate over all triples? 128 choose 3 is ~340k. That's fine.
    # But we can do it smarter.
    # For each pair (i, j), L = lcm(i, j).
    # Any k such that lcm(i, k) == L and lcm(j, k) == L forms a bad triple.
    # Condition: k divides L, and for each prime power p^e in L, if i or j has p^e, then k must have p^e? No.
    # lcm(i, k) == L implies k divides L and for all p, max(v_p(i), v_p(k)) = v_p(L).
    # This means if v_p(i) < v_p(L), then v_p(k) must be v_p(L).
    # If v_p(i) == v_p(L), v_p(k) can be anything <= v_p(L).
    
    # Let's just iterate k for each pair (i, j) to be safe and simple, N is small.
    for i in range(N):
        for j in range(i+1, N):
            L = lcm_mat[i][j]
            # k must be distinct from i, j
            for k in range(N):
                if k == i or k == j:
                    continue
                # Check if {i, j, k} is bad
                if lcm_mat[i][k] == L and lcm_mat[j][k] == L:
                    pair_bad[i][j].add(k)
                    pair_bad[j][i].add(k)
                    # Also update for pairs (i,k) and (j,k)
                    pair_bad[i][k].add(j)
                    pair_bad[k][i].add(j)
                    pair_bad[j][k].add(i)
                    pair_bad[k][j].add(i)
                    
    # 3. Branch and Bound
    # We want to maximize the number of selected elements.
    # State:
    #   selected: bitmask of selected elements
    #   candidates: bitmask of elements not yet decided (initially all 1s)
    #   current_count: number of bits in selected
    
    # To speed up, we can order candidates.
    # A good heuristic: try elements with fewest conflicts first? Or most?
    # Usually, in MIS, picking high degree nodes first helps prune.
    # Here, "conflict" is forming a bad triple with 2 others.
    
    # Let's use a recursive function.
    # To avoid deep recursion limits and overhead, we can use a stack or just standard recursion with sys.setrecursionlimit.
    # N=128, depth max 128.
    
    best_count = 0
    best_set = 0
    
    # Precompute list of bits for iteration
    # We will iterate through candidates.
    
    def get_conflicts_with_selected(new_bit, selected_mask):
        # Check if adding new_bit to selected_mask creates any bad triple.
        # A bad triple is {a, b, new_bit} where a, b are in selected_mask.
        # This is bad if new_bit is in pair_bad[a][b].
        # So we need to check if there exist a, b in selected_mask such that new_bit in pair_bad[a][b].
        
        # Iterate over bits in selected_mask
        # selected_mask is an int.
        # We can iterate over set bits.
        
        # Optimization: For each new_bit, we can precompute a mask of "forbidden pairs"?
        # No, pair_bad is a mapping from pair to set of elements.
        # Let's invert it: for each element x, what pairs (a,b) are "dangerous" for x?
        # dangerous_pairs[x] = set of (a,b) such that x in pair_bad[a][b].
        # Then adding x is invalid if any pair (a,b) from dangerous_pairs[x] is fully contained in selected_mask.
        
        pass

    # Let's precompute dangerous_pairs_mask
    # For each x, we want to know if there is a pair (a,b) in selected_mask such that x conflicts with them.
    # This is equivalent to: Is there any a in selected_mask such that there is a b in selected_mask and x in pair_bad[a][b]?
    
    # Let's create for each x, a list of pairs (a,b) that conflict with x.
    # conflict_pairs[x] = list of (a, b)
    
    conflict_pairs = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            for k in pair_bad[i][j]:
                conflict_pairs[k].append((i, j))
                
    def is_valid_addition(x, selected_mask):
        # Check if x conflicts with any pair in selected_mask
        for a, b in conflict_pairs[x]:
            # Check if both a and b are in selected_mask
            # Bit a is 1 << a
            # Bit b is 1 << b
            if (selected_mask & (1 << a)) and (selected_mask & (1 << b)):
                return False
        return True

    # Heuristic for candidate order:
    # We can sort candidates by some heuristic.
    # Let's try a simple DFS with branching.
    # To improve pruning, we can use an upper bound based on the number of remaining candidates.
    # A stronger bound: Greedy coloring of the conflict graph induced by remaining candidates?
    # Given N=128 and low density, simple bound might suffice if we branch well.
    
    # Let's try to include elements in an order that maximizes constraints early.
    # Elements that are part of many bad triples should be decided early.
    
    # Calculate "degree" (number of bad triples involving x)
    degrees = [0] * N
    for i in range(N):
        for j in range(i+1, N):
            for k in pair_bad[i][j]:
                degrees[k] += 1
                # Note: each bad triple {i,j,k} contributes to degrees of i, j, k.
                # The loop above adds k for pair (i,j).
                # We need to be careful not to double count or miss.
                # Actually, for a specific triple {i,j,k}, it is counted in pair_bad[i][j] (adds k), pair_bad[i][k] (adds j), pair_bad[j][k] (adds i).
                # So the loop `for k in pair_bad[i][j]: degrees[k] += 1` counts each incidence of k in a bad triple with i,j.
                # This is correct for "number of bad triples containing x".
                
    # Sort indices by degree descending
    order = sorted(range(N), key=lambda x: degrees[x], reverse=True)
    
    # Map original index to position in order?
    # Or just iterate through `order` in the recursion?
    # If we iterate through `order`, we need to maintain the state of which ones are decided.
    # Let's use a bitmask `decided` and `selected`.
    # `candidates` are those not in `decided`.
    
    # Recursive function
    def solve(idx, selected_mask, count):
        nonlocal best_count, best_set
        
        # Upper bound: count + (N - idx)
        # Since we process in fixed order `order`, the remaining elements are order[idx:]
        if count + (N - idx) <= best_count:
            return
            
        if idx == N:
            if count > best_count:
                best_count = count
                best_set = selected_mask
            return
            
        x = order[idx]
        
        # Option 1: Exclude x
        # The next index is idx + 1
        solve(idx + 1, selected_mask, count)
        
        # Option 2: Include x
        # Check if valid
        if is_valid_addition(x, selected_mask):
            solve(idx + 1, selected_mask | (1 << x), count + 1)
            
    # Start
    solve(0, 0, 0)
    
    # Convert best_set to list of integers
    result = []
    for i in range(N):
        if best_set & (1 << i):
            result.append(i + 1)
            
    if not result:
        # Fallback if something went wrong (should not happen)
        return [1]
        
    return result

# EVOLVE-BLOCK-END
