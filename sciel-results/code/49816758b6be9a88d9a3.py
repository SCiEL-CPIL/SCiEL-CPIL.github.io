import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    N = 128
    # Precompute LCMs
    # lcm[i][j] for 1 <= i,j <= N
    # Using 1-based indexing for simplicity, 0 is unused
    lcm_table = [[1] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            l = i * j // gcd(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l
            
    S = set(range(1, N + 1))
    S_list = sorted(S) # Maintain sorted list for efficiency

    def find_bad_triple_and_degrees(s_set, s_list):
        """
        Finds bad triples. To save time, we don't necessarily need ALL of them if we just want to prune.
        However, for degree counting, we need them.
        Optimization: Early exit if we just want *a* triple? 
        The current strategy uses degrees. Let's keep it but make it fast.
        """
        n = len(s_list)
        counts = {}
        # We will just find ONE bad triple and return it, and also do a fast degree count?
        # No, let's just find one bad triple. The "degree" heuristic is expensive.
        # Alternative heuristic: Remove the largest element in the first bad triple found.
        # This is O(N^3) worst case, but often finds one early.
        
        for i in range(n):
            a = s_list[i]
            for j in range(i + 1, n):
                b = s_list[j]
                lab = lcm_table[a][b]
                for k in range(j + 1, n):
                    c = s_list[k]
                    if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                        return (a, b, c)
        return None

    # Pruning Phase
    # Strategy: Find a bad triple, remove the largest element from it.
    # Why largest? In many combinatorial designs, removing the "maximal" element in a conflict
    # preserves the "minimal" elements which are more constrained.
    # Let's test this logic. If it fails, we can revert to degree counting.
    # Given the time, simple is better.
    
    while time.time() < deadline:
        triple = find_bad_triple_and_degrees(S, S_list)
        if triple is None:
            break
        
        a, b, c = triple
        # Remove the largest element
        remove_val = max(a, b, c)
        S.remove(remove_val)
        S_list.remove(remove_val) # O(N) but N is small

    # Polish Phase: Try to add back removed elements
    # We need to check if adding x creates a bad triple.
    
    def can_add(x, current_set, current_list):
        if x in current_set:
            return True
        # Check if x forms a bad triple with any pair in current_set
        # We need to check all pairs (a,b) in current_list
        n = len(current_list)
        for i in range(n):
            a = current_list[i]
            for j in range(i + 1, n):
                b = current_list[j]
                # Check if {a, b, x} is a bad triple
                lab = lcm_table[a][b]
                lax = lcm_table[a][x]
                if lab != lax:
                    continue
                lxb = lcm_table[b][x]
                if lab == lxb:
                    return False
        return True

    # Try to add elements
    # Iterate from 1 to N. 
    # To potentially get a better set, we might want to try adding in an order that
    # allows more elements. Smaller elements are more "conflicting".
    # If we add smaller elements first, we might block larger ones.
    # If we add larger elements first, we might block smaller ones.
    # Usually, it's better to add elements that are "harder to place" first?
    # Or just iterate 1..N.
    
    # Let's do multiple passes.
    for _ in range(2): # 2 passes
        if time.time() > deadline:
            break
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if x not in S:
                if can_add(x, S, S_list):
                    S.add(x)
                    S_list.append(x)
                    S_list.sort() # Keep it sorted

    return list(S)
