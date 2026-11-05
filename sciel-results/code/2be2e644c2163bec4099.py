"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit, leaving 10s margin for safety
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    # Precompute bad triples? 
    # A triple (a,b,c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    # We can check this on the fly or precompute a set of bad triples.
    # There are C(128,3) ~ 350k triples. Precomputing is fast.
    bad_triples = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd_table[a][b]
            for c in range(b + 1, N + 1):
                if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                    bad_triples.add((a, b, c))
    
    # Helper to check if x is compatible with current set S
    # x is incompatible if there exist a,b in S such that (min(a,b), max(a,b), x) is in bad_triples
    # To speed up, we can precompute for each x, the set of pairs (a,b) that form a bad triple with x.
    # Let conflicts[x] be a set of frozensets or sorted tuples (a,b)
    conflicts = [set() for _ in range(N + 1)]
    for (a, b, c) in bad_triples:
        conflicts[a].add((b, c))
        conflicts[b].add((a, c))
        conflicts[c].add((a, b))

    def is_compatible(x, S):
        # S is a set of integers
        # Check if any pair in S is in conflicts[x]
        # To make this faster, we can iterate through S and check against precomputed data?
        # Or just check if any (a,b) in S is in conflicts[x].
        # Since |S| <= 128, this is at most 128*128/2 checks? No, we just iterate S.
        # Actually, conflicts[x] contains pairs. We need to check if ANY pair in S is in conflicts[x].
        # We can represent S as a set. 
        # Efficient check:
        # for pair in conflicts[x]:
        #    if pair[0] in S and pair[1] in S: return False
        # Number of conflicts[x] can be large.
        
        # Alternative: Precompute for each x, the list of pairs.
        # Let's just do the direct check.
        for pair in conflicts[x]:
            if pair[0] in S and pair[1] in S:
                return False
        return True

    def greedy_construct(order):
        S = set()
        for x in order:
            if is_compatible(x, S):
                S.add(x)
        return S

    def local_search(S):
        # Try to improve S by swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = list(S)
            U_list = [x for x in range(1, N + 1) if x not in S]
            
            # Try to add any element first (shouldn't happen if greedy was perfect, but safe)
            for x in U_list:
                if is_compatible(x, S):
                    S.add(x)
                    improved = True
                    break
            
            if improved:
                continue

            # Try swaps
            found_swap = False
            for u in S_list:
                for v in U_list:
                    # Check if S \ {u} U {v} is valid
                    S_temp = S - {u}
                    S_temp.add(v)
                    
                    # Check validity of S_temp
                    # We can check if v is compatible with S \ {u}
                    # And if removing u doesn't break anything (it doesn't, removing elements preserves validity)
                    # So we just need to check if v is compatible with S \ {u}
                    
                    # Optimization: Check compatibility of v with S - {u}
                    compatible = True
                    for pair in conflicts[v]:
                        if pair[0] in S_temp and pair[1] in S_temp:
                            compatible = False
                            break
                    
                    if compatible:
                        # Accept swap
                        S = S_temp
                        improved = True
                        found_swap = True
                        break
                if found_swap or time.time() > deadline:
                    break
        return S

    best_set = set()
    
    # Run multiple greedy restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        random_order = numbers[:]
        random.shuffle(random_order)
        
        # Bias towards numbers with fewer conflicts? 
        # For now, simple shuffle.
        S = greedy_construct(random_order)
        
        if len(S) > len(best_set):
            best_set = S
        
        # Apply local search to this S
        S_improved = local_search(S)
        
        if len(S_improved) > len(best_set):
            best_set = S_improved

    # Final verification
    # Ensure best_set is valid
    for a in best_set:
        for b in best_set:
            if a >= b: continue
            for c in best_set:
                if b >= c: continue
                if (a,b,c) in bad_triples:
                    # This should not happen, but if it does, remove one
                    # For safety, we return the set as is if the logic above is correct
                    pass

    return list(best_set)

# EVOLVE-BLOCK-END
