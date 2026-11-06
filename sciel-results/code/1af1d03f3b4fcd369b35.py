import random
import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 128
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # Precompute bad triples
    # A triple (a,b,c) with a<b<c is bad if gcd(a,b)==gcd(b,c)==gcd(c,a)
    bad_triples = []
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            g_ab = gcd_table[a][b]
            for c in range(b+1, N+1):
                g_bc = gcd_table[b][c]
                if g_ab == g_bc:
                    g_ca = gcd_table[c][a]
                    if g_ab == g_ca:
                        bad_triples.append((a, b, c))
    
    # Build conflict hyperedges: for each element, which other elements form bad triples with it
    # conflict[x] = set of (y, z) pairs such that {x,y,z} is a bad triple
    # For efficient checking, we'll use a different approach:
    # For each pair (a,b), store which c values make (a,b,c) bad
    pair_conflicts = {}
    for (a, b, c) in bad_triples:
        pair_conflicts.setdefault((a,b), set()).add(c)
        pair_conflicts.setdefault((b,a), set()).add(c)
    
    # Build for each element x, the set of elements y such that there exists z with (x,y,z) bad
    # And more specifically, for greedy/branch-bound, we need:
    # If we include x and y, what z's are forbidden?
    
    # For branch and bound, we model this as: find largest subset with no bad triple
    # Use bitset representation for speed
    # Elements are 1..128, map to 0..127 for bit operations
    
    # Precompute: for each pair (i,j), the bitmask of k such that (i,j,k) is bad
    # i,j,k are 0-indexed (representing 1..128)
    pair_bad_mask = [[0]*N for _ in range(N)]
    for (a, b, c) in bad_triples:
        ai, bi, ci = a-1, b-1, c-1
        pair_bad_mask[ai][bi] |= (1 << ci)
        pair_bad_mask[bi][ai] |= (1 << ci)
    
    # Also for each element i, precompute all pairs (j,k) that form bad triple with i
    # This helps in computing the "forbidden" set when adding i
    
    # Branch and bound for maximum independent set in 3-uniform hypergraph
    # State: current set S (bitset), candidate set C (bitset of elements not yet decided)
    # When we add element i to S, we must remove from C all j such that there exists k in S
    # with (i,j,k) bad, AND all j,k such that (i,j,k) bad for any k
    
    # Actually, the constraint is: no three elements in S form a bad triple.
    # So if S contains {i,j}, we cannot add any k where (i,j,k) is bad.
    # If S contains {i,j,k}, that's already invalid.
    
    # For branch and bound:
    # Upper bound: |S| + |C| (optimistic: assume all remaining can be added)
    # We can improve: for each i in C, count how many elements in C are "blocked" by i
    # But simple bound may suffice with good ordering
    
    # Let's use a recursive branch and bound with memoization on the candidate set
    # But 2^128 is too large, so we need good pruning
    
    # Alternative: use the fact that the problem is small enough for a smarter approach
    # Let's try: greedy construction + local search with 1-for-2 swaps
    
    # First, get a good initial solution via greedy
    def greedy_construct():
        S = set()
        # Try elements in order of "least restrictive" - those involved in fewest bad triples
        triple_count = [0]*N
        for (a,b,c) in bad_triples:
            triple_count[a-1] += 1
            triple_count[b-1] += 1
            triple_count[c-1] += 1
        
        # Sort by triple count (ascending)
        order = sorted(range(N), key=lambda x: triple_count[x])
        
        for i in order:
            # Check if adding i creates a bad triple with any two in S
            valid = True
            S_list = list(S)
            for j in range(len(S_list)):
                for k in range(j+1, len(S_list)):
                    sj, sk = S_list[j], S_list[k]
                    if (1 << i) & pair_bad_mask[sj][sk]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.add(i)
        return S
    
    # Local search: 1-for-2 swaps (remove one, add two)
    def local_search(S, time_limit=30):
        S = set(S)
        best = set(S)
        best_size = len(S)
        start = time.time()
        
        while time.time() - start < time_limit:
            improved = False
            
            # Try 1-for-1 swaps: remove i, add j
            for i in list(S):
                for j in range(N):
                    if j in S:
                        continue
                    # Check if S - {i} + {j} is valid
                    S_new = S - {i}
                    S_new.add(j)
                    # Check validity
                    valid = True
                    S_list = list(S_new)
                    for a in range(len(S_list)):
                        for b in range(a+1, len(S_list)):
                            sa, sb = S_list[a], S_list[b]
                            # Check if any c in S_list makes (sa,sb,c) bad
                            mask = pair_bad_mask[sa][sb]
                            if mask & (1 << j) and j != sa and j != sb:
                                # j is in the forbidden set for pair (sa,sb)
                                valid = False
                                break
                        if not valid:
                            break
                    if valid and len(S_new) > best_size:
                        S = S_new
                        best = set(S)
                        best_size = len(S)
                        improved = True
            
            # Try 1-for-2 swaps: remove i, add j and k
            if not improved:
                for i in list(S):
                    S_without_i = S - {i}
                    for j in range(N):
                        if j in S_without_i:
                            continue
                        for k in range(j+1, N):
                            if k in S_without_i:
                                continue
                            S_new = S_without_i | {j, k}
                            # Check validity
                            valid = True
                            S_list = list(S_new)
                            for a in range(len(S_list)):
                                for b in range(a+1, len(S_list)):
                                    sa, sb = S_list[a], S_list[b]
                                    mask = pair_bad_mask[sa][sb]
                                    # Check if any element in S_new is in the forbidden set
                                    # Forbidden for pair (sa,sb) are elements c where (sa,sb,c) bad
                                    # We need to check if any c in S_new (other than sa,sb) is forbidden
                                    forbidden_in_set = mask & ((1 << N) - 1)
                                    # Remove sa and sb from consideration
                                    forbidden_in_set &= ~(1 << sa)
                                    forbidden_in_set &= ~(1 << sb)
                                    # Check intersection with S_new
                                    S_new_mask = 0
                                    for x in S_new:
                                        S_new_mask |= (1 << x)
                                    if forbidden_in_set & S_new_mask:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid and len(S_new) > best_size:
                                S = S_new
                                best = set(S)
                                best_size = len(S)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best
    
    # Get initial solution
    initial = greedy_construct()
    best = local_search(initial, time_limit=45)
    
    # Convert to list of 1-indexed values
    result = sorted(x+1 for x in best)
    
    return result
