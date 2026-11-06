# EVOLVE-BLOCK-START
import random
from math import gcd

def construct():
    N = 128
    # Precompute LCMs
    lcms = [[0]*N for _ in range(N)]
    for i in range(1, N):
        for j in range(i+1, N):
            lcms[i-1][j-1] = (i*j)//gcd(i,j)
            lcms[j-1][i-1] = lcms[i-1][j-1]
            
    # Precompute all bad triples
    bad_triples = []
    triples_by_node = {i: [] for i in range(1, N+1)}
    for a in range(1, N):
        for b in range(a+1, N):
            L_ab = lcms[a-1][b-1]
            for c in range(b+1, N):
                if lcms[a-1][c-1] == L_ab and lcms[b-1][c-1] == L_ab:
                    t = (a, b, c)
                    bad_triples.append(t)
                    triples_by_node[a].append(t)
                    triples_by_node[b].append(t)
                    triples_by_node[c].append(t)
                    
    best_set = set()
    best_size = 0
    
    # Randomized Greedy Construction with Local Search
    # We perform multiple restarts to find a larger set
    num_restarts = 5000
    
    for _ in range(num_restarts):
        S = set()
        # Shuffle order to break symmetry
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        
        # Greedy Add
        for x in candidates:
            # Check if x can be added
            conflict = False
            for t in triples_by_node[x]:
                # t is (a,b,c) containing x
                # Check if other two are in S
                # Optimization: t has 3 elements.
                # If x == t[0], check t[1], t[2]
                # If x == t[1], check t[0], t[2]
                # If x == t[2], check t[0], t[1]
                if x == t[0]:
                    if t[1] in S and t[2] in S:
                        conflict = True
                        break
                elif x == t[1]:
                    if t[0] in S and t[2] in S:
                        conflict = True
                        break
                else: # x == t[2]
                    if t[0] in S and t[1] in S:
                        conflict = True
                        break
            
            if not conflict:
                S.add(x)
        
        # Local Search: Try to add remaining elements
        # We iterate through elements not in S and try to add them.
        # If we can't add directly, we might try swaps later, but for now just additions.
        # To make it stronger, we can try to add elements in random order again.
        remaining = [x for x in range(1, N+1) if x not in S]
        random.shuffle(remaining)
        
        for x in remaining:
            conflict = False
            for t in triples_by_node[x]:
                if x == t[0]:
                    if t[1] in S and t[2] in S:
                        conflict = True
                        break
                elif x == t[1]:
                    if t[0] in S and t[2] in S:
                        conflict = True
                        break
                else:
                    if t[0] in S and t[1] in S:
                        conflict = True
                        break
            
            if not conflict:
                S.add(x)
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    return sorted(best_set)
# EVOLVE-BLOCK-END
