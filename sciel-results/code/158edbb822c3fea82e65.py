import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 128
    # Precompute GCD matrix
    G = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            G[i][j] = g
            G[j][i] = g
            
    # Precompute BadPairs for each x
    # BadPairs[x] is a set of tuples (a, b) with a < b
    # such that G[a,b] == G[a,x] == G[b,x]
    BadPairs = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x: continue
            for b in range(a + 1, N + 1):
                if b == x: continue
                if G[a][b] == G[a][x] and G[a][b] == G[b][x]:
                    BadPairs[x].add((a, b))
                    
    # Helper to add a pair to PairsInS in canonical order
    def add_pair_to_set(pairs_set, a, b):
        if a > b:
            a, b = b, a
        pairs_set.add((a, b))

    # Greedy Initial Construction
    S = []
    InS = [False] * (N + 1)
    PairsInS = set()
    
    for x in range(1, N + 1):
        # Check if x can be added
        valid = True
        # Check against all pairs in S
        # For speed, we can just check BadPairs[x] intersection with PairsInS
        if PairsInS & BadPairs[x]:
            valid = False
        if valid:
            S.append(x)
            InS[x] = True
            # Update PairsInS
            for a in S[:-1]:
                add_pair_to_set(PairsInS, a, x)
                
    best_S = S.copy()
    best_size = len(S)
    
    # Local Search
    iterations = 0
    
    while time.time() < deadline:
        iterations += 1
        
        # Randomly choose move type
        if random.random() < 0.5:
            # Try Add
            candidates = [x for x in range(1, N + 1) if not InS[x]]
            if not candidates:
                continue
            x = random.choice(candidates)
            
            if not (PairsInS & BadPairs[x]):
                # Valid add
                S.append(x)
                InS[x] = True
                for a in S[:-1]:
                    add_pair_to_set(PairsInS, a, x)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S.copy()
        else:
            # Try Swap
            if not S:
                continue
            u = random.choice(S)
            candidates = [x for x in range(1, N + 1) if not InS[x]]
            if not candidates:
                continue
            v = random.choice(candidates)
            
            # Check if (S \ {u}) U {v} is valid
            valid_swap = True
            for pair in BadPairs[v]:
                if pair in PairsInS:
                    a, b = pair
                    if a != u and b != u:
                        valid_swap = False
                        break
            
            if valid_swap:
                # Accept swap
                S.remove(u)
                InS[u] = False
                # Remove pairs involving u from PairsInS
                to_remove = [p for p in PairsInS if u in p]
                for p in to_remove:
                    PairsInS.discard(p)
                
                # Add v
                S.append(v)
                InS[v] = True
                for a in S[:-1]:
                    add_pair_to_set(PairsInS, a, v)
                
                # Check if this new state allows immediate improvement or is just for exploration
                # We don't update best_S here as size is same, but it might enable future adds.

    # Final Validation and Cleanup
    # Ensure the returned set is valid. If not, remove elements from forbidden triples.
    result = best_S.copy()
    InResult = [False] * (N + 1)
    for x in result:
        InResult[x] = True
        
    # Re-verify validity and fix if necessary
    changed = True
    while changed and time.time() < deadline - 1.0:
        changed = False
        # Find a forbidden triple
        R = result
        forbidden_found = None
        for i in range(len(R)):
            for j in range(i + 1, len(R)):
                for k in range(j + 1, len(R)):
                    a, b, c = R[i], R[j], R[k]
                    if G[a][b] == G[a][c] and G[a][b] == G[b][c]:
                        forbidden_found = (a, b, c)
                        break
                if forbidden_found:
                    break
            if forbidden_found:
                break
        
        if forbidden_found:
            a, b, c = forbidden_found
            # Remove one of them (e.g., the largest or random)
            to_remove = max(a, b, c)
            result.remove(to_remove)
            InResult[to_remove] = False
            changed = True
            
    return result
