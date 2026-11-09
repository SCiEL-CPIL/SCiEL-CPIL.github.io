import time
from math import lcm
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin
    
    N = 128
    all_nums = list(range(1, N + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[i][j] = lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Precompute forbidden triples
    # A triple (a,b,c) is forbidden if lcm(a,b)==lcm(a,c)==lcm(b,c)
    # For each pair (a,b), find all c that form a forbidden triple with them
    # Then for each element, we know which pairs it can't be in with
    
    # Better: for each element x, precompute the set of pairs (a,b) 
    # such that {a,b,x} is forbidden
    # This lets us check in O(1) per pair whether adding x creates a violation
    
    # Actually, let's precompute: for each x, list of (a,b) where a<b, 
    # and lcm(a,b)==lcm(a,x)==lcm(b,x)
    # When checking if x can join S, we check if any (a,b) in this list 
    # has both a and b in S
    
    forb_pairs = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            ab_lcm = lcm_table[a][b]
            for c in range(b + 1, N + 1):
                if lcm_table[a][c] == ab_lcm and lcm_table[b][c] == ab_lcm:
                    forb_pairs[a].append((b, c))
                    forb_pairs[b].append((a, c))
                    forb_pairs[c].append((a, b))
    
    def can_add(x, S):
        """Check if x can be added to set S (represented as boolean list)."""
        for a, b in forb_pairs[x]:
            if S[a] and S[b]:
                return False
        return True
    
    def greedy(order):
        """Greedy construction given an ordering of elements."""
        S = [False] * (N + 1)
        size = 0
        for x in order:
            if can_add(x, S):
                S[x] = True
                size += 1
        return S, size
    
    def local_search(S, size):
        """Try to improve S by removing one element and re-greedy filling."""
        current_S = S[:]
        current_size = size
        
        for u in range(1, N + 1):
            if not current_S[u]:
                continue
            # Try removing u
            current_S[u] = False
            temp_S = current_S[:]
            temp_size = current_size - 1
            
            # Re-greedy with random order on remaining
            remaining = [x for x in range(1, N + 1) if not temp_S[x]]
            random.shuffle(remaining)
            for x in remaining:
                if can_add(x, temp_S):
                    temp_S[x] = True
                    temp_size += 1
            
            if temp_size > current_size:
                return temp_S, temp_size
        
        return current_S, current_size
    
    # Multiple greedy strategies
    best_S = None
    best_size = 0
    
    # Strategy 1: Ascending order
    S, size = greedy(list(range(1, N + 1)))
    if size > best_size:
        best_size = size
        best_S = S[:]
    
    # Strategy 2: Descending order
    S, size = greedy(list(range(N, 0, -1)))
    if size > best_size:
        best_size = size
        best_S = S[:]
    
    # Strategy 3: Random restarts
    iterations = 0
    while time.time() < deadline - 10:
        order = list(range(1, N + 1))
        random.shuffle(order)
        S, size = greedy(order)
        if size > best_size:
            best_size = size
            best_S = S[:]
        iterations += 1
    
    # Local search polish
    if best_S and time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            S, size = local_search(best_S, best_size)
            if size > best_size:
                best_size = size
                best_S = S[:]
                improved = True
    
    # Convert to list
    result = [i for i in range(1, N + 1) if best_S[i]]
    return result
