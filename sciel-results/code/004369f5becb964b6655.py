import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def check_cayley_graph(p, S):
        """
        Checks if the Cayley graph on Z_p with connection set S is K4-free and has alpha <= 5.
        S is a set of integers in {1, ..., p-1} such that if x in S then p-x in S.
        """
        # Build adjacency matrix or just use the set property
        # adj(i, j) iff (i-j) % p in S
        
        # 1. Check K4-free
        # A K4 is {a, b, c, d} such that all pairwise differences are in S.
        # Due to symmetry, we can fix a=0.
        # Then we need b, c, d in S such that (c-b)%p in S, (d-b)%p in S, (d-c)%p in S.
        # Essentially, S must not contain a set {b, c, d} forming a triangle in the "difference graph".
        
        # Efficient K4 check:
        # Iterate over b, c in S.
        # If (c-b) % p in S, then {0, b, c, (c-b)} is a candidate K4?
        # No, {0, b, c, d} is a K4 if d in S, (c-b) in S, (d-b) in S, (d-c) in S.
        # Let d = c - b. Then d in S?
        # If d in S, and (d-b) = c-2b in S, and (d-c) = -b in S (which is true since S is symmetric).
        # So we just need to check if there exist b, c in S such that (c-b) in S and (c-2b) in S.
        
        S_set = set(S)
        for b in S_set:
            for c in S_set:
                diff1 = (c - b) % p
                if diff1 in S_set:
                    diff2 = (c - 2*b) % p
                    if diff2 in S_set:
                        return False, 'K4'
                        
        # 2. Check alpha <= 5 (No independent set of size 6)
        # An independent set {x1, ..., x6} means all pairwise differences are NOT in S.
        # Let T = Z_p \ S. T is also symmetric.
        # We need to check if T contains a 6-clique in the "difference graph".
        # This is equivalent to finding 6 elements in T such that all pairwise differences are in T?
        # No. An independent set in G means edges are in T (complement of S).
        # So we need a set A of size 6 such that for all x, y in A, (x-y)%p in T.
        
        # We can use a backtracking search on the subset of T.
        # To speed this up, we can use the fact that T is small (density ~0.5).
        # However, for p=35, T has size 17.
        # We can precompute the "non-edges" structure.
        
        # Actually, checking for an independent set of size 6 in a graph of size 35 is fast with bitmasks.
        # Let's build the adjacency matrix for the independent set check (which is the complement graph).
        # adj_comp[i][j] = 1 if (i-j)%p in T.
        
        # We will use a standard DFS to find a clique of size 6 in the complement graph.
        # Vertices are 0..p-1.
        
        # Optimization: Use bitmasks for neighbors in the complement graph.
        # neighbors[i] = bitmask of j such that (i-j)%p in T.
        
        neighbors = [0] * p
        for i in range(p):
            mask = 0
            for j in range(p):
                if i == j: continue
                if (i - j) % p in T:
                    mask |= (1 << j)
            neighbors[i] = mask
            
        # Find clique of size 6 in this graph
        found = [False]
        
        def dfs_clique(candidates, depth, current_size):
            if found[0]:
                return
            if current_size == 6:
                found[0] = True
                return
            if bin(candidates).count('1') < (6 - current_size):
                return
                
            # Heuristic: pick vertex with highest degree in candidates?
            # Or just iterate.
            c_mask = candidates
            while c_mask:
                if found[0]:
                    return
                lsb = c_mask & -c_mask
                v = lsb.bit_length() - 1
                c_mask ^= lsb
                
                new_candidates = c_mask & neighbors[v]
                dfs_clique(new_candidates, depth + 1, current_size + 1)
                
        dfs_clique((1 << p) - 1, 0, 0)
        
        if found[0]:
            return False, 'I6'
            
        return True, None

    def get_edges_from_cayley(p, S):
        edges = []
        for i in range(p):
            for j in range(i + 1, p):
                if (i - j) % p in S:
                    edges.append([i, j])
        return edges

    # Strategy:
    # 1. Try primes p in [31, 33, 35] (35 is not prime, but we can use Z_35? 
    #    Cayley graphs are typically defined on groups. Z_35 is a group.
    #    However, the structure of Z_35 is not as "nice" as Z_p.
    #    Exoo's 35-vertex graph is NOT a Cayley graph on Z_35.
    #    But let's try Z_31, Z_33, Z_35 anyway.
    #    Actually, R(4,6) > 35 is the record.
    #    If we find a valid graph on 35, we score 35/17 > 2.0.
    #    If we find a valid graph on 36, we score 36/17 > 2.1.
    
    # Primes/Groups to try:
    # 31, 33, 35.
    # Also 37, 39, 41? The task says n <= 40.
    
    candidates = [35, 34, 33, 32, 31, 30]
    
    best_n = 0
    best_edges = []
    
    # Fallback: Paley 17
    n_p = 17
    squares = {x * x % n_p for x in range(1, n_p)}
    p_edges = [[i, j] for i in range(n_p) for j in range(i + 1, n_p) if (j - i) % n_p in squares]
    best_edges = p_edges
    best_n = n_p
    
    for p in candidates:
        if time.time() > deadline - 10:
            break
            
        # Initialize S
        # S must be symmetric: if x in S, p-x in S.
        # Elements are 1..p-1.
        # Pairs: (1, p-1), (2, p-2), ...
        
        pairs = []
        for x in range(1, p // 2 + 1):
            pairs.append((x, (p - x) % p))
            
        # Random initial S
        S = set()
        for x, y in pairs:
            if random.random() < 0.5:
                S.add(x)
                S.add(y)
                
        # Simulated Annealing
        current_S = S
        current_valid, _ = check_cayley_graph(p, current_S)
        
        # We want to find ANY valid S.
        # Cost function: 
        # If valid, cost = 0.
        # If invalid, cost = number of violations?
        # Counting violations is hard.
        # Let's just use a "random walk with restarts" or "hill climbing on validity".
        
        # Better: Minimize the size of the largest independent set and K4.
        # But that's expensive.
        
        # Let's try a simple approach:
        # 1. Randomly flip elements of S.
        # 2. Check validity.
        # 3. If valid, return.
        # 4. If not, keep with probability exp(-delta/T) if it "improves" some metric?
        #    Since we don't have a metric, let's just do Random Search with a twist:
        #    Bias towards density 0.5.
        
        # Actually, for p=31, there are known solutions.
        # Let's try to find one.
        
        attempts = 0
        max_attempts = 50000
        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        
        # To make SA work, we need a cost.
        # Let's define cost as:
        # 10 * (max clique size - 3) + (max independent set size - 5)
        # This requires finding max clique and max independent set, which is NP-hard.
        # But for p=35, we can approximate or just check for K4 and I6.
        
        # Let's stick to:
        # If valid, cost = 0.
        # If invalid, cost = 1.
        # This is a step function. SA won't work well.
        
        # Alternative: **Greedy Construction of S**
        # Start with empty S.
        # Add pairs one by one.
        # If adding a pair creates a K4 or I6, don't add it?
        # This might result in a sparse graph, which is bad for I6.
        
        # Let's try **Local Search with Random Restarts** on the binary string representing S.
        
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            # Random init
            current_S = set()
            for x, y in pairs:
                if random.random() < 0.5:
                    current_S.add(x)
                    current_S.add(y)
                    
            valid, _ = check_cayley_graph(p, current_S)
            if valid:
                best_n = p
                best_edges = get_edges_from_cayley(p, current_S)
                return {'n': best_n, 'edges': best_edges}
                
            # Local search
            for step in range(1000):
                if time.time() > deadline - 5:
                    break
                    
                # Flip a random pair
                idx = random.randint(0, len(pairs) - 1)
                x, y = pairs[idx]
                
                if x in current_S:
                    current_S.remove(x)
                    current_S.remove(y)
                else:
                    current_S.add(x)
                    current_S.add(y)
                    
                valid, _ = check_cayley_graph(p, current_S)
                if valid:
                    best_n = p
                    best_edges = get_edges_from_cayley(p, current_S)
                    return {'n': best_n, 'edges': best_edges}

    # If no Cayley graph found, return best found or fallback
    if best_n > 17:
        return {'n': best_n, 'edges': best_edges}
        
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
