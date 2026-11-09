import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 280  # 20s safety margin
    
    # Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    # Adjacency list as bitmasks for efficiency
    # adj[i] is an integer bitmask where bit j is 1 if (i,j) is an edge
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    def get_edges(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            # Iterate set bits
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append((i, j))
                m ^= lsb
        return edges

    def find_k4s(adj, n):
        k4s = []
        # A K4 is a set of 4 vertices {a,b,c,d} such that all 6 edges exist.
        # We can iterate through combinations of 4 vertices.
        # n is small (37-42), so C(42,4) is ~100k, which is fast.
        for combo in combinations(range(n), 4):
            a, b, c, d = combo
            if (adj[a] & (1 << b)) and (adj[a] & (1 << c)) and (adj[a] & (1 << d)) and \
               (adj[b] & (1 << c)) and (adj[b] & (1 << d)) and \
               (adj[c] & (1 << d)):
                k4s.append(combo)
        return k4s

    def find_ind4s(adj, n):
        ind4s = []
        # An independent set of 4 is a set of 4 vertices with no edges between them.
        for combo in combinations(range(n), 4):
            a, b, c, d = combo
            if not (adj[a] & (1 << b)) and not (adj[a] & (1 << c)) and not (adj[a] & (1 << d)) and \
               not (adj[b] & (1 << c)) and not (adj[b] & (1 << d)) and \
               not (adj[c] & (1 << d)):
                ind4s.append(combo)
        return ind4s

    def try_add_vertex(adj, n, k4s, ind4s, deadline):
        # We want to find S subset of V(n) such that:
        # 1. No K4 is a subset of S.
        # 2. No Ind4 is disjoint from S (i.e., intersects S).
        
        # Precompute masks for K4s and Ind4s to check quickly
        k4_masks = []
        for C in k4s:
            m = 0
            for v in C:
                m |= (1 << v)
            k4_masks.append(m)
        
        ind4_masks = []
        for I in ind4s:
            m = 0
            for v in I:
                m |= (1 << v)
            ind4_masks.append(m)
            
        def fast_score(S):
            v1 = 0
            for m in k4_masks:
                if (S & m) == m:
                    v1 += 1
            v2 = 0
            for m in ind4_masks:
                if (S & m) == 0:
                    v2 += 1
            return v1 + v2

        # Simulated annealing with multiple restarts
        best_S = None
        best_score = float('inf')
        
        # Strategy: Start with a heuristic initialization
        # For each vertex, estimate how "safe" it is to include
        # A vertex is safer to include if it's in fewer K4s and in more Ind4s
        
        k4_count = [0] * n
        ind4_count = [0] * n
        for m in k4_masks:
            temp = m
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                k4_count[v] += 1
                temp ^= lsb
        for m in ind4_masks:
            temp = m
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                ind4_count[v] += 1
                temp ^= lsb
        
        # Heuristic: prefer vertices with low k4_count and high ind4_count
        # Score for including vertex v: -k4_count[v] + ind4_count[v]
        vertex_scores = [ind4_count[v] - k4_count[v] for v in range(n)]
        
        # Try multiple initializations
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            
            if restart == 0:
                # Greedy initialization: include vertices with positive score
                S = 0
                for v in range(n):
                    if vertex_scores[v] > 0:
                        S |= (1 << v)
            elif restart == 1:
                # Include top half by score
                sorted_vertices = sorted(range(n), key=lambda v: vertex_scores[v], reverse=True)
                S = 0
                for v in sorted_vertices[:n//2]:
                    S |= (1 << v)
            else:
                # Random initialization with bias
                S = 0
                for v in range(n):
                    # Probability based on score
                    prob = 0.5 + 0.3 * (vertex_scores[v] - max(vertex_scores)) / (max(vertex_scores) - min(vertex_scores) + 1)
                    if random.random() < prob:
                        S |= (1 << v)
            
            # Simulated annealing
            curr_score = fast_score(S)
            if curr_score == 0:
                return S
            best_S = S
            best_score = curr_score
            
            T = 10.0
            cooling = 0.995
            for step in range(5000):
                if time.time() > deadline - 5:
                    break
                
                if curr_score == 0:
                    return S
                
                # Pick a vertex to flip
                v = random.randint(0, n - 1)
                S_new = S ^ (1 << v)
                new_score = fast_score(S_new)
                
                delta = new_score - curr_score
                if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                    S = S_new
                    curr_score = new_score
                    if curr_score < best_score:
                        best_score = curr_score
                        best_S = S
                
                T *= cooling
            
            if best_score == 0:
                return best_S
        
        return best_S if best_score == 0 else None

    # Iteratively add vertices
    for target_n in range(n + 1, 47): # Try up to 46
        if time.time() > deadline - 10:
            break
        
        # Update K4s and Ind4s
        k4s = find_k4s(adj, n)
        ind4s = find_ind4s(adj, n)
        
        # Try to add a vertex
        added = False
        attempts = 0
        while attempts < 30 and not added and time.time() < deadline - 10:
            attempts += 1
            S = try_add_vertex(adj, n, k4s, ind4s, deadline)
            if S is not None:
                # Add new vertex n
                new_adj = adj + [0]
                # Connect new vertex to bits set in S
                for v in range(n):
                    if S & (1 << v):
                        new_adj[n] |= (1 << v)
                        new_adj[v] |= (1 << n)
                adj = new_adj
                n += 1
                added = True
                # Reset attempts for next vertex
                attempts = 0
                
        if not added:
            break
            
    # Final polish: try edge flips to see if we can extend further
    # Try flipping edges to create opportunities for adding more vertices
    for polish_iter in range(10):
        if time.time() > deadline - 5:
            break
        
        # Try to add another vertex
        k4s = find_k4s(adj, n)
        ind4s = find_ind4s(adj, n)
        S = try_add_vertex(adj, n, k4s, ind4s, deadline)
        if S is not None:
            new_adj = adj + [0]
            for v in range(n):
                if S & (1 << v):
                    new_adj[n] |= (1 << v)
                    new_adj[v] |= (1 << n)
            adj = new_adj
            n += 1
        else:
            # Try flipping a random edge to break a constraint
            if n < 46:
                # Find a random edge to flip
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j and i < j:
                    # Try flipping this edge
                    old_adj = adj[:]
                    if adj[i] & (1 << j):
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                    else:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                    
                    # Check if we can now add a vertex
                    k4s_new = find_k4s(adj, n)
                    ind4s_new = find_ind4s(adj, n)
                    S_new = try_add_vertex(adj, n, k4s_new, ind4s_new, deadline)
                    if S_new is not None:
                        new_adj = adj + [0]
                        for v in range(n):
                            if S_new & (1 << v):
                                new_adj[n] |= (1 << v)
                                new_adj[v] |= (1 << n)
                        adj = new_adj
                        n += 1
                    else:
                        # Revert the edge flip
                        adj = old_adj
            break
            
    # Return the best graph found
    edges = get_edges(adj, n)
    return n, edges

# EVOLVE-BLOCK-END
