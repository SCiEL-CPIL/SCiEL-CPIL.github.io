import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Fallback: Paley graph on 17 vertices
    def get_paley_17():
        squares = {x * x % 17 for x in range(1, 17)}
        edges = []
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    edges.append([i, j])
        return {'n': 17, 'edges': edges}

    best_result = get_paley_17()
    
    # Target n values to try
    targets = [36, 35, 34, 33, 32, 31, 30, 25, 20]
    
    for n in targets:
        if time.time() > deadline:
            break
        # Try to find a valid graph for this n
        result = try_find_graph(n, deadline)
        if result:
            # Verify the result before accepting
            if is_valid(result):
                best_result = result
                # If we found 36 or 35, we can stop as it's optimal or near-optimal
                if n >= 35:
                    return best_result
                # Continue to try for larger n? No, we iterate descending. 
                # Wait, the loop is descending. So if we find 35, we should check 36 first.
                # My loop is [36, 35, ...]. So if 36 fails, we try 35.
                # If we find 35, we return it because 36 already failed.
                # Actually, if we find 35, it's a great score. 
                # We should probably return immediately if n >= 35 to save time?
                # But what if 36 was possible but we just failed to find it?
                # Given the difficulty, finding 35 is a huge win.
                # Let's return if n >= 35.
                return best_result

    return best_result

def is_valid(graph):
    n = graph['n']
    edges = graph['edges']
    # Build adjacency matrix
    A = [[0]*n for _ in range(n)]
    for u, v in edges:
        A[u][v] = 1
        A[v][u] = 1
    
    # Check K4
    for i in range(n):
        for j in range(i+1, n):
            if A[i][j]:
                for k in range(j+1, n):
                    if A[i][k] and A[j][k]:
                        for l in range(k+1, n):
                            if A[i][l] and A[j][l] and A[k][l]:
                                return False
    
    # Check Independent 6-set
    # Equivalent to K6 in complement
    for i in range(n):
        for j in range(i+1, n):
            if not A[i][j]:
                for k in range(j+1, n):
                    if not A[i][k] and not A[j][k]:
                        for l in range(k+1, n):
                            if not A[i][l] and not A[j][l] and not A[k][l]:
                                for m in range(l+1, n):
                                    if not A[i][m] and not A[j][m] and not A[k][m] and not A[l][m]:
                                        for p in range(m+1, n):
                                            if not A[i][p] and not A[j][p] and not A[k][p] and not A[l][p] and not A[m][p]:
                                                return False
    return True

def try_find_graph(n, deadline):
    # Heuristic search
    # Start with random graph
    # p=0.5 is a good start
    
    max_iters = 5000 # Limit iterations per n to avoid hanging
    iters = 0
    
    # Try multiple restarts
    for restart in range(5):
        if time.time() > deadline:
            return None
        if iters > max_iters:
            break
            
        # Initialize
        A = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    A[i][j] = 1
                    A[j][i] = 1
        
        # Local Search
        for _ in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Find a violation
            viol, vtype = find_violation(A, n)
            if viol is None:
                # Valid graph found
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if A[i][j]:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            # Flip an edge in the violation
            if vtype == 'K4':
                # viol is a tuple (i,j,k,l)
                i,j,k,l = viol
                # Pick a random edge to remove
                edges_to_flip = [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
            else: # I6
                # viol is a tuple (i,j,k,l,m,p)
                i,j,k,l,m,p = viol
                # Pick a random non-edge to add
                edges_to_flip = [(i,j), (i,k), (i,l), (i,m), (i,p), (j,k), (j,l), (j,m), (j,p), (k,l), (k,m), (k,p), (l,m), (l,p), (m,p)]
            
            u, v = random.choice(edges_to_flip)
            A[u][v] = 1 - A[u][v]
            A[v][u] = A[u][v]
            
            iters += 1
            
    return None

def find_violation(A, n):
    # Find one K4 or one Independent 6-set
    # Randomly choose which to look for first? 
    # Or look for K4 first as it's faster?
    # Let's look for K4 first.
    
    # Find K4
    # Optimization: Use bitsets? For n=36, simple loops are okay if we break early.
    # But checking all 4-subsets is ~58k.
    # Let's iterate.
    
    # To speed up, we can precompute neighbors.
    # But simple loops might be fine for one check.
    
    # Try to find a K4
    for i in range(n):
        for j in range(i+1, n):
            if A[i][j]:
                for k in range(j+1, n):
                    if A[i][k] and A[j][k]:
                        for l in range(k+1, n):
                            if A[i][l] and A[j][l] and A[k][l]:
                                return (i,j,k,l), 'K4'
    
    # Try to find Independent 6-set (K6 in complement)
    # This is slower.
    # Use backtracking?
    # For simplicity, let's just iterate 6-subsets but with pruning.
    # Actually, iterating all 6-subsets is 1.9M. 
    # 1.9M * 15 checks = 28.5M ops. In Python, this takes ~1-2 seconds.
    # If we do this every iteration, it's too slow.
    
    # We need a faster way to find an independent set.
    # Let's use a recursive search for a clique of size 6 in the complement.
    
    def find_clique_size_6(comp_A):
        # comp_A[i][j] is 1 if NOT adjacent in A
        # We want a clique of size 6 in comp_A.
        
        # Order vertices by degree in comp_A descending
        degrees = [sum(comp_A[i]) for i in range(n)]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        # Recursion
        def backtrack(current_clique, candidates):
            if len(current_clique) == 6:
                return current_clique
            if not candidates:
                return None
            
            # Pruning: if len(candidates) + len(current_clique) < 6, return None
            if len(candidates) + len(current_clique) < 6:
                return None
                
            # Pick a vertex from candidates
            # To be fast, we can just pick the first one? Or random?
            # Let's pick the first one in the ordered list that is in candidates.
            # Actually, candidates is a list of indices.
            
            # Heuristic: Try vertices with high degree in the induced subgraph of candidates
            # For simplicity, just iterate.
            
            for idx, v in enumerate(candidates):
                # Check if v is connected to all in current_clique in comp_A
                valid = True
                for u in current_clique:
                    if not comp_A[u][v]:
                        valid = False
                        break
                if valid:
                    # New candidates are those in candidates[idx+1:] that are connected to v
                    new_candidates = [w for w in candidates[idx+1:] if comp_A[v][w]]
                    res = backtrack(current_clique + [v], new_candidates)
                    if res:
                        return res
            return None
        
        return backtrack([], order)

    # Build complement
    comp_A = [[1-A[i][j] for j in range(n)] for i in range(n)]
    # Set diagonal to 0
    for i in range(n):
        comp_A[i][i] = 0
        
    clique = find_clique_size_6(comp_A)
    if clique:
        return tuple(clique), 'I6'
        
    return None, None
