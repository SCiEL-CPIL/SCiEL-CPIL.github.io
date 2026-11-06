import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    def is_valid(n, colors):
        # colors is a list of 0/1 for pairs (i,j) i<j
        # Reconstruct adjacency matrix
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        
        # Check for K2,2,2 in color 0 and color 1
        # We can just check one color, since if one is free, the other might not be.
        # Wait, the condition is "Neither colour class may contain K2,2,2".
        # So we must check both.
        
        def has_k222(color):
            # Build adjacency list for this color
            # adj_c[i] is a set of neighbors j such that adj[i][j] == color
            adj_c = [set() for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j] == color:
                        adj_c[i].add(j)
                        adj_c[j].add(i)
            
            # Check condition:
            # For every edge (u,v) in color c:
            #   S = N(u) ∩ N(v)
            #   For every pair x,y in S:
            #     if |N(x) ∩ N(y) ∩ S| >= 2: return True
            for u in range(n):
                for v in adj_c[u]:
                    if u < v: # only check each edge once
                        S = adj_c[u].intersection(adj_c[v])
                        # S is a set of vertices
                        S_list = list(S)
                        len_S = len(S_list)
                        for i in range(len_S):
                            x = S_list[i]
                            for j in range(i+1, len_S):
                                y = S_list[j]
                                # Check common neighbors of x and y within S
                                # common = N(x) ∩ N(y) ∩ S
                                common = adj_c[x].intersection(adj_c[y]).intersection(S)
                                if len(common) >= 2:
                                    return True
            return False
        
        return not has_k222(0) and not has_k222(1)

    def paley_coloring(n):
        # Only works for prime n
        # Check if n is prime
        def is_prime(k):
            if k < 2: return False
            for i in range(2, int(k**0.5)+1):
                if k % i == 0: return False
            return True
        
        if not is_prime(n):
            return None
            
        squares = set()
        for x in range(1, n):
            squares.add((x*x) % n)
            
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                diff = (j - i) % n
                if diff in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    # Try Paley 29
    n_target = 29
    colors = paley_coloring(n_target)
    if colors:
        if is_valid(n_target, colors):
            return {'n': n_target, 'colors': colors}
        # If Paley 29 is not valid, we might need to search
        # But let's check if it's close.
        
    # Try Paley 31? No, 31 is not 1 mod 4, but let's see if a subgraph or modification works.
    # Actually, if R(K2,2,2) = 31, then n=30 is possible.
    # Let's try to construct n=30.
    # We can take the Paley 29 graph and add a vertex?
    # Or use a different construction.
    
    # Fallback: Backtracking search for n=29 or 30
    # Given the time, if Paley 29 fails, we return the best we have.
    # But Paley 29 is a very strong candidate.
    
    # If Paley 29 fails, try random search
    if time.time() > deadline:
        # Return a valid small one if we have it?
        # We don't have a guaranteed valid one yet if Paley failed.
        # Let's return the seed solution (n=17) to be safe if we run out of time
        pass

    # Let's try to repair Paley 29 if it failed
    # Or just return n=17 if nothing else works.
    # But we should try harder.
    
    # Try n=30 with a random initialization and local search
    n_try = 30
    best_n = 17
    best_colors = paley_coloring(17) # Seed solution
    
    for _ in range(100): # 100 random restarts
        if time.time() > deadline:
            break
            
        # Initialize with random or Paley-like
        # Let's use a random graph
        random_colors = [random.randint(0, 1) for _ in range(n_try*(n_try-1)//2)]
        
        # Local search: flip edges to reduce violations
        # Define a cost function: number of K2,2,2 in color 0 + number in color 1
        def count_violations(n, cols):
            adj = [[0]*n for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    c = cols[idx]
                    adj[i][j] = c
                    adj[j][i] = c
                    idx += 1
            count = 0
            for color in [0, 1]:
                adj_c = [set() for _ in range(n)]
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i][j] == color:
                            adj_c[i].add(j)
                            adj_c[j].add(i)
                for u in range(n):
                    for v in adj_c[u]:
                        if u < v:
                            S = adj_c[u].intersection(adj_c[v])
                            S_list = list(S)
                            len_S = len(S_list)
                            for i in range(len_S):
                                x = S_list[i]
                                for j in range(i+1, len_S):
                                    y = S_list[j]
                                    common = adj_c[x].intersection(adj_c[y]).intersection(S)
                                    if len(common) >= 2:
                                        count += 1
            return count

        current_violations = count_violations(n_try, random_colors)
        for step in range(1000):
            if time.time() > deadline:
                break
            if current_violations == 0:
                if n_try > best_n:
                    best_n = n_try
                    best_colors = random_colors[:]
                break
                
            # Pick a random edge to flip
            idx = random.randint(0, len(random_colors)-1)
            old_c = random_colors[idx]
            random_colors[idx] = 1 - old_c
            
            new_violations = count_violations(n_try, random_colors)
            if new_violations < current_violations:
                current_violations = new_violations
            else:
                random_colors[idx] = old_c

    if best_colors:
        return {'n': best_n, 'colors': best_colors}
    else:
        return {'n': 17, 'colors': paley_coloring(17)}

# EVOLVE-BLOCK-END
