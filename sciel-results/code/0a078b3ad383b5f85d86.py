import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Helper to check if graph is valid
    def is_valid(n, adj):
        # Check K4
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    for k in range(j+1, n):
                        if adj[j] & (1 << k):
                            # Check if i, j, k have a common neighbor l > k
                            # Or just check all 4-tuples?
                            # This is O(n^4) which is 35^4 ~ 1.5M. Acceptable for final check.
                            pass
        # The above is slow. Let's use a faster K4 check.
        # Count triangles, then for each edge (u,v), check if they share 2 neighbors.
        
        # Actually, let's just use the bitmask for K4.
        # For each pair (u,v) that is an edge, check if (adj[u] & adj[v]) has at least 2 bits set among vertices > v?
        # No, K4 can be anywhere.
        
        # Let's stick to a simple O(n^4) check for final validation, it's fast enough for one graph.
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] & (1 << j)): continue
                for k in range(j+1, n):
                    if not (adj[j] & (1 << k)): continue
                    if not (adj[i] & (1 << k)): continue
                    # i,j,k form a triangle. Check if any l is connected to all 3.
                    # Common neighbors of i,j,k
                    common = adj[i] & adj[j] & adj[k]
                    if common:
                        return False, "K4"
        
        # Check Independent Set of size 6
        # Use a simple backtracking to find if alpha >= 6
        # Or just check all subsets of size 6?
        # 35 choose 6 is 1.6M. 
        # Let's do it.
        for combo in itertools.combinations(range(n), 6):
            is_indep = True
            for u in combo:
                # Check if u is connected to any other in combo
                mask = 0
                for v in combo:
                    if u != v:
                        mask |= (1 << v)
                if adj[u] & mask:
                    is_indep = False
                    break
            if is_indep:
                return False, "Ind6"
        
        return True, "Valid"

    def count_k4(n, adj):
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    for k in range(j+1, n):
                        if adj[j] & (1 << k) and (adj[i] & (1 << k)):
                            common = adj[i] & adj[j] & adj[k]
                            count += bin(common).count('1')
        return count

    def has_ind6(n, adj):
        # Fast check using greedy + random sampling?
        # For search, we can just use a heuristic:
        # Try to find an independent set of size 6 using greedy coloring on complement?
        # Or just return False (optimistic) to guide search, and verify at end.
        # Let's use a quick randomized check.
        for _ in range(10):
            # Greedy independent set
            remaining = set(range(n))
            ind_set = []
            while remaining:
                v = random.choice(list(remaining))
                ind_set.append(v)
                remaining -= {v}
                # Remove neighbors of v
                neighbors = set()
                for u in range(n):
                    if adj[v] & (1 << u):
                        neighbors.add(u)
                remaining -= neighbors
                if len(ind_set) >= 6:
                    return True
        return False

    def score(n, adj):
        k4_count = count_k4(n, adj)
        ind6_flag = 1 if has_ind6(n, adj) else 0
        return k4_count * 10 + ind6_flag * 1000

    best_graph = None
    best_score = float('inf')
    best_n = 0

    # Try to find a graph for n=35
    for target_n in [35, 30, 25, 20, 17]:
        if time.time() > deadline:
            break
        
        # Simulated Annealing
        current_n = target_n
        adj = [0] * current_n
        for i in range(current_n):
            for j in range(i+1, current_n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_score = score(current_n, adj)
        best_local = (current_score, adj[:])
        
        temperature = 100.0
        cooling = 0.99
        
        steps = 0
        max_steps = 5000
        
        while steps < max_steps:
            if time.time() > deadline:
                break
            
            # Flip a random edge
            i = random.randint(0, current_n - 1)
            j = random.randint(0, current_n - 1)
            if i == j:
                continue
            
            if adj[i] & (1 << j):
                # Remove edge
                new_adj = adj[:]
                new_adj[i] &= ~(1 << j)
                new_adj[j] &= ~(1 << i)
            else:
                # Add edge
                new_adj = adj[:]
                new_adj[i] |= (1 << j)
                new_adj[j] |= (1 << i)
            
            new_score = score(current_n, new_adj)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < pow(2.718, -delta / max(temperature, 0.1)):
                adj = new_adj
                current_score = new_score
                if current_score < best_local[0]:
                    best_local = (current_score, adj[:])
                    # Check if valid
                    if current_score == 0:
                        valid, _ = is_valid(current_n, adj)
                        if valid:
                            # Found a valid graph!
                            edges = [[i, j] for i in range(current_n) for j in range(i+1, current_n) if adj[i] & (1 << j)]
                            return {'n': current_n, 'edges': edges}
            
            temperature *= cooling
            steps += 1
        
        # If we didn't find a valid one, store the best local found
        if best_local[0] < best_score:
            best_score = best_local[0]
            best_graph = best_local[1]
            best_n = target_n
            
    # If no valid graph found, return the best graph we have?
    # The task requires a VALID graph.
    # We must return a valid graph.
    # If we didn't find one for 35, we need to find one for smaller n.
    # The loop above tries smaller n if 35 fails.
    # But if the loop finishes and we haven't returned, we need to ensure best_graph is valid.
    # The `score` function uses heuristics.
    # We must verify `best_graph` exactly.
    
    if best_graph is not None:
        valid, _ = is_valid(best_n, best_graph)
        if valid:
            edges = [[i, j] for i in range(best_n) for j in range(i+1, best_n) if best_graph[i] & (1 << j)]
            return {'n': best_n, 'edges': edges}
    
    # Fallback: Return the Paley graph on 17 vertices (known valid)
    squares = {x * x % 17 for x in range(1, 17)}
    edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    return {'n': 17, 'edges': edges}
