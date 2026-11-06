"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # safety margin before 150s kill
    random.seed(42)
    
    def find_k4(adj, n):
        """Find one K4 in graph. Returns tuple of 4 vertices or None."""
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    common = []
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            common.append(k)
                    for a in range(len(common)):
                        for b in range(a+1, len(common)):
                            if adj[common[a]][common[b]]:
                                return (i, j, common[a], common[b])
        return None

    def find_indep6(adj, n):
        """Find one independent 6-set. Returns tuple of 6 vertices or None."""
        for combo in combinations(range(n), 6):
            indep = True
            for i in range(6):
                for j in range(i+1, 6):
                    if adj[combo[i]][combo[j]]:
                        indep = False
                        break
                if not indep:
                    break
            if indep:
                return combo
        return None

    def is_valid(adj, n):
        return find_k4(adj, n) is None and find_indep6(adj, n) is None

    def make_adj(n, edges):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = adj[v][u] = True
        return adj
    
    # Start with Paley graph on 17 vertices as baseline
    def paley_graph(n):
        # Find smallest prime >= n
        def is_prime(x):
            if x < 2: return False
            for i in range(2, int(x**0.5)+1):
                if x % i == 0:
                    return False
            return True
        
        p = n
        while not is_prime(p):
            p += 1
        # For Paley graph, need p ≡ 1 mod 4
        while p % 4 != 1:
            p += 1
        if p > 40:
            # Fallback: use random graph
            return None
        
        squares = {x*x % p for x in range(1, p)}
        edges = []
        for i in range(min(n, p)):
            for j in range(i+1, min(n, p)):
                if (j - i) % p in squares:
                    edges.append([i, j])
        return edges
    
    # Try Paley graph on 17
    best_result = None
    paley_edges = paley_graph(17)
    if paley_edges is not None:
        adj = make_adj(17, paley_edges)
        if is_valid(adj, 17):
            best_result = {'n': 17, 'edges': paley_edges}
    
    # Now try to improve via repair-based local search on larger graphs
    for target_n in range(18, 36):
        if time.time() > deadline - 5:
            break
        
        best_adj = None
        
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Start with random graph
            edge_prob = 0.4
            edges = []
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if random.random() < edge_prob:
                        edges.append([i, j])
            
            adj = make_adj(target_n, edges)
            
            # Repair loop
            for step in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Check for K4
                k4 = find_k4(adj, target_n)
                if k4:
                    # Remove a random edge from the K4
                    i, j, k, l = k4
                    pair_options = [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]
                    u, v = random.choice(pair_options)
                    adj[u][v] = adj[v][u] = False
                    continue
                
                # Check for Independent Set of size 6
                indep6 = find_indep6(adj, target_n)
                if indep6:
                    # Add a random edge within the independent set
                    u, v = random.sample(indep6, 2)
                    adj[u][v] = adj[v][u] = True
                    continue
                
                # If no violations found, graph is valid
                best_adj = [row[:] for row in adj]
                break
            
            if best_adj is not None:
                break
        
        if best_adj is not None:
            edges = []
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if best_adj[i][j]:
                        edges.append([i, j])
            # Double check validity before updating best result
            final_adj = make_adj(target_n, edges)
            if is_valid(final_adj, target_n):
                if best_result is None or target_n > best_result['n']:
                    best_result = {'n': target_n, 'edges': edges}
    
    # If we found a valid graph with n > 17, return it
    if best_result is not None and best_result['n'] > 17:
        return best_result
    
    # Otherwise return the Paley graph on 17
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i+1, 17) if (j-i) % 17 in {x*x % 17 for x in range(1, 17)}]}
# EVOLVE-BLOCK-END
