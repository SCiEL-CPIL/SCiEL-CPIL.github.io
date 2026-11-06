import random
import math
import time
import itertools

def construct():
    n = 20
    vertices = list(range(n))
    
    # Precompute neighbor lists for fast K2,2 counting
    # adj is a list of sets or list of booleans. 
    # For speed in counting, list of sets is good for membership, 
    # but for iterating neighbors, list is better.
    # We'll use a 2D list of bools for adj, and maintain neighbor lists?
    # Or just use the 2D list and iterate 0..n-1. n=20 is small.
    
    def count_k22_between(nu, nv, adj):
        """
        Count number of K2,2 subgraphs between sets nu and nv.
        nu, nv are lists of vertices.
        adj is 2D list of bools.
        A K2,2 is formed by u1, u2 in nu and v1, v2 in nv
        such that all 4 edges exist.
        """
        count = 0
        # Iterate over pairs in nv (size 2)
        # For each pair (v1, v2), count how many u in nu are connected to both.
        # If count >= 2, then we have C(count, 2) K2,2s?
        # No, a K2,2 is defined by a specific pair from nu and a specific pair from nv.
        # So for a fixed pair (v1, v2) in nv, let C be the set of u in nu such that 
        # adj[u][v1] and adj[u][v2].
        # The number of K2,2s using (v1, v2) is C(|C|, 2).
        # Summing over all pairs in nv gives the total number of K2,2s.
        
        for i in range(len(nv)):
            for j in range(i+1, len(nv)):
                v1 = nv[i]
                v2 = nv[j]
                common_count = 0
                for u in nu:
                    if adj[u][v1] and adj[u][v2]:
                        common_count += 1
                if common_count >= 2:
                    count += common_count * (common_count - 1) // 2
        return count

    def delta_e(adj, u, v, is_adding):
        """
        Calculate change in number of K3,3 subgraphs when toggling edge (u,v).
        If is_adding is True, we are adding (u,v).
        The K3,3s created/destroyed are those containing (u,v).
        This is equal to the number of K2,2s between N(v)\{u} and N(u)\{v}.
        """
        nu = [x for x in range(n) if x != v and adj[u][x]]
        nv = [x for x in range(n) if x != u and adj[v][x]]
        
        # Note: if is_adding, adj[u][v] is currently False.
        # nu and nv do not contain v and u respectively.
        # The K2,2 count is the number of K3,3s that would be completed.
        
        return count_k22_between(nu, nv, adj)

    def solve_sa(seed):
        random.seed(seed)
        
        # Initialize random graph
        adj = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i][j] = True
                    adj[j][i] = True
        
        # Calculate initial E
        # We can calculate E by summing delta_e for all existing edges?
        # No, delta_e gives the contribution of that edge to the total count?
        # Total E = sum over all edges (u,v) of (1/9) * (# K3,3 containing u,v)?
        # No, each K3,3 has 9 edges.
        # Let C(u,v) be the number of K3,3s containing edge (u,v).
        # Then Total E = (1/9) * sum_{(u,v) in E} C(u,v).
        # C(u,v) is exactly count_k22_between(N(v)\{u}, N(u)\{v}).
        
        def calc_total_E(adj):
            total = 0
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        nu = [x for x in range(n) if x != j and adj[i][x]]
                        nv = [x for x in range(n) if x != i and adj[j][x]]
                        total += count_k22_between(nu, nv, adj)
            return total // 9 # Each K3,3 is counted 9 times
        
        # Initial E
        E = calc_total_E(adj)
        m = sum(1 for i in range(n) for j in range(i+1, n) if adj[i][j])
        
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.95
        iterations = 50000
        start_time = time.time()
        
        best_E = E
        best_m = m
        best_adj = [row[:] for row in adj]
        
        for it in range(iterations):
            if time.time() - start_time > 10:
                break
            
            # Pick random edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            is_adding = not adj[u][v]
            
            # Calculate delta E
            dE = delta_e(adj, u, v, is_adding)
            
            # Logic for acceptance
            # We want to minimize E.
            # If dE < 0: Accept (E decreases)
            # If dE == 0: 
            #   If adding: m increases. Accept.
            #   If removing: m decreases. Reject (unless we want to explore, but usually bad).
            # If dE > 0: Accept with prob exp(-dE/T)
            
            accept = False
            if dE < 0:
                accept = True
            elif dE == 0:
                if is_adding:
                    accept = True
                else:
                    accept = False
            else:
                # dE > 0
                # Probability to accept
                # We use a small penalty for increasing E
                # p = exp(-dE / T)
                # To avoid overflow/underflow issues, we can just use random < p
                # But T is small eventually.
                p = math.exp(-dE / T)
                if random.random() < p:
                    accept = True
            
            if accept:
                # Apply move
                adj[u][v] = is_adding
                adj[v][u] = is_adding
                E += dE
                if is_adding:
                    m += 1
                else:
                    m -= 1
                
                # Update best
                # We prefer lower E. If E is same, prefer higher m.
                if E < best_E or (E == best_E and m > best_m):
                    best_E = E
                    best_m = m
                    best_adj = [row[:] for row in adj]
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        # Return best found
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if best_adj[i][j]:
                    edges.append([i, j])
        return edges

    # Run multiple restarts
    start_time = time.time()
    best_edges = []
    best_m = 0
    
    for i in range(50):
        if time.time() - start_time > 100:
            break
        edges = solve_sa(i)
        m = len(edges)
        if m > best_m:
            best_m = m
            best_edges = edges
            # Optional: check validity? 
            # The SA logic should ensure E=0 or low E.
            # But if E > 0, it's invalid.
            # We should only accept if E == 0.
            # Let's re-calculate E for the best candidate to be safe.
            # Actually, solve_sa returns the best state found during the run.
            # We should verify it.
            
    # Verify best_edges
    # If best_edges is empty or invalid, return a known valid graph?
    # A simple valid graph: empty graph (0 edges).
    # Or a random sparse graph.
    # But SA should find something.
    
    # Let's add a final verification step for the returned best_edges
    # If invalid, fall back to a simple construction.
    
    def is_valid(edges):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        E = calc_total_E(adj)
        return E == 0

    if not is_valid(best_edges):
        # Fallback: Greedy random
        adj = [[False]*n for _ in range(n)]
        edges = []
        for _ in range(1000):
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v or adj[u][v]:
                continue
            # Check if adding (u,v) creates K3,3
            nu = [x for x in range(n) if adj[u][x]]
            nv = [x for x in range(n) if adj[v][x]]
            # Check K2,2 between nv and nu
            # If count > 0, it creates a K3,3?
            # No, delta_e > 0 means it creates at least one K3,3.
            dE = count_k22_between(nu, nv, adj)
            if dE == 0:
                adj[u][v] = True
                adj[v][u] = True
                edges.append([u, v])
        best_edges = edges

    return best_edges
