import numpy as np
import time
import random
import math

def construct():
    n = 20
    
    # Helper to create Shrikhande graph (16 vertices, 6-regular, K2,3-free)
    # Shrikhande graph is strongly regular (16, 6, 2, 2).
    # Construction: Vertices 0..15.
    # It is the graph formed by identifying opposite vertices of the 4x4 grid?
    # Actually, Shrikhande is the complement of the Clebsch graph? No.
    # Clebsch is (16, 5, 0, 2). Complement is (16, 10, 6, 7). Not Shrikhande.
    # Shrikhande is (16, 6, 2, 2).
    # Let's just generate a random 6-regular graph on 16 vertices and verify.
    # Or use a known adjacency list.
    # Vertices: 0..15
    # Neighbors of 0: 1, 2, 3, 4, 5, 6? 
    # Let's use a simple random restart SA.
    
    def random_graph(edges_target):
        A = np.zeros((n, n), dtype=int)
        # Try to add edges randomly
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(pairs)
        # Just start with a sparse random graph
        num_edges = edges_target
        for u, v in pairs[:num_edges]:
            A[u, v] = 1
            A[v, u] = 1
        return A

    def count_edges(A):
        return int(np.sum(A)) // 2

    def is_valid(A):
        K = A @ A
        # Check if any K[i,j] > 2 for i != j
        # Diagonal is degree, ignore
        np.fill_diagonal(K, 0)
        return np.max(K) <= 2

    def get_valid_edges(A):
        K = A @ A
        np.fill_diagonal(K, 0)
        valid = []
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v] == 0:
                    # Check if adding (u,v) creates K2,3
                    # New common neighbors for (x, v) where x in N(u)
                    # K[x, v] must be <= 1
                    # New common neighbors for (u, y) where y in N(v)
                    # K[u, y] must be <= 1
                    neigh_u = np.where(A[u, :])[0]
                    neigh_v = np.where(A[v, :])[0]
                    
                    ok = True
                    for x in neigh_u:
                        if x == v: continue
                        if K[x, v] > 1:
                            ok = False
                            break
                    if not ok: continue
                    
                    for y in neigh_v:
                        if y == u: continue
                        if K[u, y] > 1:
                            ok = False
                            break
                    if ok:
                        valid.append((u, v))
        return valid

    # Simulated Annealing
    def simulated_annealing(A, T_start, T_end, steps):
        K = A @ A
        current_edges = count_edges(A)
        best_A = A.copy()
        best_edges = current_edges
        
        # Precompute neighbor lists for speed?
        # We will update K incrementally.
        
        for step in range(steps):
            T = T_start * (T_end / T_start) ** (step / steps)
            
            # Generate a random move: Add or Remove a random edge
            if random.random() < 0.5:
                # Try to add an edge
                pairs = [(u, v) for u in range(n) for v in range(u + 1, n) if A[u, v] == 0]
                if not pairs: continue
                u, v = random.choice(pairs)
                
                # Check validity
                neigh_u = np.where(A[u, :])[0]
                neigh_v = np.where(A[v, :])[0]
                
                valid = True
                for x in neigh_u:
                    if x == v: continue
                    if K[x, v] > 1:
                        valid = False
                        break
                if valid:
                    for y in neigh_v:
                        if y == u: continue
                        if K[u, y] > 1:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Apply move
                A[u, v] = 1
                A[v, u] = 1
                for x in neigh_u:
                    if x == v: continue
                    K[x, v] += 1
                    K[v, x] += 1
                for y in neigh_v:
                    if y == u: continue
                    K[u, y] += 1
                    K[y, u] += 1
                current_edges += 1
                
            else:
                # Try to remove an edge
                pairs = [(u, v) for u in range(n) for v in range(u + 1, n) if A[u, v] == 1]
                if not pairs: continue
                u, v = random.choice(pairs)
                
                # Apply move
                A[u, v] = 0
                A[v, u] = 0
                neigh_u = np.where(A[u, :])[0] # Note: A is already updated, so v is not in neigh_u
                neigh_v = np.where(A[v, :])[0]
                
                for x in neigh_u:
                    K[x, v] -= 1
                    K[v, x] -= 1
                for y in neigh_v:
                    K[u, y] -= 1
                    K[y, u] -= 1
                current_edges -= 1
            
            # Acceptance
            if current_edges > best_edges:
                best_edges = current_edges
                best_A = A.copy()
            
            # If we made a move that decreased edges (removal), we might reject it
            # But in SA, we accept with probability exp((E_new - E_old)/T)
            # Here E = -edges. So delta_E = (-(E_new) - (-(E_old))) = E_old - E_new
            # If we removed an edge, E_new = E_old - 1. delta_E = 1.
            # Prob = exp(1/T).
            # If we added an edge, E_new = E_old + 1. delta_E = -1.
            # Prob = exp(-1/T) ~ 1.
            
            # We need to track if the move was a removal.
            # This logic is getting complicated with the inline updates.
            # Let's simplify: Just do Hill Climbing with Random Restarts.
            pass

    # The SA implementation above is messy. Let's do a cleaner **Multi-Start Local Search**.
    # 1. Start with a random graph or Shrikhande-based graph.
    # 2. Run greedy add until stuck.
    # 3. Run random removals (perturbation).
    # 4. Repeat.
    # 5. Keep best.
    
    # This is what the current program does. Why is it only 2.1?
    # Maybe it's not exploring enough.
    # Let's try to improve the *quality* of the perturbation.
    
    # New Idea: **Iterated Local Search (ILS)**
    # 1. Local Search (Greedy Add)
    # 2. Perturbation (Random Walk of length L)
    # 3. Local Search again
    # 4. Accept if better, or with prob p
    
    def local_search(A):
        K = A @ A
        edges = count_edges(A)
        improved = True
        while improved:
            improved = False
            # Find all valid edges
            # Vectorized check for all pairs is hard, so we iterate
            # But we can be smart: only check pairs that are "close" to being valid?
            # No, just iterate all non-edges.
            for u in range(n):
                for v in range(u + 1, n):
                    if A[u, v] == 0:
                        # Check
                        neigh_u = np.where(A[u, :])[0]
                        neigh_v = np.where(A[v, :])[0]
                        
                        valid = True
                        for x in neigh_u:
                            if x == v: continue
                            if K[x, v] > 1:
                                valid = False
                                break
                        if valid:
                            for y in neigh_v:
                                if y == u: continue
                                if K[u, y] > 1:
                                    valid = False
                                    break
                        
                        if valid:
                            # Add
                            A[u, v] = 1
                            A[v, u] = 1
                            for x in neigh_u:
                                if x == v: continue
                                K[x, v] += 1
                                K[v, x] += 1
                            for y in neigh_v:
                                if y == u: continue
                                K[u, y] += 1
                                K[y, u] += 1
                            edges += 1
                            improved = True
                            # Break inner loops to restart with new K
                            # This is O(n^2) per add. Total O(n^4) worst case per LS.
                            # n=20, n^4 = 160,000. Very fast.
                            break
                if improved:
                    break
        return A, edges

    best_A = np.zeros((n, n), dtype=int)
    best_edges = 0
    start_time = time.time()
    timeout = 115
    
    # Initial population: Random graphs and Shrikhande-like
    # We'll just do random restarts.
    
    while time.time() - start_time < timeout:
        # Random start
        A = np.zeros((n, n), dtype=int)
        # Add random edges
        num_initial = random.randint(10, 40)
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(pairs)
        for u, v in pairs[:num_initial]:
             A[u, v] = 1
             A[v, u] = 1
        
        # Local Search
        A, edges = local_search(A)
        
        if edges > best_edges:
            best_edges = edges
            best_A = A.copy()
            
        # Perturbation: Remove random edges
        # This is effectively a new random start, but let's try to keep structure
        # Actually, just restarting with a new random graph is a valid "Multi-Start" strategy.
        # To make it "Iterated", we perturb the best solution.
        
        if best_edges > 0:
            A_pert = best_A.copy()
            K_pert = A_pert @ A_pert
            # Remove 3-10 random edges
            edges_list = [(u, v) for u in range(n) for v in range(u + 1, n) if A_pert[u, v] == 1]
            random.shuffle(edges_list)
            num_remove = random.randint(3, 10)
            for u, v in edges_list[:num_remove]:
                A_pert[u, v] = 0
                A_pert[v, u] = 0
                # Update K_pert
                neigh_u = np.where(A_pert[u, :])[0]
                neigh_v = np.where(A_pert[v, :])[0]
                for x in neigh_u:
                    K_pert[x, v] -= 1
                    K_pert[v, x] -= 1
                for y in neigh_v:
                    K_pert[u, y] -= 1
                    K_pert[y, u] -= 1
            
            # Local Search on perturbed
            A_pert, edges_pert = local_search(A_pert)
            if edges_pert > best_edges:
                best_edges = edges_pert
                best_A = A_pert.copy()

    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u, v] == 1:
                edges.append([u, v])
    return edges
