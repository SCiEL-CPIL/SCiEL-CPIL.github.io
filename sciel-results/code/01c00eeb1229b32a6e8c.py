import time
import random
import numpy as np
import sys

def construct():
    n = 40
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # We will work with adjacency matrices for fast C6 checks.
    # A C6 exists if there is a path of length 4 between u and v.
    # Path length 4 count between u and v is (A^4)[u][v].
    # If (A^4)[u][v] > 0, then adding edge (u,v) creates a C6 (since the path u-x-y-z-v plus edge v-u forms a C6).
    # Wait, adding edge (u,v) to a C6-free graph creates a C6 iff there is a path of length 4 between u and v.
    # So we can maintain A^4 dynamically? No, A^4 changes when A changes.
    # Recomputing A^4 is O(n^3) or O(n^6) depending on method. 
    # n=40 is small. A^4 can be computed via matrix multiplication.
    # A^2 = A @ A
    # A^4 = A^2 @ A^2
    # This is 2 matrix multiplications of 40x40 matrices. Very fast.
    
    # Strategy:
    # 1. Start with a random graph with ~90 edges, ensuring it is C6-free.
    #    We can start with 8 K5s (80 edges) which is C6-free.
    # 2. Use Simulated Annealing.
    #    Moves:
    #    a) Add a random non-edge. If it creates a C6, reject (or accept with prob exp(-delta/T) if we allowed violations? 
    #       No, we must stay in the feasible region for the final answer. 
    #       Standard Simulated Annealing allows infeasible states, but we need to return a valid graph.
    #       Better: Use a "repair" heuristic or just reject infeasible moves. 
    #       Given the density (105/780 ~ 13%), the feasible region is likely connected enough.
    #       Let's try "Add if valid, else try Swap".
    #    b) Swap: Remove edge e1, Add edge e2. 
    #       This maintains edge count. 
    #       We can try to increase edge count by adding edges.
    
    # Let's stick to a simpler, robust approach:
    # Randomized Greedy with Backtracking/Restart, but optimized with NumPy.
    # Actually, the current program's bottleneck is the Python loop in `would_create_c6`.
    # I will replace that with a NumPy-based check.
    
    # Pre-compute all pairs
    # A is 40x40 numpy array
    
    def is_c6_free(adj_matrix):
        # Check if any (A^4)[u][v] > 0 for u != v?
        # No, C6-free means no cycle of length 6.
        # A cycle of length 6 exists if there are u,v such that u!=v and there is a path of length 5 between them?
        # No. A C6 is a subgraph.
        # If the graph contains a C6, then for some u,v on the cycle, the distance is <= 5.
        # Specifically, if u and v are part of a C6, they are connected by a path of length 5 (the rest of the cycle).
        # So if (A^5)[u][v] > 0 and (A)[u][v] == 1? No.
        # If (A^k)[u][v] > 0, there is a walk of length k.
        # A simple cycle of length 6 implies that for every pair of vertices on the cycle, there is a path of length <= 5.
        # More directly: A graph is C6-free iff for all u != v, the number of walks of length 5 between u and v is 0?
        # No, that's not right. A C6 has paths of length 1, 2, 3, 4, 5 between its vertices.
        # If there is a C6, then for some u,v, there is a path of length 5.
        # Conversely, if there is a path of length 5 between u and v, does it imply a C6?
        # Not necessarily. It could be a walk with repeated vertices.
        # However, if we maintain the graph as C6-free, and we want to check if adding (u,v) creates a C6:
        # It creates a C6 iff there is a path of length 4 between u and v in the current graph.
        # So we just need to check if (A^4)[u][v] > 0.
        
        A2 = adj_matrix @ adj_matrix
        A4 = A2 @ A2
        # We only care about u != v
        # If A4[u][v] > 0, there is a walk of length 4. 
        # Does a walk of length 4 imply a simple path of length 4?
        # In a C6-free graph, any walk of length 4 between distinct vertices u,v 
        # must contain a simple path of length <= 4.
        # If the simple path length is 4, adding edge (u,v) creates a C6.
        # If the simple path length is < 4, adding edge (u,v) creates a C3 or C4 or C5, which is allowed?
        # The task says "containing no six-cycle". It does NOT say "no cycles of length < 6".
        # So creating a C4 is fine.
        # Therefore, we only reject if adding (u,v) creates a C6.
        # This happens iff there is a path of length 4 between u and v.
        # If there is a walk of length 4, does it guarantee a simple path of length 4?
        # If the walk is u-a-b-c-v.
        # If all distinct, it's a simple path of length 4.
        # If not distinct, e.g., u-a-u-b-v (length 4 walk). 
        # This implies edges (u,a) and (u,b) and (b,v).
        # This doesn't create a C6 when we add (u,v).
        # So checking A4 > 0 is a SUFFICIENT condition for rejection (safe), but might be NECESSARY?
        # If A4[u][v] == 0, there is no walk of length 4, so no path of length 4, so safe.
        # If A4[u][v] > 0, there is a walk. It might not be a simple path.
        # However, in dense graphs, it's highly likely to be a simple path or contain one.
        # To be 100% safe, we should verify if the walk is simple?
        # Given n=40, we can do a quick check if A4 > 0. 
        # If A4 > 0, we can try to find the actual path? 
        # Or just reject. Rejection might be too conservative if it's a "fake" walk.
        # But for maximizing edges, being conservative is safer than invalid output.
        # Let's assume A4 > 0 implies C6 risk and reject.
        
        # Actually, let's just use the NumPy check for the *candidate* edge.
        pass

    # Let's refine the search:
    # 1. Start with 8 K5s (80 edges).
    # 2. Compute A4.
    # 3. Iterate:
    #    - Pick random non-edge (u,v).
    #    - Check if A4[u][v] == 0.
    #    - If yes, add edge. Update A, A2, A4.
    #    - If no, try a swap: remove random edge (x,y), add (u,v).
    #      Check if new graph is C6-free.
    
    # Updating A4 after an edge change is tricky. 
    # A = A + delta.
    # A2 = (A+delta)^2 = A^2 + A@delta + delta@A + delta^2.
    # A4 = (A2+delta2)^2 ...
    # It's easier to just recompute A4 from A every time? 
    # n=40. 40x40 matmul is ~64k ops. 
    # In Python/NumPy, this is microseconds.
    # 100,000 iterations * 100us = 10s. Feasible.
    
    def compute_A4(adj):
        A = np.array(adj, dtype=np.int8)
        A2 = A @ A
        A4 = A2 @ A2
        return A4

    # Initial setup
    adj = [[0]*n for _ in range(n)]
    edges = []
    
    # Build 8 K5s
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u = base + a
                v = base + b
                adj[u][v] = 1
                adj[v][u] = 1
                edges.append([u, v])
    
    best_edges = list(edges)
    best_count = len(best_edges)
    
    # Precompute list of all possible edges
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Main Loop: Randomized Greedy with NumPy Check
    # We will try to add edges. If we can't add any more, we do a swap.
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        # Reset to base for diversity? Or continue from best?
        # Let's continue from best to improve.
        # But if stuck, reset to base with random noise?
        # Let's just run a long loop of "Add if possible, else Swap".
        
        if restart == 0:
            current_adj = [row[:] for row in adj]
            current_edges = list(edges)
            current_count = len(current_edges)
        else:
            # Restart from best found so far
            current_adj = [row[:] for row in adj]
            current_edges = list(best_edges)
            current_count = len(best_edges)
            
        # Build set of current edges for fast lookup
        edge_set = set()
        for e in current_edges:
            edge_set.add((e[0], e[1]))
            edge_set.add((e[1], e[0]))
            
        # Compute initial A4
        A_np = np.array(current_adj, dtype=np.int8)
        A2 = A_np @ A_np
        A4 = A2 @ A2
        
        no_improve_count = 0
        
        for step in range(5000): # Max steps per restart
            if time.time() > deadline - 2:
                break
                
            # 1. Try to add a random non-edge
            # Generate random pair
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            
            if (u, v) in edge_set:
                continue
                
            # Check if adding (u,v) creates C6
            # If A4[u][v] == 0, it's safe (likely)
            if A4[u][v] == 0:
                # Add edge
                current_adj[u][v] = 1
                current_adj[v][u] = 1
                current_edges.append([u, v])
                edge_set.add((u, v))
                edge_set.add((v, u))
                current_count += 1
                no_improve_count = 0
                
                # Update A4
                # Efficient update:
                # A[u][v] += 1, A[v][u] += 1
                # A2 changes: 
                # A2[i][j] = sum_k A[i][k]A[k][j]
                # New A2 = A2 + A[:,u]*A[v,:] + A[:,v]*A[u,:] + (A[u][u]+A[v][v] terms? No, u!=v)
                # Actually, just recompute A4. It's fast enough.
                A_np = np.array(current_adj, dtype=np.int8)
                A2 = A_np @ A_np
                A4 = A2 @ A2
                
                if current_count > best_count:
                    best_count = current_count
                    best_edges = list(current_edges)
                    adj = [row[:] for row in current_adj]
                    
                if best_count >= 105:
                    return best_edges
            else:
                # 2. Try a Swap
                # Pick a random edge to remove
                if not current_edges:
                    break
                idx = random.randint(0, len(current_edges)-1)
                rem_u, rem_v = current_edges[idx]
                
                # Candidate: Remove (rem_u, rem_v), Add (u, v)
                # Check if resulting graph is C6-free.
                # This is harder to check quickly without removing first.
                # Let's try the swap and check the whole graph?
                # Or just check if the new edge (u,v) creates a C6 in the graph WITHOUT (rem_u, rem_v)?
                # This is complex. 
                # Alternative: Just reject the add, and try another random pair.
                # This is effectively a "lazy" search.
                # To improve, we can do a local search: 
                # If we can't add, try to remove an edge that is "critical" and add a different one.
                # For simplicity and speed, let's just retry with a different random pair.
                # This is basically random sampling of the feasible region.
                no_improve_count += 1
                if no_improve_count > 200:
                    break

    return best_edges

# EVOLVE-BLOCK-END
