import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for safety
    
    # 1. Construct Polarity Graph of PG(2,5)
    points = []
    for x in range(5):
        for y in range(5):
            points.append((x, y, 1))
    for m in range(5):
        points.append((1, m, 0))
    points.append((0, 1, 0))
    
    n = len(points)
    A = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(i + 1, n):
            val = (points[i][0] * points[j][0] + 
                   points[i][1] * points[j][1] + 
                   points[i][2] * points[j][2]) % 5
            if val == 0:
                A[i, j] = 1
                A[j, i] = 1
                
    # 2. Find the best edge to add (minimizing 3-paths)
    A2 = A @ A
    A3 = A2 @ A
    
    min_val = float('inf')
    best_pair = None
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0:
                val = A3[i, j]
                if val < min_val:
                    min_val = val
                    best_pair = (i, j)
                    
    A[best_pair[0], best_pair[1]] = 1
    A[best_pair[1], best_pair[0]] = 1
    
    # 3. Local Search (Hill Climbing / Simulated Annealing lite)
    # Objective: Minimize C4 count
    # Move: Swap an edge (u,v) in the graph with a non-edge (x,y)
    
    def count_c4(A):
        # Number of C4s = (1/8) * sum_{i,j} (A^2)_{ij}^2 - (1/2) * sum_i deg(i)^2 + ... 
        # Actually, simpler: 
        # C4 count = 1/8 * ( sum_{i,j} (A^2)_{ij}^2 - sum_i deg(i)^2 - sum_i A_{ii}^2 )
        # Wait, the standard formula is:
        # #C4 = 1/8 * ( ||A^2||_F^2 - sum_i d_i^2 - 2*sum_i A_ii )
        # For simple graphs A_ii=0.
        # ||A^2||_F^2 = sum_{i,j} (A^2)_{ij}^2
        # sum_i d_i^2 = sum_i (A^2)_{ii}
        # So #C4 = 1/8 * ( sum_{i != j} (A^2)_{ij}^2 )
        A2 = A @ A
        total_sq = np.sum(A2**2)
        diag_sq = np.sum(np.diag(A2)**2)
        return (total_sq - diag_sq) // 8

    current_A = A.copy()
    current_c4 = count_c4(current_A)
    best_A = current_A.copy()
    best_c4 = current_c4
    
    # Precompute list of edges and non-edges for faster sampling
    # But we need to update them.
    
    # To speed up, we can use the fact that swapping (u,v) and (x,y)
    # only affects local neighborhoods.
    # However, for simplicity and robustness within time limit, 
    # we can do a few iterations of random swaps and recalculate C4.
    
    # Let's try a more efficient update.
    # Delta C4 when adding edge (u,v):
    # Adds paths of length 3 between u and v.
    # Let N(u) be neighbors of u, N(v) be neighbors of v.
    # New C4s formed by adding (u,v) = |N(u) \cap N(v)| * (something)?
    # No, adding (u,v) creates C4s of the form u-a-v-b-u.
    # This requires a neighbor a of u and b of v such that a-b is an edge? No.
    # The C4 is u-v-b-a-u.
    # So we need a in N(u) and b in N(v) such that (a,b) is an edge.
    # Count = sum_{a in N(u)} sum_{b in N(v)} A[a,b]
    
    def delta_c4_add(A, u, v):
        # A[u,v] is 0
        # Count paths of length 3 from u to v
        # (A^3)[u,v]
        # But we can compute it faster using neighbors
        # Neighbors of u: Nu
        # Neighbors of v: Nv
        # We need number of edges between Nu and Nv.
        # This is sum_{a in Nu} sum_{b in Nv} A[a,b]
        # Note: u and v are not in Nu/Nv respectively if graph is simple.
        # But a could be v? No, A[u,v]=0 so v not in Nu.
        # b could be u? No, A[u,v]=0 so u not in Nv.
        # So we are counting edges between two sets of vertices.
        Nu = np.where(A[u] == 1)[0]
        Nv = np.where(A[v] == 1)[0]
        # Count edges between Nu and Nv
        # Submatrix A[Nu, Nv]
        # sum of A[Nu, Nv]
        return np.sum(A[np.ix_(Nu, Nv)])

    def delta_c4_remove(A, u, v):
        # A[u,v] is 1
        # Removing (u,v) removes C4s that used this edge.
        # These are C4s of form u-v-b-a-u where (a,b) is edge, a in N(u)\{v}, b in N(v)\{u}.
        # Count = sum_{a in N(u)\{v}} sum_{b in N(v)\{u}} A[a,b]
        Nu = np.where(A[u] == 1)[0]
        Nv = np.where(A[v] == 1)[0]
        # Exclude u and v from sets?
        # u is in Nv? Yes, A[v,u]=1. So u is in Nv.
        # v is in Nu? Yes, A[u,v]=1. So v is in Nu.
        # We want a != v and b != u.
        # The term A[v,u] is 1, but a=v, b=u is excluded.
        # Are there other terms involving u or v?
        # a=v: sum_{b in Nv} A[v,b]. This counts edges incident to v.
        # b=u: sum_{a in Nu} A[a,u]. This counts edges incident to u.
        # The formula sum_{a in Nu} sum_{b in Nv} A[a,b] includes:
        # a=v, b=u: A[v,u] = 1.
        # a=v, b != u: A[v,b] = 1 if b in Nv. These are edges (v,b).
        # a != v, b=u: A[a,u] = 1 if a in Nu. These are edges (a,u).
        # We want to exclude cases where the path goes through u or v again?
        # The C4 is u-v-b-a-u.
        # Vertices are u, v, b, a. Distinct.
        # So a != u, a != v, b != u, b != v.
        # Since a in N(u), a != u is true. a=v is possible if (u,v) edge.
        # Since b in N(v), b != v is true. b=u is possible if (u,v) edge.
        # So we must exclude a=v and b=u.
        # The sum S = sum_{a in Nu} sum_{b in Nv} A[a,b]
        # S includes A[v,u] (a=v, b=u) which is 1.
        # S includes A[v,b] for b in Nv, b!=u. These are 1s.
        # S includes A[a,u] for a in Nu, a!=v. These are 1s.
        # We want S' = S - (edges incident to v in Nv) - (edges incident to u in Nu) + A[v,u]?
        # Actually, let's just compute the count directly by iterating or using a mask.
        # For speed, let's just use the A^3 value if we update A^3? No, A^3 update is expensive.
        # Let's stick to the submatrix sum and subtract the unwanted terms.
        # Unwanted: a=v or b=u.
        # Sum_{a in Nu, a!=v} Sum_{b in Nv, b!=u} A[a,b]
        # Let's just compute the full sum and subtract:
        # Sum_{a in Nu, a!=v} A[a,u]  (b=u)
        # Sum_{b in Nv, b!=u} A[v,b]  (a=v)
        # Note A[u,u]=0, A[v,v]=0.
        # Sum_{a in Nu, a!=v} A[a,u] = deg(u) - 1 (since A[u,v]=1 is the only one? No, A[a,u]=1 for all a in Nu. Sum is deg(u). But we exclude a=v? No, a is in Nu. If a=v, A[v,u]=1. So sum is deg(u). We want to exclude b=u. So we subtract Sum_{a in Nu} A[a,u] = deg(u).
        # Similarly for a=v, we subtract Sum_{b in Nv} A[v,b] = deg(v).
        # But we double subtracted A[v,u] which is 1?
        # Let's check:
        # Total S = sum_{a in Nu} sum_{b in Nv} A[a,b]
        # We want S_clean = sum_{a in Nu \ {v}} sum_{b in Nv \ {u}} A[a,b]
        # S = S_clean + sum_{b in Nv \ {u}} A[v,b] + sum_{a in Nu \ {v}} A[a,u] + A[v,u]
        # sum_{b in Nv \ {u}} A[v,b] = deg(v) - 1 (since A[v,u]=1)
        # sum_{a in Nu \ {v}} A[a,u] = deg(u) - 1 (since A[u,v]=1)
        # A[v,u] = 1
        # So S = S_clean + (deg(v) - 1) + (deg(u) - 1) + 1
        # S_clean = S - deg(v) - deg(u) + 1
        
        deg_u = A[u].sum()
        deg_v = A[v].sum()
        S = np.sum(A[np.ix_(Nu, Nv)])
        return S - deg_u - deg_v + 1

    # Local Search Loop
    # We will try to improve by swapping edges.
    # Pick a random edge (u,v) and a random non-edge (x,y).
    # Calculate delta C4.
    
    edges = [(i, j) for i in range(n) for j in range(i+1, n) if A[i,j] == 1]
    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if A[i,j] == 0]
    
    import random
    random.seed(42)
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        # Pick random edge to remove
        u, v = random.choice(edges)
        # Pick random non-edge to add
        x, y = random.choice(non_edges)
        
        # Check if swap is valid (u,v) != (x,y) obviously true.
        # Also ensure we don't add an edge that is already there (guaranteed by non_edges).
        # Ensure we don't remove an edge that isn't there (guaranteed by edges).
        
        # Calculate delta
        # Remove (u,v): -delta_c4_remove(A, u, v)
        # Add (x,y): +delta_c4_add(A, x, y)
        # But these are not independent if they share vertices.
        # If {u,v} intersects {x,y}, the calculation is tricky.
        # For simplicity, if they share a vertex, we can just recalculate C4 for the new graph.
        # Or skip the move.
        # Given n=31, probability of sharing vertex is moderate.
        # Let's just recalculate if they share a vertex, or use a generic update.
        
        # Actually, delta_c4_add/remove assumes the rest of the graph is fixed.
        # If we remove (u,v) and add (x,y), and they are disjoint, the deltas add up.
        # If they share a vertex, say u=x, then the neighborhood changes.
        
        # To be safe and fast, let's just do a "1-swap" where we remove one edge and add one.
        # If they share a vertex, we skip this pair in the fast path and maybe do a slow recalc?
        # Or just skip.
        
        if u in (x, y) or v in (x, y):
            continue
            
        d_rem = delta_c4_remove(A, u, v)
        d_add = delta_c4_add(A, x, y)
        
        delta = d_add - d_rem
        
        if delta < 0:
            # Apply move
            A[u, v] = 0
            A[v, u] = 0
            A[x, y] = 1
            A[y, x] = 1
            
            # Update lists
            edges.remove((u, v))
            edges.append((x, y))
            non_edges.remove((x, y))
            non_edges.append((u, v))
            
            current_c4 += delta
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_A = A.copy()
                
            # Accept the move (Hill Climbing)
            # We don't need to reset steps for hill climbing, but maybe for SA?
            # Let's stick to strict hill climbing for now.
            
        # Occasionally try a random restart or larger move?
        # For now, just keep going.
        
        if steps % 1000 == 0:
             # Check time
             if time.time() > deadline:
                 break

    # Convert best_A to edge list
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_A[i, j] == 1:
                final_edges.append([i, j])
                
    return final_edges
