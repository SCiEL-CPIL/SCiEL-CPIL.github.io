import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = 31
    target_edges = 91
    
    # 1. Construct a base graph using the Polarity Graph of the Finite Field F_{q^2}
    # q = 5, n = q^2 + q + 1 = 25 + 5 + 1 = 31
    # Vertices: 0..24 represent elements of F_25, 25..30 are extra? 
    # Actually, standard polarity graph on F_{q^2} has q^2+q+1 vertices.
    # Let's use the standard construction for v = q^2+q+1.
    
    q = 5
    # We need to represent F_{25}.
    # F_{25} = F_5[x] / (x^2 + x + 2) since x^2+x+2 is irreducible over F_5.
    # Let's verify irreducibility:
    # f(0)=2, f(1)=1+1+2=4, f(2)=4+2+2=8=3, f(3)=9+3+2=14=4, f(4)=16+4+2=22=2.
    # No roots in F_5, so it's irreducible.
    
    # We will map vertices 0..30 to:
    # 0..24: elements of F_25
    # 25..30: This mapping is tricky for a direct polarity graph which is usually on q^2+q+1 points of PG(2, q).
    # A simpler approach for C4 minimization is the "Norm Graph" or "Polarity Graph" structure.
    
    # Alternative Strategy: 
    # Use a greedy heuristic or simulated annealing starting from a random or structured graph.
    # Given the small size (31 nodes, 91 edges), a local search is very feasible.
    
    # Initial Graph Construction:
    # To minimize 4-cycles, we want the graph to be as "triangle-free" and "4-cycle-free" as possible.
    # The extremal graph ex(31, C4) has 90 edges. We need 91.
    # A good starting point is a random graph with 91 edges, but structured is better.
    # Let's try a "Blow-up" of a small C4-free graph or just use a random start with SA.
    # However, a known C4-free graph with 31 vertices and 90 edges is the polarity graph of PG(2,5).
    # Let's construct the Polarity Graph of PG(2,5).
    
    # Points of PG(2,5) are non-zero triples (x,y,z) in F_5^3 modulo scaling.
    # Number of points = (5^3-1)/(5-1) = 124/4 = 31.
    # Polarity: (x,y,z) is orthogonal to (x',y',z') if xx' + yy' + zz' = 0? 
    # Standard polarity in PG(2,q) for q odd:
    # Points are (x,y,1) for x,y in F_q, (x,1,0) for x in F_q, and (1,0,0).
    # Total: q^2 + q + 1 = 31.
    
    # Let's define the points explicitly.
    points = []
    # Type 1: (x, y, 1) for x, y in 0..4
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    # Type 2: (x, 1, 0) for x in 0..4
    for x in range(q):
        points.append((x, 1, 0))
    # Type 3: (1, 0, 0)
    points.append((1, 0, 0))
    
    # Verify count
    assert len(points) == 31
    
    # Define the polarity relation.
    # In PG(2, q) with q odd, a standard polarity is given by the symmetric bilinear form:
    # B(P, Q) = x1*x2 + y1*y2 + z1*z2 ? No, that's not a polarity in the sense of a self-dual correlation unless we adjust.
    # Actually, for q odd, the standard polarity is:
    # (x,y,z) ~ (x',y',z') iff xx' + yy' + zz' = 0 is NOT a polarity (it's symmetric but not alternating).
    # Wait, a polarity maps points to lines. A point P is absolute if P is on its pole line.
    # For the standard polarity in PG(2, q) (q odd), the absolute points form a conic? No.
    # Let's use the specific construction for the C4-free graph:
    # Vertices are points of PG(2, q).
    # Edge between P and Q if P is on the pole of Q?
    # Or simply: P ~ Q iff B(P, Q) = 0 for a suitable symmetric bilinear form that defines a polarity.
    # For q=5, the form x1*x2 + y1*y2 + z1*z2 is symmetric.
    # Is it a polarity? A polarity is a correlation of order 2.
    # The map P -> line L_P where L_P = {Q | B(P,Q)=0} is a polarity if the matrix is symmetric and non-singular.
    # The matrix is Identity, which is symmetric and non-singular.
    # So P is adjacent to Q if B(P,Q) = 0 AND P != Q.
    # Note: If B(P,P) = 0, P is an absolute point and is adjacent to itself? No, simple graphs have no self-loops.
    # So we exclude self-loops.
    
    edges = []
    adj_matrix = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            q_pt = points[j]
            # B(P, Q) = x1*x2 + y1*y2 + z1*z2 mod 5
            val = (p[0]*q_pt[0] + p[1]*q_pt[1] + p[2]*q_pt[2]) % q
            if val == 0:
                edges.append([i, j])
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
                
    # This graph should have ex(31, C4) edges, which is 90.
    # Let's check the count.
    if len(edges) == 90:
        # We need 91 edges.
        # We will use Simulated Annealing to add one edge and then optimize.
        # Or, we can just add the "best" edge that minimizes the increase in C4s.
        
        # Calculate current C4 count (should be 0 or minimal)
        def count_c4(adj):
            # C4 count = 1/2 * sum_{u,v} (common_neighbors(u,v) choose 2) ? No.
            # Number of C4s = 1/4 * sum_{u,v} (d(u,v) choose 2) where d(u,v) is number of common neighbors?
            # Actually, each C4 is counted 4 times if we sum over pairs of opposite vertices?
            # Let's use: Count = 0
            # For each pair of vertices (u, v), let k be the number of common neighbors.
            # Each C4 contains 2 pairs of opposite vertices.
            # So total C4s = 1/2 * sum_{u < v} C(k_uv, 2).
            
            total = 0
            # Use matrix multiplication for speed?
            # M2 = A^2. M2[u,v] = number of common neighbors of u and v.
            # C4s = 0.5 * sum_{u < v} C(M2[u,v], 2)
            # Note: Diagonal elements M2[u,u] = degree(u). C(deg, 2) counts triangles, not C4s.
            # We only sum for u != v.
            
            A2 = np.dot(adj, adj)
            # Zero out diagonal
            np.fill_diagonal(A2, 0)
            
            # For each u < v, add C(A2[u,v], 2)
            # Vectorized:
            # We need sum of x*(x-1)/2 for all u < v.
            # Let S = sum_{u < v} A2[u,v]*(A2[u,v]-1)
            # Then C4s = S / 4? 
            # Let's trace: A C4 is u-v-w-x-u.
            # Pairs of opposite vertices: (u,w) and (v,x).
            # Common neighbors of u and w are v and x. So C(2,2)=1.
            # Common neighbors of v and x are u and w. So C(2,2)=1.
            # Sum over all u < v of C(k_uv, 2) counts each C4 exactly twice (once for each pair of opposite vertices).
            # So C4s = 0.5 * Sum.
            
            # Calculate sum of C(k, 2) for all u < v
            # k = A2[u,v]
            # C(k,2) = k*(k-1)/2
            # Sum = 0.5 * sum_{u<v} k*(k-1)
            # Total C4s = 0.5 * Sum = 0.25 * sum_{u<v} k*(k-1)
            
            # Since A2 is symmetric, sum_{u<v} k*(k-1) = 0.5 * (sum_{u!=v} k*(k-1))
            # So Total C4s = 0.25 * 0.5 * sum_{u!=v} k*(k-1) = 0.125 * sum_{u!=v} k*(k-1)
            
            # Let's just compute it directly to be safe with small n.
            c4_count = 0
            for u in range(n):
                for v in range(u + 1, n):
                    k = A2[u, v]
                    if k >= 2:
                        c4_count += k * (k - 1) // 2
            return c4_count // 2 # Wait, the formula derived:
            # Each C4 contributes 1 to C(k_uw, 2) and 1 to C(k_vx, 2).
            # Sum over all pairs (u,v) of C(k_uv, 2) counts each C4 twice.
            # So C4s = Sum / 2.
            # My loop calculates Sum. So return Sum / 2.
            # Let's re-verify.
            # C4: 1-2-3-4-1.
            # Opposite pairs: (1,3) and (2,4).
            # Common neighbors of 1,3: 2,4. k=2. C(2,2)=1.
            # Common neighbors of 2,4: 1,3. k=2. C(2,2)=1.
            # Sum = 1 + 1 = 2.
            # C4s = 2 / 2 = 1. Correct.
            
        current_c4s = count_c4(adj_matrix)
        
        # We have 90 edges. We need 91.
        # Find the edge (u,v) not in graph that adds the minimum number of new C4s.
        # Adding edge (u,v) creates new C4s if there are common neighbors.
        # Specifically, if u and v have k common neighbors, adding (u,v) creates C(k, 2) new C4s?
        # No. A new C4 involving edge (u,v) must be of the form u-w-x-v-u.
        # w and x must be common neighbors of u and v?
        # If we add (u,v), a new C4 is u-v-x-w-u.
        # This requires (v,x), (x,w), (w,u) to exist.
        # Since (u,v) is new, the other 3 edges must exist.
        # This forms a triangle v-x-w-v? No.
        # The C4 is u-v-x-w-u.
        # Edges: (u,v) [new], (v,x) [old], (x,w) [old], (w,u) [old].
        # So x is a neighbor of v, w is a neighbor of u, and (x,w) is an edge.
        # This is not simply C(common_neighbors, 2).
        
        # However, a simpler metric for "potential" is the number of common neighbors.
        # If u and v have many common neighbors, adding (u,v) likely creates many C4s.
        # Let's evaluate all missing edges.
        
        best_edge = None
        min_c4_increase = float('inf')
        
        # Precompute degrees and adjacency lists for speed
        # But n=31 is small.
        
        for u in range(n):
            for v in range(u + 1, n):
                if adj_matrix[u, v] == 0:
                    # Calculate increase in C4s if we add (u,v)
                    # New C4s are of the form u-v-x-w-u
                    # This is equivalent to counting paths of length 2 from v to u that do not use the edge (u,v) (which doesn't exist yet)
                    # AND such that the path v-x-w-u is valid.
                    # Actually, any pair (x,w) such that v-x, x-w, w-u exist forms a C4 with the new edge u-v.
                    # Wait, u-v-x-w-u is a C4 if u!=x, v!=w, etc.
                    # Since it's a simple graph, u,v,x,w are distinct?
                    # If x=u, then v-u-w-u, not a C4.
                    # So we need distinct vertices.
                    
                    # Count pairs (x, w) such that:
                    # adj[v,x] == 1
                    # adj[x,w] == 1
                    # adj[w,u] == 1
                    # And {u,v,x,w} are distinct.
                    
                    # This can be computed as:
                    # Sum_{x in N(v)} Sum_{w in N(u)} [adj[x,w]==1]
                    # Subtract cases where vertices are not distinct.
                    
                    # Given small n, we can just iterate.
                    inc = 0
                    for x in range(n):
                        if x == u or x == v: continue
                        if adj_matrix[v, x] == 0: continue
                        for w in range(n):
                            if w == u or w == v or w == x: continue
                            if adj_matrix[x, w] == 0: continue
                            if adj_matrix[w, u] == 0: continue
                            inc += 1
                            
                    # Each C4 is counted once in this iteration?
                    # The C4 is u-v-x-w-u.
                    # The reverse is u-w-x-v-u.
                    # Does my loop count both?
                    # Loop counts (x,w).
                    # If we swap x and w, we get (w,x).
                    # adj[v,w] must be 1? No, adj[v,x] was 1.
                    # The C4 u-v-x-w-u uses edge (v,x) and (w,u).
                    # The C4 u-w-x-v-u uses edge (v,u) [new] and (w,x) and (x,v)?
                    # No, the C4 is defined by the cycle.
                    # The set of edges is {(u,v), (v,x), (x,w), (w,u)}.
                    # My loop finds this specific ordering.
                    # Does it find the same C4 again?
                    # If we pick x' = w and w' = x:
                    # Need adj[v,w]==1? Not necessarily.
                    # So each C4 is counted exactly once for a fixed orientation of the new edge (u,v)?
                    # Actually, the C4 is undirected.
                    # The structure is u-v-x-w-u.
                    # Is there another x', w' that generates the same set of edges?
                    # Only if the graph has symmetries, but x and w are specific vertices.
                    # So `inc` is the exact number of new C4s.
                    
                    if inc < min_c4_increase:
                        min_c4_increase = inc
                        best_edge = (u, v)
                        
        # Add the best edge
        if best_edge:
            u, v = best_edge
            edges.append([u, v])
            adj_matrix[u, v] = 1
            adj_matrix[v, u] = 1
            
        # Now we have 91 edges.
        # We can try to improve by swapping edges.
        # Simulated Annealing:
        # Move: Remove edge (a,b), Add edge (c,d) where (c,d) is not in graph.
        # Constraint: Keep edge count 91.
        # Objective: Minimize C4s.
        
        # To speed up C4 counting, we can maintain a counter.
        # But recalculating from scratch is O(n^3) ~ 31^3 ~ 30k ops. Very fast.
        # 150s allows millions of steps.
        
        current_score = count_c4(adj_matrix)
        best_score = current_score
        best_adj = adj_matrix.copy()
        best_edges = edges.copy()
        
        temp = 1.0
        cooling_rate = 0.999
        
        # Time check
        while time.time() < deadline - 10:
            # Pick random edge to remove
            e_idx = random.randint(0, len(edges) - 1)
            u, v = edges[e_idx]
            
            # Pick random non-edge to add
            # We need to find a non-edge quickly.
            # Since density is high (~91/465 ~ 20%), most pairs are edges.
            # We can just pick random u, v and check.
            found = False
            attempts = 0
            while not found and attempts < 50:
                a = random.randint(0, n - 1)
                b = random.randint(0, n - 1)
                if a != b and adj_matrix[a, b] == 0:
                    # Ensure we don't remove the same edge we are adding (impossible if non-edge)
                    # Ensure (a,b) is distinct from (u,v)
                    found = True
                attempts += 1
            
            if not found:
                continue
                
            # Apply move
            adj_matrix[u, v] = 0
            adj_matrix[v, u] = 0
            adj_matrix[a, b] = 1
            adj_matrix[b, a] = 1
            
            # Update edges list? 
            # For simplicity, we can just use adj_matrix for counting.
            # But we need to return the list of edges.
            # We can reconstruct edges from adj_matrix at the end if needed, 
            # or maintain the list. Maintaining list is O(1) for removal, O(1) for add.
            
            # Remove (u,v) from edges
            # Find index
            # To make this fast, we can use a set of tuples for O(1) lookup?
            # But we need to return list of lists.
            # Let's just re-derive edges from adj_matrix if we do many swaps?
            # No, 91 edges is small.
            
            # Let's just use adj_matrix for the score and reconstruct at the end.
            # But we need to know which edges to remove/add for the next step.
            # We'll maintain `edges` as a list of tuples for easier manipulation, 
            # and convert to list of lists at the end.
            
            # Wait, I'm modifying adj_matrix directly.
            # I need to update `edges` list to keep it in sync for random selection.
            
            # This is getting complex to manage state perfectly with simple list ops.
            # Alternative: Just use adj_matrix for everything.
            # To pick a random edge to remove:
            # Pick random u, v. If adj[u,v]==1, it's an edge.
            
            # Let's restart the SA loop logic to be cleaner.
            pass

        # The above while loop was a sketch. Let's implement a cleaner SA.
        
        # Reset to best known so far (which is the polarity graph + 1 edge)
        adj_matrix = best_adj
        current_score = best_score
        
        temp = 1.0
        cooling_rate = 0.9995
        
        # Precompute list of all possible edges for faster random non-edge selection?
        # Or just try random pairs.
        
        while time.time() < deadline - 5:
            # Pick random edge to remove
            while True:
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v and adj_matrix[u, v] == 1:
                    break
            
            # Pick random non-edge to add
            while True:
                a = random.randint(0, n - 1)
                b = random.randint(0, n - 1)
                if a != b and adj_matrix[a, b] == 0:
                    # Avoid trivial swaps (removing u-v and adding u-v)
                    if (u == a and v == b) or (u == b and v == a):
                        continue
                    break
            
            # Calculate delta in C4s
            # Removing (u,v) decreases C4s. Adding (a,b) increases C4s.
            # It's easier to just recalculate.
            
            # Apply
            adj_matrix[u, v] = 0
            adj_matrix[v, u] = 0
            adj_matrix[a, b] = 1
            adj_matrix[b, a] = 1
            
            new_score = count_c4(adj_matrix)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = adj_matrix.copy()
            else:
                # Metropolis criterion
                delta = new_score - current_score
                if random.random() < np.exp(-delta / temp):
                    current_score = new_score
                else:
                    # Revert
                    adj_matrix[u, v] = 1
                    adj_matrix[v, u] = 1
                    adj_matrix[a, b] = 0
                    adj_matrix[b, a] = 0
            
            temp *= cooling_rate
            if temp < 0.01:
                temp = 0.01 # Floor

        # Reconstruct edge list from best_adj
        final_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if best_adj[i, j] == 1:
                    final_edges.append([i, j])
                    
        return final_edges

    else:
        # Fallback if polarity graph construction failed or edge count mismatch
        # Just return a random graph or the initial lexicographic one
        # But we should try to make it valid.
        # If len(edges) != 90, we can't proceed with the specific "add 1 edge" logic.
        # Let's just return a random graph with 91 edges as a fallback.
        # This is unlikely to happen with the correct PG(2,5) construction.
        
        # Generate random graph
        all_pairs = [[i, j] for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        return all_pairs[:91]

# Test helper
if __name__ == "__main__":
    edges = construct()
    print(f"Generated {len(edges)} edges")
    # Count C4s
    n = 31
    adj = np.zeros((n, n), dtype=int)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    A2 = np.dot(adj, adj)
    np.fill_diagonal(A2, 0)
    c4s = 0
    for u in range(n):
        for v in range(u + 1, n):
            k = A2[u, v]
            if k >= 2:
                c4s += k * (k - 1) // 2
    print(f"C4 count: {c4s // 2}")
