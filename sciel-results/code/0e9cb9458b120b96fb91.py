import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer
    
    q = 5
    n = 31
    target_edges = 91
    
    # 1. Generate 31 points of PG(2,5)
    points = []
    points.append((1, 0, 0))
    points.append((0, 1, 0))
    points.append((0, 0, 1))
    for y in range(1, q):
        points.append((1, y, 0))
    for z in range(1, q):
        points.append((1, 0, z))
    for z in range(1, q):
        points.append((0, 1, z))
    for y in range(1, q):
        for z in range(1, q):
            points.append((1, y, z))
            
    # 2. Construct Polarity Graph
    adj = [set() for _ in range(n)]
    base_edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            p_i = points[i]
            p_j = points[j]
            if (p_i[0]*p_j[0] + p_i[1]*p_j[1] + p_i[2]*p_j[2]) % q == 0:
                adj[i].add(j)
                adj[j].add(i)
                base_edges.append((i, j))
                
    # 3. Identify non-edges
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                non_edges.append((i, j))
                
    # 4. Efficient C4 counting function
    # A 4-cycle is counted by pairs of vertices with 2+ common neighbors.
    # Specifically, for each pair (u,v), let c(u,v) be the number of common neighbors.
    # The number of 4-cycles containing the "diagonal" (u,v) is C(c(u,v), 2).
    # Wait, a 4-cycle u-x-v-y-u has diagonals (u,v) and (x,y).
    # So each 4-cycle contributes 1 to C(c(u,v),2) for its two diagonals.
    # Thus, Total C4 = (1/2) * sum_{u<v} C(c(u,v), 2).
    
    def count_c4s(adj_sets):
        # Precompute common neighbor counts for all pairs
        # c[u][v] = len(adj[u] & adj[v])
        # We can compute this efficiently.
        c = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                c[u][v] = len(adj_sets[u] & adj_sets[v])
                c[v][u] = c[u][v]
        
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                k = c[u][v]
                if k >= 2:
                    total += k * (k - 1) // 2
        return total // 2

    # 5. Find the best edge to add
    # Cost of adding edge (u,v) is the number of new C4s created.
    # New C4s are of the form u-x-v-y-u where x, y are common neighbors of u and v?
    # No. Adding (u,v) creates a C4 if there exist x, y such that (u,x), (x,v), (v,y), (y,u) are edges.
    # This requires x in N(u) \cap N(v) and y in N(u) \cap N(v) and (x,y) is an edge?
    # Wait. The cycle is u-x-v-y-u.
    # Edges: (u,x), (x,v), (v,y), (y,u).
    # So x must be a common neighbor of u and v.
    # y must be a common neighbor of u and v.
    # AND (x,y) must be an edge? No, the cycle is u-x-v-y-u. The edges are (u,x), (x,v), (v,y), (y,u).
    # The pair (x,y) is NOT an edge in the cycle. The "diagonal" of this C4 is (u,v) and (x,y).
    # Wait, if I add (u,v), the new C4s must use (u,v).
    # A C4 using (u,v) looks like u-v-x-y-u.
    # Edges: (u,v), (v,x), (x,y), (y,u).
    # So x must be in N(v), y must be in N(u).
    # AND (x,y) must be an edge.
    # So the number of new C4s is the number of edges (x,y) such that x in N(v) and y in N(u).
    # Note: x and y must be distinct and not u,v.
    # Since (u,v) was not an edge, N(u) and N(v) don't contain each other.
    # But x could be in N(u) \cap N(v)? If x is in N(u) \cap N(v), then (u,x) and (v,x) are edges.
    # Then u-v-x-u is a triangle. We need a 4-cycle.
    # The cycle is u-v-x-y-u.
    # x in N(v), y in N(u), and (x,y) is an edge.
    # Also x != y, x != u, x != v, y != u, y != v.
    # Since (u,v) is not an edge, u is not in N(v) and v is not in N(u).
    # So we just need to count edges (x,y) with x in N(v) and y in N(u).
    # Let A = N(v), B = N(u).
    # Count = sum_{x in A} |N(x) \cap B|.
    # This counts pairs (x,y). Since (x,y) is an undirected edge, and x in A, y in B.
    # If x in A \cap B and y in A \cap B, and (x,y) is an edge, it is counted once as (x,y) with x in A, y in B?
    # If we sum over x in A, and count y in N(x) \cap B.
    # If x in A \cap B, y in A \cap B, and (x,y) edge.
    # When x is the first vertex, y is counted.
    # When y is the first vertex (if y in A), x is counted.
    # So edges within A \cap B are counted twice?
    # Let's check.
    # Sum_{x in A} |N(x) \cap B|.
    # If x in A, y in B, and (x,y) edge.
    # If x in B as well, then when we iterate x', if x' = y, and y in A, then x is in N(y) \cap B.
    # So yes, edges with both endpoints in A \cap B are counted twice.
    # Edges with one endpoint in A \ B and one in B \ A are counted once.
    # Edges with one in A \ B and one in A \cap B?
    # If x in A \ B, y in A \cap B. (x,y) edge.
    # x in A, y in B. Counted when x is iterated.
    # y in A, x in B? No, x not in B. So not counted when y is iterated (since x not in B).
    # So only edges within A \cap B are double counted.
    # So New C4s = (Sum - E_CC) where E_CC is edges within A \cap B?
    # Wait. If x,y in A \cap B.
    # Iterating x: y is in N(x) \cap B. Count 1.
    # Iterating y: x is in N(y) \cap B. Count 1.
    # Total 2. But it's 1 edge.
    # So we subtract E_CC?
    # Sum = E_AB + E_AC + E_BC + 2*E_CC?
    # Let A = (A\B) \cup (A\cap B). B = (B\A) \cup (A\cap B).
    # Edges (x,y).
    # x in A\B, y in B\A: 1 count.
    # x in A\B, y in A\cap B: 1 count (when x iterated). y in A, x not in B.
    # x in A\cap B, y in B\A: 1 count (when y iterated? No, y in B, x in A. When y iterated, x in N(y) \cap A? No, we check N(y) \cap B. x in B? No. So 0 counts?
    # Let's re-evaluate.
    # Sum_{x in A} |N(x) \cap B|.
    # Term for x: counts y in N(x) such that y in B.
    # If x in A\B, y in B\A: y in B. Counted.
    # If x in A\B, y in A\cap B: y in B. Counted.
    # If x in A\cap B, y in B\A: y in B. Counted.
    # If x in A\cap B, y in A\cap B: y in B. Counted.
    # So every edge (x,y) with x in A, y in B is counted once for x.
    # But since graph is undirected, edge (y,x) is the same.
    # If y in A and x in B, then when we iterate y, x is in N(y) \cap B. So it's counted again.
    # So an edge (x,y) is counted:
    # 1 if (x in A and y in B) XOR (y in A and x in B).
    # 2 if (x in A and y in B) AND (y in A and x in B). i.e., x,y in A \cap B.
    # 0 if neither.
    # So Sum = (Edges with exactly one endpoint in A\cap B and the other in A\B or B\A) + 2*(Edges in A\cap B).
    # We want the number of distinct edges (x,y) with x in A, y in B.
    # This is (Edges with exactly one in A\cap B) + (Edges in A\cap B).
    # Let S1 = edges with exactly one in A\cap B.
    # Let S2 = edges in A\cap B.
    # Sum = S1 + 2*S2.
    # We want S1 + S2.
    # So New C4s = Sum - S2.
    # S2 is the number of edges in the induced subgraph of A \cap B.
    
    best_edge = None
    min_cost = float('inf')
    
    for u, v in non_edges:
        A = adj[v]
        B = adj[u]
        
        S = 0
        for x in A:
            S += len(adj[x] & B)
            
        C = A & B
        # Count edges in C
        E_CC = 0
        for x in C:
            E_CC += len(adj[x] & C)
        E_CC //= 2
        
        cost = S - E_CC
        
        if cost < min_cost:
            min_cost = cost
            best_edge = (u, v)
            
    if best_edge is None:
        best_edge = non_edges[0]
        
    # Add the best edge
    u_add, v_add = best_edge
    adj[u_add].add(v_add)
    adj[v_add].add(u_add)
    
    # Now we have 91 edges.
    # Let's try to improve by swapping edges.
    # We can remove one edge and add another non-edge.
    # Or perform a 2-opt: remove two, add two.
    # Given time, let's do a simple local search:
    # While time remains:
    #   Pick a random edge to remove.
    #   Find the best non-edge to add (minimizing new C4s).
    #   If total C4s decreases, accept.
    
    current_c4s = count_c4s(adj)
    
    # Precompute list of edges for faster removal
    # We'll maintain a set of edges for O(1) check
    edge_set = set()
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edge_set.add((i, j))
                
    # Add the added edge to edge_set
    edge_set.add((u_add, v_add))
    
    # List of all edges
    all_edges = list(edge_set)
    
    # List of all non-edges
    all_pairs = set()
    for i in range(n):
        for j in range(i+1, n):
            all_pairs.add((i, j))
    non_edges_set = all_pairs - edge_set
    
    # Local Search
    # We want to minimize C4s.
    # Move: Remove edge e1, Add edge e2.
    # Cost change = C4s(new) - C4s(old).
    # If we remove e1=(a,b), we lose C4s containing (a,b).
    # If we add e2=(c,d), we gain C4s containing (c,d).
    # But interactions exist.
    # It's safer to just recount if we have time, or use the incremental formula.
    # Incremental is complex for swaps.
    # Let's try a few random swaps and recount.
    # Recounting is O(n^3) ~ 30000 ops. Very fast.
    
    random.seed(42)
    best_c4s = current_c4s
    best_adj = [set(s) for s in adj]
    best_edge_set = set(edge_set)
    
    iterations = 0
    while time.time() < deadline - 10:
        # Pick a random edge to remove
        if not all_edges: break
        e1 = random.choice(all_edges)
        a, b = e1
        
        # Find best non-edge to add
        # We can restrict to non-edges that are "close" or just pick random non-edges
        # Let's pick a few random non-edges and evaluate
        best_swap_c4s = best_c4s
        best_swap_e2 = None
        
        # Sample some non-edges
        non_edges_list = list(non_edges_set)
        if not non_edges_list: break
        
        # Try top 10 random non-edges
        samples = random.sample(non_edges_list, min(10, len(non_edges_list)))
        
        for e2 in samples:
            c, d = e2
            
            # Simulate swap
            # Remove e1
            adj[a].discard(b)
            adj[b].discard(a)
            # Add e2
            adj[c].add(d)
            adj[d].add(c)
            
            new_c4s = count_c4s(adj)
            
            # Revert
            adj[c].discard(d)
            adj[d].discard(c)
            adj[a].add(b)
            adj[b].add(a)
            
            if new_c4s < best_swap_c4s:
                best_swap_c4s = new_c4s
                best_swap_e2 = e2
        
        if best_swap_e2 is not None:
            # Apply swap
            a, b = e1
            c, d = best_swap_e2
            
            adj[a].discard(b)
            adj[b].discard(a)
            adj[c].add(d)
            adj[d].add(c)
            
            edge_set.discard(e1)
            edge_set.add(best_swap_e2)
            
            all_edges.remove(e1)
            all_edges.append(best_swap_e2)
            
            non_edges_set.discard(best_swap_e2)
            non_edges_set.add(e1)
            
            best_c4s = best_swap_c4s
            best_adj = [set(s) for s in adj]
            best_edge_set = set(edge_set)
            
        iterations += 1
        if iterations % 100 == 0:
            # Optional: print progress if debugging
            pass
            
    # Finalize
    final_edges = []
    for u, v in best_edge_set:
        final_edges.append([u, v])
        
    return final_edges

# Helper for testing
if __name__ == "__main__":
    edges = construct()
    print(len(edges))
