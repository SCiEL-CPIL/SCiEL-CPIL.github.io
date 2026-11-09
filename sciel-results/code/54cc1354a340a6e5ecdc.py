import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140  # Safety margin
    n = 20
    
    # Precompute all triples of vertices for K3,3 check
    # A K3,3 exists if there are 3 vertices with >= 3 common neighbors.
    # We need to check for any triple (a,b,c) if |N(a) & N(b) & N(c)| >= 3.
    
    # To vectorize:
    # Let A be the adjacency matrix (n x n boolean).
    # For a fixed triple (a,b,c), common neighbors are given by A[a] & A[b] & A[c].
    # We want to know if sum(A[a] & A[b] & A[c]) >= 3.
    
    # A faster way to check "is graph K3,3-free" is to iterate over all pairs of vertices (u,v)
    # and look at their common neighbors.
    # If any pair (u,v) has >= 3 common neighbors, say w1, w2, w3, 
    # then we must ensure that no two of w1,w2,w3 are connected to each other? 
    # No, K3,3 is bipartite. The vertices in the two sets of size 3 must not have internal edges?
    # Actually, K3,3 is a subgraph. It doesn't matter if there are extra edges.
    # A K3,3 subgraph exists iff there are two disjoint sets of 3 vertices, S1 and S2,
    # such that every vertex in S1 is connected to every vertex in S2.
    
    # The condition "no three vertices have three common neighbours" is equivalent to being K3,3-free?
    # Let's verify.
    # If there is a K3,3, let the partitions be A={a1,a2,a3} and B={b1,b2,b3}.
    # Every ai is connected to every bj.
    # Consider the triple of vertices {b1, b2, b3}. Their common neighbors include a1, a2, a3.
    # So |N(b1) & N(b2) & N(b3)| >= 3.
    # Conversely, if there are 3 vertices {x,y,z} with 3 common neighbors {u,v,w},
    # then the subgraph induced by {x,y,z} U {u,v,w} contains a K3,3 (specifically, the edges between the two sets form a K3,3).
    # Yes, the condition is equivalent.
    
    # Efficient check:
    # For each pair (u,v), calculate common = A[u] & A[v].
    # If sum(common) >= 3, we need to check if any 3 of these common neighbors form an independent set?
    # No. The definition of K3,3 requires the bipartite structure.
    # If u and v are in the same partition of K3,3, they don't need to be connected.
    # But the condition "3 vertices have 3 common neighbors" implies:
    # Let the 3 vertices be S. Let the 3 common neighbors be T.
    # Then every s in S is connected to every t in T.
    # This forms a K3,3 subgraph on S U T.
    # So yes, checking if any triple has >= 3 common neighbors is sufficient and necessary.
    
    # Vectorized approach for counting violations:
    # We can't easily vectorize over all triples (1140 of them) in a single numpy op without huge memory or complex indexing.
    # However, n=20 is small.
    # Alternative: Iterate over all pairs (u,v). There are 190 pairs.
    # For each pair, get common neighbors mask.
    # If count >= 3, we have a potential K3,3 centered at pair (u,v) being in the same partition?
    # Wait. If u,v are in partition A, and w1,w2,w3 in partition B.
    # Then u,v share w1,w2,w3 as neighbors.
    # So if ANY pair (u,v) has >= 3 common neighbors, does that guarantee a K3,3?
    # Yes, because we can pick any 3 of the common neighbors. Let them be w1,w2,w3.
    # Then u,v,w1,w2,w3 form a K3,3?
    # We need u connected to w1,w2,w3. Yes.
    # We need v connected to w1,w2,w3. Yes.
    # Do we need w1 connected to w2? No.
    # Do we need u connected to v? No.
    # So yes, if any pair has >= 3 common neighbors, the graph contains a K3,3.
    
    # This reduces the check from 1140 triples to 190 pairs!
    # And the check for each pair is just counting bits in a 20-bit integer or boolean sum.
    
    def is_k33_free(adj):
        """
        Returns True if the graph is K3,3-free.
        adj: numpy array of shape (n,n) with dtype bool or int.
        """
        # Check all pairs
        # We can vectorize this too if we want, but 190 iterations is very fast in Python if inner is fast.
        # Even better: use numpy to compute common neighbors for all pairs at once?
        # Common neighbors of u and v is A[u] & A[v].
        # We want to know if max(sum(A[u] & A[v] for all u,v)) >= 3.
        
        # Let's try a vectorized approach using matrix multiplication?
        # (A @ A.T)[u,v] gives the number of common neighbors of u and v.
        # If any entry (u,v) with u!=v is >= 3, then it's NOT K3,3 free.
        # Wait, (A @ A.T) counts common neighbors.
        # Diagonal entries are degrees.
        # So we just need to check if max(A @ A.T) >= 3 (excluding diagonal, but diagonal is degree which is likely > 3).
        # Actually, we only care about u != v.
        # If any off-diagonal entry of A @ A.T is >= 3, then there is a K3,3.
        
        common_counts = adj @ adj.T
        # Mask diagonal
        np.fill_diagonal(common_counts, 0)
        return np.max(common_counts) < 3

    def count_violations(adj):
        """
        Returns the number of pairs (u,v) with >= 3 common neighbors.
        This is a proxy for the number of K3,3s (not exact, but correlated and fast).
        Actually, for the energy function, we just want to penalize violations.
        Number of violating pairs is a good metric.
        """
        common_counts = adj @ adj.T
        np.fill_diagonal(common_counts, 0)
        # Count how many pairs have count >= 3
        return np.sum(common_counts >= 3)

    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    # 1. Greedy Construction
    # Start with empty graph
    adj = np.zeros((n, n), dtype=np.int8)
    
    # Randomly shuffle edges to break symmetry
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    
    # Add edges greedily if they don't create a violation
    for u, v in all_edges:
        # Tentatively add edge
        adj[u][v] = 1
        adj[v][u] = 1
        if is_k33_free(adj):
            pass # Keep
        else:
            # Remove
            adj[u][v] = 0
            adj[v][u] = 0
            
    current_edges = np.sum(adj) // 2
    best_adj = adj.copy()
    best_edges = current_edges
    
    # 2. Local Search (2-opt style swaps)
    # We try to replace an existing edge with a missing edge to increase density or maintain density while fixing violations?
    # Since we are valid, we want to increase edges.
    # But we can't just add edges, we might be stuck.
    # We can try to "swap" an edge that is "less useful" with a missing edge?
    # Or simply try to add missing edges, and if it fails, remove a random edge and try again?
    # This is a random walk.
    
    # Better: Try to add missing edges. If it creates a violation, we are stuck.
    # To escape, we can remove a random existing edge, then try to add the missing one?
    # This might decrease edge count temporarily.
    
    # Let's use Simulated Annealing on the state space, but with the FAST check.
    # Move: Flip a single edge.
    # Energy: -edges + penalty * violations.
    # Since check is fast (matrix mult), we can do many steps.
    
    T_start = 1.0
    T_end = 0.01
    alpha = 0.9995
    T = T_start
    
    current_violations = count_violations(adj)
    
    while time.time() < deadline - 5 and T > T_end:
        # Pick random edge to flip
        u, v = random.choice(all_edges)
        
        # Flip
        if adj[u][v]:
            adj[u][v] = 0
            adj[v][u] = 0
            delta_edges = -1
        else:
            adj[u][v] = 1
            adj[v][u] = 1
            delta_edges = 1
            
        new_violations = count_violations(adj)
        new_edges = current_edges + delta_edges
        
        E_old = -current_edges + 100 * current_violations
        E_new = -new_edges + 100 * new_violations
        
        delta_E = E_new - E_old
        
        if delta_E < 0 or random.random() < np.exp(-delta_E / T):
            # Accept
            current_violations = new_violations
            current_edges = new_edges
            
            if current_violations == 0 and current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj.copy()
        else:
            # Reject
            if adj[u][v]:
                adj[u][v] = 0
                adj[v][u] = 0
            else:
                adj[u][v] = 1
                adj[v][u] = 1
                
        T *= alpha
        
    # Final Greedy Polish
    if best_violations_check(best_adj): # Need a helper or inline
        pass

    # Ensure we return a valid graph
    # If best_adj is not valid (shouldn't happen if we only update best when valid), fallback
    if not is_k33_free(best_adj):
        # Re-run greedy or use a known valid structure
        # Fallback: Random restarts of greedy
        for _ in range(10):
            if time.time() > deadline - 10: break
            adj = np.zeros((n, n), dtype=np.int8)
            random.shuffle(all_edges)
            for u, v in all_edges:
                adj[u][v] = 1
                adj[v][u] = 1
                if is_k33_free(adj):
                    pass
                else:
                    adj[u][v] = 0
                    adj[v][u] = 0
            e = np.sum(adj) // 2
            if e > best_edges:
                best_edges = e
                best_adj = adj.copy()
    
    return get_edges(best_adj)

# EVOLVE-BLOCK-END
