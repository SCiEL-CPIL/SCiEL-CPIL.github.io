import numpy as np
import random

def count_4_cycles(A):
    """
    Count the number of 4-cycles in the graph represented by adjacency matrix A.
    A 4-cycle is a subgraph isomorphic to C4.
    Formula: Sum over all pairs of vertices (i, j) of C(common_neighbors(i, j), 2) / 4
    """
    A2 = A @ A
    # A2[i, j] is the number of common neighbors (length-2 paths) between i and j.
    # Each 4-cycle u-v-w-x-u contributes 2 to A2[u, w] (paths u-v-w and u-x-w)
    # and 2 to A2[v, x].
    # So each 4-cycle is counted 4 times in sum_{i<j} C(A2[i,j], 2).
    # Wait, C(k, 2) counts pairs of distinct common neighbors.
    # For u, w, common neighbors are v and x. C(2, 2) = 1.
    # So each 4-cycle contributes 1 to sum_{i<j} C(A2[i,j], 2) for the pair (u,w)
    # and 1 for the pair (v,x).
    # Total contribution per 4-cycle is 2.
    # So N(C4) = 0.5 * sum_{i<j} C(A2[i,j], 2).
    
    # Let's re-verify.
    # Cycle 1-2-3-4-1.
    # Pairs: (1,3) common neighbors {2,4}. C(2,2)=1.
    # Pair (2,4) common neighbors {1,3}. C(2,2)=1.
    # Other pairs (1,2) common neighbors? No, unless there are chords.
    # In a simple C4, only opposite vertices have 2 common neighbors.
    # So sum is 2.
    # N(C4) = 2 / 2 = 1. Correct.
    
    # What if there are multiple 4-cycles sharing edges?
    # The formula holds.
    
    # Efficient calculation:
    # Sum_{i<j} (A2[i,j] * (A2[i,j] - 1)) / 2
    # Then divide by 2.
    
    # We only need upper triangle for i<j.
    # Or sum all and handle symmetry.
    # A2 is symmetric.
    # Sum_{i != j} C(A2[i,j], 2) / 2?
    # Sum_{i<j} C(A2[i,j], 2) is half of Sum_{i != j} C(A2[i,j], 2).
    # So Total = 0.5 * (0.5 * Sum_{i != j} C(A2[i,j], 2)) = 0.25 * Sum_{i != j} C(A2[i,j], 2).
    
    # Let's just compute sum over all i,j and adjust.
    # Sum_{i,j} C(A2[i,j], 2) includes i=j terms where C(d_i, 2) counts triangles? No, C4s?
    # A2[i,i] is d_i. C(d_i, 2) counts pairs of neighbors of i.
    # This corresponds to length-2 paths i-a-i and i-b-i? No, a-b edge?
    # If a,b are neighbors of i, and a-b is an edge, it's a triangle.
    # We don't want triangles.
    # So we must exclude i=j.
    
    # Let S = sum_{i != j} C(A2[i,j], 2).
    # Then N(C4) = S / 4.
    
    A2 = A @ A
    # Zero out diagonal to ignore i=j
    np.fill_diagonal(A2, 0)
    
    # Calculate sum of C(k, 2) for all off-diagonal elements
    # C(k, 2) = k(k-1)/2
    # Sum = 0.5 * sum(A2 * (A2 - 1))
    # Since A2 is symmetric, sum_{i!=j} is 2 * sum_{i<j}.
    # We want sum_{i<j} C(A2[i,j], 2).
    # Let's just sum the whole matrix off-diagonal and divide by 2 to get i<j sum.
    
    sum_pairs = np.sum(A2 * (A2 - 1))
    # This sum is over all i,j. Since A2 is symmetric, it's 2 * sum_{i<j}.
    # sum_{i<j} C(A2[i,j], 2) = 0.5 * sum_{i<j} A2[i,j](A2[i,j]-1)
    # = 0.5 * (0.5 * sum_{i!=j} A2[i,j](A2[i,j]-1))
    # = 0.25 * sum_pairs
    
    sum_ij = 0.25 * sum_pairs
    
    # N(C4) = sum_ij / 2
    return int(sum_ij / 2)

def construct():
    n = 31
    m = 91
    
    best_edges = None
    best_score = float('inf')
    
    # Random restarts to avoid local minima
    # Each restart: random init + hill climbing
    num_restarts = 20
    max_iters_per_restart = 5000
    
    for r in range(num_restarts):
        # Initialize random graph
        A = np.zeros((n, n), dtype=int)
        edges_set = set()
        
        # Generate m random edges
        while len(edges_set) < m:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                if u > v: u, v = v, u
                if (u, v) not in edges_set:
                    edges_set.add((u, v))
                    A[u, v] = 1
                    A[v, u] = 1
        
        current_score = count_4_cycles(A)
        if current_score < best_score:
            best_score = current_score
            best_edges = [(u, v) for u, v in edges_set]
            
        # Hill Climbing
        # List of current edges and non-edges for fast sampling
        # We can just sample randomly and check if edge exists
        
        for i in range(max_iters_per_restart):
            # Pick a random edge to remove
            # To do this efficiently, we might maintain a list, but with 91 edges,
            # sampling from set is fine.
            # However, we need a random non-edge to add.
            
            # Convert edges_set to list for random choice? 
            # Or just pick random u,v and check.
            
            # Better: Maintain a list of edges.
            # But we are using set for A construction.
            # Let's just pick random pairs.
            
            # Remove edge
            # Pick random existing edge
            # We need a list of edges for O(1) random selection.
            # Let's maintain a list `edge_list` alongside `A`.
            
            # This is getting complicated to manage state in the loop without refactoring.
            # Let's just use the set.
            # Random edge:
            # Pick random u, v. If edge exists, remove. Else skip?
            # This is inefficient if graph is dense. 91/465 is ~20%.
            # So 80% chance of picking non-edge.
            
            # Strategy:
            # 1. Pick random u, v.
            # 2. If A[u,v] == 1, remove it.
            # 3. Pick random x, y.
            # 4. If A[x,y] == 0, add it.
            # 5. If both succeeded, check score.
            
            # To ensure we actually try a swap, we might need to retry.
            
            success = False
            for _ in range(10): # max retries for finding valid swap
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v or A[u, v] == 0:
                    continue
                    
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y or A[x, y] == 1:
                    continue
                
                # Potential swap: remove (u,v), add (x,y)
                # Check if (x,y) is same as (u,v) - impossible since A[x,y]==0 and A[u,v]==1
                
                # Apply swap
                A[u, v] = 0
                A[v, u] = 0
                A[x, y] = 1
                A[y, x] = 1
                
                new_score = count_4_cycles(A)
                
                if new_score < current_score:
                    current_score = new_score
                    edges_set.discard((min(u,v), max(u,v)))
                    edges_set.add((min(x,y), max(x,y)))
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = [(a, b) for a, b in edges_set]
                    success = True
                    break
                else:
                    # Revert
                    A[u, v] = 1
                    A[v, u] = 1
                    A[x, y] = 0
                    A[y, x] = 0
            
            if not success:
                # If we can't find an improving move easily, we might be stuck.
                # With 5000 iters, we should explore well.
                pass
        
        # Early stopping if we hit 0? (Impossible for m=91 > ex)
        if best_score == 0:
            break

    return best_edges
