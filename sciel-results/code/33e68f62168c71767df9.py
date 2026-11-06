import numpy as np
import random

def construct():
    n = 31
    target_edges = 91
    
    def greedy_construct(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        # Adjacency matrix
        A = np.zeros((n, n), dtype=int)
        edges = []
        
        # Precompute all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        for _ in range(target_edges):
            best_edge = None
            min_cost = float('inf')
            # Shuffle to break ties randomly
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                if A[u, v] == 1:
                    continue
                
                # Calculate cost: number of 4-cycles formed by adding (u, v)
                # This is the number of edges (w, z) in current graph 
                # such that w in N(u), z in N(v), and w != z.
                # Since A is symmetric and zero-diagonal, 
                # cost = sum(A[w, z] for w in N(u) for z in N(v))
                
                N_u = np.nonzero(A[u])[0]
                N_v = np.nonzero(A[v])[0]
                
                if len(N_u) == 0 or len(N_v) == 0:
                    cost = 0
                else:
                    # Use submatrix sum
                    cost = A[np.ix_(N_u, N_v)].sum()
                
                if cost < min_cost:
                    min_cost = cost
                    best_edge = (u, v)
                    # Early exit if we find a zero cost edge? 
                    # It's rare to have many zero cost edges late in the game.
                    # But if min_cost == 0, we can break? 
                    # No, because we want to minimize total cycles, 
                    # picking any 0-cost edge is equally good locally.
                    # But picking one might affect future costs.
                    # Greedy usually just picks the first min or random min.
                    # Let's just find the global min in the shuffled list.
            
            if best_edge is None:
                # Fallback: pick any remaining edge
                for u, v in all_edges:
                    if A[u, v] == 0:
                        best_edge = (u, v)
                        break
            
            u, v = best_edge
            A[u, v] = 1
            A[v, u] = 1
            edges.append([u, v])
            
            # Remove from all_edges to speed up next iterations?
            # all_edges is a list of tuples, removing is O(N).
            # With 465 edges, it's fine.
            try:
                all_edges.remove(best_edge)
            except ValueError:
                pass
        
        return edges

    # Try a few seeds to see if we can get a better result
    # The greedy algorithm is deterministic for a given seed if we don't shuffle, 
    # but we shuffle to handle ties.
    # Actually, the cost calculation is deterministic. 
    # The only randomness is in tie-breaking.
    # Let's run 10 different seeds and pick the one with the lowest total 4-cycles.
    
    def count_4_cycles(edges):
        n = 31
        A = np.zeros((n, n), dtype=int)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
        # Number of 4-cycles = (1/8) * trace(A^4) - terms for triangles and 2-cycles?
        # Formula: C4 = (1/8) * (sum(A^4) - 4*E - 2*V + ...) 
        # Simpler: 
        # C4 = 1/8 * sum_{i,j} (A^4)_{ij} - (terms for non-4-cycles)
        # Actually, (A^4)_{ii} counts closed walks of length 4.
        # Closed walks:
        # 1. i-j-k-i (triangle) x 2 (directions)? No, i-j-k-i is length 3.
        # Length 4 closed walks:
        # a) i-j-k-i-j (back and forth) -> 2 * deg(i) * deg(j) ... complicated.
        # b) i-j-k-l-i (4-cycle) -> 8 per 4-cycle (4 starting vertices * 2 directions)
        # c) i-j-i-k-i (back and forth twice) -> 2 * deg(i)
        # d) i-j-k-j-i (back and forth) -> 2 * deg(j)
        
        # There is a known formula:
        # #C4 = (1/8) * (trace(A^4) - 4*sum(A^2) - 2*sum(A)) ... wait.
        # Let's use the standard formula:
        # #C4 = (1/8) * (trace(A^4) - 4 * sum_{i} (A^2)_{ii} - 2 * sum_{i,j} A_{ij})
        # (A^2)_{ii} = deg(i)
        # sum_{i,j} A_{ij} = 2 * E
        # So #C4 = (1/8) * (trace(A^4) - 4 * sum(deg(i)) - 4 * E)
        # Wait, sum(deg(i)) = 2E. So 4 * sum(deg(i)) = 8E.
        # Let's verify.
        # trace(A^4) = sum of closed walks of length 4.
        # Each 4-cycle contributes 8.
        # Each "back and forth" walk i-j-i-k-i: 
        #   i-j-i is 2 ways? No, i-j-i is one walk. 
        #   Number of such walks: for each edge (i,j), we can go i-j-i. Then from i to k and back.
        #   This is getting messy.
        
        # Alternative: Count directly.
        # For each pair of vertices (u, v), let c_uv be the number of common neighbors.
        # A 4-cycle is determined by two opposite vertices u, v and two common neighbors w, z.
        # For each pair (u, v), the number of 4-cycles where u and v are opposite is C(c_uv, 2).
        # Total 4-cycles = sum_{u < v} C(c_uv, 2).
        # c_uv = (A^2)_{uv}.
        
        A2 = A @ A
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = A2[i, j]
                if c >= 2:
                    total += c * (c - 1) // 2
        return total

    best_edges = None
    best_score = float('inf')
    
    # Run a few iterations
    for seed in range(10):
        edges = greedy_construct(seed)
        score = count_4_cycles(edges)
        if score < best_score:
            best_score = score
            best_edges = edges
            
    return best_edges
