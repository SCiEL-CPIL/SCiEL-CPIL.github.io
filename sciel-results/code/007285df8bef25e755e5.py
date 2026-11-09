import time
import random
import numpy as np

def construct():
    n = 31
    m = 91
    deadline = time.time() + 140  # Safety margin
    
    rng = random.Random(42)
    
    def count_c4_fast(adj_mat):
        A = adj_mat.astype(np.int32)
        common = A @ A
        # C4 count is sum over all pairs (u,v) of C(common[u,v], 2) / 2
        # Only consider u < v to avoid double counting pairs, then divide by 2 for the two ways to traverse the cycle?
        # Actually, each C4 is counted 4 times in the sum of C(common, 2) over all ordered pairs?
        # Let's use the formula: Total C4s = 0.5 * sum_{u < v} C(common[u,v], 2)
        # Wait, if u and v are adjacent, they form triangles, not C4s directly with common neighbors?
        # A C4 is u-v-w-x-u. u and w are opposite. v and x are opposite.
        # The pairs of opposite vertices are (u,w) and (v,x).
        # For a pair (u,w) to be opposite in a C4, they must have at least 2 common neighbors.
        # Each C4 has 2 pairs of opposite vertices.
        # So sum_{u < v} C(common[u,v], 2) counts each C4 exactly 2 times (once for each pair of opposite vertices).
        # So Total C4s = 0.5 * sum_{u < v} C(common[u,v], 2).
        
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        cv = common[mask]
        total = np.sum(cv * (cv - 1) // 2)
        return total // 2

    def find_c4_free_graph():
        best_adj = None
        best_c4 = float('inf')
        
        for attempt in range(200): # More attempts
            if time.time() > deadline - 10: break
            
            # Start with a random 90-edge graph
            all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            edges = rng.sample(all_pairs, 90)
            adj = np.zeros((n, n), dtype=np.int8)
            for u, v in edges:
                adj[u, v] = 1
                adj[v, u] = 1
                
            c4s = count_c4_fast(adj)
            
            if c4s < best_c4:
                best_c4 = c4s
                best_adj = adj.copy()
                
            if c4s == 0:
                return adj
            
            # Local search
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                c4s = count_c4_fast(adj)
                if c4s == 0:
                    break
                
                # Find a C4
                A = adj.astype(np.int32)
                common = A @ A
                mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                cv = common[mask]
                indices = np.where(cv >= 2)
                
                if len(indices[0]) == 0:
                    break
                
                # Pick a random C4-defining pair (u, v)
                idx = rng.randint(0, len(indices[0]) - 1)
                u, v = indices[0][idx], indices[1][idx]
                
                # Common neighbors
                cn = np.where((adj[u] == 1) & (adj[v] == 1))[0]
                
                # Try to remove an edge incident to u or v that is part of a C4
                # And add a random edge to maintain count
                best_swap_c4 = c4s
                best_swap = None
                
                # Candidates to remove: edges (u, cn[k]) or (v, cn[k])
                # We need to ensure we don't remove an edge that doesn't help
                for k in range(len(cn)):
                    w = cn[k]
                    # Try removing (u, w)
                    if adj[u, w]:
                        # Pick a random non-edge to add
                        for _ in range(10):
                            a = rng.randint(0, n-1)
                            b = rng.randint(0, n-1)
                            if a == b: continue
                            if a > b: a, b = b, a
                            if adj[a, b] == 1: continue
                            if (a, b) == (u, w): continue
                            
                            # Swap
                            adj[u, w] = 0
                            adj[w, u] = 0
                            adj[a, b] = 1
                            adj[b, a] = 1
                            
                            new_c4s = count_c4_fast(adj)
                            
                            if new_c4s < best_swap_c4:
                                best_swap_c4 = new_c4s
                                best_swap = (u, w, a, b)
                                
                            # Revert
                            adj[u, w] = 1
                            adj[w, u] = 1
                            adj[a, b] = 0
                            adj[b, a] = 0
                            
                    # Try removing (v, w)
                    if adj[v, w]:
                        for _ in range(10):
                            a = rng.randint(0, n-1)
                            b = rng.randint(0, n-1)
                            if a == b: continue
                            if a > b: a, b = b, a
                            if adj[a, b] == 1: continue
                            if (a, b) == (v, w): continue
                            
                            # Swap
                            adj[v, w] = 0
                            adj[w, v] = 0
                            adj[a, b] = 1
                            adj[b, a] = 1
                            
                            new_c4s = count_c4_fast(adj)
                            
                            if new_c4s < best_swap_c4:
                                best_swap_c4 = new_c4s
                                best_swap = (v, w, a, b)
                                
                            # Revert
                            adj[v, w] = 1
                            adj[w, v] = 1
                            adj[a, b] = 0
                            adj[b, a] = 0
                            
                if best_swap:
                    u1, v1, a, b = best_swap
                    adj[u1, v1] = 0
                    adj[v1, u1] = 0
                    adj[a, b] = 1
                    adj[b, a] = 1
                    improved = True
                    c4s = count_c4_fast(adj)
                    if c4s < best_c4:
                        best_c4 = c4s
                        best_adj = adj.copy()
                    if c4s == 0:
                        return adj
                        
        return best_adj

    base_adj = find_c4_free_graph()
    
    if base_adj is None:
        # Fallback: use a random 90-edge graph
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        edges = rng.sample(all_pairs, 90)
        base_adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            base_adj[u, v] = 1
            base_adj[v, u] = 1
    
    # Add the 91st edge
    A = base_adj.astype(np.int32)
    A2 = A @ A
    A3 = A @ A2
    
    best_total_c4 = float('inf')
    best_final_adj = base_adj.copy()
    
    for i in range(n):
        for j in range(i+1, n):
            if base_adj[i, j] == 0:
                paths = A3[i, j]
                if paths < best_total_c4:
                    best_total_c4 = paths
                    best_final_adj = base_adj.copy()
                    best_final_adj[i, j] = 1
                    best_final_adj[j, i] = 1

    # Final check and return
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_final_adj[i, j] == 1:
                final_edges.append([i, j])
                
    return final_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
