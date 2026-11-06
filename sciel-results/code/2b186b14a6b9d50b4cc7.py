import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Minimizes the number of 4-cycles in a graph with 31 vertices and 91 edges.
    Strategy: Greedy local search with random restarts and simulated annealing.
    Uses NumPy for vectorized calculation of 4-cycle counts.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    n = 31
    m = 91
    
    # Pre-compute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Helper to count 4-cycles
    def count_c4(A):
        # Number of 4-cycles = sum over all pairs (u, v) of C(common_neighbors(u, v), 2)
        # common_neighbors(u, v) = sum(A[u] * A[v])
        # We only need to count each 4-cycle once.
        # A 4-cycle u-v-w-x-u is counted by pairs (u,w) and (v,x).
        # Actually, the standard formula is:
        # C4 = 0.5 * sum_{u < v} ( (A^2)_{uv} choose 2 ) is WRONG for simple graphs?
        # Let's use the property: 
        # Number of C4s = sum_{u < v} C(k(u,v), 2) / 2 ? No.
        # Let's derive:
        # A 4-cycle is u-v-w-x-u. The pairs of opposite vertices are (u,w) and (v,x).
        # For a pair (u,w), let k be the number of common neighbors. 
        # Each pair of common neighbors (v, x) forms a 4-cycle u-v-w-x-u.
        # So sum_{u<w} C(k(u,w), 2) counts each 4-cycle exactly twice (once for u,w and once for v,x).
        # So C4 = 0.5 * sum_{u<w} C(k(u,w), 2).
        
        A2 = A @ A
        # Extract upper triangle
        triu_idx = np.triu_indices(n, 1)
        k_vals = A2[triu_idx]
        # C(k, 2) = k * (k - 1) / 2
        # We need sum of C(k,2)
        c4_count = np.sum(k_vals * (k_vals - 1)) / 4.0
        return int(c4_count)

    # Helper to get edge list from matrix
    def get_edges(A):
        edges = []
        triu_idx = np.triu_indices(n, 1)
        mask = A[triu_idx] == 1
        edge_indices = np.where(mask)[0]
        for idx in edge_indices:
            u, v = triu_idx[0][idx], triu_idx[1][idx]
            edges.append([u, v])
        return edges

    best_edges = None
    best_score = float('inf')
    
    # Initialize with a random graph or a heuristic start
    # A good start is a random regular-like graph or just random
    # Let's try a few restarts
    
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
            
        # Random initialization
        A = np.zeros((n, n), dtype=np.int8)
        random.shuffle(all_edges)
        chosen_edges = all_edges[:m]
        for u, v in chosen_edges:
            A[u, v] = 1
            A[v, u] = 1
            
        current_score = count_c4(A)
        
        # Local search
        T = 1.0  # Temperature
        cooling = 0.99
        min_T = 0.01
        
        # Precompute edge indices in matrix for faster swap?
        # We can iterate over all pairs of (edge, non-edge)
        # To speed up, we can just pick random edge and random non-edge
        
        while time.time() < deadline - 2.0:
            if T < min_T:
                break
                
            # Pick a random edge to remove
            # We need a list of current edges for random selection
            # Re-calculating edge list every step is O(n^2). 
            # With n=31, n^2=961. This is fine.
            curr_edges = get_edges(A)
            if len(curr_edges) != m:
                # Should not happen if we maintain count
                pass
                
            u1, v1 = curr_edges[random.randint(0, len(curr_edges) - 1)]
            
            # Pick a random non-edge to add
            # Find a non-edge
            # To avoid O(n^2) scan for non-edge, we can just try random pairs
            found = False
            attempts = 0
            while not found and attempts < 100:
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 < v2 and A[u2, v2] == 0 and (u2, v2) != (u1, v1) and (v2, u2) != (u1, v1):
                    # Ensure it's not the same edge
                    if not (u1 == u2 and v1 == v2):
                        found = True
                attempts += 1
                
            if not found:
                continue
            
            # Perform swap
            A[u1, v1] = 0
            A[v1, u1] = 0
            A[u2, v2] = 1
            A[v2, u2] = 1
            
            new_score = count_c4(A)
            delta = new_score - current_score
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edges = get_edges(A)
            else:
                # Reject: revert
                A[u1, v1] = 1
                A[v1, u1] = 1
                A[u2, v2] = 0
                A[v2, u2] = 0
                
            T *= cooling
            
    # If best_edges is still None (shouldn't happen), return random
    if best_edges is None:
        A = np.zeros((n, n), dtype=np.int8)
        random.shuffle(all_edges)
        for u, v in all_edges[:m]:
            A[u, v] = 1
            A[v, u] = 1
        best_edges = get_edges(A)
        
    return best_edges
# EVOLVE-BLOCK-END
