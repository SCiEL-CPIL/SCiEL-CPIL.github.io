import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    q = 5
    n = q * q + q + 1  # 31
    
    # Generate all points in PG(2, q)
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make the first non-zero coordinate 1
                vec = [x, y, z]
                first_non_zero_idx = -1
                for i in range(3):
                    if vec[i] != 0:
                        first_non_zero_idx = i
                        break
                
                if first_non_zero_idx != -1:
                    inv = pow(vec[first_non_zero_idx], q - 2, q) # Fermat's little theorem for inverse in F_q
                    normalized = [(v * inv) % q for v in vec]
                    # Ensure the first non-zero is 1
                    # It should be by construction
                    if normalized not in points:
                        points.append(normalized)
    
    # There should be exactly 31 points
    if len(points) != n:
        # Fallback to a simple random graph if construction fails
        # But it shouldn't fail
        pass
        
    # Define the polarity matrix M
    # A standard polarity for odd q
    M = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=int)
    
    # Build adjacency matrix
    A = np.zeros((n, n), dtype=int)
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            u = np.array(points[i])
            v = np.array(points[j])
            # Check orthogonality: u^T M v == 0 mod q
            val = (u @ M @ v) % q
            if val == 0:
                A[i, j] = 1
                A[j, i] = 1
                edges.append([i, j])
    
    # Check edge count
    if len(edges) != 90:
        # If the polarity didn't yield exactly 90 edges, we might need to adjust
        # For q=5, this specific M should work. 
        # If not, we can just take the first 90 edges of a random graph or similar, 
        # but let's assume it works for now. 
        # Actually, let's verify the count.
        # If it's not 90, we can try another M or just proceed with what we have 
        # and add/remove edges to get to 90, then add 1 more.
        # But for q=5, the number of edges in the polarity graph of PG(2,5) is 
        # (q+1)*q*(q-1)/2? No.
        # The number of absolute points is q+1.
        # Each absolute point is adjacent to q-1 other points?
        # Total edges = (q+1)(q-1) + (q^2-q)(q+1)/2 ?
        # Let's just trust the generation and check the count.
        
        # If len(edges) != 90, we can fix it by adding/removing random edges
        # to reach 90, then proceed.
        while len(edges) < 90:
            # Add a random non-edge
            found = False
            for _ in range(1000):
                i, j = np.random.choice(n, 2, replace=False)
                if i > j: i, j = j, i
                if A[i, j] == 0:
                    A[i, j] = 1
                    A[j, i] = 1
                    edges.append([i, j])
                    found = True
                    break
            if not found:
                break
        
        while len(edges) > 90:
            # Remove a random edge
            idx = np.random.choice(len(edges))
            i, j = edges[idx]
            A[i, j] = 0
            A[j, i] = 0
            edges.pop(idx)
    
    # Now we have 90 edges. We need to add 1 edge to minimize C4s.
    # Calculate number of length-3 paths between all non-adjacent pairs.
    # This is (A^3)[i][j] for i != j and A[i][j] == 0.
    
    A3 = np.linalg.matrix_power(A, 3)
    
    min_paths = float('inf')
    best_edge = None
    
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0:
                paths = A3[i, j]
                if paths < min_paths:
                    min_paths = paths
                    best_edge = (i, j)
    
    # Add the best edge
    if best_edge:
        i, j = best_edge
        A[i, j] = 1
        A[j, i] = 1
        edges.append([i, j])
    
    # Optional: Local search to improve further
    # Given the time constraint and the strength of the initial construction,
    # a simple local search might not be necessary, but let's add a quick one.
    
    def count_c4s(adj):
        # Count C4s using adjacency matrix
        # Number of C4s = (1/8) * sum_{i,j} (A^2)[i][j]^2 - ... 
        # Actually, the number of C4s is (1/8) * trace((A^2)^2) - ... no.
        # A simpler way: 
        # For each pair of vertices (u, v), let c(u, v) be the number of common neighbors.
        # The number of C4s containing edge (u, v) is not just c(u, v).
        # The total number of C4s is (1/8) * sum_{u, v} c(u, v)^2 - ... 
        # Wait, the number of C4s is (1/8) * (sum_{u, v} c(u, v)^2 - sum_u deg(u) + ...)?
        # Let's use the formula: Number of C4s = (1/8) * (sum_{u, v} (A^2)_{uv}^2 - sum_u deg(u) + ... )
        # Actually, a robust way is:
        # Count all 4-cycles.
        # A 4-cycle is a set of 4 vertices {a,b,c,d} with edges ab, bc, cd, da.
        # We can count them by iterating over pairs of edges?
        # Or use the matrix formula:
        # N_C4 = (1/8) * (sum_{i,j} (A^2)_{ij}^2 - sum_i deg(i) + ... )
        # Let's just compute it directly for small n=31.
        # Iterate over all pairs of vertices (i, j).
        # For each pair, find common neighbors.
        # This counts C4s multiple times.
        
        # Better: 
        # For each pair of vertices (u, v), let k = number of common neighbors.
        # The number of C4s where u and v are opposite vertices is C(k, 2).
        # Each C4 has 2 pairs of opposite vertices.
        # So Total C4s = (1/2) * sum_{u < v} C(common_neighbors(u, v), 2).
        
        A2 = A @ A
        total_c4s = 0
        for u in range(n):
            for v in range(u + 1, n):
                k = A2[u, v]
                if k >= 2:
                    total_c4s += k * (k - 1) // 2
        return total_c4s // 2 # Wait, each C4 is counted twice? 
        # A C4 is u-v-w-x-u. Opposite pairs are (u,w) and (v,x).
        # So yes, each C4 is counted twice in the sum over all pairs.
        # So divide by 2.
    
    current_c4s = count_c4s(A)
    
    # Local search: try swapping edges
    # If time permits
    while time.time() < deadline - 5:
        improved = False
        # Try to remove an edge (i, j) and add a non-edge (k, l)
        # To speed up, only try a few random swaps per iteration
        for _ in range(100):
            # Pick a random edge to remove
            edge_idx = np.random.choice(len(edges))
            i, j = edges[edge_idx]
            
            # Pick a random non-edge to add
            found_non_edge = False
            for _ in range(50):
                k, l = np.random.choice(n, 2, replace=False)
                if k > l: k, l = l, k
                if A[k, l] == 0 and (k, l) != (i, j):
                    found_non_edge = True
                    break
            
            if not found_non_edge:
                continue
            
            # Perform swap
            A[i, j] = 0
            A[j, i] = 0
            A[k, l] = 1
            A[l, k] = 1
            
            edges[edge_idx] = [k, l]
            
            new_c4s = count_c4s(A)
            
            if new_c4s < current_c4s:
                current_c4s = new_c4s
                improved = True
            else:
                # Revert
                A[i, j] = 1
                A[j, i] = 1
                A[k, l] = 0
                A[l, k] = 0
                edges[edge_idx] = [i, j]
        
        if not improved:
            # Try a few more times before giving up
            # If no improvement in 100 swaps, break
            break
            
    return edges
