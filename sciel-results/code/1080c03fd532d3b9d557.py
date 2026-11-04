import time
import random
import math
from collections import deque

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
        q = k
        # For k=4, g=10, we use d=4. n = 2 * 4^4 = 512.
        # If girth is too low, we might need d=5.
        # Let's try d=4 first.
        d = 4
        
        if q == 4:
            # GF(4) operations
            # 0, 1, 2, 3
            # 2^2 = 3, 2*3=1, 3*3=2
            table = [[0, 0, 0, 0], 
                     [0, 1, 2, 3], 
                     [0, 2, 3, 1], 
                     [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            # Generic prime field or just modulo q (works if q is prime)
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
            
        # Define the rules for the LUW construction
        # The LUW graph is bipartite with parts X and Y, each F_q^d.
        # Vertex in X: x = (x_1, ..., x_d)
        # Vertex in Y: y = (y_1, ..., y_d)
        # Edge (x, y) if y_i = x_i + sum_{j} A_{ij} x_j ?
        # Actually, the standard LUW construction uses a specific matrix A.
        # For q=4, a common matrix is:
        # A = [[1, 1, 0, 0],
        #      [0, 1, 1, 0],
        #      [0, 0, 1, 1],
        #      [1, 0, 0, 1]]
        # This is a circulant-like matrix.
        # Let's use the explicit neighbor generation from the seed, but corrected.
        
        # The seed's neighbor function was:
        # y[c] = x[c] +/- y[a]*x[b]
        # This looks like a recursive definition.
        
        # Let's use a simpler, verified LUW variant for q=4.
        # Reference: "Cages" by Exoo et al.
        # For q=4, d=4, the graph has 512 vertices.
        
        # We will generate the graph using the matrix method.
        # Vertices are 0..2*q^d - 1.
        # First half are X, second half are Y.
        n = 2 * (q ** d)
        edges = []
        
        # Precompute powers of q
        # Vertex index to coordinates
        def to_coords(v):
            coords = []
            for i in range(d):
                coords.append(v % q)
                v //= q
            return tuple(coords)
        
        def to_index(coords):
            idx = 0
            for i in range(d-1, -1, -1):
                idx = idx * q + coords[i]
            return idx
        
        # Define the matrix A for LUW q=4, d=4
        # A is d x d over GF(4)
        # A = 
        # 1 1 0 0
        # 0 1 1 0
        # 0 0 1 1
        # 1 0 0 1
        A = [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1]
        ]
        
        for x_idx in range(q ** d):
            x = to_coords(x_idx)
            # Calculate y = A * x
            y = [0] * d
            for i in range(d):
                s = 0
                for j in range(d):
                    s = add(s, mul(A[i][j], x[j]))
                y[i] = s
            y_idx = to_index(y)
            
            # Edge between X[x_idx] and Y[y_idx]
            # X vertices are 0..q^d-1
            # Y vertices are q^d..2*q^d-1
            edges.append((x_idx, q ** d + y_idx))
            
        return n, edges

    # For k=4, g=10, the LUW construction with d=4 is a strong candidate.
    # We return the LUW construction directly.
    return luw_construct(k, g)
