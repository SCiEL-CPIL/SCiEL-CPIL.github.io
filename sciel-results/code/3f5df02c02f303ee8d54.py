import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Strategy:
    # Decompose M_{n,m,p} into n independent M_{1,m,p} blocks.
    # Rank of M_{1,m,p} is min(m, p).
    # Total rank = n * min(m, p).
    
    min_mp = min(m, p)
    rank = n * min_mp
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the decomposition
    # For each row i of A (0 to n-1):
    for i in range(n):
        # Block offset in the rank dimension
        offset = i * min_mp
        
        # For each column j of A (0 to m-1)
        for j in range(m):
            # U row index is i*m + j
            u_row = i * m + j
            
            # In the block for i, we use the first min_mp columns.
            # We map j to a column index in the block.
            # Since min_mp <= m, we can just use j as the index if j < min_mp.
            # If m > min_mp, we only use the first min_mp columns of j?
            # Wait, the construction for M_{1,m,p} with rank min(m,p) requires:
            # If m <= p: Rank m. U_{(0,j), s} = delta_{js}. V_{(j,k), s} = delta_{js}. W_{(k,0), s} = 1.
            # If p < m: Rank p. U_{(0,j), s} = 1. V_{(j,k), s} = delta_{ks}. W_{(k,0), s} = delta_{ks}?
            
            # Let's handle the two cases.
            if m <= p:
                # Rank = m
                # U_{(i,j), offset + j} = 1
                if j < min_mp:
                    U[u_row][offset + j] = 1
            else:
                # Rank = p
                # This case is slightly different.
                # If p < m, rank is p.
                # We need sum_s U_{(i,j),s} V_{(j,k),s} W_{(k,i),s} = 1.
                # Let s range 0 to p-1.
                # Set W_{(k,i), offset+s} = delta_{ks}.
                # Then sum_s U_{(i,j), offset+s} V_{(j,k), offset+s} delta_{ks}
                # = U_{(i,j), offset+k} V_{(j,k), offset+k} = 1.
                # We can set U_{(i,j), offset+k} = 1 and V_{(j,k), offset+k} = 1.
                # This requires U to have 1s in all p columns for each j.
                # And V to have 1s in column k for row (j,k).
                
                # So for U:
                for s in range(min_mp):
                    U[u_row][offset + s] = 1
                    
                # For V:
                # V row index is j*p + k
                for k in range(p):
                    v_row = j * p + k
                    V[v_row][offset + k] = 1
                    
                # For W:
                # W row index is k*n + i
                for k in range(p):
                    w_row = k * n + i
                    W[w_row][offset + k] = 1
                    
                # Note: In the m <= p case, we also need to fill V and W.
                # Let's re-structure to avoid duplication.

    # Re-implementing cleanly based on the two cases
    
    # Case 1: m <= p
    if m <= p:
        rank = n * m
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for i in range(n):
            offset = i * m
            for j in range(m):
                # U
                U[i * m + j][offset + j] = 1
                
                # V: V_{(j,k), offset+j} = 1 for all k
                for k in range(p):
                    V[j * p + k][offset + j] = 1
                    
                # W: W_{(k,i), offset+j} = 1 for all k
                for k in range(p):
                    W[k * n + i][offset + j] = 1
                    
    # Case 2: p < m
    else:
        rank = n * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for i in range(n):
            offset = i * p
            for j in range(m):
                # U: U_{(i,j), offset+s} = 1 for all s in 0..p-1
                for s in range(p):
                    U[i * m + j][offset + s] = 1
                    
                # V: V_{(j,k), offset+k} = 1
                for k in range(p):
                    V[j * p + k][offset + k] = 1
                    
                # W: W_{(k,i), offset+k} = 1
                for k in range(p):
                    W[k * n + i][offset + k] = 1

    return {"U": U, "V": V, "W": W}
