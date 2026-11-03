"""Program for matrix multiplication 2x3 by 3x5 using tensor code / Berlekamp-Massey approach."""
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Dimensions of factor matrices
    U_dim = n * m  # 6
    V_dim = m * p  # 15
    W_dim = p * n  # 10

    # We decompose M_{2,3,5} into M_{2,2,5} and M_{2,1,5}
    # M_{2,1,5} has rank 1 (outer product)
    # M_{2,2,5} is decomposed into M_{2,2,2} (Strassen, rank 7) and M_{2,2,3} (rank 11)
    # M_{2,2,3} is decomposed into M_{2,2,2} (Strassen, rank 7) and M_{2,2,1} (schoolbook, rank 4)
    # Total rank = 7 + 7 + 4 + 1 = 19

    # Let's build the decomposition step by step.
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)

    # Strategy: 
    # Split j into j1 in {0,1} and j2 in {2}
    # For j1 in {0,1}, split k into k1 in {0,1} and k2 in {2,3,4}
    # For j2 = 2, it's a rank 1 component

    # Component 1: j in {0,1}, k in {0,1} -> M_{2,2,2} Strassen (rank 7)
    # Component 2: j in {0,1}, k in {2,3,4} -> M_{2,2,3} (rank 11)
    #   - Component 2a: k in {2,3} -> M_{2,2,2} Strassen (rank 7)
    #   - Component 2b: k in {4} -> M_{2,2,1} schoolbook (rank 4)
    # Component 3: j in {2}, k in {0,1,2,3,4} -> M_{2,1,5} (rank 1)

    # Total: 7 + 7 + 4 + 1 = 19

    # Let's define the Strassen decomposition for M_{2,2,2}
    # For matrices A (2x2), B (2x2), C = A*B
    # Strassen uses 7 multiplications:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # In tensor form, for each multiplication Mc, we have:
    # U entries: linear combination of A entries
    # V entries: linear combination of B entries
    # W entries: linear combination of C entries (with signs)

    # Let's build the full decomposition.
    # We'll use a list of (U_entries, V_entries, W_entries) for each rank component.
    # U_entries: dict mapping (i,j) to coefficient
    # V_entries: dict mapping (j,k) to coefficient
    # W_entries: dict mapping (k,i) to coefficient

    components = []

    # Helper to add a Strassen component for a sub-problem
    # sub_i: list of i indices
    # sub_j: list of j indices
    # sub_k: list of k indices
    # offset_i, offset_j, offset_k: offsets in the full index space
    def add_strassen(sub_i, sub_j, sub_k, offset_i, offset_j, offset_k):
        # sub_i = [i0, i1], sub_j = [j0, j1], sub_k = [k0, k1]
        i0, i1 = sub_i
        j0, j1 = sub_j
        k0, k1 = sub_k
        
        # M1 = (A[i0,j0] + A[i1,j1])(B[j0,k0] + B[j1,k1])
        # C[i0,k0] += M1, C[i1,k1] += M1
        u1 = {(i0, j0): 1, (i1, j1): 1}
        v1 = {(j0, k0): 1, (j1, k1): 1}
        w1 = {(k0, i0): 1, (k1, i1): 1}
        components.append((u1, v1, w1))
        
        # M2 = (A[i1,j0] + A[i1,j1])B[j0,k0]
        # C[i1,k0] += M2
        u2 = {(i1, j0): 1, (i1, j1): 1}
        v2 = {(j0, k0): 1}
        w2 = {(k0, i1): 1}
        components.append((u2, v2, w2))
        
        # M3 = A[i0,j0](B[j0,k1] - B[j1,k1])
        # C[i0,k1] += M3
        u3 = {(i0, j0): 1}
        v3 = {(j0, k1): 1, (j1, k1): -1}
        w3 = {(k1, i0): 1}
        components.append((u3, v3, w3))
        
        # M4 = A[i1,j1](B[j1,k0] - B[j0,k0])
        # C[i1,k0] += M4
        u4 = {(i1, j1): 1}
        v4 = {(j1, k0): 1, (j0, k0): -1}
        w4 = {(k0, i1): 1}
        components.append((u4, v4, w4))
        
        # M5 = (A[i0,j0] + A[i0,j1])B[j1,k1]
        # C[i0,k0] -= M5, C[i0,k1] += M5
        u5 = {(i0, j0): 1, (i0, j1): 1}
        v5 = {(j1, k1): 1}
        w5 = {(k0, i0): -1, (k1, i0): 1}
        components.append((u5, v5, w5))
        
        # M6 = (A[i1,j0] - A[i0,j0])(B[j0,k0] + B[j0,k1])
        # C[i1,k1] += M6
        u6 = {(i1, j0): 1, (i0, j0): -1}
        v6 = {(j0, k0): 1, (j0, k1): 1}
        w6 = {(k1, i1): 1}
        components.append((u6, v6, w6))
        
        # M7 = (A[i0,j1] - A[i1,j1])(B[j1,k0] + B[j1,k1])
        # C[i0,k0] += M7
        u7 = {(i0, j1): 1, (i1, j1): -1}
        v7 = {(j1, k0): 1, (j1, k1): 1}
        w7 = {(k0, i0): 1}
        components.append((u7, v7, w7))

    # Helper to add a schoolbook component for M_{2,2,1}
    # sub_i = [i0, i1], sub_j = [j0, j1], sub_k = [k0]
    def add_schoolbook_221(sub_i, sub_j, sub_k):
        i0, i1 = sub_i
        j0, j1 = sub_j
        k0 = sub_k[0]
        
        # C[i0,k0] = A[i0,j0]*B[j0,k0] + A[i0,j1]*B[j1,k0]
        # C[i1,k0] = A[i1,j0]*B[j0,k0] + A[i1,j1]*B[j1,k0]
        
        # Term 1: A[i0,j0]*B[j0,k0] -> C[i0,k0]
        u = {(i0, j0): 1}
        v = {(j0, k0): 1}
        w = {(k0, i0): 1}
        components.append((u, v, w))
        
        # Term 2: A[i0,j1]*B[j1,k0] -> C[i0,k0]
        u = {(i0, j1): 1}
        v = {(j1, k0): 1}
        w = {(k0, i0): 1}
        components.append((u, v, w))
        
        # Term 3: A[i1,j0]*B[j0,k0] -> C[i1,k0]
        u = {(i1, j0): 1}
        v = {(j0, k0): 1}
        w = {(k0, i1): 1}
        components.append((u, v, w))
        
        # Term 4: A[i1,j1]*B[j1,k0] -> C[i1,k0]
        u = {(i1, j1): 1}
        v = {(j1, k0): 1}
        w = {(k0, i1): 1}
        components.append((u, v, w))

    # Helper to add a rank 1 component for M_{2,1,p}
    # sub_i = [i0, i1], sub_j = [j0], sub_k = [k0, ..., k_{p-1}]
    def add_rank1_21p(sub_i, sub_j, sub_k):
        i0, i1 = sub_i
        j0 = sub_j[0]
        
        # C[i,k] = A[i,j0]*B[j0,k] for all i, k
        # This is rank 1: u_i = 1 for all i, v_k = 1 for all k, w_{k,i} = 1 for all k,i
        # Wait, T[i,j0,k] = 1. 
        # We need sum_c U[i,j0,c] * V[j0,k,c] * W[k,i,c] = 1
        # If we set U[i,j0,0] = 1, V[j0,k,0] = 1, W[k,i,0] = 1, then product = 1.
        u = {}
        for i in sub_i:
            u[(i, j0)] = 1
        v = {}
        for k in sub_k:
            v[(j0, k)] = 1
        w = {}
        for k in sub_k:
            for i in sub_i:
                w[(k, i)] = 1
        components.append((u, v, w))

    # Now build the components:
    # 1. j in {0,1}, k in {0,1}: Strassen
    add_strassen([0, 1], [0, 1], [0, 1], 0, 0, 0)
    
    # 2. j in {0,1}, k in {2,3}: Strassen
    add_strassen([0, 1], [0, 1], [2, 3], 0, 0, 0)
    
    # 3. j in {0,1}, k in {4}: Schoolbook 2x2x1
    add_schoolbook_221([0, 1], [0, 1], [4])
    
    # 4. j in {2}, k in {0,1,2,3,4}: Rank 1
    add_rank1_21p([0, 1], [2], [0, 1, 2, 3, 4])

    # Total components: 7 + 7 + 4 + 1 = 19
    rank = len(components)
    
    # Now build the U, V, W matrices
    U = [[0] * rank for _ in range(U_dim)]
    V = [[0] * rank for _ in range(V_dim)]
    W = [[0] * rank for _ in range(W_dim)]
    
    for c, (u_entries, v_entries, w_entries) in enumerate(components):
        for (i, j), coeff in u_entries.items():
            U[i * m + j][c] = coeff
        for (j, k), coeff in v_entries.items():
            V[j * p + k][c] = coeff
        for (k, i), coeff in w_entries.items():
            W[k * n + i][c] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
