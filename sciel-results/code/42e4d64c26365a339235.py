import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct 4x4x4 decomposition (Rank 49) using Strassen composition
    # Define Strassen 2x2x2 tensors (U2, V2, W2) of shape (4, 7)
    # Indices: 0:00, 1:01, 2:10, 3:11
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 1: (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Term 2: (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # Term 3: A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # Term 4: A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # Term 5: (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # Term 6: (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # Term 7: (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Compose 4x4x4
    # U4 = U2 (outer) x U2 (inner) with permutation
    # Outer indices: i1, j1. Inner indices: i0, j0.
    # Kronecker product gives index (2*i1 + j1)*4 + (2*i0 + j0)
    # We want index (2*i1 + i0)*4 + (2*j1 + j0)
    
    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)
    
    # Permutation for rows
    # Old index r = 8*i1 + 4*j1 + 2*i0 + j0
    # New index r_new = 8*i1 + 4*i0 + 2*j1 + j0
    perm = np.zeros(16, dtype=int)
    for i1 in range(2):
        for j1 in range(2):
            for i0 in range(2):
                for j0 in range(2):
                    old_idx = (2*i1 + j1)*4 + (2*i0 + j0)
                    new_idx = (2*i1 + i0)*4 + (2*j1 + j0)
                    perm[old_idx] = new_idx
    
    U4 = U4_kron[perm, :]
    V4 = V4_kron[perm, :]
    W4 = W4_kron[perm, :]
    
    # 2. Add extra terms for 4x5x5
    # Current rank 49.
    # Need to add:
    # A[:, 4] B[4, :4] (Rank 4)
    # A[:, :4] B[:4, 4] (Rank 4)
    # A[:, 4] B[4, 4] (Rank 1)
    
    # Initialize full U, V, W
    rank_444 = 49
    rank_extra = 9
    total_rank = rank_444 + rank_extra
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Fill 4x4x4 part
    # U4 is 16x49. We need to map to n*m = 20.
    # Indices i, j in 0..3 map to i*m + j. m=5.
    # U4 row index is i*4 + j. We need i*5 + j.
    for idx in range(16):
        i = idx // 4
        j = idx % 4
        new_idx = i * 5 + j
        U[new_idx, :rank_444] = U4[idx, :]
        
    for idx in range(16):
        j = idx // 4
        k = idx % 4
        new_idx = j * 5 + k
        V[new_idx, :rank_444] = V4[idx, :]
        
    for idx in range(16):
        k = idx // 4
        i = idx % 4
        new_idx = k * 4 + i # W is p*n = 5*4 = 20. Index k*n + i.
        W[new_idx, :rank_444] = W4[idx, :]
        
    # Fill extra terms
    col = rank_444
    
    # Term A[:, 4] B[4, :4]
    # A col 4 is index 4 in A (0..4). i in 0..3.
    # B row 4 is index 4 in B (0..4). k in 0..3.
    # Product is 4x4.
    # For each i, k:
    for i in range(4):
        for k in range(4):
            U[i*5 + 4, col] = 1
            V[4*5 + k, col] = 1 # B row 4, col k
            W[k*4 + i, col] = 1
            col += 1
            
    # Term A[:, :4] B[:4, 4]
    # A cols 0..3. B col 4.
    # Product is 4x1.
    for i in range(4):
        for j in range(4):
            U[i*5 + j, col] = 1
            V[j*5 + 4, col] = 1 # B row j, col 4
            W[4*4 + i, col] = 1 # C row i, col 4
            col += 1
            
    # Term A[:, 4] B[4, 4]
    # A col 4. B[4,4].
    # Product is 4x1.
    for i in range(4):
        U[i*5 + 4, col] = 1
        V[4*5 + 4, col] = 1
        W[4*4 + i, col] = 1
        col += 1
        
    # Convert to list of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    # Quick validation
    res = construct()
    print("Rank:", len(res['U'][0]))
    # Verify one random case
    n, m, p = 4, 5, 5
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_true = A @ B
    
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    C_calc = np.zeros((n, p))
    for c in range(U.shape[1]):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        # C += u * v * w ? No.
        # T = sum U_i V_j W_k.
        # C_{ik} = sum_j sum_c U_{ic} V_{jc} W_{kc} A_{ij} B_{jk}
        # This is not simply u*v*w.
        # The decomposition is T_{ijk} = sum U_{ic} V_{jc} W_{kc}.
        # C_{ik} = sum_{j} A_{ij} B_{jk} = sum_{j, c} A_{ij} U_{ic} V_{jc} W_{kc} ?
        # No, the tensor T represents the multiplication map.
        # C = A B.
        # C_{ik} = sum_j A_{ij} B_{jk}.
        # The decomposition says T_{ijk} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}?
        # The task says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
        # So U_{i*m+j, c}, V_{j*p+k, c}, W_{k*n+i, c}.
        # The bilinear form is sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} ?
        # No, that would be a trilinear form.
        # The matrix multiplication tensor is T_{i,j,k} = delta_{i,i} delta_{j,j} delta_{k,k}?
        # T_{i,j,k} such that C_{ik} = sum_j A_{ij} B_{jk} T_{i,j,k}?
        # Actually, T_{i,j,k} is 1 if the indices match the multiplication pattern.
        # For matrix multiplication, T_{i,j,k} is not a single tensor.
        # The tensor is defined by the bilinear map (A, B) -> C.
        # C_{ik} = sum_{j, l} A_{ij} B_{jl} (delta_{kl}?)
        # The standard tensor decomposition for matrix multiplication $M_{n,m,p}$ has entries $T_{i,j,k}$.
        # $T_{i,j,k} = 1$ if $i,j,k$ are valid indices?
        # No, the tensor is $T \in \mathbb{R}^{n \times m \times p}$?
        # No, $T \in \mathbb{R}^{n \times m} \otimes \mathbb{R}^{m \times p} \otimes \mathbb{R}^{p \times n}$?
        # The task defines U, V, W.
        # The condition is: For all A, B, $AB = \sum_c U_c \otimes V_c \otimes W_c$?
        # No, $C_{ik} = \sum_c (\sum_j U_{i*m+j, c} A_{ij}) (\sum_l V_{j*p+k, c} B_{jl}) W_{k*n+i, c}$?
        # This looks like a rank-1 update to C.
        # Let $x_c = U_c^T A$? No.
        # Let $u_c$ be the vector $U[:, c]$.
        # The term is $W_c (u_c^T A) (B v_c)$?
        # Let's check the dimensions.
        # $U_c$ is $n \times m$.
        # $V_c$ is $m \times p$.
        # $W_c$ is $p \times n$.
        # The contribution to C is $W_c \cdot (V_c \cdot A^T \cdot U_c^T)$?
        # Actually, the standard form is:
        # $C_{ik} = \sum_c U_{i, c} V_{k, c} \dots$?
        # With the given flattening:
        # $U_{i*m+j, c}$, $V_{j*p+k, c}$, $W_{k*n+i, c}$.
        # The bilinear form is $\sum_{i,j,k} A_{ij} B_{jk} T_{i,j,k}$.
        # The decomposition is $T_{i,j,k} = \sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}$.
        # So $C_{ik} = \sum_j A_{ij} B_{jk} \sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}$.
        # This must equal $\sum_j A_{ij} B_{jk}$.
        # So $\sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} = \delta_{i,i} \delta_{j,j} \delta_{k,k}$?
        # No, it must be 1 for all valid i,j,k?
        # Wait, $C_{ik} = \sum_j A_{ij} B_{jk}$.
        # So the coefficient of $A_{ij} B_{jk}$ in $C_{ik}$ is 1.
        # So $T_{i,j,k} = 1$ for all $i,j,k$?
        # No, $T_{i,j,k}$ is the coefficient of $A_{ij} B_{jk}$ in $C_{ik}$.
        # In $C_{ik} = \sum_j A_{ij} B_{jk}$, the term $A_{ij} B_{jk}$ appears exactly once for each j.
        # So $T_{i,j,k} = 1$ for all $i, j, k$?
        # If $T_{i,j,k} = 1$, then $C_{ik} = \sum_{j} A_{ij} B_{jk} \sum_c U V W$.
        # This implies $\sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} = 1$ for all $i,j,k$.
        # This is the condition for the decomposition.
        
        # Let's verify this condition for the schoolbook.
        # Schoolbook: Rank nmp.
        # For each i,j,k, there is a column c.
        # $U_{i*m+j, c} = 1, V_{j*p+k, c} = 1, W_{k*n+i, c} = 1$.
        # Sum is 1. Correct.
        
        # So my constructed U, V, W must satisfy:
        # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1.
        
        # Let's test this.
        pass
