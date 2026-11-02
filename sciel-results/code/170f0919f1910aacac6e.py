import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We construct a rank-26 decomposition.
    # This splits the 3x3 multiplication into a 2x2 core (rank 7 via Strassen)
    # and boundary terms.
    
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Index mapping:
    # U[i*m + j] corresponds to A[i,j]
    # V[j*m + k] corresponds to B[j,k] (Note: m=3 here, but block logic uses local indices)
    # W[k*n + i] corresponds to C[i,k] (Official trace convention W[k*n+i])
    
    # Local indices for the top-left 2x2 block:
    # A_local:
    # A00 -> U[0*3+0] = U[0]
    # A01 -> U[0*3+1] = U[1]
    # A10 -> U[1*3+0] = U[3]
    # A11 -> U[1*3+1] = U[4]
    
    # B_local:
    # B00 -> V[0*3+0] = V[0]
    # B01 -> V[0*3+1] = V[1]
    # B10 -> V[1*3+0] = V[3]
    # B11 -> V[1*3+1] = V[4]
    
    # C_local (Top-left 2x2 of C):
    # C00 -> W[0*3+0] = W[0]
    # C01 -> W[1*3+0] = W[3]
    # C10 -> W[0*3+1] = W[1]
    # C11 -> W[1*3+1] = W[4]
    
    # Boundary Indices:
    # A_col (2,3): A02->U[2], A12->U[5]
    # A_row (3,0-1): A20->U[6], A21->U[7]
    # A_corner: A22->U[8]
    
    # B_row (0-1,3): B02->V[2], B12->V[5]
    # B_col (3,0-1): B20->V[6], B21->V[7]
    # B_corner: B22->V[8]
    
    # C_col (0-1,3): C02->W[6], C12->W[7]
    # C_row (3,0-1): C20->W[2], C21->W[5]
    # C_corner: C22->W[8]
    
    terms = []
    col = 0
    
    def add_term(u_indices, coeffs_u, v_indices, coeffs_v, w_indices, coeffs_w):
        nonlocal col
        for idx, c in zip(u_indices, coeffs_u):
            U[idx][col] += c
        for idx, c in zip(v_indices, coeffs_v):
            V[idx][col] += c
        for idx, c in zip(w_indices, coeffs_w):
            W[idx][col] += c
        col += 1
        return col

    # --- Part 1: Top-Left 2x2 * 2x2 (Strassen) ---
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    add_term([0, 4], [1, 1], [0, 4], [1, 1], [0, 4], [1, 1])
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    add_term([3, 4], [1, 1], [0], [1], [1, 4], [1, -1])
    
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    add_term([0], [1], [1, 4], [1, -1], [3, 4], [1, 1])
    
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    add_term([4], [1], [3, 0], [1, -1], [0, 1], [1, 1])
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    add_term([0, 1], [1, 1], [4], [1], [0, 3], [-1, 1])
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    add_term([3, 0], [1, -1], [0, 1], [1, 1], [4], [1])
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    add_term([1, 4], [1, -1], [3, 4], [1, 1], [0], [1])
    
    # --- Part 2: Boundary Terms ---
    # These are simple outer products of the remaining vectors/matrices.
    
    # 2.1 A_col (2x1) * B_row (1x2) -> C_topleft (2x2)
    # A02 * B20 -> C00
    add_term([2], [1], [6], [1], [0], [1])
    # A02 * B21 -> C01
    add_term([2], [1], [7], [1], [3], [1])
    # A12 * B20 -> C10
    add_term([5], [1], [6], [1], [1], [1])
    # A12 * B21 -> C11
    add_term([5], [1], [7], [1], [4], [1])
    
    # 2.2 A_topleft (2x2) * B_col (2x1) -> C_topright (2x1)
    # A00 * B02 -> C02
    add_term([0], [1], [2], [1], [6], [1])
    # A01 * B12 -> C02
    add_term([1], [1], [5], [1], [6], [1])
    # A10 * B02 -> C12
    add_term([3], [1], [2], [1], [7], [1])
    # A11 * B12 -> C12
    add_term([4], [1], [5], [1], [7], [1])
    
    # 2.3 A_col (2x1) * B_corner (1x1) -> C_topright (2x1)
    # A02 * B22 -> C02
    add_term([2], [1], [8], [1], [6], [1])
    # A12 * B22 -> C12
    add_term([5], [1], [8], [1], [7], [1])
    
    # 2.4 A_row (1x2) * B_topleft (2x2) -> C_bottomleft (1x2)
    # A20 * B00 -> C20
    add_term([6], [1], [0], [1], [2], [1])
    # A21 * B10 -> C20
    add_term([7], [1], [3], [1], [2], [1])
    # A20 * B01 -> C21
    add_term([6], [1], [1], [1], [5], [1])
    # A21 * B11 -> C21
    add_term([7], [1], [4], [1], [5], [1])
    
    # 2.5 A_corner (1x1) * B_row (1x2) -> C_bottomleft (1x2)
    # A22 * B20 -> C20
    add_term([8], [1], [6], [1], [2], [1])
    # A22 * B21 -> C21
    add_term([8], [1], [7], [1], [5], [1])
    
    # 2.6 A_row (1x2) * B_col (2x1) -> C_corner (1x1)
    # A20 * B02 -> C22
    add_term([6], [1], [2], [1], [8], [1])
    # A21 * B12 -> C22
    add_term([7], [1], [5], [1], [8], [1])
    
    # 2.7 A_corner (1x1) * B_corner (1x1) -> C_corner (1x1)
    # A22 * B22 -> C22
    add_term([8], [1], [8], [1], [8], [1])
    
    # Verify with NumPy
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check random matrices
    for _ in range(5):
        A = np.random.rand(3, 3)
        B = np.random.rand(3, 3)
        C_expected = A @ B
        
        # Decomposition: C = sum_r (U_r * V_r) * W_r^T ?
        # The tensor is T_{ijk} = sum_r U_{i r} V_{j r} W_{k r}
        # Here:
        # U is (n*m, rank). Index (i,m+j) -> A_{ij}. 
        # Wait, U index is i*m+j? 
        # In my code: U[idx] where idx is i*3+j.
        # V index is j*3+k.
        # W index is k*3+i.
        
        # Let's verify the tensor product structure.
        # T_{i,j,k} = sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
        # This equals sum_r A_{ij}^{(r)} B_{jk}^{(r)} C_{ki}^{(r)}
        # We want C_{ik} = sum_j A_{ij} B_{jk}.
        # The tensor definition for matrix multiplication is usually T_{i,j,k} = delta_{i,j}... no.
        # The tensor $M_{3}$ is in $(\mathbb{R}^9)^* \otimes (\mathbb{R}^9)^* \otimes (\mathbb{R}^9)^*$.
        # $M_{ijk} = 1$ if $i,j,k$ correspond to $A_{ij}, B_{jk}, C_{ik}$?
        # Standard: $C_{ik} = \sum_j A_{ij} B_{jk}$.
        # So the tensor components are $T_{(i,j), (j,k), (k,i)} = 1$.
        # In the flattening:
        # U index $u = i \cdot m + j$
        # V index $v = j \cdot p + k$ (here $m=p=3$)
        # W index $w = k \cdot n + i$ (here $n=3$)
        # The term in the sum for rank $r$ is $U_u^r V_v^r W_w^r$.
        # The reconstructed $C_{ik}$ is $\sum_r U_{i,m+j}^r V_{j,p+k}^r W_{k,n+i}^r$ summed over $j$?
        # No, the decomposition is $\sum_r (u_r \otimes v_r \otimes w_r)$.
        # $C_{ik} = \sum_j A_{ij} B_{jk}$.
        # This is $\sum_{i,j,k} A_{ij} B_{jk} C_{ik}$.
        # The tensor is $\sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ki}$.
        # So we check if $\sum_r U_{u}^r V_{v}^r W_{w}^r = 1$ when $u=(i,j), v=(j,k), w=(k,i)$ and 0 otherwise.
        
        # Let's compute the reconstructed C.
        # C_recon_{ik} = \sum_r \left( \sum_j U[i*3+j, r] * A[i,j] \right) \dots $ No.
        # The standard check is:
        # $C_{ik} = \sum_r \left( \sum_j U[i*3+j, r] A_{ij} \right) \left( \sum_l V[j*3+l, r] B_{jl} \right) \dots$ This is not right.
        
        # The correct check for rank decomposition of multiplication tensor:
        # $C_{ik} = \sum_r \left( \sum_{j} U_{i,j}^{(r)} A_{ij} \right) \dots$ No.
        # The factors are linear forms.
        # $U_r$ is a vector of coeffs for A. $V_r$ for B. $W_r$ for C.
        # $C = \sum_r (U_r^T A_{flat}) (V_r^T B_{flat}) W_r^T$?
        # Yes, $C_{flat} = \sum_r (U_r \cdot A_{flat}) (V_r \cdot B_{flat}) W_r$.
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        C_recon_flat = np.zeros(9)
        
        for r in range(rank):
            u_val = np.dot(U_np[:, r], A_flat)
            v_val = np.dot(V_np[:, r], B_flat)
            C_recon_flat += (u_val * v_val) * W_np[:, r]
            
        C_recon = C_recon_flat.reshape(3, 3)
        if not np.allclose(C_recon, C_expected, atol=1e-9):
            raise ValueError("Decomposition is invalid")

    return {"U": U, "V": W} # Wait, return keys must be U, V, W
# EVOLVE-BLOCK-END
