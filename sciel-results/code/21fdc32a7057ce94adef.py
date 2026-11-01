import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor M_{2,4,4}.
    Uses a vectorized NumPy approach to generate the schoolbook basis,
    structurally distinct from loop-based implementations.
    """
    n, m, p = DIMENSIONS
    
    # The rank of the schoolbook decomposition is n*m*p
    rank = n * m * p
    
    # Create index arrays for the tensor indices i, j, k
    # i ranges from 0 to n-1
    # j ranges from 0 to m-1
    # k ranges from 0 to p-1
    
    # We need to create matrices U, V, W of shape (dim, rank)
    # U has dimensions (n*m, rank)
    # V has dimensions (m*p, rank)
    # W has dimensions (p*n, rank)
    
    # The schoolbook decomposition corresponds to the basis tensors:
    # T_c = e_i e_j e_k where c = i*m + j? No, the index mapping is usually:
    # U row index: i*m + j
    # V row index: j*p + k
    # W row index: k*n + i
    # Tensor term: U[row_u, c] * V[row_v, c] * W[row_w, c] = 1
    
    # Let's generate the column indices for each basis element
    # There are n*m*p basis elements.
    # We can create the matrices by iterating over the flattened index 'c'
    # But to be "vectorized" and structurally different, let's build them via broadcasting.
    
    # Actually, constructing sparse-ish dense matrices via NumPy loops is still loops.
    # Let's use a different structural approach: 
    # Create a 3D array of shape (n, m, p) representing the tensor T[i,j,k] = 1.
    # Then we "unfold" this tensor into the Khatri-Rao product form.
    
    # T is not a tensor in the Khatri-Rao sense directly, but the matrix multiplication tensor
    # is defined by T[i,j,k] such that C = A*B implies C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor entries are T[i, j, k] = delta_{i,?} no, the standard definition is:
    # The bilinear map is (A, B) -> A*B.
    # The tensor is T in V^* \otimes W^* \otimes Z^*.
    # T(i, j, k) = 1 if the product A_{ij} B_{jk} contributes to C_{ik}.
    # Wait, the standard decomposition basis vectors are:
    # U_c has a 1 at (i, j) corresponding to the term A_{ij}
    # V_c has a 1 at (j, k) corresponding to the term B_{jk}
    # W_c has a 1 at (k, i) corresponding to the term C_{ki} (note index order in W)
    
    # So for each triple (i, j, k), we have a rank-1 term.
    # Total rank = n * m * p.
    
    # To make the code structurally different, we will construct the columns
    # using NumPy's advanced indexing and flattening.
    
    # Generate all combinations of i, j, k
    i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Calculate row indices for U, V, W
    # U rows are indexed by i, j -> i * m + j
    # V rows are indexed by j, k -> j * p + k
    # W rows are indexed by k, i -> k * n + i
    
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Use NumPy's advanced indexing to set the values
    # We want U[u_rows[c], c] = 1 for all c
    # This is equivalent to:
    # U[u_rows, np.arange(rank)] = 1
    U[u_rows, np.arange(rank)] = 1.0
    V[v_rows, np.arange(rank)] = 1.0
    W[w_rows, np.arange(rank)] = 1.0
    
    # Convert to lists of lists as required by the interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
