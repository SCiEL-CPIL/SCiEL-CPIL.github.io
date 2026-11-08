import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Rank 28 solution from seed, slightly optimized for speed
    def make_rank28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        # Part 1: A2 B2 (j=2, k=0..4)
        for i in range(n):
            for k in range(p):
                U[i * m + 2][r] = 1
                V[2 * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
        # Part 2: A1 B13 (j=0,1, k=4)
        for i in range(n):
            for j in range(2):
                U[i * m + j][r] = 1
                V[j * p + 4][r] = 1
                W[4 * n + i][r] = 1
                r += 1
        # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen
        U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 0][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = 1; W[1 * n + 1][r] = 1; r += 1
        U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 0][r] = 1; W[0 * n + 1][r] = 1; W[1 * n + 1][r] = -1; r += 1
        U[0 * m + 0][r] = 1; V[0 * p + 1][r] = 1; V[1 * p + 1][r] = -1; W[1 * n + 0][r] = 1; W[1 * n + 1][r] = 1; r += 1
        U[1 * m + 1][r] = 1; V[1 * p + 0][r] = 1; V[0 * p + 0][r] = -1; W[0 * n + 0][r] = 1; W[0 * n + 1][r] = 1; r += 1
        U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = -1; W[1 * n + 0][r] = 1; r += 1
        U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1; V[0 * p + 0][r] = 1; V[0 * p + 1][r] = 1; W[1 * n + 1][r] = 1; r += 1
        U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1; V[1 * p + 0][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = 1; r += 1
        # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen
        U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 2][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = 1; W[3 * n + 1][r] = 1; r += 1
        U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 2][r] = 1; W[2 * n + 1][r] = 1; W[3 * n + 1][r] = -1; r += 1
        U[0 * m + 0][r] = 1; V[0 * p + 3][r] = 1; V[1 * p + 3][r] = -1; W[3 * n + 0][r] = 1; W[3 * n + 1][r] = 1; r += 1
        U[1 * m + 1][r] = 1; V[1 * p + 2][r] = 1; V[0 * p + 2][r] = -1; W[2 * n + 0][r] = 1; W[2 * n + 1][r] = 1; r += 1
        U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = -1; W[3 * n + 0][r] = 1; r += 1
        U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1; V[0 * p + 2][r] = 1; V[0 * p + 3][r] = 1; W[3 * n + 1][r] = 1; r += 1
        U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1; V[1 * p + 2][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = 1; r += 1
        
        return {"U": U, "V": V, "W": W}

    def validate(U, V, W, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Vectorized validation
        # We need to check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        # This is a tensor contraction.
        # Let's construct the target tensor T (size n x m x p) filled with 1s? No, T[i,j,k] is 1 if the indices correspond to the multiplication.
        # The multiplication tensor T_{ijk} is 1 if we are computing C_{ik} from A_{ij} and B_{jk}.
        # Actually, the tensor to decompose is T_{ijk} where C_{ik} = sum_j A_{ij} B_{jk}.
        # The term in the sum is A_{ij} B_{jk}.
        # So the tensor T is defined by T_{ijk} = 1 for all valid i,j,k?
        # No, the tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k such that C_{ik} = sum_j (A \otimes B) ...
        # The standard definition: The multiplication map is a bilinear map A x B -> C.
        # The tensor is T = sum_{i,j,k} e_i^* \otimes e_j^* \otimes e_k ... wait.
        # The evaluator checks: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1.
        # This implies the target tensor T_{ijk} is 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
        # Wait, is that right?
        # If T_{ijk} = 1 for all i,j,k, then we are decomposing the all-ones tensor.
        # But matrix multiplication is not the all-ones tensor.
        # Let's re-read the task: "exactly decompose the matrix multiplication tensor".
        # The matrix multiplication tensor for n x m and m x p has non-zero entries T_{ijk} = 1 if the indices (i,j,k) form a valid path in the multiplication.
        # Actually, for a fixed j, the term is A_{ij} B_{jk}.
        # So the tensor is sum_j (sum_i e_i \otimes e_j \otimes sum_k e_k ...).
        # The entry T_{ijk} is 1 if it contributes to C_{ik}.
        # Every (i,j,k) triple contributes exactly 1 to the coefficient of A_{ij} B_{jk} in C_{ik}.
        # So yes, T_{ijk} = 1 for all i, j, k.
        
        # However, the indices in U, V, W are flattened.
        # U[i*m + j, r]
        # V[j*p + k, r]
        # W[k*n + i, r]
        
        # We can check this by constructing the product tensor from U, V, W and comparing to ones.
        # Product tensor P_{ijk} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We can use np.einsum or reshape/multiply.
        
        # Reshape U, V, W to 3D? No, they are 2D.
        # Let's compute P_{ijk} for all i,j,k.
        # Loop is small (2*3*5 = 30).
        # We can vectorize over r.
        
        # U_flat: (n*m, rank)
        # V_flat: (m*p, rank)
        # W_flat: (p*n, rank)
        
        # We want to compute S[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's do it with a loop over j, as that links the indices.
        valid = True
        for j in range(m):
            # Indices for U where column index is j: i*m + j for i in 0..n-1
            idx_u = np.array([i * m + j for i in range(n)])
            # Indices for V where row index is j: j*p + k for k in 0..p-1
            idx_v = np.array([j * p + k for k in range(p)])
            # Indices for W where column index is i? No, W index is k*n + i.
            # For a fixed j, we have i in 0..n-1, k in 0..p-1.
            # We need W[k*n + i, r].
            
            # Let's build the matrices for this j.
            # U_j: (n, rank)
            # V_j: (p, rank)
            # W_j: (p*n, rank) -> we need to extract W[k*n + i]
            
            U_j = U[idx_u, :] # (n, rank)
            V_j = V[idx_v, :] # (p, rank)
            
            # Construct W_j matrix of shape (n, p, rank) where W_j[i, k, r] = W[k*n + i, r]
            # Index in W: k*n + i
            # Create index array for W
            w_indices = []
            for k in range(p):
                for i in range(n):
                    w_indices.append(k * n + i)
            w_indices = np.array(w_indices)
            # This gives a flat list of length p*n in order k, then i.
            # We want a matrix (n, p) of indices?
            # Let's just create the full index array for all i, k.
            # i is 0..n-1, k is 0..p-1
            # We want W_idx[i, k] = k*n + i
            # I = np.arange(n)[:, np.newaxis]
            # K = np.arange(p)[np.newaxis, :]
            # W_idx = K * n + I  # shape (n, p)
            
            I = np.arange(n)
            K = np.arange(p)
            # Meshgrid to get all combinations
            II, KK = np.meshgrid(I, K, indexing='ij') # II is (n, p), KK is (n, p)
            W_flat_indices = KK * n + II # (n, p)
            W_j = W[W_flat_indices.ravel(), :].reshape(n, p, rank)
            
            # Now compute S[i, k] = sum_r U_j[i, r] * V_j[k, r] * W_j[i, k, r]
            # U_j: (n, rank)
            # V_j: (p, rank)
            # W_j: (n, p, rank)
            
            # S[i, k] = sum_r (U_j[i, r] * W_j[i, k, r]) * V_j[k, r]
            # Let M[i, k, r] = U_j[i, r] * W_j[i, k, r]
            M = U_j[:, np.newaxis, :] * W_j # (n, 1, rank) * (n, p, rank) -> (n, p, rank)
            
            # Now dot with V_j along r
            # V_j is (p, rank). We need to transpose it to (rank, p) or just use einsum
            # S = np.einsum('ikr,kr->ik', M, V_j)
            S = np.einsum('ikr,kr->ik', M, V_j)
            
            # Check if S is all ones
            if not np.allclose(S, 1.0, atol=1e-9):
                return False
                
        return True

    best = make_rank28()
    best_rank = 28
    
    # Try to find lower rank
    # Use integer values primarily, maybe some half-integers if needed
    # Given the structure, integer solutions are more likely to be stable
    vals = [-2, -1, 0, 1, 2]
    
    # Pre-compute indices for speed?
    # The validation function is now vectorized over r, so it should be fast.
    
    for rank in range(27, best_rank):
        if time.time() > deadline:
            break
        found = False
        # Limit attempts to avoid timeout if rank 27 is impossible
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            U = np.random.choice(vals, size=(n*m, rank))
            V = np.random.choice(vals, size=(m*p, rank))
            W = np.random.choice(vals, size=(p*n, rank))
            if validate(U, V, W, rank):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                found = True
                break
        if found:
            break
            
    return best

# EVOLVE-BLOCK-END
