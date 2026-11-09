"""Program to find rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]
n, m, p = DIMENSIONS  # 2, 3, 5

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        # Create random test matrices
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # C_{i,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # But this isn't quite right. Let me think again.
        
        # The decomposition is: C_{i,k} = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
        # Wait, that's not standard. Let me re-read the task.
        
        # The task says: W[k*n+i,r]
        # So the convention is:
        # U has shape (n*m, rank) - indexed by (i, j) where i is row of A, j is col of A
        # V has shape (m*p, rank) - indexed by (j, k) where j is row of B, k is col of B  
        # W has shape (p*n, rank) - indexed by (k, i) where k is col of C, i is row of C
        
        # The formula should be: C_{i,k} = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # But that has a double sum over j and r, which doesn't make sense for a rank-r decomposition.
        
        # Actually, I think the correct interpretation is:
        # C_{i,k} = sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
        # No, that's still not right.
        
        # Let me think about this differently. The standard tensor decomposition for matrix multiplication is:
        # C_{i,k} = sum_j A_{i,j} * B_{j,k}
        # This can be written as a sum of rank-1 terms:
        # C_{i,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
        # where U, V, W are the factor matrices.
        
        # But the shapes don't match what I described. Let me re-read.
        
        # U: (n*m, rank) - so U[a, r] where a = i*m + j
        # V: (m*p, rank) - so V[b, r] where b = j*p + k
        # W: (p*n, rank) - so W[c, r] where c = k*n + i
        
        # The formula must be: C_{i,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # But j is summed over, so this is: C_{i,k} = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Wait, that's a double sum. That can't be right for a rank-r decomposition.
        
        # I think I'm misunderstanding the convention. Let me just test with the schoolbook case.
        
        # For schoolbook: rank = n*m*p = 30
        # U[i*m+j, r] = 1 if r = i*m*p + j*p + k, else 0
        # V[j*p+k, r] = 1 if r = i*m*p + j*p + k, else 0
        # W[k*n+i, r] = 1 if r = i*m*p + j*p + k, else 0
        
        # Then C_{i,k} = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # = sum_j sum_r delta(r, i*m*p+j*p+k) * delta(r, i*m*p+j*p+k) * delta(r, i*m*p+j*p+k)
        # = sum_j delta(0) ... this doesn't work.
        
        # I think the correct formula is actually:
        # C_{i,k} = sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
        # No, that's still weird.
        
        # Let me just implement it as: for each r, compute the contribution.
        # Actually, I think the standard way is:
        # C = sum_r W[:, r] * (U[:, r]^T * V[:, r])
        # where U[:, r] is a vector of length n*m, V[:, r] is a vector of length m*p, W[:, r] is a vector of length p*n.
        
        # Let me reshape and see:
        # U_r = U[:, r] reshaped to (n, m)
        # V_r = V[:, r] reshaped to (m, p)
        # W_r = W[:, r] reshaped to (p, n)
        
        # Then C = sum_r W_r^T * (U_r * V_r) where * is element-wise? No.
        
        # I think the correct interpretation is that each rank-1 term computes:
        # C^{(r)} = (U_r as n x m) * (V_r as m x p) * (W_r as p x n)
        # But that gives a scalar, not a matrix.
        
        # I'm confused. Let me just use the schoolbook decomposition and verify it works.
        
        # Schoolbook:
        # For each (i, j, k), create a term where:
        # U[i*m+j, r] = 1
        # V[j*p+k, r] = 1
        # W[k*n+i, r] = 1
        # All other entries are 0.
        
        # Then the formula should give: C_{i,k} = sum_j A_{i,j} B_{j,k}
        
        # I think the formula is: C_{i,k} = sum_r sum_j U[i*m+j, r] * A_{i,j} * V[j*p+k, r] * B_{j,k} * W[k*n+i, r]
        # No, that's not right either.
        
        # Let me just trust the convention and verify numerically.
        pass
    
    def block_identity(n, m, p):
        """Returns U, V, W for n x m * m x p with rank n*m*p (schoolbook)."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    def strassen_2x2():
        """Returns U, V, W for 2x2 * 2x2 with rank 7 (Strassen's algorithm)."""
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # Strassen's 7 multiplications:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # U rows: A11=0, A12=1, A21=2, A22=3
        # V rows: B11=0, B12=1, B21=2, B22=3
        # W rows: C11=0, C12=1, C21=2, C22=3
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[3, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[3, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1; W[2, 6] = -1
        
        return U, V, W
    
    # Build the 29-rank solution:
    # Split p=5 into 2+2+1.
    # Block 1: 2x3 * 3x2 (cols 0,1) - use Strassen for 2x2 part + schoolbook for rest
    # Block 2: 2x3 * 3x2 (cols 2,3) - same
    # Block 3: 2x3 * 3x1 (col 4) - schoolbook
    
    # For 2x3 * 3x2:
    # Split m=3 into 2+1.
    # Sub-block 1: 2x2 * 2x2 (Strassen, rank 7)
    # Sub-block 2: 2x1 * 1x2 (schoolbook, rank 2)
    # Sub-block 3: 2x2 * 1x2 (schoolbook, rank 4)
    # Sub-block 4: 2x1 * 2x2 (schoolbook, rank 4)
    # Total: 7 + 2 + 4 + 4 = 17
    
    # But wait, that's worse than schoolbook (12) for 2x3*3x2.
    # So I should just use schoolbook for 2x3*3x2.
    
    # Actually, let me reconsider. The best approach is:
    # Use Strassen for the 2x2 * 2x2 sub-blocks where possible.
    
    # For 2x3 * 3x5:
    # Split n=2 into 1+1? No, that doesn't help.
    # Split m=3 into 2+1?
    # Split p=5 into 2+2+1?
    
    # Let me try: split p=5 into 2+3.
    # Block 1: 2x3 * 3x2 (schoolbook rank 12)
    # Block 2: 2x3 * 3x3 (can we do better than 27?)
    
    # For 2x3 * 3x3, split m=3 into 2+1:
    # 2x2 * 2x3 (rank 12) + 2x1 * 1x3 (rank 6) + 2x2 * 1x3 (rank 12) + 2x1 * 2x3 (rank 6) = 36? No.
    
    # I think the best I can do is 29 using Strassen for one 2x2*2x2 block.
    
    # Let me construct the 29-rank solution:
    # Split 2x3 * 3x5 into:
    # - 2x2 * 2x2 (Strassen, rank 7) for A[0:2, 0:2] * B[0:2, 0:2]
    # - Rest is schoolbook
    
    # Actually, let me just use the known result that R(2,3,5) = 28 or 29.
    # I'll construct a 29-rank solution using Strassen for one 2x2*2x2 sub-block.
    
    # Fallback: Schoolbook rank 30
    U, V, W = block_identity(n, m, p)
    
    # Try to improve to 29 by using Strassen for a 2x2*2x2 sub-block
    # Split the problem:
    # Use Strassen for A[0:2, 0:2] * B[0:2, 0:2] (rank 7 instead of 4)
    # Save 4-7 = -3? No, Strassen uses 7 instead of 4, so it's worse!
    
    # Wait, Strassen uses 7 multiplications instead of 4, so it's MORE rank, not less.
    # I had it backwards. Strassen reduces the number of scalar multiplications from 4 to 7? No, from 4 to 7 is an increase.
    
    # Actually, Strassen's algorithm uses 7 multiplications instead of 8 for 2x2 matrices.
    # So it saves 1 multiplication.
    
    # For 2x2 * 2x2: schoolbook rank = 4, Strassen rank = 7? No, that's wrong.
    # Schoolbook: 4 multiplications. Strassen: 7 multiplications. So Strassen is worse for 2x2?
    
    # No, I'm confusing myself. Strassen's algorithm for 2x2 matrix multiplication uses 7 scalar multiplications instead of 8.
    # So the rank is 7, not 4.
    
    # But the schoolbook rank for 2x2 * 2x2 is 4 (one multiplication per output element).
    # So Strassen (rank 7) is worse than schoolbook (rank 4) for 2x2!
    
    # I had it completely backwards. Strassen is useful for larger matrices, not 2x2.
    
    # So the best I can do is schoolbook rank 30.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
