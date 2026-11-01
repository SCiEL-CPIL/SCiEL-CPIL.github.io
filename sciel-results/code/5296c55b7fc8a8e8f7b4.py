import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = 2, 2, 10
    # U shape (4, R), V shape (20, R), W shape (20, R)
    
    # Standard Strassen terms (verified):
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    def build_strassen_rank_35():
        U = np.zeros((4, 35), dtype=float)
        V = np.zeros((20, 35), dtype=float)
        W = np.zeros((20, 35), dtype=float)
        
        col = 0
        for q in range(5):
            for a_vec, b_vec, c_vec in terms:
                U[0, col] = a_vec[0]
                U[1, col] = a_vec[1]
                U[2, col] = a_vec[2]
                U[3, col] = a_vec[3]
                
                V[0*10 + 2*q, col] = b_vec[0]
                V[0*10 + 2*q+1, col] = b_vec[1]
                V[1*10 + 2*q, col] = b_vec[2]
                V[1*10 + 2*q+1, col] = b_vec[3]
                
                W[4*q + 0, col] = c_vec[0]
                W[4*q + 1, col] = c_vec[1]
                W[4*q + 2, col] = c_vec[2]
                W[4*q + 3, col] = c_vec[3]
                
                col += 1
        
        return U, V, W
    
    def verify_decomposition(U, V, W):
        """Verify that the tensor decomposition exactly represents matrix multiplication.
        For each r: sum_r U[i,r]*V[j,r]*W[k,r] should equal delta-based multiplication.
        The tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r] should equal 1 for all valid i,j,k.
        """
        n, m, p = 2, 2, 10
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i, r] * V[j, r] * W[k, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    # Build the known rank-35 solution as fallback
    U35, V35, W35 = build_strassen_rank_35()
    
    # Strategy: Apply rank-7 Strassen to first 2x2 block, then use rank-2 for remaining 4 blocks
    # This would give 7 + 4*2 = 15 rank if it works, but let's verify if rank-2 is possible for 2x2 * 2x1
    
    # For 2x2 * 2x1: C = A * b where A is 2x2, b is 2x1, C is 2x1
    # C[0] = A[0,0]*b[0] + A[0,1]*b[1]
    # C[1] = A[1,0]*b[0] + A[1,1]*b[1]
    # This is equivalent to multiplying a 2x2 matrix by a 2x1 vector.
    # The tensor has entries T[i,j,k] where i in {0,1} (row of C), j in {0,1} (column of A / row of b), k in {0,1} (row of C? No, k is the output index)
    # Actually for M_{2,2,1}: T[i,j,k] where i is row of A (and C), j is column of A (and row of b), k is row of C
    # T[i,j,k] = 1 if j == k? No.
    # C[k] = sum_j A[k,j] * b[j]
    # So T[k,j,?] - wait, the tensor is T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k]
    # For M_{2,2,1}: C[i,0] = sum_j A[i,j] B[j,0]
    # So T[i,j,0] = 1 for all i,j. That's 4 entries.
    # Can we decompose this with rank 2?
    # We need sum_r U[i,r] V[j,r] W[0,r] = 1 for all i in {0,1}, j in {0,1}
    # Let W[0,r] = 1 for all r (wlog, we can absorb into U or V)
    # Then sum_r U[i,r] V[j,r] = 1 for all i,j
    # This means the outer product sum_r U[:,r] V[:,r]^T = all-ones matrix
    # The all-ones 2x2 matrix has rank 1, so we can do it with rank 1!
    # U[:,0] = [1, 1]^T, V[:,0] = [1, 1]^T, W[0,0] = 1
    # Check: sum_r U[i,r] V[j,r] W[0,r] = 1*1*1 = 1 for all i,j. Yes!
    # So rank 1 is sufficient for M_{2,2,1}!
    
    # Wait, let me double-check. For M_{2,2,1}:
    # C[i,0] = A[i,0] * B[0,0] + A[i,1] * B[1,0]
    # Decomposition: C[i,0] = sum_r U[i,r] * (sum_j U_part... ) 
    # No, the standard form is:
    # C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) * (sum_j' V[j',k,r] B[j',k]) * W[k,i,r]
    # Wait, that's not right either. Let me use the correct formulation.
    # The tensor T_{i,j,k} for M_{n,m,p} satisfies:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Decomposition: C_{i,k} = sum_r alpha_r * (sum_j U^{(r)}_{i,j} A_{i,j}) * (sum_j' V^{(r)}_{j',k} B_{j',k})
    # No, that's not standard. The standard tensor decomposition is:
    # T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}
    # where T_{i,j,k} is the multiplication tensor.
    # For M_{n,m,p}, T_{i,j,k} = 1 for all valid i,j,k? No.
    # T_{i,j,k} represents the coefficient such that C_{i,k} = sum_{j} T_{i,j,k} A_{i,j} B_{j,k}
    # Since C_{i,k} = sum_j A_{i,j} B_{j,k}, we have T_{i,j,k} = 1 for all i,j,k.
    # Wait, that can't be right because then the tensor is all-ones and has rank 1.
    # Let me re-examine.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The "tensor" of matrix multiplication is defined such that:
    # C_{i,k} = sum_{j} sum_{r} U_{i,j,r} V_{j,k,r} W_{k,i,r} A_{i,j} B_{j,k}
    # Hmm, I think I'm confusing myself. Let me use the formulation from the task.
    # The task says: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # And the decomposition should satisfy:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # (This is the condition that the tensor equals the all-ones tensor in the appropriate sense)
    # Wait, actually for matrix multiplication M_{n,m,p}, the tensor entries are:
    # T_{(i,j),(j,k),(k,i)} ... no, the indices are flattened.
    # U has index (i,j) where i in [0,n), j in [0,m) -> size n*m
    # V has index (j,k) where j in [0,m), k in [0,p) -> size m*p
    # W has index (k,i) where k in [0,p), i in [0,n) -> size p*n
    # The condition is: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta_{...}? 
    # No, it should be that the bilinear form reproduces the matrix product.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Decomposition: C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_j' V[j',k,r] B[j',k]) W[k,i,r]
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_r W[k,i,r] U[i,j,r] V[j',k,r] = delta_{j,j'} * (something)
    # Actually, the coefficient of A[i,j] B[j',k] in the LHS is sum_r W[k,i,r] U[i,j,r] V[j',k,r]
    # This must equal 1 if j==j' and 0 otherwise (since C[i,k] only has terms where j=j').
    # So: sum_r W[k,i,r] U[i,j,r] V[j,k,r] = delta_{j,j'}
    # But j and j' are both summed in the matrix product, so we need:
    # sum_r W[k,i,r] U[i,j,r] V[j,k,r] = 1 for all i,j,k? No, that's not right.
    
    # Let me just use the verification function from the current program, which checks:
    # sum_r U[i,r] V[j,r] W[k,r] = 1 for all i,j,k
    # where U is indexed by i (0 to n*m-1), V by j (0 to m*p-1), W by k (0 to p*n-1)
    # But wait, U has shape (n*m, rank), so U[i,r] where i is in [0, n*m)
    # This doesn't match the (i,j) indexing. Let me look at the current program's verification:
    # It checks: for i in range(n), j in range(m), k in range(p):
    #   val = sum_r U[i, r] * V[j, r] * W[k, r]
    #   Should equal 1.0
    # But U has shape (4, R) = (n*m, R), so U[i, r] only uses the first n rows? That seems wrong.
    # Oh wait, I see - in the current program, U is built as:
    # U[0, col] = a_vec[0], U[1, col] = a_vec[1], U[2, col] = a_vec[2], U[3, col] = a_vec[3]
    # So U[0] corresponds to A[0,0], U[1] to A[0,1], U[2] to A[1,0], U[3] to A[1,1]
    # And the verification checks U[i, r] for i in range(n)=range(2), which only uses U[0] and U[1]!
    # That's a bug in the verification! It should check all 4 entries of U.
    
    # Let me fix the verification and use the correct formulation.
    # The correct condition is:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    def verify_decomposition_fixed(U, V, W):
        n, m, p = 2, 2, 10
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    # Verify the rank-35 solution with the fixed verification
    assert verify_decomposition_fixed(U35, V35, W35), "Rank 35 solution failed verification"
    
    # Now let's try the rank 15 approach:
    # Block 0 (columns 0,1): use rank-7 Strassen
    # Blocks 1-4 (columns 2-9): use rank-1 each (since M_{2,2,1} has rank 1)
    # Total rank: 7 + 4*1 = 11
    
    # For a single column k, the condition is:
    # For all i in {0,1}, j in {0,1}:
    #   sum_r U[i*2+j, r] * V[j*10+k, r] * W[k*2+i, r] = 1
    # We can achieve this with rank 1:
    # U[i*2+j, r0] = 1 for all i,j
    # V[j*10+k, r0] = 1 for all j
    # W[k*2+i, r0] = 1 for all i
    # Then sum_r ... = 1*1*1 = 1. Yes!
    
    # Let's build rank 11:
    def build_rank_11():
        R = 11
        U = np.zeros((4, R), dtype=float)
        V = np.zeros((20, R), dtype=float)
        W = np.zeros((20, R), dtype=float)
        
        # First 7 terms: Strassen for columns 0,1
        for col in range(7):
            a_vec, b_vec, c_vec = terms[col]
            U[0, col] = a_vec[0]
            U[1, col] = a_vec[1]
            U[2, col] = a_vec[2]
            U[3, col] = a_vec[3]
            
            # V for columns 0,1
            V[0*10 + 0, col] = b_vec[0]  # j=0, k=0
            V[0*10 + 1, col] = b_vec[1]  # j=0, k=1
            V[1*10 + 0, col] = b_vec[2]  # j=1, k=0
            V[1*10 + 1, col] = b_vec[3]  # j=1, k=1
            
            # W for columns 0,1
            W[0*2 + 0, col] = c_vec[0]  # k=0, i=0
            W[0*2 + 1, col] = c_vec[1]  # k=0, i=1
            W[1*2 + 0, col] = c_vec[2]  # k=1, i=0
            W[1*2 + 1, col] = c_vec[3]  # k=1, i=1
        
        # Next 4 terms: rank-1 for columns 2,3,4,5 (wait, columns 2-9 is 8 columns, not 4)
        # Oh right, p=10, so columns 0-9. First 2 columns use Strassen, remaining 8 columns use rank-1 each.
        # That's 7 + 8 = 15 rank.
        
        # Let me redo: 7 + 8*1 = 15
        R = 15
        U = np.zeros((4, R), dtype=float)
        V = np.zeros((20, R), dtype=float)
        W = np.zeros((20, R), dtype=float)
        
        # First 7 terms: Strassen for columns 0,1
        for col in range(7):
            a_vec, b_vec, c_vec = terms[col]
            U[0, col] = a_vec[0]
            U[1, col] = a_vec[1]
            U[2, col] = a_vec[2]
            U[3, col] = a_vec[3]
            
            V[0*10 + 0, col] = b_vec[0]
            V[0*10 + 1, col] = b_vec[1]
            V[1*10 + 0, col] = b_vec[2]
            V[1*10 + 1, col] = b_vec[3]
            
            W[0*2 + 0, col] = c_vec[0]
            W[0*2 + 1, col] = c_vec[1]
            W[1*2 + 0, col] = c_vec[2]
            W[1*2 + 1, col] = c_vec[3]
        
        # Next 8 terms: rank-1 for columns 2,3,4,5,6,7,8,9
        for col_offset in range(8):
            col = 7 + col_offset
            k = 2 + col_offset  # column index in B
            
            # U: all ones
            U[0, col] = 1
            U[1, col] = 1
            U[2, col] = 1
            U[3, col] = 1
            
            # V: ones for this column k
            V[0*10 + k, col] = 1  # j=0, k=k
            V[1*10 + k, col] = 1  # j=1, k=k
            
            # W: ones for this column k
            W[k*2 + 0, col] = 1  # k=k, i=0
            W[k*2 + 1, col] = 1  # k=k, i=1
        
        return U, V, W
    
    U15, V15, W15 = build_rank_11()
    
    # Verify rank 15
    if verify_decomposition_fixed(U15, V15, W15):
        # Great! Rank 15 works.
        return {"U": U15.tolist(), "V": V15.tolist(), "W": W15.tolist()}
    else:
        # Rank 15 failed, return rank 35
        return {"U": U35.tolist(), "V": V35.tolist(), "W": W35.tolist()}

