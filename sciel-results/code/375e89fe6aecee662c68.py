"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # 1. Build the Hybrid Strassen/Schoolbook Decomposition
    # Structure: 
    # A is 2x3, B is 3x7. 
    # Split B into B1 (3x2) and B2 (3x5).
    # Split A into A1 (2x2) and A2 (2x1).
    # C = A*B = A1*B1 + A2*B2.
    
    # We can apply Strassen to A1 (2x2) and B1 (2x2) part? 
    # Actually, A1 is 2x2, B1 is 2x2. Result is 2x2.
    # A2 is 2x1, B2 is 1x5. Result is 2x5.
    
    # Let's be more precise with indices to ensure exactness.
    # A[i, j], B[j, k] -> C[i, k]
    # i in 0..1, j in 0..2, k in 0..6
    
    # Strategy:
    # The term A1*B1 involves j=0,1 and k=0,1. 
    # The term A2*B2 involves j=2 and k=0..6.
    # Wait, this split is not quite right for Strassen which requires square matrices.
    # Let's split the inner dimension m=3 into m1=2 and m2=1.
    # A = [A1 (2x2) | A2 (2x1)]
    # B = [B1 (2x7); B2 (1x7)]  -> B is 3x7. So B1 is 2x7, B2 is 1x7.
    # C = A1 @ B1 + A2 @ B2.
    
    # A1 is 2x2, B1 is 2x7. This is not square. Strassen is for square.
    # However, we can split the output k dimension as well?
    # Let's try a different structural decomposition.
    
    # Known result for 2x3x7:
    # Rank is likely 39 or lower.
    # Schoolbook is 2*3*7 = 42.
    
    # Let's use the "Strassen for 2x2x2" insight from the cell findings.
    # We can group the computation.
    # The tensor T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    
    # Let's try to construct a rank 39 decomposition explicitly.
    # We can view the 2x3x7 multiplication as:
    # 2 * (3x7) multiplications? No.
    
    # Let's try to find a decomposition by solving a linear system for specific structures.
    # But first, let's try the hybrid approach mentioned in the prompt's "knowledge".
    # "splitting the inner dimension and using Strassen for 2x2x2 blocks"
    
    # Let's split the inner dimension j=0,1,2.
    # Let's split the output dimension k=0..6.
    
    # Consider the sub-matrix of A: A[0:2, 0:2] (2x2)
    # Consider the sub-matrix of B: B[0:2, 0:2] (2x2)
    # This contributes to C[0:2, 0:2].
    # We can use Strassen for this 2x2x2 block. Rank 7 instead of 8.
    # Saving: 1 term.
    
    # What about the rest?
    # A[0:2, 2] (2x1) * B[2, 0:7] (1x7) -> C[0:2, 0:7] contribution. Rank 2*1*7 = 14.
    # A[0:2, 0:2] (2x2) * B[0:2, 2:7] (2x5) -> C[0:2, 2:7] contribution. Rank 2*2*5 = 20.
    
    # Total Rank = 7 (for C[0:2, 0:2] via Strassen) + 14 (for j=2 part) + 20 (for j=0,1, k=2..6 part)
    # Total = 7 + 14 + 20 = 41.
    # This is rank 41.
    
    # Can we do better?
    # The "Strassen" part handles j=0,1 and k=0,1.
    # The "Schoolbook" part for j=0,1 and k=2..6 is rank 20.
    # The "Schoolbook" part for j=2 and k=0..6 is rank 14.
    
    # Is there a way to combine?
    # Maybe use Strassen on a different block?
    # If we split k into 2+5.
    # B[0:2, 0:2] is 2x2. A[0:2, 0:2] is 2x2.
    
    # Let's implement the Rank 41 decomposition first, as it is a solid improvement over 42.
    # Then we can try to optimize further or verify if 40 is possible.
    
    def build_hybrid_decomposition():
        U = np.zeros((n * m, 0))
        V = np.zeros((m * p, 0))
        W = np.zeros((p * n, 0))
        
        def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
            nonlocal U, V, W
            c = U.shape[1]
            U = np.column_stack((U, np.zeros(n * m)))
            V = np.column_stack((V, np.zeros(m * p)))
            W = np.column_stack((W, np.zeros(p * n)))
            U[u_idx, c] = u_val
            V[v_idx, c] = v_val
            W[w_idx, c] = w_val
            
        # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        # Indices: i in 0,1; j in 0,1; k in 0,1.
        # Strassen formulas for 2x2 multiplication:
        # C00 = M1 - M2 + M4 - M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M4 - M6
        # Where:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B10)
        # M7 = (A01-A11)*(B10+B11)
        
        # Note: Our tensor decomposition is T[i,j,k] = sum U[i,j,c] V[j,k,c] W[k,i,c]
        # This is equivalent to C[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_j U[i,j,c] B[j,k]) (sum_j A[i,j] V[j,k,c])? 
        # No, the standard decomposition is:
        # C[i,k] = sum_c U[i,j,c] is not right.
        # The tensor T_{ijk} = \sum_c U_{ij}^c V_{jk}^c W_{ki}^c.
        # This means for each c, we form matrices U_c, V_c, W_c? No.
        # It means we form outer products?
        # Let's stick to the coefficient definition.
        
        # For Strassen, we have 7 multiplications.
        # Each multiplication corresponds to a term in the sum.
        # We need to map the Strassen intermediates to the U,V,W coefficients.
        
        # Let's define the 7 terms for the 2x2x2 block (i=0,1; j=0,1; k=0,1).
        # The contribution to C[i,k] from this block is sum_{j=0,1} A[i,j] B[j,k].
        # We want to represent this as sum_{c=0}^6 U[i,j,c] V[j,k,c] W[k,i,c].
        # Wait, the W factor depends on k and i. In the standard matrix multiplication tensor,
        # the "third leg" is usually just 1 if we are doing A*B.
        # But here the tensor is defined with W[k*n+i].
        # If we set W[k*n+i, c] = 1 for all c, then sum_c U[i,j,c] V[j,k,c] = C[i,k].
        # This is the standard bilinear complexity.
        
        # So for the Strassen block:
        # We need U[i,j,c] and V[j,k,c] such that sum_c U[i,j,c] V[j,k,c] = (A1*B1)[i,k].
        # And W[k*n+i, c] = 1.
        
        # Let's populate U, V, W for the Strassen block (i in 0,1; j in 0,1; k in 0,1).
        # c=0..6
        
        # M1: (A00+A11)*(B00+B11)
        # A part: A00+A11 -> U[0,0,0]=1, U[1,1,0]=1
        # B part: B00+B11 -> V[0,0,0]=1, V[1,1,0]=1
        # C part: contributes to all C[i,k] in 2x2.
        # We need to distribute this product to C[i,k].
        # This is tricky because Strassen combines sums of A and sums of B.
        # The decomposition T = sum U_c \otimes V_c \otimes W_c.
        # If W_c is all 1s, then U_c \otimes V_c must sum to the matrix product.
        # But Strassen uses 7 products of *sums*.
        # This implies that the "rank" in the tensor sense (where we can put coefficients in U, V, W) 
        # allows us to implement Strassen.
        
        # Let's verify the Strassen decomposition in tensor form.
        # T_{ijk} for 2x2x2.
        # We want \sum_{c=0}^6 U_{ij}^c V_{jk}^c W_{ki}^c = \delta_{i,k}? No, T_{ijk} is the multiplication tensor.
        # T_{ijk} = 1 if the indices correspond to a valid term in the sum?
        # Actually, the tensor T_{ijk} for matrix multiplication is 1 if we are multiplying A_{ij} and B_{jk} to contribute to C_{ik}.
        # So T_{ijk} = 1 for all i,j,k in the block.
        
        # We need \sum_c U_{ij}^c V_{jk}^c W_{ki}^c = 1 for all i,j,k in {0,1}^3.
        
        # Let's use the known Strassen tensor decomposition.
        # The 7 terms are:
        # 1. (A00+A11)(B00+B11) -> contributes to C00, C01, C10, C11 with signs.
        # This requires W to carry the signs and distribution.
        
        # Let's look up the explicit tensor decomposition for Strassen.
        # U (2x2x7), V (2x2x7), W (2x2x7) such that sum U[i,j,c] V[j,k,c] W[k,i,c] = 1.
        
        # c=0: M1 = (A00+A11)(B00+B11)
        # U[0,0,0]=1, U[1,1,0]=1
        # V[0,0,0]=1, V[1,1,0]=1
        # W[0,0,0]=1, W[1,0,0]=-1, W[0,1,0]=-1, W[1,1,0]=1  (To get C00=+, C01=-, C10=-, C11=+)
        
        # c=1: M2 = (A10+A11)B00
        # U[1,0,1]=1, U[1,1,1]=1
        # V[0,0,1]=1
        # W[0,1,1]=1, W[1,1,1]=1 (Contributes to C01 and C11? No, M2 contributes to C10 and C11? 
        # Let's check Strassen:
        # C00 = M1 - M2 + M4 - M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M4 - M6
        
        # So M2 contributes to C10 and C11.
        # W[0,1,1] is C[1,0]? No, W[k*n+i]. n=2.
        # W[0*2+1] = W[1] is C[1,0]? 
        # Indexing: W[k*n+i]. 
        # C[i,k] is stored at W[k*n+i].
        # So C[1,0] is at W[0*2+1] = W[1].
        # C[1,1] is at W[1*2+1] = W[3].
        # So W[1,1]=1, W[3,1]=1.
        
        # This is getting complex to derive manually. 
        # Let's use a simpler approach: 
        # Since we know the rank is likely 39, and we have a 120s budget, 
        # we can try to *solve* for a rank-39 decomposition using a heuristic search 
        # starting from the rank-41 hybrid structure.
        
        # But first, let's just implement the Rank 41 hybrid correctly.
        # If we can't get Strassen right, we fall back to Schoolbook (42).
        
        # Let's try a different tactic: 
        # Use the "Rank 39" hint. 
        # 2x3x7.
        # Maybe split 7 into 2+2+3?
        # Or use the fact that 2x2x2 has rank 7.
        
        # Let's try to construct the Rank 41 decomposition explicitly using the block logic.
        # Block 1: i=0,1; j=0,1; k=0,1. (2x2x2). Use Strassen (Rank 7).
        # Block 2: i=0,1; j=0,1; k=2..6. (2x2x5). Schoolbook (Rank 20).
        # Block 3: i=0,1; j=2; k=0..6. (2x1x7). Schoolbook (Rank 14).
        
        # Total Rank = 7 + 20 + 14 = 41.
        
        # Let's implement this.
        
        U = np.zeros((n * m, 41))
        V = np.zeros((m * p, 41))
        W = np.zeros((p * n, 41))
        
        c = 0
        
        # --- Block 1: Strassen for i in 0,1; j in 0,1; k in 0,1 ---
        # We need 7 terms.
        # We will hardcode the Strassen coefficients.
        
        # c=0: M1
        U[0*m+0, c] = 1; U[1*m+1, c] = 1
        V[0*p+0, c] = 1; V[1*p+1, c] = 1
        W[0*n+0, c] = 1; W[1*n+0, c] = -1; W[0*n+1, c] = -1; W[1*n+1, c] = 1
        c += 1
        
        # c=1: M2
        U[1*m+0, c] = 1; U[1*m+1, c] = 1
        V[0*p+0, c] = 1
        W[0*n+1, c] = 1; W[1*n+1, c] = 1 # C[1,0] and C[1,1]
        c += 1
        
        # c=2: M3
        U[0*m+0, c] = 1
        V[0*p+1, c] = 1; V[1*p+1, c] = -1
        W[0*n+0, c] = 1; W[1*n+0, c] = -1 # C[0,0] and C[1,0]? 
        # Wait, M3 = A00(B01-B11). Contributes to C01 and C11?
        # C01 = M3 + M5. C11 = M1 - M3 + M4 - M6.
        # So M3 contributes to C01 (+1) and C11 (-1).
        # C[0,1] is W[1*n+0] = W[2].
        # C[1,1] is W[1*n+1] = W[3].
        W[1*n+0, c] = 1; W[1*n+1, c] = -1
        c += 1
        
        # c=3: M4
        U[1*m+1, c] = 1
        V[1*p+0, c] = 1; V[0*p+0, c] = -1
        # M4 = A11(B10-B00). Contributes to C10 (+1) and C00 (-1).
        # C[1,0] is W[0*n+1] = W[1].
        # C[0,0] is W[0*n+0] = W[0].
        W[0*n+1, c] = 1; W[0*n+0, c] = -1
        c += 1
        
        # c=4: M5
        U[0*m+0, c] = 1; U[0*m+1, c] = 1
        V[1*p+1, c] = 1
        # M5 = (A00+A01)B11. Contributes to C00 (-1) and C01 (+1).
        # C[0,0] is W[0]. C[0,1] is W[2].
        W[0*n+0, c] = -1; W[1*n+0, c] = 1
        c += 1
        
        # c=5: M6
        U[1*m+0, c] = 1; U[0*m+0, c] = -1
        V[0*p+0, c] = 1; V[1*p+0, c] = 1
        # M6 = (A10-A00)(B00+B10). Contributes to C11 (-1).
        # C[1,1] is W[3].
        W[1*n+1, c] = -1
        c += 1
        
        # c=6: M7
        U[0*m+1, c] = 1; U[1*m+1, c] = -1
        V[1*p+0, c] = 1; V[1*p+1, c] = 1
        # M7 = (A01-A11)(B10+B11). 
        # Wait, standard Strassen uses M7 for C01 and C11?
        # C01 = M3 + M5. C11 = M1 - M3 + M4 - M6.
        # Where is M7? 
        # Ah, standard Strassen is:
        # C00 = M1 - M2 + M4 - M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M4 - M6
        # This uses 6 multiplications? No, M1..M7 are 7.
        # Let's re-verify Strassen.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = (A00)(B01-B11)
        # M4 = (A11)(B10-B00)
        # M5 = (A00+A01)(B11)
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M2 + M4 - M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M4 - M6
        # Wait, M7 is not used in C? 
        # Actually, C11 = M1 - M3 + M4 - M6 is correct for 6?
        # No, Strassen uses 7. 
        # C11 = M1 - M3 + M4 - M6? 
        # Let's check C11 = A00B01 + A01B11 + A10B01 + A11B11? No.
        # C11 = A10B01 + A11B11.
        # M1 - M3 + M4 - M6 = (A00+A11)(B00+B11) - A00(B01-B11) + A11(B10-B00) - (A10-A00)(B00+B10)
        # = A00B00 + A00B11 + A11B00 + A11B11 - A00B01 + A00B11 + A11B10 - A11B00 - A10B00 - A10B10 + A00B00 + A00B10
        # This doesn't look like C11.
        
        # Correct Strassen:
        # C00 = M1 - M2 + M4 - M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M4 - M6
        # Wait, I am missing M7.
        # Ah, C11 = M1 - M3 + M4 - M6 is for 6 multiplications? No.
        # The standard 7th multiplication M7 is used in C11?
        # Actually, C11 = M1 - M3 + M4 - M6 + M7? No.
        
        # Let's just use a verified Strassen tensor decomposition.
        # I will skip the manual derivation and use a known valid one or fall back to schoolbook for the 2x2 block if I can't get it right quickly.
        # Given the time, let's just use Schoolbook for the 2x2 block (Rank 8) to ensure correctness, 
        # which gives Rank 8 + 20 + 14 = 42.
        # This is not an improvement.
        
        # Let's try to find a rank 41 decomposition by removing one term from schoolbook?
        # No, that's hard.
        
        # Let's try a different structural split.
        # 2x3x7.
        # What if we use a known low-rank decomposition for 2x3x7?
        # Literature suggests rank(2,3,7) <= 39.
        
        # Since I cannot guarantee a correct Strassen implementation in the time, 
        # I will implement a **Random Search** starting from the Schoolbook decomposition 
        # and trying to merge terms or find linear dependencies to reduce rank.
        
        # However, the prompt suggests "Formulate as linear code optimization...".
        # This is too complex for 120s.
        
        # Let's try a **Greedy Sparsification** approach.
        # Start with Schoolbook (Rank 42).
        # Try to set one term to zero and adjust others to compensate?
        # This is a local search on the space of decompositions.
        
        # Better: Use the "Hybrid" idea but with a verified Strassen implementation.
        # I will hardcode the Strassen coefficients from a reliable source.
        
        # Verified Strassen Tensor (2x2x2):
        # U (2x2x7), V (2x2x7), W (2x2x7)
        
        # c=0: M1
        # U: [[1,0],[0,1]]
        # V: [[1,0],[0,1]]
        # W: [[1,-1],[-1,1]]
        
        # c=1: M2
        # U: [[0,0],[1,1]]
        # V: [[1,0],[0,0]]
        # W: [[0,0],[1,1]]
        
        # c=2: M3
        # U: [[1,0],[0,0]]
        # V: [[0,1],[0,-1]]
        # W: [[0,1],[0,-1]]
        
        # c=3: M4
        # U: [[0,0],[0,1]]
        # V: [[-1,0],[1,0]]
        # W: [[-1,0],[1,0]]
        
        # c=4: M5
        # U: [[1,1],[0,0]]
        # V: [[0,0],[0,1]]
        # W: [[-1,0],[0,1]]
        
        # c=5: M6
        # U: [[-1,0],[1,0]]
        # V: [[1,1],[0,0]]
        # W: [[0,0],[0,-1]]
        
        # c=6: M7
        # U: [[0,1],[0,-1]]
        # V: [[0,0],[1,1]]
        # W: [[0,0],[0,1]]
        
        # Let's verify c=0:
        # Sum U[i,j] V[j,k] W[k,i]
        # i=0,k=0: j=0: 1*1*1 = 1. j=1: 0*0*-1 = 0. Total 1. (C00 gets +1 from M1)
        # i=0,k=1: j=0: 1*0*-1 = 0. j=1: 0*1*1 = 0. Total 0. (C01 gets 0 from M1? Wait)
        # M1 contributes to C00, C01, C10, C11?
        # In Strassen, M1 is added to C00 and C11, and subtracted from C01 and C10?
        # C00 = M1 - M2 + M4 - M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M4 - M6
        # So M1 contributes +1 to C00, -1 to C10? No, C10 = M2+M4. M1 is not in C10.
        # My W matrix for c=0 was [[1,-1],[-1,1]].
        # W[0,0]=1 (C00), W[0,1]=-1 (C01), W[1,0]=-1 (C10), W[1,1]=1 (C11).
        # So M1 contributes to C01 and C10?
        # But C01 = M3 + M5. No M1.
        # So the W matrix for M1 should be [[1,0],[0,1]]?
        # If W is [[1,0],[0,1]], then M1 contributes to C00 and C11.
        # C00 = M1... C11 = M1...
        # This matches.
        
        # So c=0 W is [[1,0],[0,1]].
        
        # Let's re-verify c=1 (M2):
        # M2 contributes to C10 and C11?
        # C10 = M2 + M4. C11 = ... - M6. No M2 in C11.
        # So M2 contributes to C10 only?
        # C10 = M2 + M4.
        # So W for M2 should be [[0,0],[1,0]]? (C10 is W[0,1] if W[k,i]? No W[k*n+i]. C[1,0] is W[0*2+1]=W[1].)
        # So W[1] = 1. Others 0.
        
        # This is too error-prone. 
        # I will fall back to the **Schoolbook** decomposition (Rank 42) 
        # and then perform a **local search** to try to reduce the rank to 41 or 40.
        
        def build_schoolbook():
            terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
            rank = len(terms)
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for c, (i, j, k) in enumerate(terms):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
            return U, V, W

        def verify(U, V, W, tol=1e-9):
            rank = U.shape[1]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_val = U[i * m + j]
                        v_val = V[j * p + k]
                        w_val = W[k * n + i]
                        total = np.sum(u_val * v_val * w_val)
                        if abs(total - 1.0) > tol:
                            return False
            return True

        # Start with Schoolbook
        U, V, W = build_schoolbook()
        best_rank = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if not verify(best_U, best_V, best_W):
            raise ValueError("Schoolbook decomposition failed verification")
        
        # Try to reduce rank by 1 (to 41) using a random search with small perturbations
        # and local optimization.
        
        np.random.seed(42)
        candidate_vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        
        # Attempt to find a rank 41 decomposition
        target_rank = 41
        found = False
        
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values
            U_rand = np.random.choice(candidate_vals, size=(n*m, target_rank))
            V_rand = np.random.choice(candidate_vals, size=(m*p, target_rank))
            W_rand = np.random.choice(candidate_vals, size=(p*n, target_rank))
            
            # Local search
            best_err = np.inf
            for _ in range(100):
                if time.time() > deadline - 10:
                    break
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_val = U_rand[i * m + j]
                            v_val = V_rand[j * p + k]
                            w_val = W_rand[k * n + i]
                            total = np.sum(u_val * v_val * w_val)
                            err += (total - 1.0)**2
                if err < best_err:
                    best_err = err
                    if err < 1e-9:
                        if verify(U_rand, V_rand, W_rand):
                            best_rank = target_rank
                            best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                            found = True
                            break
                # Perturb
                mat = np.random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    r, c = np.random.randint(0, n*m), np.random.randint(0, target_rank)
                    old = U_rand[r, c]
                    U_rand[r, c] = np.random.choice(candidate_vals)
                elif mat == 'V':
                    r, c = np.random.randint(0, m*p), np.random.randint(0, target_rank)
                    old = V_rand[r, c]
                    V_rand[r, c] = np.random.choice(candidate_vals)
                else:
                    r, c = np.random.randint(0, p*n), np.random.randint(0, target_rank)
                    old = W_rand[r, c]
                    W_rand[r, c] = np.random.choice(candidate_vals)
            
            if found:
                break
        
        # Final verification
        if not verify(best_U, best_V, best_W):
            best_U, best_V, best_W = build_schoolbook()
            best_rank = n * m * p
        
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
# EVOLVE-BLOCK-END
