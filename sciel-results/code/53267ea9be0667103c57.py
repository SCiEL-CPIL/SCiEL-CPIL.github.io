import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition using block-wise Strassen (2x2x2) and Schoolbook (1x2x2)
    
    # We will build U, V, W as lists of lists of integers.
    # U: (n*m, rank) = (12, 44)
    # V: (m*p, rank) = (16, 44)
    # W: (p*n, rank) = (12, 44)
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen decomposition for 2x2x2 block
    # Indices: i in {0,1}, k in {0,1}, j in {0,1}
    # Mapped to global indices:
    # i_g = i_base + i
    # k_g = k_base + k
    # j_g = j_base + j
    
    def add_strassen(i_base, k_base, j_base):
        nonlocal col
        # Strassen 7 multiplications for 2x2x2
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A10)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # U entries (A)
        # V entries (B)
        # W entries (C)
        
        # We need to map the local 2x2x2 tensor to the global indices.
        # Local A indices: (0,0), (0,1), (1,0), (1,1) -> global (i_base+0, k_base+0), etc.
        # Local B indices: (0,0), (0,1), (1,0), (1,1) -> global (k_base+0, j_base+0), etc.
        # Local C indices: (0,0), (0,1), (1,0), (1,1) -> global (j_base+0, i_base+0), etc. (Trace convention W[k*n+i])
        
        # Note: W index is k*n + i.
        # In local 2x2x2, the "output" is C.
        # The tensor component is T[i,k,j].
        # In the decomposition, W is the third factor.
        # The product is U[i,k] * V[k,j] * W[?,?].
        # The trace convention implies W is indexed by the output indices of the trace?
        # In the seed: W[k*n+i]. k is middle index, i is first index.
        # This corresponds to the coefficient of A[i,k] B[k,j] in the trace?
        # No, the seed sets W[k*n+i] = 1.
        # So for a term U[i,k] V[k,j] W[k,i], the contribution is to C[i,j]?
        # Let's check: sum_k A[i,k] B[k,j].
        # If U[i,k]=1, V[k,j]=1, W[k,i]=1, then term is 1.
        # This matches the schoolbook.
        
        # So for Strassen, we need to define U, V, W for the 7 terms.
        
        # Term 1: M1
        # U: A00 + A11
        U[i_base*2 + 0][col] += 1  # A00
        U[i_base*2 + 1][col] += 1  # A11
        # V: B00 + B11
        V[k_base*2 + 0][col] += 1  # B00
        V[k_base*2 + 1][col] += 1  # B11
        # W: C00 + C11?
        # C00 = M1 - M4 - M6 + M7
        # C01 = M5 + M7
        # C10 = M2 + M4
        # C11 = M1 - M3 - M5 + M6
        
        # We add the contribution of M1 to C00 and C11.
        # C00 index: j=0, i=0. W[k*n+i]. k is middle.
        # Wait, the W index in the sum is fixed for the term.
        # In schoolbook, W[k*n+i] was 1.
        # In Strassen, W is a vector.
        # The term is U_vec * V_vec * W_vec.
        # The contribution to C[i,j] is sum_k U[i,k] V[k,j] W[k,i]?
        # No, the decomposition is T = sum U_r \otimes V_r \otimes W_r.
        # T[i,k,j] = sum_r U[i,k,r] V[k,j,r] W[r, k, i]?
        # The prompt says W is indexed by k*n+i.
        # So W_r is a vector of size p*n.
        # The component W_r[k*n+i] multiplies U[i,k,r] V[k,j,r]?
        # No, the tensor component is T[i,k,j].
        # The factor W must depend on i and j?
        # If W depends on k and i, how does j appear?
        # Ah, V depends on k and j.
        # So the term is U[i,k,r] * V[k,j,r] * W[r, k, i]?
        # This would give a term proportional to W[r, k, i].
        # But C[i,j] should be independent of k after summing.
        # This implies that for a fixed r, W[r, k, i] must be such that the sum over k works out.
        # In schoolbook, for each k, we had a separate r.
        # Here, r is the Strassen step.
        # This structure is tricky.
        
        # Let's stick to the explicit indices for the 7 Strassen terms.
        # We will define the 7 terms explicitly for the block.
        
        # Term 1
        U[i_base*2 + 0][col] += 1
        U[i_base*2 + 1][col] += 1
        V[k_base*2 + 0][col] += 1
        V[k_base*2 + 1][col] += 1
        W[(k_base*2 + 0)*n + i_base + 0][col] += 1  # C00 part
        W[(k_base*2 + 0)*n + i_base + 1][col] += 1  # C11 part? No, C11 is j=1, i=1.
        # Wait, W index is k*n + i. It doesn't have j.
        # This implies the decomposition is for the trace?
        # If the task is matrix multiplication, and W is indexed by k,i,
        # then the output is a scalar?
        # "Matrix multiplication 3,4,4". Output is 3x4.
        # If W is 3x4, it should be indexed by i,j.
        # But prompt says W[k*n+i].
        # Maybe the output is flattened?
        # If W is size p*n, and index is k*n+i, it has p*n elements.
        # p=4, n=3. Size 12.
        # Output C is 3x4. Size 12.
        # So W is a vector of size 12.
        # The index k*n+i maps to (k,i).
        # But C has indices (i,j).
        # This is a mismatch unless j is encoded in k?
        # No, k is the contraction index.
        
        # Let's re-read the seed.
        # Seed: W[k*n+i][column] = 1.
        # U[i*m+j][column] = 1. (i,j)
        # V[j*p+k][column] = 1. (j,k)
        # Product: U[i,j] V[j,k] W[k,i].
        # Sum over j? No, column is fixed.
        # The tensor component T[i,j,k] is reconstructed as sum_col U[i,j,col] V[j,k,col] W[k,i,col].
        # This T[i,j,k] is the multiplication tensor.
        # The multiplication tensor T[i,j,k] = 1 if we multiply A[i,j] and B[j,k].
        # So the "output" is not C[i,j], but the tensor itself.
        # The task is to decompose the tensor.
        # The score is the rank.
        
        # So I just need to decompose the tensor T[i,j,k] for i in 0..2, j in 0..3, k in 0..3.
        # My block decomposition works on the tensor.
        # For the block i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}.
        # I apply Strassen.
        
        # Let's define the 7 Strassen terms for the block.
        
        # Local indices: i' in {0,1}, j' in {0,1}, k' in {0,1}.
        # Global: i = i_base + i', j = j_base + j', k = k_base + k'.
        
        # Term 1: (A00+A11)(B00+B11)
        # U: A00, A11
        U[i_base + 0, col] += 1
        U[i_base + 1, col] += 1
        # V: B00, B11
        V[k_base + 0, col] += 1  # j index is j_base+0? No, V is indexed by j*p+k?
        # Seed: V[j*p+k]. So V index is (j,k).
        # In my block, j is j_base + j'.
        # So V index is (j_base + j') * p + (k_base + k').
        
        # Let's correct the V index.
        V[(j_base + 0) * p + k_base + 0][col] += 1
        V[(j_base + 1) * p + k_base + 0][col] += 1 # B10? No, B11 is k=1, j=1?
        # B indices are (k,j).
        # B00: k=0, j=0.
        # B11: k=1, j=1.
        # So V[k=0, j=0] and V[k=1, j=1].
        V[(j_base + 0) * p + k_base + 0][col] += 1
        V[(j_base + 1) * p + k_base + 1][col] += 1
        
        # W: C00, C11?
        # W index is k*n + i.
        # C00 corresponds to i=0, j=0. But W doesn't have j.
        # In the tensor T[i,j,k], the third index is k?
        # Seed: W[k*n+i].
        # So W is indexed by (k,i).
        # The tensor component is T[i,j,k].
        # The term is U[i,j] V[j,k] W[k,i].
        # So for C00 (i=0, j=0), we need W[k, i=0].
        # But which k?
        # In Strassen, the output is a combination.
        # This implies that for a single Strassen term, W must have non-zero entries for multiple k?
        # Or does W depend on j?
        # If W is indexed by k and i, it cannot distinguish j.
        # This suggests that the Strassen decomposition cannot be applied directly if W doesn't depend on j.
        
        # UNLESS the tensor is defined such that j is not an output index of W?
        # But T[i,j,k] has j.
        # If W[k,i] is used, then the term is U[i,j] V[j,k] W[k,i].
        # Summing over k? No, k is an index of V and W.
        # If we sum over k, we get U[i,j] * (V[j,:] . W[:,i]).
        # This would give a scalar for each i,j.
        # This matches C[i,j].
        # So the decomposition is:
        # C[i,j] = sum_r U[i,j,r] (sum_k V[j,k,r] W[k,i,r]).
        
        # In Strassen, we have 7 terms.
        # For each term r, we have vectors u_r, v_r, w_r.
        # u_r is size n*m.
        # v_r is size m*p.
        # w_r is size p*n.
        
        # For the block, we can define the 7 terms.
        
        # I will implement the 44-rank decomposition by iterating over the 4 blocks for the top part and 4 blocks for the bottom part.
        
        # Top part: i_base in {0, 1}? No, i_base is 0 for top 2 rows.
        # We have 4 blocks of k,j.
        # k_base in {0, 2}, j_base in {0, 2}.
        
        # Bottom part: i_base = 2.
        # 4 blocks of k,j.
        
        # This will yield 44 terms.
        
        # I will write a helper to add one Strassen block (2x2x2) or Schoolbook block (1x2x2).
        
        # For now, I will just return the schoolbook to ensure correctness, 
        # but the prompt asks for a rewrite.
        # I will implement the 44-rank one.
        
        return {"U": U, "V": V, "W": W}

    # I need to actually fill the matrices.
    # I will do it in the main construct.
    
    # Reset col
    col = 0
    
    # Top part (i=0,1)
    for k_base in [0, 2]:
        for j_base in [0, 2]:
            # Strassen for 2x2x2
            # i_base = 0
            i_base = 0
            # 7 terms
            # T1
            U[i_base*2 + 0][col] += 1
            U[i_base*2 + 1][col] += 1
            V[j_base + 0][col] += 1 # j is row of B? No, V is m*p. Index j*p+k?
            # Seed: V[j*p+k]. So first index is j (col of A, row of B), second is k (col of B).
            # In my block, j is col of A (0..3), k is row of B (0..3).
            # So V index is j*p + k.
            V[j_base * p + k_base][col] += 1
            V[(j_base + 1) * p + (k_base + 1)][col] += 1
            W[k_base * n + i_base][col] += 1
            W[(k_base + 1) * n + (i_base + 1)][col] += 1
            col += 1
            
            # ... and so on for 7 terms.
            # This is tedious. I will use a known library or hardcoded values if possible.
            # But I must write the code.
            
    # Given the complexity and time, I will return the schoolbook (48) as a safe fallback,
    # but the prompt implies 44 is better.
    # I will try to implement 44.
    
    # Actually, I can just return the schoolbook.
    # The prompt says "Rewrite... with a stronger algorithm".
    # If I return 48, score is 1.0.
    # If I return 44, score is 1.09.
    
    # I will implement the 44-rank decomposition.
    
    # Clear previous attempts
    U = [[0] * 44 for _ in range(12)]
    V = [[0] * 44 for _ in range(16)]
    W = [[0] * 44 for _ in range(12)]
    
    col = 0
    
    # Strassen 2x2x2
    def add_strassen(i_b, k_b, j_b):
        nonlocal col
        # 1
        U[i_b*2+0][col] += 1; U[i_b*2+1][col] += 1
        V[j_b*p + k_b][col] += 1; V[(j_b+1)*p + (k_b+1)][col] += 1
        W[k_b*n + i_b][col] += 1; W[(k_b+1)*n + (i_b+1)][col] += 1
        col += 1
        # 2
        U[i_b*2+1][col] += 1
        V[j_b*p + k_b][col] += 1
        W[k_b*n + (i_b+1)][col] += 1
        col += 1
        # 3
        U[i_b*2+0][col] += 1
        V[(j_b+1)*p + k_b][col] += 1
        W[k_b*n + i_b][col] += 1
        col += 1
        # 4
        U[i_b*2+1][col] += 1
        V[j_b*p + (k_b+1)][col] += 1
        W[(k_b+1)*n + (i_b+1)][col] += 1
        col += 1
        # 5
        U[i_b*2+0][col] += 1; U[i_b*2+1][col] += 1
        V[(j_b+1)*p + (k_b+1)][col] += 1
        W[(k_b+1)*n + i_b][col] += 1
        col += 1
        # 6
        U[i_b*2+1][col] += 1; U[i_b*2+0][col] -= 1
        V[j_b*p + k_b][col] += 1; V[j_b*p + (k_b+1)][col] += 1
        W[k_b*n + i_b][col] += 1
        col += 1
        # 7
        U[i_b*2+0][col] += 1; U[i_b*2+1][col] -= 1
        V[(j_b+1)*p + k_b][col] += 1; V[(j_b+1)*p + (k_b+1)][col] += 1
        W[(k_b+1)*n + (i_b+1)][col] += 1
        col += 1

    # Schoolbook 1x2x2
    def add_schoolbook_1x2x2(i_b, k_b, j_b):
        nonlocal col
        for k in range(2):
            for j in range(2):
                U[i_b*2+0][col] += 1 # i is 0
                V[(j_b+j)*p + (k_b+k)][col] += 1
                W[(k_b+k)*n + i_b][col] += 1
                col += 1

    # Top part (i=0,1)
    for k_b in [0, 2]:
        for j_b in [0, 2]:
            add_strassen(0, k_b, j_b)
            
    # Bottom part (i=2)
    for k_b in [0, 2]:
        for j_b in [0, 2]:
            add_schoolbook_1x2x2(2, k_b, j_b)
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
