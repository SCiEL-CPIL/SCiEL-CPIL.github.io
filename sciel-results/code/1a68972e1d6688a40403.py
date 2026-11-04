"""Rewritten program for 4x5x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor entries
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries.append((a, b, c))
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    def strassen2_decomp():
        """2x2x2 Strassen, rank 7"""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # M1=(A11+A22)(B11+B22): C11+=M1, C22+=M1
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        # M2=(A21+A22)B21: C21+=M2
        U[2][1]=1; U[3][1]=1; V[2][1]=1; W[2][1]=1
        # M3=A11(B12-B22): C12+=M3
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1
        # M4=A22(B21-B11): C11+=M4, C21+=M4
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        # M5=(A11+A12)B22: C11-=M5, C12+=M5
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        # M6=(A21-A11)(B11+B12): C22+=M6
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        # M7=(A12-A22)(B21+B22): C11+=M7, C22+=M7
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1; W[3][6]=1
        return U, V, W
    
    def schoolbook_block(nn, mm, pp):
        """Schoolbook decomposition for nn x mm x pp block, rank nn*mm*pp"""
        rank = nn * mm * pp
        U = [[0]*rank for _ in range(nn*mm)]
        V = [[0]*rank for _ in range(mm*pp)]
        W = [[0]*rank for _ in range(pp*nn)]
        col = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i*mm+j][col] = 1
                    V[j*pp+k][col] = 1
                    W[k*nn+i][col] = 1
                    col += 1
        return U, V, W
    
    # Partition: n=4=2+2, m=5=3+2, p=6=3+3
    # 8 blocks:
    # (0,0,0): 2x3x3 -> rank 18
    # (0,0,1): 2x3x3 -> rank 18
    # (0,1,0): 2x2x3 -> rank 12
    # (0,1,1): 2x2x3 -> rank 12
    # (1,0,0): 2x3x3 -> rank 18
    # (1,0,1): 2x3x3 -> rank 18
    # (1,1,0): 2x2x3 -> rank 12
    # (1,1,1): 2x2x3 -> rank 12
    # Total schoolbook: 120
    
    # Better: Use Strassen for the 2x2x3 blocks (blocks with m=2, p=3)
    # A 2x2x3 block can be viewed as two 2x2x1 blocks or one 2x2x2 + one 2x2x1
    # Actually, let's think about this differently.
    # 
    # For a 2x2x3 block: multiply 2x2 by 2x3 to get 2x2
    # We can split the 2x3 into 2x2 + 2x1
    # 2x2x2: Strassen rank 7
    # 2x2x1: rank 4 (each output element needs one scalar mult)
    # But wait, 2x2x1 means multiplying 2x2 by 2x1 to get 2x1
    # That's 2*2*1 = 4 multiplications
    # Total: 7 + 4 = 11 vs schoolbook 12
    
    # Actually, let me reconsider. The block (2,2,3) means:
    # A is 2x2, B is 2x3, C is 3x2 (since output is p x n = 3 x 2)
    # Wait no. Let me re-examine the indexing.
    # 
    # For block (i_block, j_block, k_block) with sizes (ni, mj, pk):
    # A indices: i in [0,ni), j in [0,mj) -> a = i*m + j
    # B indices: j in [0,mj), k in [0,pk) -> b = j*p + k  
    # C indices: k in [0,pk), i in [0,ni) -> c = k*n + i
    # 
    # So for block (2,2,3): ni=2, mj=2, pk=3
    # A: 2x2 (rows 0-1, cols 0-1 within the block)
    # B: 2x3 (rows 0-1, cols 0-2 within the block)
    # C: 3x2 (rows 0-2, cols 0-1 within the block)
    # 
    # This is multiplying a 2x2 matrix by a 2x3 matrix to get a 3x2 result?
    # No wait. Standard matrix multiplication: (ni x mj) * (mj x pk) = (ni x pk)
    # But here C has shape pk x ni, which is the transpose convention.
    # 
    # The tensor entry is: T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    # So C[k,i] = sum_j A[i,j] * B[j,k]
    # This is standard: C = A * B where A is ni x mj, B is mj x pk, C is ni x pk
    # But the output index is c = k*n + i, so C is stored as pk x ni (transposed)
    # 
    # Anyway, for rank purposes, multiplying 2x2 by 2x3:
    # We can decompose B into B[:, :2] (2x2) and B[:, 2:3] (2x1)
    # C[:, :2] = A * B[:, :2]  -> 2x2 * 2x2 = 2x2, Strassen rank 7
    # C[:, 2] = A * B[:, 2]   -> 2x2 * 2x1 = 2x1, rank 4
    # Total: 11 vs schoolbook 12
    # 
    # So for each 2x2x3 block, we can save 1 rank.
    # There are 4 such blocks: (0,1,0), (0,1,1), (1,1,0), (1,1,1)
    # Savings: 4
    # New total: 120 - 4 = 116
    
    # Can we do better for 2x3x3 blocks?
    # 2x3x3: A is 2x3, B is 3x3, C is 3x2
    # Split A into A[:, :2] (2x2) and A[:, 2:3] (2x1)
    # C[:, :2] = A[:, :2] * B[:, :2]  -> 2x2 * 2x2 = 2x2, Strassen rank 7
    # C[:, 2] = A[:, 2] * B[:, 2]    -> 2x1 * 1x2 = 2x2, rank 4
    # Wait, that's not right. Let me think again.
    # 
    # C[k,i] = sum_j A[i,j] * B[j,k]
    # For 2x3x3: i in [0,2), j in [0,3), k in [0,3)
    # 
    # Split j into j in [0,2) and j=2:
    # C[k,i] = sum_{j=0}^{1} A[i,j]*B[j,k] + A[i,2]*B[2,k]
    # 
    # First part: 2x2 * 2x2 = 2x2 (for each k, but B is 2x3, so B[:,:2] is 2x2)
    # Actually B[j,k] for j in [0,2) and k in [0,3) is 2x3
    # So A[:,:2] (2x2) * B[:,:2] (2x3) = 2x3
    # This is 2x2 * 2x3, same as before! Rank 11 with Strassen trick.
    # 
    # Second part: A[:,2] (2x1) * B[2,:] (1x3) = 2x3
    # This is outer product, rank 1 per element... no.
    # C2[k,i] = A[i,2] * B[2,k] for i in [0,2), k in [0,3)
    # This is a 2x3 matrix that's rank 1. To decompose:
    # C2[k,i] = A[i,2] * B[2,k]
    # We need: sum_r U[i,2 offset]*V[2,k offset]*W[k,i offset] = A[i,2]*B[2,k]
    # This can be done with rank 1 if we can set U[i,2]=A[i,2], V[2,k]=B[2,k], but
    # W would need to be 1 for all (k,i), which gives sum = (sum_i A[i,2])*(sum_k B[2,k])
    # That's not right.
    # 
    # Actually, for C2[k,i] = A[i,2] * B[2,k], this is a separable matrix.
    # We can write it as: for each (i,k) pair, one rank-1 term.
    # But that's 2*3 = 6 terms.
    # 
    # Alternatively: C2[k,i] = A[i,2] * B[2,k]
    # Set U[i,2] = A[i,2], V[2,k] = B[2,k], W[k,i] = 1 for all k,i
    # Then sum_r U[i,2]*V[2,k]*W[k,i] = A[i,2]*B[2,k]*1 = A[i,2]*B[2,k] ✓
    # But W[k,i] = 1 for all k,i means W is all ones, which is a single column.
    # So this is rank 1! One term suffices.
    # 
    # Wait, but U, V, W are fixed matrices. We can't set them to depend on A and B.
    # In tensor decomposition, U, V, W are fixed, and the "multiplication" is:
    # C[k,i] = sum_r U[a,b] * V[b,c] * W[c,a] where a=(i,j), b=(j,k), c=(k,i)
    # 
    # For the term A[i,2]*B[2,k]:
    # a = (i, 2), b = (2, k), c = (k, i)
    # U[(i,2), r] * V[(2,k), r] * W[(k,i), r] = A[i,2] * B[2,k]
    # 
    # If we set U[(i,2), r] = 1, V[(2,k), r] = 1, W[(k,i), r] = 1 for a specific r,
    # then the contribution is 1*1*1 = 1, not A[i,2]*B[2,k].
    # 
    # I'm confusing myself. In tensor decomposition for matrix multiplication,
    # the entries U, V, W are constants, and the "algorithm" is:
    # C = sum_r (U[:,r] * A) * (B * V[:,r]) * W[:,r]  (roughly)
    # 
    # More precisely, for each rank-1 term r:
    # U[:,r] defines a linear combination of A entries
    # V[:,r] defines a linear combination of B entries  
    # W[:,r] defines how the product contributes to C
    # 
    # The product for term r is: (linear_comb_A) * (linear_comb_B)
    # and this scalar is added to C entries weighted by W.
    # 
    # So for C2[k,i] = A[i,2] * B[2,k]:
    # We need a term where linear_comb_A = A[i,2] and linear_comb_B = B[2,k]
    # But linear_comb_A is the same for all k, and linear_comb_B is the same for all i.
    # So we can't get A[i,2]*B[2,k] with a single term unless i and k are fixed.
    # 
    # We need 2*3 = 6 separate terms, one for each (i,k) pair.
    # For term (i0, k0): U[(i0,2),r]=1, V[(2,k0),r]=1, W[(k0,i0),r]=1
    # 
    # So 2x3x3 rank = 11 (for 2x2*2x3 part) + 6 (for 2x1*1x3 part) = 17 vs schoolbook 18.
    # Savings: 1 per 2x3x3 block.
    # There are 4 such blocks: (0,0,0), (0,0,1), (1,0,0), (1,0,1)
    # Savings: 4
    # 
    # Total savings: 4 + 4 = 8
    # New total: 120 - 8 = 112
    
    # Can we do even better? Let me try to use Strassen more aggressively.
    # 
    # For 2x2x3: rank 11 (as computed)
    # For 2x3x3: rank 17 (as computed)
    # 
    # Alternative for 2x3x3: Split B into B[:,:2] (3x2) and B[:,2] (3x1)
    # C[:, :2] = A * B[:,:2] -> 2x3 * 3x2 = 2x2, Strassen rank 7
    # C[:, 2] = A * B[:,2] -> 2x3 * 3x1 = 2x1
    # For 2x3 * 3x1: C[i] = sum_j A[i,j]*B[j,2] for i in [0,2)
    # This is 2 scalar multiplications... no, it's 2 dot products of length 3.
    # Rank: each C[i] = sum_j A[i,j]*B[j,2]
    # We can write this as: for each j, a term A[i,j]*B[j,2]
    # But A[i,j] depends on i, B[j,2] depends on j.
    # Term for (i0, j0): U[(i0,j0),r]=1, V[(j0,2),r]=1, W[(2,i0),r]=1
    # We need 2*3 = 6 such terms.
    # Total: 7 + 6 = 13 vs schoolbook 18. Savings: 5!
    # 
    # Wait, that's better. Let me verify.
    # C[k,i] = sum_j A[i,j] * B[j,k]
    # For k in [0,2): C[k,i] = sum_{j=0}^{2} A[i,j]*B[j,k]
    # For k=2: C[2,i] = sum_{j=0}^{2} A[i,j]*B[j,2]
    # 
    # Part 1 (k in [0,2)): A (2x3) * B[:,:2] (3x2) = C[:,:2] (2x2)
    # This is 2x3 * 3x2. Can we use Strassen? Strassen is for 2x2*2x2.
    # We can split A into A[:,:2] (2x2) and A[:,2] (2x1)
    # C[:,:2] = A[:,:2]*B[:,:2] + A[:,2]*B[2,:2]
    # A[:,:2]*B[:,:2]: 2x2*2x2, Strassen rank 7
    # A[:,2]*B[2,:2]: 2x1 * 1x2 = 2x2, rank 4 (one term per (i,k) pair)
    # Total for part 1: 7 + 4 = 11
    # 
    # Part 2 (k=2): C[2,:] = A * B[:,2] = (2x3)*(3x1) = 2x1
    # C[2,i] = sum_j A[i,j]*B[j,2]
    # Rank 6 (one term per (i,j) pair)
    # 
    # Total: 11 + 6 = 17. Same as before.
    # 
    # Hmm, so 17 is the best I can get for 2x3x3 with this approach.
    # 
    # Let me try yet another split for 2x3x3.
    # Split A into A[0,:] (1x3) and A[1,:] (1x3)
    # C[k,0] = A[0,:]*B[:,k] = sum_j A[0,j]*B[j,k] for k in [0,3)
    # C[k,1] = A[1,:]*B[:,k] = sum_j A[1,j]*B[j,k] for k in [0,3)
    # Each is a 1x3 * 3x3 = 1x3, which is 3 dot products of length 3.
    # Rank per row: 3*3 = 9? No.
    # C[k,0] = sum_j A[0,j]*B[j,k]
    # For fixed i=0: 3 terms (one per j) for each k... no.
    # For each (j,k) pair: U[(0,j),r]=1, V[(j,k),r]=1, W[(k,0),r]=1
    # That's 3*3 = 9 terms for i=0.
    # Similarly 9 terms for i=1.
    # Total: 18 = schoolbook. No savings.
    # 
    # OK so 17 for 2x3x3 and 11 for 2x2x3 seems to be the best with this approach.
    # Total: 4*17 + 4*11 = 68 + 44 = 112.
    
    def build_strassen_hybrid():
        """Build decomposition using Strassen where beneficial.
        Partition: n=2+2, m=3+2, p=3+3
        2x3x3 blocks: rank 17 each (4 blocks)
        2x2x3 blocks: rank 11 each (4 blocks)
        Total: 112
        """
        total_rank = 4*17 + 4*11  # 112
        U_full = [[0] * total_rank for _ in range(nm)]
        V_full = [[0] * total_rank for _ in range(mp)]
        W_full = [[0] * total_rank for _ in range(pn)]
        
        col = 0
        
        # Helper: add a 2x3x3 block at (ni_offset, mj_offset, pk_offset)
        def add_2x3x3(ni_off, mj_off, pk_off):
            nonlocal col
            # A: ni=2, mj=3; B: mj=3, pk=3; C: pk=3, ni=2
            # Part 1: A[:,:2] (2x2) * B[:,:2] (3x2) -> C[:,:2] (2x2)
            # Use Strassen for 2x2*2x2
            sU, sV, sW = strassen2_decomp()
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U_full[(ni_off+i)*m + (mj_off+j)][col] += sU[i*2+j][r]
                for j in range(2):
                    for k in range(2):
                        V_full[(mj_off+j)*p + (pk_off+k)][col] += sV[j*2+k][r]
                for k in range(2):
                    for i in range(2):
                        W_full[(pk_off+k)*n + (ni_off+i)][col] += sW[k*2+i][r]
                col += 1
            # Part 1b: A[:,2] (2x1) * B[2,:2] (1x2) -> C[:,:2] (2x2)
            # C[k,i] += A[i,2] * B[2,k] for k in [0,2), i in [0,2)
            for i in range(2):
                for k in range(2):
                    U_full[(ni_off+i)*m + (mj_off+2)][col] = 1
                    V_full[(mj_off+2)*p + (pk_off+k)][col] = 1
                    W_full[(pk_off+k)*n + (ni_off+i)][col] = 1
                    col += 1
            # Part 2: A (2x3) * B[:,2] (3x1) -> C[:,2] (2x1)
            # C[2,i] = sum_j A[i,j]*B[j,2] for i in [0,2)
            for i in range(2):
                for j in range(3):
                    U_full[(ni_off+i)*m + (mj_off+j)][col] = 1
                    V_full[(mj_off+j)*p + (pk_off+2)][col] = 1
                    W_full[(pk_off+2)*n + (ni_off+i)][col] = 1
                    col += 1
        
        # Helper: add a 2x2x3 block at (ni_offset, mj_offset, pk_offset)
        def add_2x2x3(ni_off, mj_off, pk_off):
            nonlocal col
            # A: ni=2, mj=2; B: mj=2, pk=3; C: pk=3, ni=2
            # Part 1: A (2x2) * B[:,:2] (2x2) -> C[:,:2] (2x2)
            # Use Strassen for 2x2*2x2
            sU, sV, sW = strassen2_decomp()
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U_full[(ni_off+i)*m + (mj_off+j)][col] += sU[i*2+j][r]
                for j in range(2):
                    for k in range(2):
                        V_full[(mj_off+j)*p + (pk_off+k)][col] += sV[j*2+k][r]
                for k in range(2):
                    for i in range(2):
                        W_full[(pk_off+k)*n + (ni_off+i)][col] += sW[k*2+i][r]
                col += 1
            # Part 2: A (2x2) * B[:,2] (2x1) -> C[:,2] (2x1)
            # C[2,i] = sum_j A[i,j]*B[j,2] for i in [0,2)
            for i in range(2):
                for j in range(2):
                    U_full[(ni_off+i)*m + (mj_off+j)][col] = 1
                    V_full[(mj_off+j)*p + (pk_off+2)][col] = 1
                    W_full[(pk_off+2)*n + (ni_off+i)][col] = 1
                    col += 1
        
        # Add all 8 blocks
        # (0,0,0): 2x3x3, ni_off=0, mj_off=0, pk_off=0
        add_2x3x3(0, 0, 0)
        # (0,0,1): 2x3x3, ni_off=0, mj_off=0, pk_off=3
        add_2x3x3(0, 0, 3)
        # (0,1,0): 2x2x3, ni_off=0, mj_off=3, pk_off=0
        add_2x2x3(0, 3, 0)
        # (0,1,1): 2x2x3, ni_off=0, mj_off=3, pk_off=3
        add_2x2x3(0, 3, 3)
        # (1,0,0): 2x3x3, ni_off=2, mj_off=0, pk_off=0
        add_2x3x3(2, 0, 0)
        # (1,0,1): 2x3x3, ni_off=2, mj_off=0, pk_off=3
        add_2x3x3(2, 0, 3)
        # (1,1,0): 2x2x3, ni_off=2, mj_off=3, pk_off=0
        add_2x2x3(2, 3, 0)
        # (1,1,1): 2x2x3, ni_off=2, mj_off=3, pk_off=3
        add_2x2x3(2, 3, 3)
        
        # Verify
        if col != total_rank:
            return None
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(total_rank):
                        s += U_full[i*m+j][r] * V_full[j*p+k][r] * W_full[k*n+i][r]
                    if abs(s - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            return {"U": U_full, "V": V_full, "W": W_full}
        return None
    
    # Try the Strassen hybrid
    result = build_strassen_hybrid()
    if result is not None:
        best_result = result
        best_rank = 112
    
    # Try to reduce further by exploiting linear dependencies
    # In the 2x2x3 blocks, the "Part 2" (A * B[:,2]) has 4 terms.
    # Can any of these be combined with terms from other blocks?
    # 
    # For block (0,1,0): Part 2 terms are for i in [0,2), j in [0,2)
    #   U[(i)*5 + (3+j)][r] = 1, V[(3+j)*6 + k][r] = 1, W[k*4 + i][r] = 1
    #   for k=2 (the third column of B)
    # 
    # For block (1,1,0): Part 2 terms are for i in [0,2), j in [0,2)
    #   U[(2+i)*5 + (3+j)][r] = 1, V[(3+j)*6 + k][r] = 1, W[k*4 + (2+i)][r] = 1
    #   for k=2
    # 
    # These use different U and W indices, so they can't be directly combined.
    # 
    # What about combining across the p dimension?
    # Block (0,1,0) Part 2: V[(3+j)*6 + 0:3][r] = 1 for k in [0,2), and V[(3+j)*6+2][r]=1
    # Block (0,1,1) Part 2: V[(3+j)*6 + 3:6][r] = 1 for k in [3,5), and V[(3+j)*6+5][r]=1
    # Different V indices, so no direct combination.
    
    # Try a randomized search for rank 111 or lower
    import random
    random.seed(42)
    
    def try_reduced_rank(target_rank):
        """Try to find a decomposition with target_rank by random search."""
        if time.time() > deadline - 30:
            return None
        
        U_full = [[0] * target_rank for _ in range(nm)]
        V_full = [[0] * target_rank for _ in range(mp)]
        W_full = [[0] * target_rank for _ in range(pn)]
        
        col = 0
        # Start with the 112-rank decomposition and try to merge columns
        # by introducing linear dependencies
        
        # Build the 112-rank decomposition first
        temp_U = [[0] * 112 for _ in range(nm)]
        temp_V = [[0] * 112 for _ in range(mp)]
        temp_W = [[0] * 112 for _ in range(pn)]
        
        col2 = 0
        def add_2x3x3_temp(ni_off, mj_off, pk_off):
            nonlocal col2
            sU, sV, sW = strassen2_decomp()
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        temp_U[(ni_off+i)*m + (mj_off+j)][col2] += sU[i*2+j][r]
                for j in range(2):
                    for k in range(2):
                        temp_V[(mj_off+j)*p + (pk_off+k)][col2] += sV[j*2+k][r]
                for k in range(2):
                    for i in range(2):
                        temp_W[(pk_off+k)*n + (ni_off+i)][col2] += sW[k*2+i][r]
                col2 += 1
            for i in range(2):
                for k in range(2):
                    temp_U[(ni_off+i)*m + (mj_off+2)][col2] = 1
                    temp_V[(mj_off+2)*p + (pk_off+k)][col2] = 1
                    temp_W[(pk_off+k)*n + (ni_off+i)][col2] = 1
                    col2 += 1
            for i in range(2):
                for j in range(3):
                    temp_U[(ni_off+i)*m + (mj_off+j)][col2] = 1
                    temp_V[(mj_off+j)*p + (pk_off+2)][col2] = 1
                    temp_W[(pk_off+2)*n + (ni_off+i)][col2] = 1
                    col2 += 1
        
        def add_2x2x3_temp(ni_off, mj_off, pk_off):
            nonlocal col2
            sU, sV, sW = strassen2_decomp()
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        temp_U[(ni_off+i)*m + (mj_off+j)][col2] += sU[i*2+j][r]
                for j in range(2):
                    for k in range(2):
                        temp_V[(mj_off+j)*p + (pk_off+k)][col2] += sV[j*2+k][r]
                for k in range(2):
                    for i in range(2):
                        temp_W[(pk_off+k)*n + (ni_off+i)][col2] += sW[k*2+i][r]
                col2 += 1
            for i in range(2):
                for j in range(2):
                    temp_U[(ni_off+i)*m + (mj_off+j)][col2] = 1
                    temp_V[(mj_off+j)*p + (pk_off+2)][col2] = 1
                    temp_W[(pk_off+2)*n + (ni_off+i)][col2] = 1
                    col2 += 1
        
        add_2x3x3_temp(0, 0, 0)
        add_2x3x3_temp(0, 0, 3)
        add_2x2x3_temp(0, 3, 0)
        add_2x2x3_temp(0, 3, 3)
        add_2x3x3_temp(2, 0, 0)
        add_2x3x3_temp(2, 0, 3)
        add_2x2x3_temp(2, 3, 0)
        add_2x2x3_temp(2, 3, 3)
        
        # Now try to reduce by merging columns
        # For each pair of columns, check if one can be expressed as a linear combination
        # of the other plus a "null" vector (one that contributes 0 to all tensor entries)
        
        # Simpler approach: try to find a column that is a linear combination of others
        # such that the tensor equation is still satisfied
        
        # This is complex, so let's just try a few random combinations
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            
            # Pick a random column to try to eliminate
            elim_col = random.randint(0, 111)
            
            # Try to express column elim_col as a linear combination of other columns
            # with small integer/half-integer coefficients
            # This is a hard problem, so let's try a heuristic:
            # Check if column elim_col is already a linear combination of others
            
            # For now, just copy the 112-rank solution and remove one column
            # if it's linearly dependent
            
            # This is getting too complex. Let's just return the 112-rank solution.
            pass
        
        return None
    
    # For now, return the best result we have
    return best_result
# EVOLVE-BLOCK-END
