import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(actual - expected) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # safety margin
    
    # Block decomposition: partition {0,1,2,3,4} into {0,1}, {2,3}, {4}
    # Block sizes: 2, 2, 1
    block_sizes = [2, 2, 1]
    num_blocks = len(block_sizes)
    
    # We'll build U, V, W by concatenating block decompositions
    U_blocks = []  # list of (rows, rank) matrices
    V_blocks = []
    W_blocks = []
    
    total_rank = 0
    
    for alpha in range(num_blocks):
        for beta in range(num_blocks):
            for gamma in range(num_blocks):
                a = block_sizes[alpha]  # rows of A block
                b = block_sizes[beta]   # cols of A block / rows of B block
                c = block_sizes[gamma]  # cols of B block
                
                if a * b * c <= 8:
                    # Use Strassen for 2x2x2
                    if a == 2 and b == 2 and c == 2:
                        # Strassen decomposition
                        # M1 = (A11+A22)(B11+B22)
                        # M2 = (A21+A22)B11
                        # M3 = A11(B12-B22)
                        # M4 = A22(B21-B11)
                        # M5 = (A11+A12)B22
                        # M6 = (A21-A11)(B11+B12)
                        # M7 = (A12-A22)(B21+B22)
                        # C11 = M1 + M4 - M5 + M7
                        # C12 = M3 + M5
                        # C21 = M2 + M4
                        # C22 = M1 - M2 + M3 + M6
                        
                        # Each M is a product of a 2x2 "left" and 2x2 "right"
                        # We need to express each M as a sum of rank-1 terms
                        # M1 = (A11+A22)(B11+B22)
                        #   = A11*B11 + A11*B22 + A22*B11 + A22*B22
                        #   But we want to use the Strassen trick: compute (A11+A22) as one matrix
                        #   and (B11+B22) as one matrix, then multiply them
                        #   The product of two 2x2 matrices has rank at most 4 (schoolbook)
                        #   So M1 contributes rank 4
                        #   Wait, but Strassen says the whole 2x2x2 has rank 7, not 7*4=28
                        #   
                        # I'm confusing myself. Let me think again.
                        # 
                        # In Strassen's algorithm, we compute 7 products of 2x2 matrices.
                        # Each product of two 2x2 matrices is itself a 2x2x2 tensor of rank at most 4.
                        # So the total rank would be 7 * 4 = 28? No, that's not right either.
                        #
                        # Actually, in the tensor decomposition framework:
                        # The 2x2x2 multiplication tensor has rank 7.
                        # This means we can write T[i,j,k] = sum_{r=1}^{7} U[i,j,r] * V[j,k,r] * W[k,i,r]
                        # where U is 4x7, V is 4x7, W is 4x7.
                        #
                        # The standard Strassen decomposition gives explicit U, V, W matrices.
                        # Let me look up the exact form.
                        
                        # For 2x2 matrix multiplication:
                        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
                        # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
                        #
                        # Strassen:
                        # M1 = (a+d)(e+h)
                        # M2 = (c+d)e
                        # M3 = a(f-h)
                        # M4 = d(g-c)
                        # M5 = (a+b)h
                        # M6 = (c-a)(e+f)
                        # M7 = (b-d)(g+h)
                        #
                        # C11 = M1 + M4 - M5 + M7
                        # C12 = M3 + M5
                        # C21 = M2 + M4
                        # C22 = M1 - M2 + M3 + M6
                        #
                        # In tensor form, T[i,j,k] = 1 if j==k for 2x2 (indices 0,1)
                        # Wait no, T[i,j,k] for 2x2 matrix mult:
                        # C[i,k] = sum_j A[i,j] * B[j,k]
                        # So the tensor is T[i,j,k] = 1 (it's always 1 for valid indices)
                        # No wait, the tensor T[i,j,k] represents the coefficient:
                        # C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
                        # So T[i,j,k] = 1 for all i,j,k in {0,1}
                        # 
                        # Hmm, that's the multiplication tensor. Its rank is 7 (Strassen).
                        #
                        # The U, V, W matrices for Strassen:
                        # U[i*m+j, r] corresponds to the coefficient of A[i,j] in the r-th product
                        # V[j*p+k, r] corresponds to the coefficient of B[j,k] in the r-th product
                        # W[k*n+i, r] corresponds to the coefficient in C[k,i]... wait
                        #
                        # Let me use the convention from the problem:
                        # W[k*n+i, r] - so W is indexed by (k, i) where k is col of B / row of C, i is row of A / col of C
                        # Actually re-reading: "the output factor follows the official trace convention W[k*n+i,r]"
                        # And U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
                        # So U[i*m+j, r] for i in [0,n), j in [0,m)
                        # V[j*p+k, r] for j in [0,m), k in [0,p)
                        # W[k*n+i, r] for k in [0,p), i in [0,n)
                        #
                        # And the decomposition is: C[k,i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        # Wait, that doesn't look right dimensionally.
                        #
                        # Let me re-read the verify function:
                        # actual = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                        # This should equal 1.0 for all i, j, k
                        #
                        # So the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # for all i in [0,n), j in [0,m), k in [0,p)
                        #
                        # This is the multiplication tensor where T[i,j,k] = 1 for all indices.
                        # Wait, that can't be right for matrix multiplication.
                        #
                        # Oh I see, for matrix multiplication C = A * B:
                        # C[i,k] = sum_j A[i,j] * B[j,k]
                        # So the tensor is T[i,j,k] = 1 (the coefficient is always 1)
                        # And the decomposition T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                        # where W is indexed by (k,i) not (k*n+i)... 
                        #
                        # Wait, in the verify: W[k*n+i, r]
                        # So W has p*n rows, indexed by k*n+i where k in [0,p), i in [0,n)
                        # This means W[k,i,r] in mathematical notation
                        #
                        # And the formula is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # for all i in [0,n), j in [0,m), k in [0,p)
                        #
                        # So this is indeed the "all ones" tensor, which is the multiplication tensor
                        # for the case where we're just computing the sum.
                        #
                        # OK so for 2x2x2 (n=m=p=2):
                        # T[i,j,k] = 1 for all i,j,k in {0,1}
                        # Rank is 7 (Strassen)
                        
                        # Strassen U, V, W for 2x2x2:
                        # U[i*2+j, r] for i,j in {0,1}, r in {0,...,6}
                        # V[j*2+k, r] for j,k in {0,1}, r in {0,...,6}
                        # W[k*2+i, r] for k,i in {0,1}, r in {0,...,6}
                        
                        # Let me derive from Strassen:
                        # M1 = (A00+A11)(B00+B11)
                        #   U: A00 coeff=1, A11 coeff=1 => U[0,0]=1, U[3,0]=1 (i=0,j=0 and i=1,j=1)
                        #   V: B00 coeff=1, B11 coeff=1 => V[0,0]=1, V[3,0]=1 (j=0,k=0 and j=1,k=1)
                        #   W: contributes to C00 and C11
                        #   C00 += M1, C11 += M1
                        #   W[k*2+i]: C00 is k=0,i=0 => W[0,0]=1; C11 is k=1,i=1 => W[3,0]=1
                        
                        # M2 = (A10+A11)B00
                        #   U: A10 coeff=1, A11 coeff=1 => U[2,1]=1, U[3,1]=1
                        #   V: B00 coeff=1 => V[0,1]=1
                        #   W: C10 += M2 => k=0,i=1 => W[2,1]=1; C11 += M2... 
                        #   Wait, C11 = M1 - M2 + M3 + M6
                        #   So C10 += M2, and C11 -= M2
                        #   W[2,1] = 1 (for C10), W[3,1] = -1 (for C11)
                        
                        # This is getting complicated. Let me just hardcode the known Strassen decomposition.
                        
                        # Known Strassen decomposition for 2x2x2:
                        # Using the standard formulation where:
                        # U rows are (i,j) pairs: (0,0),(0,1),(1,0),(1,1)
                        # V rows are (j,k) pairs: (0,0),(0,1),(1,0),(1,1)  
                        # W rows are (k,i) pairs: (0,0),(0,1),(1,0),(1,1)
                        
                        U_strassen = [
                            [1,  1,  1,  0,  1,  0,  0],  # A00
                            [0,  0,  1,  0,  1,  0,  1],  # A01
                            [0,  1,  0,  1,  0,  1,  0],  # A10
                            [1,  1,  0,  1,  0,  0,  1],  # A11
                        ]
                        V_strassen = [
                            [1,  1,  0,  1,  0,  1,  0],  # B00
                            [0,  0,  1,  0,  1,  1,  0],  # B01
                            [0,  0,  0,  1,  0,  0,  1],  # B10
                            [1,  0,  1,  1,  1,  0,  1],  # B11
                        ]
                        W_strassen = [
                            [1,  0,  0,  1,  0,  0,  0],  # C00
                            [0,  0,  0,  0,  1,  0,  0],  # C01
                            [0,  1,  0,  0,  0,  0,  0],  # C10
                            [1,  0,  0,  0,  0,  1,  1],  # C11
                        ]
                        
                        # Verify: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should = 1 for all i,j,k
                        # Let me check (0,0,0): U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[0,1] + ... 
                        # = 1*1*1 + 1*0*0 + 1*0*0 + 0*1*0 + 1*1*0 + 0*1*0 + 0*0*0 = 1 ✓
                        
                        # Check (0,0,1): U[0,0]*V[0,1]*W[1,0] + U[0,1]*V[0,2]*W[1,1] + ...
                        # Wait, W is indexed by (k,i), so W[1,0] means k=1,i=0 => row k*n+i = 1*2+0 = 2
                        # I'm getting confused with indexing. Let me just verify numerically.
                        
                        U_s = np.array(U_strassen, dtype=float)
                        V_s = np.array(V_strassen, dtype=float)
                        W_s = np.array(W_strassen, dtype=float)
                        
                        for ii in range(2):
                            for jj in range(2):
                                for kk in range(2):
                                    val = sum(U_s[ii*2+jj, r] * V_s[jj*2+kk, r] * W_s[kk*2+ii, r] for r in range(7))
                                    if abs(val - 1.0) > 1e-9:
                                        # Fallback: use schoolbook for this block
                                        U_s = None
                                        break
                                if U_s is None:
                                    break
                            if U_s is None:
                                break
                        if U_s is None:
                            # Schoolbook 2x2x2
                            U_s = np.zeros((4, 8))
                            V_s = np.zeros((4, 8))
                            W_s = np.zeros((4, 8))
                            for ii in range(2):
                                for jj in range(2):
                                    for kk in range(2):
                                        col = ii*4 + jj*2 + kk
                                        U_s[ii*2+jj, col] = 1
                                        V_s[jj*2+kk, col] = 1
                                        W_s[kk*2+ii, col] = 1
                            rank_block = 8
                        else:
                            rank_block = 7
                    else:
                        # Schoolbook for a*b*c
                        rank_block = a * b * c
                        U_s = np.zeros((a*b, rank_block))
                        V_s = np.zeros((b*c, rank_block))
                        W_s = np.zeros((c*a, rank_block))
                        for ii in range(a):
                            for jj in range(b):
                                for kk in range(c):
                                    col = ii*b*c + jj*c + kk
                                    U_s[ii*b+jj, col] = 1
                                    V_s[jj*c+kk, col] = 1
                                    W_s[kk*a+ii, col] = 1
                else:
                    # Shouldn't happen with our block sizes
                    rank_block = a * b * c
                    U_s = np.zeros((a*b, rank_block))
                    V_s = np.zeros((b*c, rank_block))
                    W_s = np.zeros((c*a, rank_block))
                    for ii in range(a):
                        for jj in range(b):
                            for kk in range(c):
                                col = ii*b*c + jj*c + kk
                                U_s[ii*b+jj, col] = 1
                                V_s[jj*c+kk, col] = 1
                                W_s[kk*a+ii, col] = 1
                
                U_blocks.append(U_s)
                V_blocks.append(V_s)
                W_blocks.append(W_s)
                total_rank += rank_block
    
    # Now we need to assemble the full U, V, W matrices
    # The full U has n*m = 25 rows
    # Row i*m+j where i is the global row index, j is the global col index
    # We need to map global (i,j) to local (alpha, ii) and (beta, jj)
    
    # Block offsets
    block_offsets = [0]
    for s in block_sizes[:-1]:
        block_offsets.append(block_offsets[-1] + s)
    
    # Build full U (25 x total_rank)
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    rank_offset = 0
    for alpha in range(num_blocks):
        for beta in range(num_blocks):
            for gamma in range(num_blocks):
                a = block_sizes[alpha]
                b = block_sizes[beta]
                c = block_sizes[gamma]
                
                U_s = U_blocks[alpha * num_blocks * num_blocks + beta * num_blocks + gamma]
                V_s = V_blocks[alpha * num_blocks * num_blocks + beta * num_blocks + gamma]
                W_s = W_blocks[alpha * num_blocks * num_blocks + beta * num_blocks + gamma]
                
                # Map local indices to global
                # U local row ii*b+jj -> global row (block_offsets[alpha]+ii)*m + (block_offsets[beta]+jj)
                # V local row jj*c+kk -> global row (block_offsets[beta]+jj)*p + (block_offsets[gamma]+kk)
                # W local row kk*a+ii -> global row (block_offsets[gamma]+kk)*n + (block_offsets[alpha]+ii)
                
                for local_row in range(a * b):
                    ii = local_row // b
                    jj = local_row % b
                    global_row = (block_offsets[alpha] + ii) * m + (block_offsets[beta] + jj)
                    U_full[global_row, rank_offset:rank_offset + U_s.shape[1]] = U_s[local_row, :]
                
                for local_row in range(b * c):
                    jj = local_row // c
                    kk = local_row % c
                    global_row = (block_offsets[beta] + jj) * p + (block_offsets[gamma] + kk)
                    V_full[global_row, rank_offset:rank_offset + V_s.shape[1]] = V_s[local_row, :]
                
                for local_row in range(c * a):
                    kk = local_row // a
                    ii = local_row % a
                    global_row = (block_offsets[gamma] + kk) * n + (block_offsets[alpha] + ii)
                    W_full[global_row, rank_offset:rank_offset + W_s.shape[1]] = W_s[local_row, :]
                
                rank_offset += U_s.shape[1]
    
    # Convert to lists
    result = {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist()
    }
    
    # Verify
    if _verify_decomposition(U_full, V_full, W_full, n, m, p):
        return result
    
    # Fallback to schoolbook
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
