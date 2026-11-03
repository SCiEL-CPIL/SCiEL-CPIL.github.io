DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    
    # We are multiplying an (n x m) matrix A by an (m x p) matrix B to get (n x p) C.
    # The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    # T[i,j,k] = 1 if we are computing C[i,k] = sum_j A[i,j] B[j,k].
    # The decomposition is sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] (note W index order per contract: W[k*n+i, r])
    # Wait, the contract says:
    # U: (n*m, rank), index i*m+j
    # V: (m*p, rank), index j*p+k
    # W: (p*n, rank), index k*n+i
    # And the tensor entry T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This corresponds to C[i,k] = sum_j (sum_r U[i,j,r] V[j,k,r]) B[j,k] ... no.
    # Standard tensor decomposition for matrix multiplication:
    # C = A * B
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as sum_r (sum_j A[i,j] U[j,r]) (sum_k B[k,r'] ... ) No.
    # The bilinear form is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # A rank-r decomposition means:
    # C[i,k] = sum_{r=1}^R ( sum_j A[i,j] u_{j,r} ) ( sum_k v_{k,r} B[j,k] ... ) 
    # Actually, the standard form is:
    # C[i,k] = sum_{r=1}^R alpha_{i,r} beta_{k,r} gamma_{j,r} ... no.
    # Let's look at the indices in the contract.
    # U has shape (n*m, R). Let's say U[i*m+j, r].
    # V has shape (m*p, R). Let's say V[j*p+k, r].
    # W has shape (p*n, R). Let's say W[k*n+i, r].
    # The product is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # This sums over r. The result is a tensor T[i,j,k].
    # For this to equal the matrix multiplication tensor, we need:
    # T[i,j,k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # The matrix multiplication tensor T_MM[i,j,k] is 1 for all valid i,j,k.
    # So we need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    
    # This is equivalent to decomposing the identity map on the tensor product space.
    # We can use a known decomposition for 5x5 matrix multiplication.
    # The best known rank for 5x5 is 19 (Landsberg 2006).
    # However, implementing the specific rank-19 decomposition is complex.
    # A simpler approach: use the rank-25 schoolbook decomposition.
    # Can we do better?
    # We can use a block decomposition.
    # 5 = 2 + 2 + 1.
    # We can use Strassen for the 2x2 blocks and schoolbook for the rest.
    # Let's partition the indices into blocks of size 2 and 1.
    # i in {0,1,2,3,4}, j in {0,1,2,3,4}, k in {0,1,2,3,4}.
    # Blocks: (0,1), (2,3), (4).
    # There are 3x3x3 = 27 blocks.
    # Each block is of size s_i x s_j x s_k.
    # We can decompose each block independently and sum the ranks.
    # For a 2x2x2 block, we can use Strassen (rank 7).
    # For a 2x2x1 block, it's a 2x2 matrix times a 1x2 vector? No, it's a 2x2 submatrix of A, 2x1 of B?
    # Let's be precise.
    # A is n x m. B is m x p.
    # Block A[i_block, j_block] is s_i x s_j.
    # Block B[j_block, k_block] is s_j x s_k.
    # Block C[i_block, k_block] is s_i x s_k.
    # The multiplication is C_block = A_block * B_block.
    # This is a matrix multiplication of size s_i x s_k with intermediate s_j.
    # The tensor rank of this block is at most s_i * s_j * s_k (schoolbook).
    # For 2x2x2, rank 7 (Strassen).
    # For 2x2x1, it's a 2x2 matrix times a 2x1 vector. Rank is 4? Or can we do better?
    # A 2x2 matrix A and 2x1 vector v. C = A v. C is 2x1.
    # C[0,0] = A[0,0]v[0,0] + A[0,1]v[1,0]
    # C[1,0] = A[1,0]v[0,0] + A[1,1]v[1,0]
    # This is a 2x2x1 tensor. The rank is 4 (schoolbook). Can it be lower?
    # The tensor is T[i,j,k] where i in {0,1}, j in {0,1}, k={0}.
    # T[i,j,0] = 1.
    # We need sum_r U[i,j,r] V[j,0,r] W[0,i,r] = 1.
    # This is equivalent to decomposing the identity map from R^2 (j) to R^2 (i) via R^2 (j)? No.
    # It's a bilinear map R^2 x R^2 -> R^2 (i).
    # The rank is 4.
    # For 2x1x2, rank 4.
    # For 1x2x2, rank 4.
    # For 1x1x1, rank 1.
    # For 1x2x1, rank 2.
    # For 2x1x1, rank 2.
    # For 1x1x2, rank 2.
    
    # Let's sum the ranks for the 3x3x3 block structure.
    # Block sizes: 2, 2, 1.
    # Number of 2x2x2 blocks: 1. Rank 7.
    # Number of 2x2x1 blocks: 1 (i=2, j=2, k=1). Rank 4.
    # Number of 2x1x2 blocks: 1 (i=2, j=1, k=2). Rank 4.
    # Number of 1x2x2 blocks: 1 (i=1, j=2, k=2). Rank 4.
    # Number of 2x2x0? No, k=0 is size 1.
    # Let's list all 27 blocks:
    # i_block: 0(size 2), 1(size 2), 2(size 1)
    # j_block: 0(size 2), 1(size 2), 2(size 1)
    # k_block: 0(size 2), 1(size 2), 2(size 1)
    
    # Ranks:
    # (2,2,2): 7
    # (2,2,1): 4
    # (2,1,2): 4
    # (1,2,2): 4
    # (2,1,1): 2
    # (1,2,1): 2
    # (1,1,2): 2
    # (2,1,0)? No.
    # (1,1,1): 1
    
    # Count:
    # (2,2,2): 1 block. Rank 7.
    # (2,2,1): 1 block (i=0,j=0,k=2). Rank 4.
    # (2,2,1): 1 block (i=0,j=1,k=2). Rank 4.
    # (2,2,1): 1 block (i=1,j=0,k=2). Rank 4.
    # (2,2,1): 1 block (i=1,j=1,k=2). Rank 4.
    # Total (2,2,1): 4 blocks. Rank 16.
    
    # (2,1,2): 1 block (i=0,j=2,k=0). Rank 4.
    # (2,1,2): 1 block (i=0,j=2,k=1). Rank 4.
    # (2,1,2): 1 block (i=1,j=2,k=0). Rank 4.
    # (2,1,2): 1 block (i=1,j=2,k=1). Rank 4.
    # Total (2,1,2): 4 blocks. Rank 16.
    
    # (1,2,2): 1 block (i=2,j=0,k=0). Rank 4.
    # (1,2,2): 1 block (i=2,j=0,k=1). Rank 4.
    # (1,2,2): 1 block (i=2,j=1,k=0). Rank 4.
    # (1,2,2): 1 block (i=2,j=1,k=1). Rank 4.
    # Total (1,2,2): 4 blocks. Rank 16.
    
    # (2,1,1): 1 block (i=0,j=2,k=2). Rank 2.
    # (2,1,1): 1 block (i=1,j=2,k=2). Rank 2.
    # Total (2,1,1): 2 blocks. Rank 4.
    
    # (1,2,1): 1 block (i=2,j=0,k=2). Rank 2.
    # (1,2,1): 1 block (i=2,j=1,k=2). Rank 2.
    # Total (1,2,1): 2 blocks. Rank 4.
    
    # (1,1,2): 1 block (i=2,j=2,k=0). Rank 2.
    # (1,1,2): 1 block (i=2,j=2,k=1). Rank 2.
    # Total (1,1,2): 2 blocks. Rank 4.
    
    # (2,1,0)? No, k=0 is size 2.
    # (1,1,1): 1 block (i=2,j=2,k=2). Rank 1.
    
    # Total rank = 7 + 16 + 16 + 16 + 4 + 4 + 4 + 1 = 68.
    # This is better than 75.
    
    # Let's verify the block decomposition.
    # We need to construct U, V, W for each block and then assemble them.
    # For each block (ib, jb, kb) with sizes (si, sj, sk), we have a local decomposition of rank R_block.
    # The global U, V, W will have rank sum(R_block).
    # We need to map the local indices to global indices.
    
    # Block definitions:
    # i_blocks: [(0, 2), (2, 2), (4, 1)] -> start, size
    # j_blocks: [(0, 2), (2, 2), (4, 1)]
    # k_blocks: [(0, 2), (2, 2), (4, 1)]
    
    i_blocks = [(0, 2), (2, 2), (4, 1)]
    j_blocks = [(0, 2), (2, 2), (4, 1)]
    k_blocks = [(0, 2), (2, 2), (4, 1)]
    
    # We need a function to get the decomposition for a given block size (si, sj, sk).
    # For (2,2,2), use Strassen.
    # For others, use schoolbook.
    
    def get_decomp(si, sj, sk):
        """
        Returns U, V, W for a si x sj x sk block.
        U: (si*sj, R)
        V: (sj*sk, R)
        W: (sk*si, R)
        """
        if si == 2 and sj == 2 and sk == 2:
            # Strassen decomposition
            # C = A B
            # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
            # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
            # Strassen:
            # m1 = (a+d)(e+h)
            # m2 = (c+d)e
            # m3 = a(f-h)
            # m4 = d(g-e)
            # m5 = (a+b)h
            # m6 = (c-d)(f+g)
            # m7 = (b-d)(e+g)
            # C11 = m1 - m4 + m5 + m7
            # C12 = m3 + m5
            # C21 = m2 + m4
            # C22 = m1 - m3 + m2 - m6
            
            # We need U, V, W such that sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
            # Note the index order in W: W[k*n+i, r]. Here local indices are k, i.
            # So W_local[k, i, r].
            
            # Let's define the 7 terms.
            # Term 1: m1 = (a+d)(e+h)
            # U: a=1, d=1 -> U[0,0]=1, U[1,1]=1
            # V: e=1, h=1 -> V[0,0]=1, V[1,1]=1
            # W: contributes to C11 (+1), C22 (+1) -> W[0,0]=1, W[1,1]=1
            
            # Term 2: m2 = (c+d)e
            # U: c=1, d=1 -> U[1,0]=1, U[1,1]=1
            # V: e=1 -> V[0,0]=1
            # W: contributes to C21 (+1) -> W[0,1]=1
            
            # Term 3: m3 = a(f-h)
            # U: a=1 -> U[0,0]=1
            # V: f=1, h=-1 -> V[0,1]=1, V[1,1]=-1
            # W: contributes to C12 (+1), C22 (-1) -> W[1,0]=1, W[1,1]=-1
            
            # Term 4: m4 = d(g-e)
            # U: d=1 -> U[1,1]=1
            # V: g=1, e=-1 -> V[1,0]=1, V[0,0]=-1
            # W: contributes to C11 (-1), C21 (+1) -> W[0,0]=-1, W[0,1]=1
            
            # Term 5: m5 = (a+b)h
            # U: a=1, b=1 -> U[0,0]=1, U[0,1]=1
            # V: h=1 -> V[1,1]=1
            # W: contributes to C11 (+1), C12 (+1) -> W[0,0]=1, W[1,0]=1
            
            # Term 6: m6 = (c-d)(f+g)
            # U: c=1, d=-1 -> U[1,0]=1, U[1,1]=-1
            # V: f=1, g=1 -> V[0,1]=1, V[1,0]=1
            # W: contributes to C22 (-1) -> W[1,1]=-1
            
            # Term 7: m7 = (b-d)(e+g)
            # U: b=1, d=-1 -> U[0,1]=1, U[1,1]=-1
            # V: e=1, g=1 -> V[0,0]=1, V[1,0]=1
            # W: contributes to C11 (+1) -> W[0,0]=1
            
            R = 7
            U = np.zeros((si*sj, R))
            V = np.zeros((sj*sk, R))
            W = np.zeros((sk*si, R))
            
            # Term 1
            U[0, 0] = 1; U[3, 0] = 1  # a, d
            V[0, 0] = 1; V[3, 0] = 1  # e, h
            W[0, 0] = 1; W[3, 0] = 1  # C11, C22 (k=0,i=0 and k=1,i=1)
            
            # Term 2
            U[2, 1] = 1; U[3, 1] = 1  # c, d
            V[0, 1] = 1               # e
            W[1, 1] = 1               # C21 (k=0,i=1)
            
            # Term 3
            U[0, 2] = 1               # a
            V[1, 2] = 1; V[3, 2] = -1 # f, -h
            W[0, 2] = 1; W[3, 2] = -1 # C12 (+1), C22 (-1) (k=1,i=0 and k=1,i=1)
            
            # Term 4
            U[3, 3] = 1               # d
            V[2, 3] = 1; V[0, 3] = -1 # g, -e
            W[0, 3] = -1; W[1, 3] = 1 # C11 (-1), C21 (+1)
            
            # Term 5
            U[0, 4] = 1; U[1, 4] = 1  # a, b
            V[3, 4] = 1               # h
            W[0, 4] = 1; W[2, 4] = 1  # C11 (+1), C12 (+1) (k=0,i=0 and k=1,i=0)
            
            # Term 6
            U[2, 5] = 1; U[3, 5] = -1 # c, -d
            V[1, 5] = 1; V[2, 5] = 1  # f, g
            W[3, 5] = -1              # C22 (-1)
            
            # Term 7
            U[1, 6] = 1; U[3, 6] = -1 # b, -d
            V[0, 6] = 1; V[2, 6] = 1  # e, g
            W[0, 6] = 1               # C11 (+1)
            
            return U, V, W
        else:
            # Schoolbook decomposition
            R = si * sj * sk
            U = np.zeros((si*sj, R))
            V = np.zeros((sj*sk, R))
            W = np.zeros((sk*si, R))
            col = 0
            for ii in range(si):
                for jj in range(sj):
                    for kk in range(sk):
                        U[ii*sj+jj, col] = 1
                        V[jj*sk+kk, col] = 1
                        W[kk*si+ii, col] = 1
                        col += 1
            return U, V, W

    # Assemble the global U, V, W
    total_rank = 0
    # Pre-calculate total rank
    for ib, (i_start, i_size) in enumerate(i_blocks):
        for jb, (j_start, j_size) in enumerate(j_blocks):
            for kb, (k_start, k_size) in enumerate(k_blocks):
                _, _, W_block = get_decomp(i_size, j_size, k_size)
                total_rank += W_block.shape[1]
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    current_rank = 0
    for ib, (i_start, i_size) in enumerate(i_blocks):
        for jb, (j_start, j_size) in enumerate(j_blocks):
            for kb, (k_start, k_size) in enumerate(k_blocks):
                U_block, V_block, W_block = get_decomp(i_size, j_size, k_size)
                R_block = U_block.shape[1]
                
                # Map local indices to global indices
                for li in range(i_size):
                    for lj in range(j_size):
                        global_i = i_start + li
                        global_j = j_start + lj
                        for r in range(R_block):
                            U[global_i*m + global_j, current_rank + r] = U_block[li*j_size + lj, r]
                
                for lj in range(j_size):
                    for lk in range(k_size):
                        global_j = j_start + lj
                        global_k = k_start + lk
                        for r in range(R_block):
                            V[global_j*p + global_k, current_rank + r] = V_block[lj*k_size + lk, r]
                
                for lk in range(k_size):
                    for li in range(i_size):
                        global_k = k_start + lk
                        global_i = i_start + li
                        for r in range(R_block):
                            W[global_k*n + global_i, current_rank + r] = W_block[lk*i_size + li, r]
                
                current_rank += R_block
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
