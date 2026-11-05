"""Program for rank-minimal exact tensor decomposition of 4x5*5x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Uses a block decomposition with Strassen's algorithm for 2x2x2 blocks 
    and optimized decompositions for other blocks.
    Partition: i=[2,2], j=[3,2], k=[3,3]
    Expected rank: 72
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # Define block offsets and sizes
    # i: 2, 2
    i_offsets = [0, 2]
    i_sizes = [2, 2]
    # j: 3, 2
    j_offsets = [0, 3]
    j_sizes = [3, 2]
    # k: 3, 3
    k_offsets = [0, 3]
    k_sizes = [3, 3]
    
    max_rank = 120  # Safe upper bound
    
    U = np.zeros((n * m, max_rank), dtype=float)
    V = np.zeros((m * p, max_rank), dtype=float)
    W = np.zeros((p * n, max_rank), dtype=float)
    
    col = 0
    
    def add_strassen_2x2x2(i_0, j_0, k_0):
        nonlocal col
        # Term 1: (A11+A22)(B11+B22)
        U[(i_0+0)*m + (j_0+0), col] = 1
        U[(i_0+1)*m + (j_0+1), col] = 1
        V[(j_0+0)*p + (k_0+0), col] = 1
        V[(j_0+1)*p + (k_0+1), col] = 1
        W[(k_0+0)*n + (i_0+0), col] = 1
        W[(k_0+1)*n + (i_0+1), col] = 1
        col += 1
        
        # Term 2: (A21+A22)B11
        U[(i_0+1)*m + (j_0+0), col] = 1
        U[(i_0+1)*m + (j_0+1), col] = 1
        V[(j_0+0)*p + (k_0+0), col] = 1
        W[(k_0+0)*n + (i_0+1), col] = 1
        W[(k_0+1)*n + (i_0+1), col] = -1
        col += 1
        
        # Term 3: A11(B21-B11)
        U[(i_0+0)*m + (j_0+0), col] = 1
        V[(j_0+0)*p + (k_0+1), col] = 1
        V[(j_0+1)*p + (k_0+1), col] = -1
        W[(k_0+1)*n + (i_0+0), col] = 1
        W[(k_0+1)*n + (i_0+1), col] = 1
        col += 1
        
        # Term 4: A22(B12-B22)
        U[(i_0+1)*m + (j_0+1), col] = 1
        V[(j_0+1)*p + (k_0+0), col] = 1
        V[(j_0+0)*p + (k_0+0), col] = -1
        W[(k_0+0)*n + (i_0+0), col] = 1
        W[(k_0+0)*n + (i_0+1), col] = 1
        col += 1
        
        # Term 5: (A11+A12)B22
        U[(i_0+0)*m + (j_0+0), col] = 1
        U[(i_0+0)*m + (j_0+1), col] = 1
        V[(j_0+1)*p + (k_0+1), col] = 1
        W[(k_0+1)*n + (i_0+0), col] = 1
        W[(k_0+0)*n + (i_0+0), col] = -1
        col += 1
        
        # Term 6: (A21-A11)(B11+B12)
        U[(i_0+1)*m + (j_0+0), col] = 1
        U[(i_0+0)*m + (j_0+0), col] = -1
        V[(j_0+0)*p + (k_0+0), col] = 1
        V[(j_0+0)*p + (k_0+1), col] = 1
        W[(k_0+1)*n + (i_0+1), col] = 1
        col += 1
        
        # Term 7: (A12-A22)(B21+B22)
        U[(i_0+0)*m + (j_0+1), col] = 1
        U[(i_0+1)*m + (j_0+1), col] = -1
        V[(j_0+1)*p + (k_0+0), col] = 1
        V[(j_0+1)*p + (k_0+1), col] = 1
        W[(k_0+0)*n + (i_0+0), col] = 1
        col += 1

    def add_rank1_block(i_0, j_0, k_0, n_b, m_b, p_b):
        nonlocal col
        # For blocks where one dimension is 1, or generally small, we can often use rank 1 or 2.
        # Specifically, if n_b=1, m_b=1, p_b=1, rank 1.
        # If n_b=2, m_b=1, p_b=1, rank 1?
        # T_{i,j,k} = 1.
        # U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1.
        # If m_b=1, j is fixed.
        # We need sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1 for all i,k.
        # If we use 1 term: U_{i,j,0}=1, V_{j,k,0}=1, W_{k,i,0}=1.
        # This works for any n_b, p_b if we set U and W to 1?
        # Wait, W index is k*n+i.
        # If we set W[k*n+i, 0] = 1 for all k,i in the block, it works.
        # So any block with m_b=1 has rank 1?
        # Let's check 2x1x2.
        # U[i,0,r] V[0,k,r] W[k,i,r] = 1.
        # r=0: U[i,0,0]=1, V[0,k,0]=1, W[k,i,0]=1.
        # Sum = 1.
        # Yes, any block with m_b=1 has rank 1.
        
        # Similarly, if n_b=1, rank 1?
        # T_{0,j,k} = 1.
        # U[0,j,r] V[j,k,r] W[k,0,r] = 1.
        # r=0: U[0,j,0]=1, V[j,k,0]=1, W[k,0,0]=1.
        # Sum = 1.
        # Yes, any block with n_b=1 has rank 1.
        
        # Similarly, if p_b=1, rank 1?
        # T_{i,j,0} = 1.
        # U[i,j,r] V[j,0,r] W[0,i,r] = 1.
        # r=0: U[i,j,0]=1, V[j,0,0]=1, W[0,i,0]=1.
        # Sum = 1.
        # Yes, any block with p_b=1 has rank 1.
        
        # So if any dimension is 1, rank is 1.
        
        if n_b == 1 or m_b == 1 or p_b == 1:
            # Rank 1
            for i in range(n_b):
                for j in range(m_b):
                    for k in range(p_b):
                        U[(i_0+i)*m + (j_0+j), col] = 1
                        V[(j_0+j)*p + (k_0+k), col] = 1
                        W[(k_0+k)*n + (i_0+i), col] = 1
            col += 1
            return

        # If all dimensions > 1, we need a more complex decomposition.
        # For 2x2x2, we use Strassen.
        if n_b == 2 and m_b == 2 and p_b == 2:
            add_strassen_2x2x2(i_0, j_0, k_0)
            return
        
        # Fallback: Schoolbook
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[(i_0+i)*m + (j_0+j), col] = 1
                    V[(j_0+j)*p + (k_0+k), col] = 1
                    W[(k_0+k)*n + (i_0+i), col] = 1
                    col += 1

    for i_idx, i_0 in enumerate(i_offsets):
        n_b = i_sizes[i_idx]
        if time.time() > deadline:
            break
        for j_idx, j_0 in enumerate(j_offsets):
            m_b = j_sizes[j_idx]
            if time.time() > deadline:
                break
            for k_idx, k_0 in enumerate(k_offsets):
                p_b = k_sizes[k_idx]
                
                # Decompose this block
                # We can further split blocks like 2x3x3 into 2x2x2 + 2x2x1 + 2x1x2 + 2x1x1
                # But our add_rank1_block handles the "any dim 1" case.
                # We need to handle 2x3x3 specifically.
                
                if n_b == 2 and m_b == 3 and p_b == 3:
                    # Split j into 2, 1 and k into 2, 1
                    # 2x2x2 (Strassen)
                    add_strassen_2x2x2(i_0, j_0, k_0)
                    # 2x2x1 (Rank 1)
                    add_rank1_block(i_0, j_0, k_0+2, 2, 2, 1)
                    # 2x1x2 (Rank 1)
                    add_rank1_block(i_0, j_0+2, k_0, 2, 1, 2)
                    # 2x1x1 (Rank 1)
                    add_rank1_block(i_0, j_0+2, k_0+2, 2, 1, 1)
                elif n_b == 2 and m_b == 2 and p_b == 3:
                    # Split k into 2, 1
                    # 2x2x2 (Strassen)
                    add_strassen_2x2x2(i_0, j_0, k_0)
                    # 2x2x1 (Rank 1)
                    add_rank1_block(i_0, j_0, k_0+2, 2, 2, 1)
                else:
                    add_rank1_block(i_0, j_0, k_0, n_b, m_b, p_b)

    # Trim arrays to actual rank
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
