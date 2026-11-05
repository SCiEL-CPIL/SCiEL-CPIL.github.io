"""Matrix multiplication tensor decomposition search for 3x4x6."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]
START_TIME = time.time()
DEADLINE = 135.0  # Safety margin before 150s limit

def construct():
    global START_TIME
    start_time = time.time()
    n, m, p = DIMENSIONS
    N, M, P = n, m, p
    NM, MP, PN = N * M, M * P, P * N
    
    # Build target tensor T of shape (N*M, M*P, P*N)
    T = np.zeros((NM, MP, PN))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                u = i * M + j
                v = j * P + k
                w = k * N + i
                T[u, v, w] = 1.0
    
    def check_exact(U_arr, V_arr, W_arr, tol=1e-10):
        """Check if the decomposition exactly reconstructs T."""
        if U_arr.size == 0:
            return False
        T_rec = np.einsum('uc,vc,wc->uvw', U_arr, V_arr, W_arr)
        return np.allclose(T_rec, T, atol=tol)
    
    def strassen_222():
        """Strassen decomposition for 2x2x2. Returns U, V, W of shape (4, 7)."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 1: (a00+a11)(b00+b11)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # Term 2: (a10+a11)b00
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1
        
        # Term 3: a00(b01-b11)
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1; W[3,2] = 1
        
        # Term 4: a11(b10-b00)
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[2,3] = 1
        
        # Term 5: (a00+a01)b11
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[1,4] = 1; W[3,4] = -1
        
        # Term 6: (a10-a00)(b00+b01)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1
        
        # Term 7: (a01-a11)(b10+b11)
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1
        
        return U, V, W
    
    def rank1_1mp(M_local, P_local):
        """Rank-1 decomposition for 1xm x px1 tensor."""
        U = np.zeros((M_local, 1))
        V = np.zeros((M_local * P_local, 1))
        W = np.zeros((P_local, 1))
        U[0, 0] = 1
        V[:, 0] = 1
        W[:, 0] = 1
        return U, V, W
    
    # Strategy: Partition the indices and use block decompositions.
    # We partition i into blocks of size 1 or 2, j into blocks of size 1 or 2,
    # and k into blocks of size 1 or 2.
    # For each combination of blocks, we use an appropriate decomposition.
    # 1x1x1: rank 1
    # 1x1x2: rank 2 (or can be done with rank 2)
    # 1x2x1: rank 2
    # 2x1x1: rank 2
    # 1x1x1: rank 1
    # 1x2x2: rank 4? Actually 1x2x2 is just two 1x1x2 blocks, so rank 4.
    # But wait, 1x2x2 tensor T[i=0, j, k].
    # T[0, j, k] = 1 for all j, k.
    # This is a rank 1 tensor? No.
    # T is a 1x2x2 array of all ones.
    # Rank of all-ones 1x2x2 tensor is 1?
    # U[0,0]=1, V[j,0]=1, W[k,0]=1.
    # Yes, rank 1.
    
    # Let's re-evaluate the rank of sub-tensors.
    # T_sub[i', j', k'] = 1 for all i', j', k' in the block.
    # This is a tensor of all ones of shape (ni, nj, nk).
    # The rank of an all-ones tensor of shape (ni, nj, nk) is 1?
    # U[i', 0] = 1, V[j', 0] = 1, W[k', 0] = 1.
    # Yes, rank 1.
    
    # So if we partition into blocks, and for each block we use rank 1,
    # the total rank is the number of blocks.
    # But this is only valid if the blocks are disjoint in the index sets
    # and we can sum the decompositions.
    # Since T is the sum of its blocks (each block is 1s, and they don't overlap),
    # we can just sum the rank-1 decompositions of each block.
    
    # So the problem reduces to: partition {0..N-1} x {0..M-1} x {0..P-1}
    # into rectangular boxes (ni x nj x nk) such that the sum of the ranks
    # of the boxes is minimized.
    # Since each box of size ni x nj x nk has rank 1 (all ones),
    # the total rank is just the number of boxes.
    # So we want to cover the N x M x P grid with the minimum number of
    # rectangular boxes.
    
    # Wait, is the rank of an all-ones ni x nj x nk tensor always 1?
    # Yes, because it's the outer product of three all-ones vectors.
    
    # So the minimum rank is the minimum number of boxes needed to cover
    # the N x M x P grid.
    # For N=3, M=4, P=6:
    # We can cover with 1 box of size 3x4x6? No, that's the whole thing.
    # But the tensor is not all ones!
    # T[i, j, k] is 1 only if the indices match the multiplication pattern.
    # T[u, v, w] = 1 where u=i*M+j, v=j*P+k, w=k*N+i.
    # So T is not an all-ones tensor. It's a permutation-like tensor.
    
    # So the block decomposition approach with rank 1 per block is wrong.
    # We need to use the actual tensor structure.
    
    # Let's go back to the seed program's approach.
    # The seed program uses:
    # - For i=0 (n_block=1): 3 blocks of k (each size 2), each using rank1_1mp(4, 2)
    #   This gives rank 1 per block, total rank 3.
    # - For i={1,2} (n_block=2): 3 blocks of k (each size 2), 2 blocks of j (each size 2)
    #   Each uses strassen_222() with rank 7.
    #   Total rank 3*2*7 = 42.
    # Total rank = 3 + 42 = 45.
    
    # The schoolbook rank is 3*4*6 = 72.
    # So the seed program achieves rank 45, which is better than 72.
    
    # Can we do better?
    # Let's try different partitions.
    
    # Partition i into {0}, {1}, {2} (three blocks of size 1)
    # For each i-block of size 1:
    #   Partition j into {0}, {1}, {2}, {3} (four blocks of size 1)
    #   For each j-block of size 1:
    #     Partition k into {0}, {1}, ..., {5} (six blocks of size 1)
    #     Each 1x1x1 block has rank 1.
    #     Total rank per i-block: 4*6 = 24.
    #     Total rank: 3*24 = 72. (Same as schoolbook)
    
    # Partition i into {0}, {1,2}
    # i=0: 1x4x6 tensor.
    #   T[0, j, k] = 1 for all j, k.
    #   This is an all-ones 1x4x6 tensor, rank 1.
    #   So rank 1 for i=0.
    # i={1,2}: 2x4x6 tensor.
    #   T[i, j, k] = 1 for i in {1,2}, all j, k.
    #   This is an all-ones 2x4x6 tensor, rank 1.
    #   So rank 1 for i={1,2}.
    # Total rank = 1 + 1 = 2.
    
    # Wait, is T[0, j, k] = 1 for all j, k?
    # T[u, v, w] = 1 where u=i*M+j, v=j*P+k, w=k*N+i.
    # For i=0: u=j, v=j*P+k, w=k*3+0 = 3k.
    # So T[j, j*P+k, 3k] = 1.
    # This is NOT an all-ones tensor!
    # It's a sparse tensor with 1s at specific positions.
    
    # So the block decomposition approach is more complex.
    # We need to decompose the tensor for each block of (i, j, k).
    
    # Let's try a different approach: use the seed program's structure
    # but try to optimize the partitions.
    
    # For now, let's use the seed program's approach and verify it works.
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Sub-tensors with i={0} (n_block=1)
    for k_start in [0, 2, 4]:
        k_end = k_start + 2
        U_local, V_local, W_local = rank1_1mp(4, 2)
        
        U_part = np.zeros((NM, 1))
        V_part = np.zeros((MP, 1))
        W_part = np.zeros((PN, 1))
        
        for j_prime in range(4):
            for k_prime in range(2):
                u_global = 0 * M + j_prime
                v_global = j_prime * P + (k_start + k_prime)
                w_global = (k_start + k_prime) * N + 0
                
                U_part[u_global, 0] += U_local[j_prime, 0]
                V_part[v_global, 0] += V_local[j_prime * 2 + k_prime, 0]
                W_part[w_global, 0] += W_local[k_prime, 0]
        
        U_parts.append(U_part)
        V_parts.append(V_part)
        W_parts.append(W_part)
    
    # Sub-tensors with i={1,2} (n_block=2)
    for k_start in [0, 2, 4]:
        k_end = k_start + 2
        for j_start in [0, 2]:
            j_end = j_start + 2
            U_local, V_local, W_local = strassen_222()
            
            U_part = np.zeros((NM, 7))
            V_part = np.zeros((MP, 7))
            W_part = np.zeros((PN, 7))
            
            for i_prime in range(2):
                for j_prime in range(2):
                    for k_prime in range(2):
                        u_global = (1 + i_prime) * M + (j_start + j_prime)
                        v_global = (j_start + j_prime) * P + (k_start + k_prime)
                        w_global = (k_start + k_prime) * N + (1 + i_prime)
                        
                        u_local_idx = i_prime * 2 + j_prime
                        v_local_idx = j_prime * 2 + k_prime
                        w_local_idx = k_prime * 2 + i_prime
                        
                        U_part[u_global, :] += U_local[u_local_idx, :]
                        V_part[v_global, :] += V_local[v_local_idx, :]
                        W_part[w_global, :] += W_local[w_local_idx, :]
            
            U_parts.append(U_part)
            V_parts.append(V_part)
            W_parts.append(W_part)
    
    U_full = np.hstack(U_parts)
    V_full = np.hstack(V_parts)
    W_full = np.hstack(W_parts)
    
    if check_exact(U_full, V_full, W_full):
        U_list = U_full.tolist()
        V_list = V_full.tolist()
        W_list = W_full.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    rank = N * M * P
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
