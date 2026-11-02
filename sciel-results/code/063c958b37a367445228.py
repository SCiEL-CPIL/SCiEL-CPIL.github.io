"""Matrix multiplication tensor decomposition for 2x3 by 3x10.

Strategy: Divide-and-conquer via direct sum decomposition.
The tensor for M_{2,3,10} is the direct sum of tensors for M_{2,3,p_i} 
where sum(p_i) = 10. The rank of the direct sum is the sum of the ranks.
We partition p=10 into smaller parts (e.g., 4+6 or 3+3+3+1) and solve 
for the minimal rank of each sub-problem using a targeted local search 
for small dimensions, which is feasible due to the reduced size.
"""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def _try_small_rank_decomposition(n, m, p, max_rank=None, time_budget=5.0):
    """
    Attempt to find a rank decomposition for M_{n,m,p} with rank < n*m*p.
    Uses a greedy/local search approach suitable for small p.
    Returns dict with U, V, W, rank or None.
    """
    if max_rank is None:
        max_rank = n * m * p - 1
    
    nm, mp, pn = n * m, m * p, p * n
    T = _build_tensor(n, m, p)
    
    start_time = time.time()
    
    # Try ranks from lower bound to max_rank
    min_rank = max(nm, mp, pn)
    
    # If the tensor is very small, we can try to verify if schoolbook is optimal
    # by checking if any rank < nmp exists.
    
    # For M_{2,3,3}, nmp=18. Lower bound 9.
    # For M_{2,3,4}, nmp=24. Lower bound 12.
    
    # We use a simple random restart local search with projection to integers/half-integers.
    # This is heuristic but works for small sizes.
    
    for rank in range(max_rank, min_rank - 1, -1):
        if time.time() - start_time > time_budget:
            break
            
        # Try a few random initializations
        for trial in range(50):
            if time.time() - start_time > time_budget:
                break
                
            rng = random.Random(trial * 1000 + rank)
            
            # Initialize U, V, W with sparse integer/half-integer values
            U = np.zeros((nm, rank), dtype=np.float64)
            V = np.zeros((mp, rank), dtype=np.float64)
            W = np.zeros((pn, rank), dtype=np.float64)
            
            # Fill with random sparse patterns
            for k in range(rank):
                # Each column of U, V, W gets 1-2 non-zeros
                nz_u = rng.randint(1, 2)
                nz_v = rng.randint(1, 2)
                nz_w = rng.randint(1, 2)
                
                idx_u = rng.sample(range(nm), nz_u)
                idx_v = rng.sample(range(mp), nz_v)
                idx_w = rng.sample(range(pn), nz_w)
                
                for idx in idx_u:
                    U[idx, k] = rng.choice([0.5, 1, -0.5, -1])
                for idx in idx_v:
                    V[idx, k] = rng.choice([0.5, 1, -0.5, -1])
                for idx in idx_w:
                    W[idx, k] = rng.choice([0.5, 1, -0.5, -1])
            
            # Local search: minimize ||U V^T W^T - T||^2
            # Gradient descent with projection
            lr = 0.01
            for step in range(100):
                if time.time() - start_time > time_budget:
                    break
                    
                recon = np.einsum('ak,bk,ck->abc', U, V, W)
                diff = recon - T
                loss = np.sum(diff ** 2)
                
                if loss < 1e-9:
                    # Check if entries are integer/half-integer
                    if np.all(np.allclose(U * 2, np.round(U * 2))) and \
                       np.all(np.allclose(V * 2, np.round(V * 2))) and \
                       np.all(np.allclose(W * 2, np.round(W * 2))):
                        U2 = np.round(U * 2) / 2
                        V2 = np.round(V * 2) / 2
                        W2 = np.round(W * 2) / 2
                        recon2 = np.einsum('ak,bk,ck->abc', U2, V2, W2)
                        if np.allclose(recon2, T, atol=1e-9):
                            return {'U': U2, 'V': V2, 'W': W2, 'rank': rank}
                    break
                
                # Gradient
                grad_U = np.einsum('abc,bk,ck->ak', diff, V, W)
                grad_V = np.einsum('abc,ak,ck->bk', diff, U, W)
                grad_W = np.einsum('abc,ak,bk->ck', diff, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Project to grid every 10 steps
                if step % 10 == 9:
                    U = np.round(U * 2) / 2
                    V = np.round(V * 2) / 2
                    W = np.round(W * 2) / 2

    return None

def _combine_decompositions(decomp_list):
    """Combine a list of decompositions (dicts with U, V, W) into a single one.
    The tensors are direct sums, so we concatenate the factor matrices horizontally.
    """
    if not decomp_list:
        return None
        
    U_list = []
    V_list = []
    W_list = []
    total_rank = 0
    
    for d in decomp_list:
        if d is None:
            continue
        U_list.append(d['U'])
        V_list.append(d['V'])
        W_list.append(d['W'])
        total_rank += d['rank']
        
    if total_rank == 0:
        return None
        
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    return {'U': U, 'V': V, 'W': W, 'rank': total_rank}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    # Strategy: Split p=10 into parts.
    # Try different partitions of p.
    # Partitions of 10:
    # [10] -> rank 60 (schoolbook)
    # [9, 1] -> M_{2,3,9} + M_{2,3,1}
    # [8, 2] -> M_{2,3,8} + M_{2,3,2}
    # [7, 3] -> M_{2,3,7} + M_{2,3,3}
    # [6, 4] -> M_{2,3,6} + M_{2,3,4}
    # [5, 5] -> M_{2,3,5} + M_{2,3,5}
    # [4, 3, 3] -> M_{2,3,4} + M_{2,3,3} + M_{2,3,3}
    # [3, 3, 3, 1] -> M_{2,3,3} * 3 + M_{2,3,1}
    
    # We know M_{2,3,1} has rank 6 (schoolbook).
    # We suspect M_{2,3,3} might have rank < 18.
    # We suspect M_{2,3,4} might have rank < 24.
    
    # Let's try to find better ranks for the smaller blocks.
    # We will attempt to solve M_{2,3,3} and M_{2,3,4}.
    
    best_decomp = None
    best_rank = n * m * p # 60
    
    # Partition 1: [3, 3, 3, 1]
    # Solve M_{2,3,3} and M_{2,3,1}
    d_233 = _try_small_rank_decomposition(n, m, 3, time_budget=20.0)
    d_231 = _try_small_rank_decomposition(n, m, 1, time_budget=5.0)
    
    if d_233 is None:
        d_233 = _schoolbook_decomposition(n, m, 3)
    if d_231 is None:
        d_231 = _schoolbook_decomposition(n, m, 1)
        
    # Combine 3 copies of 233 and 1 copy of 231
    decomp_3331 = _combine_decompositions([d_233, d_233, d_233, d_231])
    if decomp_3331 and decomp_3331['rank'] < best_rank:
        best_rank = decomp_3331['rank']
        best_decomp = decomp_3331
        print(f"Partition [3,3,3,1] rank: {best_rank}")
        
    # Partition 2: [4, 3, 3]
    # Solve M_{2,3,4}
    d_234 = _try_small_rank_decomposition(n, m, 4, time_budget=20.0)
    if d_234 is None:
        d_234 = _schoolbook_decomposition(n, m, 4)
        
    decomp_433 = _combine_decompositions([d_234, d_233, d_233])
    if decomp_433 and decomp_433['rank'] < best_rank:
        best_rank = decomp_433['rank']
        best_decomp = decomp_433
        print(f"Partition [4,3,3] rank: {best_rank}")
        
    # Partition 3: [5, 5]
    d_235 = _try_small_rank_decomposition(n, m, 5, time_budget=20.0)
    if d_235 is None:
        d_235 = _schoolbook_decomposition(n, m, 5)
        
    decomp_55 = _combine_decompositions([d_235, d_235])
    if decomp_55 and decomp_55['rank'] < best_rank:
        best_rank = decomp_55['rank']
        best_decomp = decomp_55
        print(f"Partition [5,5] rank: {best_rank}")
        
    # Fallback to schoolbook if nothing better found
    if best_decomp is None:
        best_decomp = _schoolbook_decomposition(n, m, p)
        
    # Final verification
    recon = np.einsum('ak,bk,ck->abc', best_decomp['U'], best_decomp['V'], best_decomp['W'])
    T = _build_tensor(n, m, p)
    if not np.allclose(recon, T, atol=1e-9):
        print("Verification failed, falling back to schoolbook")
        best_decomp = _schoolbook_decomposition(n, m, p)
        
    return {'U': best_decomp['U'].tolist(), 'V': best_decomp['V'].tolist(), 'W': best_decomp['W'].tolist()}

# EVOLVE-BLOCK-END
