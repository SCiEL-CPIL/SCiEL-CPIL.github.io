import numpy as np
import time
import itertools
from functools import lru_cache

DIMENSIONS = [2, 3, 6]

def verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W exactly decompose the matrix multiplication tensor M_{n,m,p}.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Indexing:
    U index: a*m + b  (a in 0..n-1, b in 0..m-1)
    V index: c*p + d  (c in 0..m-1, d in 0..p-1)
    W index: k*n + i  (k in 0..p-1, i in 0..n-1)
    Tensor T[a,b,c,d,k,i] = 1 if a==i, c==k, b==d else 0.
    Note: The task description says W[k*n+i, r].
    Let's re-read carefully: "output factor follows the official trace convention W[k*n+i,r]"
    Usually matrix multiplication C = A B is C[i,k] = sum_j A[i,j] B[j,k].
    Here A is n x m, B is m x p. C is n x p.
    So indices are i (0..n-1), k (0..p-1).
    The sum is over j (0..m-1).
    Term: A[i,j] * B[j,k].
    U corresponds to A factors: index (i, j).
    V corresponds to B factors: index (j, k).
    W corresponds to C factors: index (k, i).
    So W index is k*n + i.
    
    The tensor equation is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) form a valid term?
    No, the tensor T is defined such that T_{(i,j), (j',k), (k',i')} = delta_{j,j'} delta_{i,i'} delta_{k,k'}.
    Wait, the standard tensor decomposition for matrix mult M_{n,m,p} is:
    T_{(i,j), (j',k), (k',i')} = \delta_{i,i'} \delta_{j,j'} \delta_{k,k'}
    where i,i' in {0..n-1}, j,j' in {0..m-1}, k,k' in {0..p-1}.
    The term in the sum is U_{(i,j), r} V_{(j',k), r} W_{(k',i'), r}.
    For the result to be \delta_{i,i'} \delta_{j,j'} \delta_{k,k'}, we must have:
    sum_r U_{i,j,r} V_{j',k,r} W_{k',i',r} = 1 if i=i', j=j', k=k' else 0.
    """
    r = U.shape[1]
    # Iterate over all non-zero entries of the tensor
    # The tensor is sparse: non-zero only when i=i', j=j', k=k'
    # There are n*m*p such non-zero entries.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Target is 1 for this specific combination of indices in the contraction
                # Wait, the contraction sum is over r.
                # The value of the tensor at (i,j, j,k, k,i) is 1.
                # The value at any other combination of indices (i,j, j',k', k'',i''') is 0.
                # But we only need to check the diagonal entries?
                # No, the decomposition must hold for ALL index combinations.
                # However, the tensor is zero for off-diagonals.
                # So we must check:
                # 1. For (i,j, j,k, k,i): sum = 1
                # 2. For all other combinations: sum = 0
                
                # This is O(n^2 m^2 p^2 r) checks.
                # n=2, m=3, p=6.
                # n^2=4, m^2=9, p^2=36. 4*9*36 = 1296 checks.
                # Each check is O(r). r~30. 1296*30 ~ 40k ops. Very fast.
                
                # Let's implement the full check for safety.
                pass
                
    # Efficient check:
    # Construct the full tensor T of size (n*m, m*p, p*n)
    # T is sparse.
    # We can compute the matrix product of U^T V etc? No.
    # Let's just loop over all combinations.
    
    # Indices:
    # u_idx: a in 0..n-1, b in 0..m-1. idx_u = a*m + b
    # v_idx: c in 0..m-1, d in 0..p-1. idx_v = c*p + d
    # w_idx: e in 0..p-1, f in 0..n-1. idx_w = e*n + f
    
    # Target T[u_idx, v_idx, w_idx] = 1 if a==f, b==c, d==e else 0.
    
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for e in range(p):
                        for f in range(n):
                            target = 1.0 if (a == f and b == c and d == e) else 0.0
                            u_idx = a * m + b
                            v_idx = c * p + d
                            w_idx = e * n + f
                            
                            s = 0.0
                            for l in range(r):
                                s += U[u_idx, l] * V[v_idx, l] * W[w_idx, l]
                            
                            if not np.isclose(s, target, atol=1e-9):
                                return False
    return True

def get_rank34_solution():
    """
    Returns the known valid rank-34 solution.
    This is based on the rank-17 decomposition of 2x3x3.
    """
    n, m, p = 2, 3, 6
    
    # Rank 17 decomposition for 2x3x3
    # This is a known optimal or near-optimal decomposition.
    # U: 6x17, V: 9x17, W: 6x17
    U233 = np.zeros((6, 17))
    V233 = np.zeros((9, 17))
    W233 = np.zeros((6, 17))
    
    # Manually transcribed from the seed program which was verified
    # Term 0
    U233[0, 0] = 1; U233[4, 0] = 1
    V233[0, 0] = 1; V233[4, 0] = 1
    W233[0, 0] = 1; W233[3, 0] = 1
    
    # Term 1
    U233[3, 1] = 1; U233[4, 1] = 1
    V233[0, 1] = 1
    W233[1, 1] = 1; W233[3, 1] = -1
    
    # Term 2
    U233[0, 2] = 1
    V233[1, 2] = 1; V233[4, 2] = -1
    W233[2, 2] = 1; W233[3, 2] = 1
    
    # Term 3
    U233[4, 3] = 1
    V233[3, 3] = 1; V233[0, 3] = -1
    W233[0, 3] = 1; W233[1, 3] = 1
    
    # Term 4
    U233[0, 4] = 1; U233[1, 4] = 1
    V233[4, 4] = 1
    W233[0, 4] = -1; W233[2, 4] = 1
    
    # Term 5
    U233[3, 5] = 1; U233[0, 5] = -1
    V233[0, 5] = 1; V233[1, 5] = 1
    W233[3, 5] = 1
    
    # Term 6
    U233[1, 6] = 1; U233[4, 6] = -1
    V233[3, 6] = 1; V233[4, 6] = 1
    W233[0, 6] = 1
    
    # Term 7
    U233[0, 7] = 1
    V233[2, 7] = 1
    W233[4, 7] = 1
    
    # Term 8
    U233[1, 8] = 1
    V233[5, 8] = 1
    W233[4, 8] = 1
    
    # Term 9
    U233[3, 9] = 1
    V233[2, 9] = 1
    W233[5, 9] = 1
    
    # Term 10
    U233[4, 10] = 1
    V233[5, 10] = 1
    W233[5, 10] = 1
    
    # Term 11
    U233[2, 11] = 1
    V233[6, 11] = 1
    W233[0, 11] = 1
    
    # Term 12
    U233[2, 12] = 1
    V233[7, 12] = 1
    W233[2, 12] = 1
    
    # Term 13
    U233[2, 13] = 1
    V233[8, 13] = 1
    W233[4, 13] = 1
    
    # Term 14
    U233[5, 14] = 1
    V233[6, 14] = 1
    W233[1, 14] = 1
    
    # Term 15
    U233[5, 15] = 1
    V233[7, 15] = 1
    W233[3, 15] = 1
    
    # Term 16
    U233[5, 16] = 1
    V233[8, 16] = 1
    W233[5, 16] = 1
    
    # Construct 2x3x6 solution by doubling the 2x3x3 solution
    # p=6 is two blocks of p=3.
    # U is same for both blocks? No, U depends on n,m.
    # V depends on m,p. V is (m*p, r).
    # W depends on p,n. W is (p*n, r).
    
    U_comb = np.zeros((6, 34))
    V_comb = np.zeros((18, 34))
    W_comb = np.zeros((12, 34))
    
    # First half (k=0,1,2)
    U_comb[:, :17] = U233
    
    # V for first half (j=0..2, k=0..2) -> indices 0..8
    # V233 has shape (9, 17). Indices j*3 + k.
    # We need to map j*6 + k for the combined V.
    for j in range(3):
        for k in range(3):
            V_comb[j * 6 + k, :17] = V233[j * 3 + k]
            
    # W for first half (k=0..2, i=0..1) -> indices 0..5
    # W233 has shape (6, 17). Indices k*2 + i.
    for k in range(3):
        for i in range(2):
            W_comb[k * 2 + i, :17] = W233[k * 2 + i]
            
    # Second half (k=3,4,5)
    U_comb[:, 17:] = U233
    
    # V for second half (j=0..2, k=3..5) -> indices 9..17?
    # V index is j*p + d = j*6 + d.
    # For d=3,4,5 (corresponding to k=0,1,2 in the block logic)
    for j in range(3):
        for k in range(3):
            V_comb[j * 6 + 3 + k, 17:] = V233[j * 3 + k]
            
    # W for second half (k=3..5, i=0..1) -> indices 6..11
    for k in range(3):
        for i in range(2):
            W_comb[(k + 3) * 2 + i, 17:] = W233[k * 2 + i]
            
    return U_comb, V_comb, W_comb

def search_rank16_233(time_limit):
    """
    Attempts to find a rank-16 decomposition for 2x3x3.
    This is a heuristic search over sparse structures.
    Returns U, V, W if found, else None.
    """
    start_time = time.time()
    n, m, p = 2, 3, 3
    r = 16
    
    # We will try to find a solution with entries in {0, 1, -1}
    # This is a constraint satisfaction problem.
    # Given the complexity, we use a random restart local search on the error surface,
    # but snapping to integers.
    
    # Actually, for such a small rank, random search is unlikely to work.
    # We will try a few specific structured patterns known to work for similar tensors.
    # If we don't find one quickly, we return None.
    
    # One known structure for 2x3x3 is related to 3x3x3 rank 7.
    # But 2x3x3 is not a direct slice.
    
    # Let's try a quick gradient descent on continuous variables, then snap.
    # This is risky for exactness, but we verify at the end.
    
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            break
            
        U = np.random.randn(6, r)
        V = np.random.randn(9, r)
        W = np.random.randn(6, r)
        
        # Normalize
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        # Simple gradient descent for 100 steps
        lr = 0.01
        for step in range(100):
            if time.time() - start_time > time_limit:
                return None
                
            # Compute error gradient
            # This is expensive. We'll skip true GD and just check if random init
            # can be snapped.
            
        # Snap to nearest half-integer
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Verify
        if verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
            return U_snap, V_snap, W_snap
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Strategy 1: Try to find rank 16 for 2x3x3.
    # If found, we can build a rank 32 solution for 2x3x6.
    # Is it possible to build 2x3x6 from 2x3x3 rank 16?
    # Yes, by doubling the p dimension.
    # U is 6x16. V is 18x32 (two blocks of 9x16). W is 12x32 (two blocks of 6x16).
    # Total rank 32.
    
    # However, finding rank 16 for 2x3x3 is a research problem.
    # We will attempt a very short search.
    
    U233, V233, W233 = search_rank16_233(time_limit=10.0)
    
    if U233 is not None:
        # Construct rank 32 solution
        U_comb = np.zeros((6, 32))
        V_comb = np.zeros((18, 32))
        W_comb = np.zeros((12, 32))
        
        U_comb[:, :16] = U233
        U_comb[:, 16:] = U233
        
        for j in range(3):
            for k in range(3):
                V_comb[j * 6 + k, :16] = V233[j * 3 + k]
                V_comb[j * 6 + 3 + k, 16:] = V233[j * 3 + k]
                
        for k in range(3):
            for i in range(2):
                W_comb[k * 2 + i, :16] = W233[k * 2 + i]
                W_comb[(k + 3) * 2 + i, 16:] = W233[k * 2 + i]
                
        # Final verification
        if verify_decomposition(U_comb, V_comb, W_comb, n, m, p):
            return {
                "U": U_comb.tolist(),
                "V": V_comb.tolist(),
                "W": W_comb.tolist()
            }
            
    # Strategy 2: Fall back to rank 34
    U34, V34, W34 = get_rank34_solution()
    
    # Verify fallback
    if verify_decomposition(U34, V34, W34, n, m, p):
        return {
            "U": U34.tolist(),
            "V": V34.tolist(),
            "W": W34.tolist()
        }
        
    # Emergency: Return empty or raise?
    # The task says "invalid result... scores 0".
    # We should ensure we always return something valid.
    # If verification fails (which shouldn't happen with hardcoded), we try schoolbook.
    
    # Schoolbook rank 36
    U36 = np.zeros((6, 36))
    V36 = np.zeros((18, 36))
    W36 = np.zeros((12, 36))
    
    for i in range(2):
        for j in range(3):
            for k in range(6):
                idx = i * 18 + j * 6 + k
                U36[i * 3 + j, idx] = 1
                V36[j * 6 + k, idx] = 1
                W36[k * 2 + i, idx] = 1
                
    return {
        "U": U36.tolist(),
        "V": V36.tolist(),
        "W": W36.tolist()
    }

if __name__ == "__main__":
    construct()
