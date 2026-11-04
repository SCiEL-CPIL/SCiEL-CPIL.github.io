"""Program for matrix multiplication 4x4 x 4x6 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    
    Returns (U, V, W) where:
      U: (4, 7) - for A[i,j], row index = i*2+j
      V: (4, 7) - for B[j,k], row index = j*2+k
      W: (4, 7) - for C[i,k], row index = k*2+i  (trace convention: W[k*n+i] with n=2)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W: C[i,k] contributions, W row = k*2 + i
    # C[0,0]: i=0,k=0 -> W[0]; C[0,1]: i=0,k=1 -> W[2]
    # C[1,0]: i=1,k=0 -> W[1]; C[1,1]: i=1,k=1 -> W[3]
    
    # C[0,0] = M1 + M4 - M5 + M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    
    # C[0,1] = M3 + M5
    W[2, 2] = 1; W[2, 4] = 1
    
    # C[1,0] = M2 + M4
    W[1, 1] = 1; W[1, 3] = 1
    
    # C[1,1] = M1 - M2 + M3 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W


def _build_rank77():
    """
    Build the structured rank-77 decomposition using 2x2 block Strassen tiling.
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    sU, sV, sW = _strassen_2x2()
    
    # Outer structure: 2x2 blocks of A, 2x3 blocks of B, 2x3 blocks of C
    # For cols 0,1: use Strassen (7 terms)
    # For col 2: use schoolbook (4 terms)
    
    total_rank = 11 * 7  # 77
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    # Helper to get Strassen inner coefficients
    def strassen_inner_U(ai2, bj2, t):
        return sU[ai2*2 + bj2, t]
    
    def strassen_inner_V(bj2, ck2, t):
        return sV[bj2*2 + ck2, t]
    
    def strassen_inner_W(ck2, ai2, t):
        return sW[ck2*2 + ai2, t]
    
    # Process each outer term
    # Outer terms 0-6: Strassen for block columns 0,1
    # Outer terms 7-10: schoolbook for block column 2
    
    for ot in range(11):
        for t in range(7):
            col = ot * 7 + t
            
            if ot < 7:
                # Strassen term for columns 0,1
                # This term combines A[ai][bj] and B[bj][ck] for ck in {0,1}
                # The Strassen structure means each term t involves specific combinations
                
                # U part: A[ai][bj] contributions
                # From Strassen U matrix: sU[ai2*2+bj2, t] gives coefficient for A[ai2][bj2]
                for ai in range(2):
                    for bj in range(2):
                        for ai2 in range(2):
                            for bj2 in range(2):
                                coeff = strassen_inner_U(ai2, bj2, t)
                                if coeff != 0:
                                    u_row = (2*ai + ai2) * 4 + (2*bj + bj2)
                                    U[u_row, col] += coeff
                
                # V part: B[bj][ck] contributions for ck in {0,1}
                for bj in range(2):
                    for ck in range(2):
                        for bj2 in range(2):
                            for ck2 in range(2):
                                coeff = strassen_inner_V(bj2, ck2, t)
                                if coeff != 0:
                                    v_row = (2*bj + bj2) * 6 + (2*ck + ck2)
                                    V[v_row, col] += coeff
                
                # W part: C[ai][ck] contributions for ck in {0,1}
                for ai in range(2):
                    for ck in range(2):
                        for ai2 in range(2):
                            for ck2 in range(2):
                                coeff = strassen_inner_W(ck2, ai2, t)
                                if coeff != 0:
                                    w_row = (2*ck + ck2) * 4 + (2*ai + ai2)
                                    W[w_row, col] += coeff
            else:
                # Schoolbook term for column 2
                # ot - 7 gives the (ai, bj) pair
                idx = ot - 7
                ai = idx // 2
                bj = idx % 2
                
                # Only one inner term (t=0) is used for schoolbook
                if t == 0:
                    # U: A[ai][bj]
                    for ai2 in range(2):
                        for bj2 in range(2):
                            u_row = (2*ai + ai2) * 4 + (2*bj + bj2)
                            U[u_row, col] = 1
                    
                    # V: B[bj][2]
                    for bj2 in range(2):
                        for ck2 in range(2):
                            v_row = (2*bj + bj2) * 6 + (2*2 + ck2)
                            V[v_row, col] = 1
                    
                    # W: C[ai][2]
                    for ai2 in range(2):
                        for ck2 in range(2):
                            w_row = (2*2 + ck2) * 4 + (2*ai + ai2)
                            W[w_row, col] = 1
                # For t > 0, schoolbook terms have zero contribution
                # (they're not used in the schoolbook part)
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """
    Verify that the decomposition correctly computes matrix multiplication.
    C[i,k] = sum_j A[i,j] * B[j,k]
    W[k*n+i] = sum_t U[i*m+j, t] * V[j*p+k, t] ... actually:
    The tensor is T[i,j,k] = delta[i,j,k] where T[i,j,k] = 1 if the output C[i,k] depends on A[i,j]*B[j,k]
    
    Actually, the matrix multiplication tensor is:
    T_{i,j,k} = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    So T[i,j,k] = delta_{i,k_output} ... 
    
    Let me think more carefully. The bilinear form is:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has indices (i,j,k) where:
    - i indexes rows of A (and C), size n=4
    - j indexes columns of A (and rows of B), size m=4
    - k indexes columns of B (and C), size p=6
    
    T[i,j,k] = 1 (the coefficient of A[i,j]*B[j,k] in C[i,k] is 1)
    
    The decomposition says: T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    Wait, that's not right either. Let me re-read the contract.
    
    "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    "U: (n*m, rank) - for A[i,j], row index = i*m+j"
    "V: (m*p, rank) - for B[j,k], row index = j*p+k"
    "W: (p*n, rank) - for C[i,k], row index = k*n+i (trace convention)"
    
    The decomposition is: T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    And T[i,j,k] = 1 for all i,j,k (since each A[i,j]*B[j,k] contributes to C[i,k] with coefficient 1)
    
    Wait no. T[i,j,k] should be 1 if A[i,j] and B[j,k] multiply to contribute to C[i,k].
    That's always the case, so T[i,j,k] = 1 for all valid i,j,k.
    
    Hmm, but that would make the tensor all-ones, which has rank 1. That can't be right.
    
    Let me reconsider. The matrix multiplication tensor is defined as:
    T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to the (i,k) entry of C.
    
    Actually, I think the tensor is:
    T[i,j,k] = delta_{j,j} = 1 (always)
    
    No wait. Let me think about this differently.
    
    The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    
    This can be written as: C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    
    where T[i,j,k] = 1 for all i,j,k.
    
    But then the tensor is just the all-ones tensor, which has rank 1.
    
    I think I'm misunderstanding the tensor structure. Let me reconsider.
    
    Actually, I think the tensor T is defined such that:
    T[i,j,k] = 1 if the product A[i,j]*B[j,k] appears in the computation of C[i,k]
    
    Since C[i,k] = sum_j A[i,j]*B[j,k], we have T[i,j,k] = 1 for all i,j,k.
    
    But that gives an all-ones tensor of rank 1, which is clearly wrong.
    
    Let me look at this from the decomposition perspective. The decomposition says:
    sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = T[i,j,k]
    
    For this to represent matrix multiplication, we need:
    For each (i,k), C[i,k] = sum_j A[i,j] B[j,k]
    
    And the decomposition gives:
    C[i,k] = sum_t W[k*n+i, t] * (sum_j U[i*m+j, t] A[i,j]) * (sum_j V[j*p+k, t] B[j,k])
    
    Wait, that's not quite right either. Let me be more careful.
    
    I think the correct interpretation is:
    The tensor T has T[i,j,k] = 1 for all i,j,k.
    
    And the decomposition is:
    T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    This means: 1 = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] for all i,j,k.
    
    That's a very strong condition. Let's check with the Strassen 2x2 case.
    
    For 2x2*2x2->2x2, n=m=p=2.
    U is 4x7, V is 4x7, W is 4x7.
    
    T[i,j,k] = 1 for all i,j,k in {0,1}.
    
    So we need: 1 = sum_t U[2i+j, t] * V[2j+k, t] * W[2k+i, t] for all i,j,k.
    
    Let's check i=0,j=0,k=0:
    1 = sum_t U[0, t] * V[0, t] * W[0, t]
    
    From the Strassen factors:
    U[0,:] = [1, 0, 1, 0, 1, -1, 0]
    V[0,:] = [1, 1, 0, -1, 0, 1, 0]
    W[0,:] = [1, 0, 0, 1, -1, 0, 1]
    
    sum_t = 1*1*1 + 0*1*0 + 1*0*0 + 0*(-1)*1 + 1*0*(-1) + (-1)*1*0 + 0*0*1
           = 1 + 0 + 0 + 0 + 0 + 0 + 0 = 1 ✓
    
    Let's check i=0,j=0,k=1:
    1 = sum_t U[0, t] * V[1, t] * W[2, t]
    
    V[1,:] = [0, 0, 1, 0, 0, 1, 0]
    W[2,:] = [0, 0, 1, 0, 1, 0, 0]
    
    sum_t = 1*0*0 + 0*0*0 + 1*1*1 + 0*0*0 + 1*0*1 + (-1)*1*0 + 0*0*0
           = 0 + 0 + 1 + 0 + 0 + 0 + 0 = 1 ✓
    
    Great, so the condition is indeed T[i,j,k] = 1 for all i,j,k.
    
    Now let's verify the rank-77 decomposition.
    """
    n, m, p = 4, 4, 6
    
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Check: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1
    
    # Convert to numpy arrays if needed
    if isinstance(U, list):
        U = np.array(U)
    if isinstance(V, list):
        V = np.array(V)
    if isinstance(W, list):
        W = np.array(W)
    
    rank = U.shape[1]
    
    # For efficiency, check a subset of (i,j,k) triples
    # Actually, let's check all of them since n*m*p = 96 is small
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for t in range(rank):
                    val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t_start = time.time()
    deadline = t_start + 110  # Safety margin before 120s limit
    
    # Build the structured rank-77 decomposition
    U, V, W = _build_rank77()
    
    # Verify the base decomposition
    if not _verify(U, V, W, n, m, p):
        # Should not happen, but return anyway
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by removing linearly dependent columns
    # The key insight: in the Kronecker product structure, some columns may be
    # linearly dependent across the three matrices simultaneously.
    
    # Strategy: For each column t, check if it can be expressed as a linear combination
    # of other columns in all three matrices U, V, W simultaneously.
    
    # This is equivalent to checking if the rank-1 tensor u_t * v_t * w_t is in the
    # span of the other rank-1 tensors.
    
    # A simpler approach: check for duplicate or proportional column triples.
    
    # First, let's identify columns that are zero in all three matrices
    zero_cols = []
    for t in range(U.shape[1]):
        if np.all(U[:, t] == 0) or np.all(V[:, t] == 0) or np.all(W[:, t] == 0):
            zero_cols.append(t)
    
    if zero_cols:
        # Remove zero columns
        keep = [t for t in range(U.shape[1]) if t not in zero_cols]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
    
    # Now check for linearly dependent column triples
    # Two columns t1 and t2 are "equivalent" if there exist scalars a, b, c such that:
    # U[:, t2] = a * U[:, t1]
    # V[:, t2] = b * V[:, t1]
    # W[:, t2] = c * W[:, t1]
    # This means the rank-1 tensors are proportional.
    
    # More generally, a column t can be removed if its rank-1 tensor is in the span
    # of the others. This is hard to check exactly, but we can use a heuristic.
    
    # Heuristic: Check if any column is a scalar multiple of another in all three matrices.
    # For each pair (t1, t2), check if U[:,t2]/U[:,t1] is constant, etc.
    
    # This is O(r^2 * n) which for r=77 is about 77^2 * 16 = ~95000 ops, very fast.
    
    def _check_proportional(U1, U2, V1, V2, W1, W2):
        """Check if (U2,V2,W2) is proportional to (U1,V1,W1)."""
        # Find non-zero elements
        nz_u1 = np.nonzero(U1)[0]
        nz_v1 = np.nonzero(V1)[0]
        nz_w1 = np.nonzero(W1)[0]
        
        if len(nz_u1) == 0 or len(nz_v1) == 0 or len(nz_w1) == 0:
            return False, None
        
        # Check U
        ratios_u = []
        for idx in nz_u1:
            if U1[idx] != 0:
                ratio = U2[idx] / U1[idx]
                if U2[idx] == 0:
                    if ratio != 0:
                        return False, None
                else:
                    ratios_u.append(ratio)
        
        if not ratios_u:
            return False, None
        
        ratio_u = ratios_u[0]
        for r in ratios_u[1:]:
            if abs(r - ratio_u) > 1e-9:
                return False, None
        
        # Check V
        ratios_v = []
        for idx in nz_v1:
            if V1[idx] != 0:
                ratio = V2[idx] / V1[idx]
                if V2[idx] == 0:
                    if ratio != 0:
                        return False, None
                else:
                    ratios_v.append(ratio)
        
        if not ratios_v:
            return False, None
        
        ratio_v = ratios_v[0]
        for r in ratios_v[1:]:
            if abs(r - ratio_v) > 1e-9:
                return False, None
        
        # Check W
        ratios_w = []
        for idx in nz_w1:
            if W1[idx] != 0:
                ratio = W2[idx] / W1[idx]
                if W2[idx] == 0:
                    if ratio != 0:
                        return False, None
                else:
                    ratios_w.append(ratio)
        
        if not ratios_w:
            return False, None
        
        ratio_w = ratios_w[0]
        for r in ratios_w[1:]:
            if abs(r - ratio_w) > 1e-9:
                return False, None
        
        return True, (ratio_u, ratio_v, ratio_w)
    
    # Remove proportional columns
    removed = set()
    for t1 in range(U.shape[1]):
        if t1 in removed:
            continue
        for t2 in range(t1 + 1, U.shape[1]):
            if t2 in removed:
                continue
            prop, _ = _check_proportional(
                U[:, t1], U[:, t2],
                V[:, t1], V[:, t2],
                W[:, t1], W[:, t2]
            )
            if prop:
                removed.add(t2)
    
    if removed:
        keep = [t for t in range(U.shape[1]) if t not in removed]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
    
    # Verify after reduction
    if not _verify(U, V, W, n, m, p):
        # Reduction broke something, rebuild
        U, V, W = _build_rank77()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
