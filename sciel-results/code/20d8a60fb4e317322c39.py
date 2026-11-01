"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _verify(U, V, W):
    """Verify that the decomposition exactly computes 4x4 matrix multiplication."""
    n, m, p = 4, 4, 4
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    rank = U.shape[1]
    U4 = U.reshape(n, m, rank)
    V4 = V.reshape(m, p, rank)
    W4 = W.reshape(p, n, rank)
    
    # Contract: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should be 1 for all i,j,k
    recon = np.einsum('ijr,jkr,kir->ijk', U4, V4, W4)
    target = np.ones((n, m, p))
    max_err = np.max(np.abs(recon - target))
    return max_err < 1e-9

def _strassen_2x2():
    """Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    Returns U2 (4x7), V2 (4x7), W2 (4x7) such that
    sum_r U2[i,j,r] * V2[j,k,r] * W2[k,i,r] = 1 for all i,j,k in {0,1}
    """
    # Strassen formulas for 2x2:
    # m0 = (a00+a11)(b00+b11)
    # m1 = (a10+a11)b00
    # m2 = a00(b10-b11)
    # m3 = a11(b01-b00)
    # m4 = (a00+a01)b11
    # m5 = (a10-a00)(b00+b01)
    # m6 = (a01-a11)(b10+b11)
    #
    # C00 = m0 + m3 - m4 + m6
    # C01 = m2 + m4
    # C10 = m1 + m3
    # C11 = m0 - m1 + m2 + m5
    
    # U2[i,j,r] = coefficient of A[i,j] in product r
    U2 = np.zeros((4, 7))
    # A flat indices: 0=A00, 1=A01, 2=A10, 3=A11
    U2[0, 0] = 1; U2[3, 0] = 1       # m0: A00 + A11
    U2[2, 1] = 1; U2[3, 1] = 1       # m1: A10 + A11
    U2[0, 2] = 1                      # m2: A00
    U2[3, 3] = 1                      # m3: A11
    U2[0, 4] = 1; U2[1, 4] = 1       # m4: A00 + A01
    U2[0, 5] = -1; U2[2, 5] = 1      # m5: A10 - A00
    U2[1, 6] = 1; U2[3, 6] = -1      # m6: A01 - A11
    
    # V2[j,k,r] = coefficient of B[j,k] in product r
    V2 = np.zeros((4, 7))
    # B flat indices: 0=B00, 1=B01, 2=B10, 3=B11
    V2[0, 0] = 1; V2[3, 0] = 1       # m0: B00 + B11
    V2[0, 1] = 1                      # m1: B00
    V2[2, 2] = 1; V2[3, 2] = -1      # m2: B10 - B11
    V2[0, 3] = -1; V2[1, 3] = 1      # m3: B01 - B00
    V2[3, 4] = 1                      # m4: B11
    V2[0, 5] = 1; V2[1, 5] = 1       # m5: B00 + B01
    V2[2, 6] = 1; V2[3, 6] = 1       # m6: B10 + B11
    
    # W2[k,i,r] = coefficient of m_r in C[i,k]
    # W flat index: k*2 + i
    W2 = np.zeros((4, 7))
    # C00 (i=0,k=0, idx=0): m0 + m3 - m4 + m6
    W2[0, 0] = 1; W2[0, 3] = 1; W2[0, 4] = -1; W2[0, 6] = 1
    # C01 (i=1,k=0, idx=1): m2 + m4
    W2[1, 2] = 1; W2[1, 4] = 1
    # C10 (i=0,k=1, idx=2): m1 + m3
    W2[2, 1] = 1; W2[2, 3] = 1
    # C11 (i=1,k=1, idx=3): m0 - m1 + m2 + m5
    W2[3, 0] = 1; W2[3, 1] = -1; W2[3, 2] = 1; W2[3, 5] = 1
    
    return U2, V2, W2

def _build_4x4_kronecker():
    """Build the rank-49 4x4 decomposition via Kronecker product of 2x2 Strassen.
    
    Uses the tensor product structure: T_4 = T_2 ⊗ T_2
    """
    U2, V2, W2 = _strassen_2x2()
    
    # For 4x4: A[i,j] where i=2*i'+i'', j=2*j'+j''
    # B[j,k] where j=2*j'+j'', k=2*k'+k''
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # The Kronecker decomposition:
    # U4[i,j, q*7+s] = U2[i',q] * U2[i'',s] where i=2*i'+i'', j=2*j'+j''
    # But wait - this isn't quite right for the tensor product of MM tensors.
    #
    # Let me think more carefully. The 4x4 MM tensor T4[i,j,k] = 1 for all i,j,k.
    # This is NOT the tensor product of two 2x2 MM tensors.
    #
    # The correct approach: 4x4 MM can be seen as 2x2 block MM where each block is 2x2.
    # At the block level, we use Strassen (7 products of 2x2 matrices).
    # Each 2x2 matrix product is decomposed with rank 7.
    # Total rank = 7 * 7 = 49.
    #
    # More precisely:
    # Let A be 4x4, partitioned into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # Each Aij is 2x2.
    # Similarly B = [[B00, B01], [B10, B11]]
    # C = AB = [[C00, C01], [C10, C11]]
    #
    # Strassen at block level:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # Each Mq is a 2x2 matrix = product of two 2x2 matrices.
    # Decompose each Mq with Strassen rank 7.
    #
    # For Mq = Xq * Yq where Xq and Yq are 2x2 linear combos:
    # Xq[i'',j''] = sum_{br,bc} a_super[q,br,bc] * A[2*br+i'', 2*bc+j'']
    # Yq[j'',k''] = sum_{kr,kc} b_super[q,kr,kc] * B[2*kr+j'', 2*kc+k'']
    #
    # Then Mq[i'',k''] = sum_{j''} Xq[i'',j''] * Yq[j'',k'']
    # = sum_{j'',br,bc,kr,kc} a_super[q,br,bc]*b_super[q,kr,kc] * A[2*br+i'',2*bc+j''] * B[2*kr+j'',2*kc+k'']
    #
    # Now decompose the 2x2 product Xq*Yq with Strassen:
    # Mq[i'',k''] = sum_s U2[i'',j'',s] * V2[j'',k'',s] * W2[k'',i'',s] ... no wait
    # Actually: Mq = Xq * Yq, and we decompose this 2x2 MM with rank 7.
    # The factorization of the 2x2 MM tensor gives:
    # (Xq * Yq)[i'',k''] = sum_s u_s[i'',j''] * v_s[j'',k''] * w_s[k'',i'']
    # where u_s, v_s, w_s are the Strassen factors for 2x2.
    #
    # So: Mq[i'',k''] = sum_s U2[i'',j'',s] * V2[j'',k'',s] * W2[k'',i'',s]
    #   where the "A" input to this 2x2 MM is Xq and "B" input is Yq.
    #
    # Therefore: Mq[i'',k''] = sum_s U2[i'',j'',s] * V2[j'',k'',s] * W2[k'',i'',s]
    #
    # And C[i,k] where i=2*br+i'', k=2*kc+k'':
    # C[i,k] = sum_q c_super[q, br, kc] * Mq[i'',k'']
    #        = sum_q sum_s c_super[q,br,kc] * U2[i'',j'',s] * V2[j'',k'',s] * W2[k'',i'',s]
    #
    # Now the full 4x4 factorization:
    # U4[i,j, q*7+s] where i=2*br+i'', j=2*bc+j'':
    #   = a_super[q,br,bc] * U2[i'',j'',s]
    #
    # V4[j,k, q*7+s] where j=2*kr+j'', k=2*kc+k'':
    #   = b_super[q,kr,kc] * V2[j'',k'',s]
    #
    # W4[k,i, q*7+s] where i=2*br+i'', k=2*kc+k'':
    #   = c_super[q,br,kc] * W2[k'',i'',s]
    #
    # Let's verify: sum_{q,s} U4[i,j,q*7+s] * V4[j,k,q*7+s] * W4[k,i,q*7+s]
    # = sum_{q,s,br,bc,kr,kc,i'',j'',k''} a_super[q,br,bc]*U2[i'',j'',s] * b_super[q,kr,kc]*V2[j'',k'',s] * c_super[q,br,kc]*W2[k'',i'',s]
    # where i=2*br+i'', j=2*bc+j''=2*kr+j'', k=2*kc+k''
    # Note: j must be consistent: 2*bc+j'' = 2*kr+j'', so bc=kr and j''=j''.
    # So sum over j'' is the same as summing over the shared index.
    #
    # = sum_q c_super[q,br,kc] * sum_s (sum_{j''} U2[i'',j'',s]*V2[j'',k'',s]) * W2[k'',i'',s] * a_super[q,br,bc]*b_super[q,kr,kc]
    #
    # Wait, this doesn't look right. The Strassen 2x2 decomposition satisfies:
    # (X*Y)[i'',k''] = sum_s U2[i'',j'',s] * V2[j'',k'',s] * W2[k'',i'',s]
    # But this is for a SPECIFIC 2x2 MM where X and Y are the inputs.
    # The factors U2, V2, W2 are fixed (they don't depend on X, Y).
    # The formula is: (X*Y)[i'',k''] = sum_s (sum_{j''} U2[i'',j'',s]*X[i'',j''] ... no.
    #
    # Let me re-derive. The 2x2 MM tensor T2[i,j,k] = 1 for all i,j,k in {0,1}.
    # A rank-7 decomposition: T2[i,j,k] = sum_r U2[i,j,r] * V2[j,k,r] * W2[k,i,r]
    #
    # For matrices X, Y: (XY)[i,k] = sum_j X[i,j] Y[j,k] = sum_j sum_r U2[i,j,r] V2[j,k,r] W2[k,i,r] X[i,j] Y[j,k]
    # Hmm, that's not right either. The factorization means:
    # (XY)[i,k] = sum_r (sum_j U2[i,j,r] X[i,j]) ... no.
    #
    # Actually the standard tensor decomposition of MM:
    # The MM tensor T[i,j,k] (where output C[i,k] = sum_j A[i,j] B[j,k]) has
    # T[i,j,k] = 1 for all valid (i,j,k).
    # A rank-r decomposition: T[i,j,k] = sum_{r=1}^r U[i,j,r] * V[j,k,r] * W[k,i,r]
    #
    # This means: C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k]) * W[k,i,r]
    # No wait, that's not right. Let me be very careful.
    #
    # The bilinear form for MM: C[i,k] = sum_j A[i,j] B[j,k]
    # A tensor decomposition of the MM operation means:
    # C[i,k] = sum_r M_r * W[k,i,r]
    # where M_r = (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k])
    #
    # So: C[i,k] = sum_r W[k,i,r] * (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k])
    #
    # For this to equal sum_j A[i,j] B[j,k] for all A, B:
    # We need: sum_r W[k,i,r] * U[i,j,r] * V[j',k,r] = delta_{j,j'} ... 
    # Hmm, this is getting confused. Let me just use the standard formulation.
    #
    # Standard: The MM tensor has entries T[i,j,k] and a factorization is:
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # where T[i,j,k] = 1 for the MM tensor (since C[i,k] = sum_j A[i,j]B[j,k]).
    #
    # Verification: sum_r U[i,j,r] V[j,k,r] W[k,i,r] should equal 1 for all i,j,k.
    #
    # For the Kronecker construction of 4x4 from 2x2:
    # The 4x4 MM tensor T4[i,j,k] = 1 for all i,j,k in {0,1,2,3}.
    # This is NOT a tensor product of T2 with itself.
    #
    # However, we can use the block structure. Let me just directly construct it.
    
    # Strassen block-level coefficients:
    a_super = np.zeros((7, 2, 2))  # a_super[q, br, bc]
    b_super = np.zeros((7, 2, 2))  # b_super[q, kr, kc]
    c_super = np.zeros((2, 2, 7))  # c_super[br, kc, q]
    
    # m0 = (A00+A11)(B00+B11)
    a_super[0] = [[1, 0], [0, 1]]
    b_super[0] = [[1, 0], [0, 1]]
    c_super[0, 0, 0] += 1; c_super[1, 1, 0] += 1
    # m1 = (A10+A11)B00
    a_super[1] = [[0, 0], [1, 1]]
    b_super[1] = [[1, 0], [0, 0]]
    c_super[1, 0, 1] += 1; c_super[1, 1, 1] += 1
    # m2 = A00(B10-B11)
    a_super[2] = [[1, 0], [0, 0]]
    b_super[2] = [[0, 0], [1, -1]]
    c_super[0, 1, 2] += 1; c_super[1, 1, 2] += 1
    # m3 = A11(B01-B00)
    a_super[3] = [[0, 0], [0, 1]]
    b_super[3] = [[-1, 1], [0, 0]]
    c_super[0, 0, 3] += 1; c_super[1, 0, 3] += 1
    # m4 = (A00+A01)B11
    a_super[4] = [[1, 1], [0, 0]]
    b_super[4] = [[0, 0], [0, 1]]
    c_super[0, 0, 4] -= 1; c_super[0, 1, 4] += 1
    # m5 = (A10-A00)(B00+B01)
    a_super[5] = [[-1, 0], [1, 0]]
    b_super[5] = [[1, 1], [0, 0]]
    c_super[1, 1, 5] += 1
    # m6 = (A01-A11)(B10+B11)
    a_super[6] = [[0, 1], [0, -1]]
    b_super[6] = [[0, 0], [1, 1]]
    c_super[0, 0, 6] += 1
    
    # Now build U4, V4, W4
    # U4[i, j, r] where r = q*7+s, i = 2*br+i2, j = 2*bc+j2
    # U4[i, j, r] = a_super[q, br, bc] * U2[i2, j2, s]
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for q in range(7):
        for s in range(7):
            r = q * 7 + s
            for br in range(2):
                for bc in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = 2 * br + i2
                            j = 2 * bc + j2
                            U4[i * 4 + j, r] = a_super[q, br, bc] * U2[i2, j2, s]
            
            for kr in range(2):
                for kc in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = 2 * kr + j2
                            k = 2 * kc + k2
                            V4[j * 4 + k, r] = b_super[q, kr, kc] * V2[j2, k2, s]
            
            for br in range(2):
                for kc in range(2):
                    for i2 in range(2):
                        for k2 in range(2):
                            i = 2 * br + i2
                            k = 2 * kc + k2
                            W4[k * 4 + i, r] = c_super[br, kc, q] * W2[k2, i2, s]
    
    return U4, V4, W4

def _try_reduce_rank(U, V, W, target_rank, time_budget):
    """Try to reduce rank by zeroing columns and solving for W.
    
    Strategy: For a given target rank, select a subset of columns to keep,
    zero out the rest in U and V, then solve for W via least squares.
    """
    n, m, p = 4, 4, 4
    deadline = time.time() + time_budget
    orig_rank = U.shape[1]
    
    # The constraint: for all (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # If we fix U and V (with some columns zeroed), we can solve for W.
    # For each (k,i) pair, the constraint is:
    # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all j
    # This gives a system: for fixed (k,i), sum_r (sum_j ... ) 
    # Actually: for fixed (k,i), we need: for all j: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # This is: sum_r W[k,i,r] * (U[i,j,r]*V[j,k,r]) = 1 for each j
    # But W[k,i,r] is the same for all j! So we need:
    # For each (k,i): the vector (sum_r W[k,i,r]*U[i,j,r]*V[j,k,r])_{j=0}^{3} = (1,1,1,1)
    # This is a system of 4 equations (one per j) with 49 unknowns (W[k,i,r]).
    # But W[k,i,r] appears in all 4 equations with different coefficients.
    #
    # Actually, let me reformulate. The full system is:
    # For all (i,j,k): sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    #
    # This is linear in W. Let's set up the linear system.
    # Unknowns: W[k,i,r] for all k,i,r. Total: 4*4*49 = 784 unknowns.
    # Equations: 4*4*4 = 64 equations.
    # Underdetermined! So we can potentially satisfy it with fewer rank terms.
    #
    # Better approach: fix U and V, then for each (k,i) independently solve:
    # min ||A_{ki} * w_{ki} - 1||^2 where w_{ki} is the vector W[k,i,:]
    # and A_{ki}[j, r] = U[i,j,r] * V[j,k,r]
    #
    # This gives 16 independent 4x49 least squares problems.
    
    n_attempts = 50
    np.random.seed(42)
    
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        # Select which columns to keep
        if target_rank >= orig_rank:
            keep = set(range(orig_rank))
        else:
            keep = set(np.random.choice(orig_rank, target_rank, replace=False))
        
        # Create masked U and V
        U_masked = U.copy()
        V_masked = V.copy()
        for r in range(orig_rank):
            if r not in keep:
                U_masked[:, r] = 0
                V_masked[:, r] = 0
        
        # Solve for W
        W_sol = np.zeros((16, orig_rank))
        valid = True
        for k in range(4):
            for i in range(4):
                # A[j, r] = U[i,j,r] * V[j,k,r]
                A_mat = np.zeros((4, orig_rank))
                for j in range(4):
                    for r in range(orig_rank):
                        A_mat[j, r] = U_masked[i * 4 + j, r] * V_masked[j * 4 + k, r]
                
                # Solve A_mat @ w = ones(4)
                try:
                    w, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, np.ones(4), rcond=None)
                    # Check if solution is valid
                    recon = A_mat @ w
                    if np.max(np.abs(recon - 1.0)) > 1e-8:
                        valid = False
                        break
                    W_sol[k * 4 + i, :] = w
                except:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Verify full solution
            W_sol_masked = W_sol.copy()
            for r in range(orig_rank):
                if r not in keep:
                    W_sol_masked[:, r] = 0
            
            if _verify(U_masked, V_masked, W_sol_masked):
                # Count actual rank (non-zero columns)
                actual_rank = sum(1 for r in keep if np.any(U_masked[:, r] != 0) or np.any(V_masked[:, r] != 0))
                return U_masked, V_masked, W_sol_masked, actual_rank
    
    return None, None, None, None

def _schoolbook():
    """Return the schoolbook rank-64 decomposition."""
    n, m, p = 4, 4, 4
    rank = 64
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Step 1: Build rank-49 Kronecker Strassen decomposition
    U49, V49, W49 = _build_4x4_kronecker()
    
    if not _verify(U49, V49, W49):
        # Fallback to schoolbook
        U, V, W = _schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 49
    best_U, best_V, best_W = U49, V49, W49
    
    # Step 2: Try to reduce rank
    # Try various target ranks
    for target_rank in [32, 36, 40, 44, 48]:
        if time.time() > deadline - 10:
            break
        
        U_r, V_r, W_r, actual_rank = _try_reduce_rank(U49, V49, W49, target_rank, 10)
        if U_r is not None and actual_rank < best_rank:
            best_rank = actual_rank
            best_U, best_V, best_W = U_r, V_r, W_r
    
    # Step 3: Try direct random search for even lower rank
    if time.time() < deadline - 20:
        for target_rank in [28, 30, 32]:
            if time.time() > deadline - 10:
                break
            
            # Random initialization with target rank
            np.random.seed(target_rank * 1000)
            U_rand = np.random.randn(16, target_rank) * 0.5
            V_rand = np.random.randn(16, target_rank) * 0.5
            
            # Solve for W
            W_sol = np.zeros((16, target_rank))
            valid = True
            for k in range(4):
                for i in range(4):
                    A_mat = np.zeros((4, target_rank))
                    for j in range(4):
                        for r in range(target_rank):
                            A_mat[j, r] = U_rand[i * 4 + j, r] * V_rand[j * 4 + k, r]
                    try:
                        w, _, _, _ = np.linalg.lstsq(A_mat, np.ones(4), rcond=None)
                        recon = A_mat @ w
                        if np.max(np.abs(recon - 1.0)) > 1e-6:
                            valid = False
                            break
                        W_sol[k * 4 + i, :] = w
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and _verify(U_rand, V_rand, W_sol):
                best_rank = target_rank
                best_U, best_V, best_W = U_rand, V_rand, W_sol
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
