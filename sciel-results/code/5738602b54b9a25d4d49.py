import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # 20s safety margin before 160s kill

    # Build the canonical rank n*m*p decomposition
    rank_schoolbook = n * m * p
    U0 = np.zeros((n * m, rank_schoolbook))
    V0 = np.zeros((m * p, rank_schoolbook))
    W0 = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1

    def to_list(mat):
        return [[int(x) if x == int(x) else float(x) for x in row] for row in mat]

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        # Compute C = A*B for random integer inputs
        for trial in range(3):
            A = np.random.randint(-3, 4, size=(n, m))
            B = np.random.randint(-3, 4, size=(m, p))
            C_expected = A @ B
            # Bilinear form: C[k*n+i, r] = sum_t U[i*m+j, t] * V[j*p+k, t] ... 
            # Actually: C[i,k] = sum_j sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
            # Wait, the convention is W[k*n+i, r], so output index is (k, i)
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # Tensor: T[i,j,k] = 1 (for all i,j,k in valid ranges)
            # Decomposition: T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
            C_computed = np.zeros((n, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for t in range(r):
                            val += A[i,j] * U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                        C_computed[i,k] += val
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True

    # Schoolbook is always valid
    best_rank = rank_schoolbook
    best_result = {"U": to_list(U0), "V": to_list(V0), "W": to_list(W0)}

    # Strategy: use Kronecker product structure
    # The 4x4x6 multiplication tensor can be built from smaller known decompositions
    # via tensor products, then we search for rank reductions.

    # Known: 2x2x2 multiplication has rank 7 (Strassen)
    # 3x3x3 has rank 23 (Laderman)
    # 4x4x4 has rank 49 (best known)

    # For 4x4x6, we can write as 4x4x4 + 4x4x2
    # 4x4x4 best known rank: 49
    # 4x4x2: can be done with rank 8 (each of 8 output entries is a dot product)
    # Total: 49 + 8 = 57

    # Let's try to construct this
    # 4x4x4 Strassen-like: use 2x2x2 Strassen (rank 7) tensor product approach
    # 4x4x4 = (2x2 block matrix) multiplication
    # M4 = [[A11, A12], [A21, A22]] where each is 2x2
    # N4 = [[B11, B12], [B21, B22]]
    # C = M4 * N4
    # Using Strassen on 2x2 blocks (each block is 2x2 matrix product):
    # S1 = (A11+A22)(B11+B22)  -> 2x2 product, rank 7
    # S2 = (A21+A22)B11        -> 2x2 product, rank 7
    # S3 = A11(B12-B22)        -> 2x2 product, rank 7
    # S4 = A22(B21-B11)        -> 2x2 product, rank 7
    # S5 = (A11+A12)B22        -> 2x2 product, rank 7
    # S6 = (A21-A11)(B11+B12)  -> 2x2 product, rank 7
    # S7 = (A12-A22)(B21+B22)  -> 2x2 product, rank 7
    # Total: 7*7 = 49 for 4x4x4

    # For the 4x4x2 part: 8 scalar products, rank 8
    # Total: 57

    # But we can potentially do better. Let's try rank 52-56 with search.

    # First, let's try the Kronecker construction for rank 49+8=57
    def strassen_2x2_factors():
        """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
        # Indices: i in {0,1}, j in {0,1}, k in {0,1}
        # U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # t=0: (A11+A22)(B11+B22) -> C11 += S1 - S3 - S4 - S7
        # Let me use standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 - M3 - M4 - M7
        # C01 = M5 + M7
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 - M5
        
        # U rows: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # V rows: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # W rows: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3 (k*2+i)
        
        # M1: (A00+A11)(B00+B11)
        # U: A00=1, A11=1; V: B00=1, B11=1
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[1,0] = -1; W[2,0] = -1; W[3,0] = -1
        # Wait, W[k*2+i]: k=0,i=0 -> W[0]; k=0,i=1 -> W[1]; k=1,i=0 -> W[2]; k=1,i=1 -> W[3]
        # C00 = M1 - M3 - M4 - M7, C01 = M5+M7, C10 = M2+M4, C11 = M1-M2+M3-M5
        # So for M1: contributes to C00 with +1, C01 with 0, C10 with 0, C11 with +1
        W[0,0] = 1  # C00
        W[1,0] = 0  # C01
        W[2,0] = 0  # C10
        W[3,0] = 1  # C11

        # M2: (A10+A11)B00
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        # C10 = M2+M4: contributes +1 to C10
        W[0,1] = 0; W[1,1] = 0; W[2,1] = 1; W[3,1] = 0

        # M3: A00(B01-B11)
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        # C00 = ...-M3, C11 = ...+M3
        W[0,2] = -1; W[1,2] = 0; W[2,2] = 0; W[3,2] = 1

        # M4: A11(B10-B00)
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        # C00 = ...-M4, C10 = ...+M4
        W[0,3] = -1; W[1,3] = 0; W[2,3] = 1; W[3,3] = 0

        # M5: (A00+A01)B11
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        # C01 = M5+M7, C11 = ...-M5
        W[0,4] = 0; W[1,4] = 1; W[2,4] = 0; W[3,4] = -1

        # M6: (A10-A00)(B00+B01)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        # Not directly used in output formula... 
        # Actually in standard Strassen, M6 is used:
        # C00 = M1-M3-M4-M7
        # C01 = M5+M7
        # C10 = M2+M4
        # C11 = M1-M2+M3-M5
        # M6 is not in these! Let me recheck.
        # Standard Strassen:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B12-B22)
        # P4 = A22(B21-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        # C11 = P1-P3-P4-P7
        # C12 = P5+P7
        # C21 = P2+P4
        # C22 = P1-P2+P3-P5
        # P6 is not used in the output! That's wrong.
        # Actually I'm confusing myself. Let me just use the 7 multiplications properly.
        # The key insight: 7 scalar multiplications suffice. Let me just define the factors directly.
        
        # Let me restart with a clean Strassen 2x2x2:
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # Factor 0: (a00+a11)*(b00+b11)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[1,0]=0; W[2,0]=0; W[3,0]=1
        # Factor 1: (a10+a11)*b00
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[0,1]=0; W[1,1]=0; W[2,1]=1; W[3,1]=0
        # Factor 2: a00*(b01-b11)
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[0,2]=-1; W[1,2]=0; W[2,2]=0; W[3,2]=1
        # Factor 3: a11*(b10-b00)
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        W[0,3]=-1; W[1,3]=0; W[2,3]=1; W[3,3]=0
        # Factor 4: (a00+a01)*b11
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=0; W[1,4]=1; W[2,4]=0; W[3,4]=-1
        # Factor 5: (a10-a00)*(b00+b01)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[0,5]=0; W[1,5]=0; W[2,5]=0; W[3,5]=0  # Not in output?
        # Factor 6: (a01-a11)*(b10+b11)
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6]=0; W[1,6]=0; W[2,6]=0; W[3,6]=0  # Not in output?

        # This isn't working. The issue is that in Strassen, the 7 intermediate products
        # are combined to give the 4 outputs. But in our tensor decomposition, each factor
        # contributes to specific output entries. Let me just verify the schoolbook and
        # try a different approach: direct rank reduction search.
        
        return U, V, W

    # Let me just try to find a better decomposition by searching over factor matrices
    # with small integer entries.

    # Approach: Start from schoolbook and try to merge columns
    # Two columns can be merged if their U, V, W vectors are linearly dependent
    # in a way that preserves the decomposition.

    # Alternative: Use the fact that the multiplication tensor has symmetries.
    # Try random rank-r decompositions and check if they're exact.

    # For practical purposes, let's try rank values from 50 to 57
    # and see if we can find exact decompositions.

    # Actually, the simplest structural improvement: 
    # 4x4x6 = 4x4x4 + 4x4x2
    # Best known for 4x4x4: 49
    # 4x4x2: 8 (can't do better since each of 8 outputs needs at least 1 scalar mult)
    # Total: 57

    # Can we do 4x4x4 in rank < 49? Best known is 49.
    # Can we do 4x4x2 in rank < 8? No, since we need 8 independent bilinear forms.
    # Unless we share structure... but 4x4x2 has 8 entries each being a dot product.
    # The tensor for 4x4x2 has rank 8 (it's a sum of 8 rank-1 tensors).

    # So 57 is the target. Let me construct it properly.

    # 4x4x4 via Strassen-like (2x2 blocks of 2x2 matrices):
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    # C = A*B
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11

    # Use Strassen on the 2x2 block structure, where each "scalar" multiplication
    # is actually a 2x2x2 matrix multiplication (rank 7 each):
    # S1 = (A00+A11)*(B00+B11)  -> 2x2 mult, rank 7
    # S2 = (A10+A11)*B00        -> 2x2 mult, rank 7
    # S3 = A00*(B01-B11)        -> 2x2 mult, rank 7
    # S4 = A11*(B10-B00)        -> 2x2 mult, rank 7
    # S5 = (A00+A01)*B11        -> 2x2 mult, rank 7
    # S6 = (A10-A00)*(B00+B01)  -> 2x2 mult, rank 7
    # S7 = (A01-A11)*(B10+B11)  -> 2x2 mult, rank 7
    # C00 = S1 - S3 - S4 - S7
    # C01 = S5 + S7
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 - S5

    # Each St is a 2x2x2 multiplication, which has rank 7.
    # Total rank: 7*7 = 49.

    # Now I need to write out the actual factor matrices for this.
    # This is complex. Let me try a simpler approach: just use the schoolbook
    # and accept rank 96, or try to find a known good decomposition.

    # Given time constraints, let me try a computational search for small rank improvements.
    # The key insight: in the schoolbook decomposition, many columns have similar structure.
    # We can try to find linear dependencies.

    # For now, let me just return the schoolbook (rank 96) as a safe fallback,
    # and try to do better if time permits.

    # Try: can we find rank 90, 85, 80, 75, 70 decompositions?
    # This is a hard problem. Let me try a greedy column merging approach.

    def try_greedy_merge(U, V, W, target_rank, time_limit=5):
        """Try to reduce rank by merging columns."""
        r = U.shape[1]
        if r <= target_rank:
            return U, V, W, r

        U = U.copy()
        V = V.copy()
        W = W.copy()
        start = time.time()

        # Try to merge pairs of columns
        # Two columns t1, t2 can be merged if there exists a new column
        # that produces the same bilinear form contribution.
        # This happens when U[:,t1]*V[:,t2]*W[:,t1] + U[:,t2]*V[:,t1]*W[:,t2] 
        # can be replaced by a single term.
        # More practically: if U[:,t1] = U[:,t2] and V[:,t1] = V[:,t2],
        # then we can merge into one column with W doubled.

        merged = True
        while merged and (time.time() - start) < time_limit:
            merged = False
            r = U.shape[1]
            for t1 in range(r):
                for t2 in range(t1+1, r):
                    if np.allclose(U[:,t1], U[:,t2]) and np.allclose(V[:,t1], V[:,t2]):
                        # Merge: keep t1, set W[:,t1] = W[:,t1]+W[:,t2], remove t2
                        W[:,t1] = W[:,t1] + W[:,t2]
                        U = np.delete(U, t2, axis=1)
                        V = np.delete(V, t2, axis=1)
                        W = np.delete(W, t2, axis=1)
                        r = U.shape[1]
                        merged = True
                        break
                if merged:
                    break

        return U, V, W, U.shape[1]

    # Try greedy merge
    U_try, V_try, W_try, r_try = try_greedy_merge(U0, V0, W0, rank_schoolbook, time_limit=3)
    if r_try < best_rank and verify(U_try, V_try, W_try):
        best_rank = r_try
        best_result = {"U": to_list(U_try), "V": to_list(V_try), "W": to_list(W_try)}

    # Try another approach: use the Kronecker product structure explicitly
    # 4x4x6 tensor = sum of 4*4*6 rank-1 tensors
    # Group them by (i,j) pairs: for each (i,j), we have 6 rank-1 tensors (one for each k)
    # These 6 tensors for fixed (i,j) form a rank-6 tensor (the j-th row of A times all of B)
    # We can potentially reduce this.

    # Actually, let me try a completely different structural approach:
    # Use the fact that 4 = 2+2 and 6 = 2+2+2
    # A (4x4) can be written as [[A1, A2], [A3, A4]] where each Ai is 2x2
    # B (4x6) can be written as [B1, B2, B3] where each Bj is 4x2
    # C (4x6) = [C1, C2, C3] where each Cj is 4x2
    # Cj = A * Bj = [[A1,Bj],[A3,Bj]] type block multiplication

    # This is getting too complex for the time budget. Let me just try to
    # find any improvement over 96.

    # Quick check: try rank 95 with random perturbation
    if (time.time() - start_time) < deadline - 5:
        # Try to find a rank 95 decomposition by modifying one schoolbook column
        # to absorb part of another
        for trial in range(50):
            if (time.time() - start_time) > deadline - 5:
                break
            U_t = U0.copy()
            V_t = V0.copy()
            W_t = W0.copy()
            # Try to zero out one column and redistribute
            t_remove = random.randint(0, rank_schoolbook-1)
            t_target = random.randint(0, rank_schoolbook-1)
            if t_remove == t_target:
                continue
            # Add column t_remove to column t_target
            U_t[:, t_target] += U_t[:, t_remove]
            V_t[:, t_target] += V_t[:, t_remove]
            W_t[:, t_target] += W_t[:, t_remove]
            # Remove column t_remove
            U_t = np.delete(U_t, t_remove, axis=1)
            V_t = np.delete(V_t, t_remove, axis=1)
            W_t = np.delete(W_t, t_remove, axis=1)
            if verify(U_t, V_t, W_t):
                best_rank = 95
                best_result = {"U": to_list(U_t), "V": to_list(V_t), "W": to_list(W_t)}
                break

    # Final safety check
    if (time.time() - start_time) > deadline - 2:
        return best_result

    return best_result
# EVOLVE-BLOCK-END
