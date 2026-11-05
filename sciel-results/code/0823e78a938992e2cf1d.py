import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140
    
    n, m, p = DIMENSIONS
    
    def build_decomposition():
        """
        Build a decomposition for 2x4 @ 4x7.
        
        Strategy: Use a structured decomposition that exploits the small
        number of rows (n=2) to share computations across output columns.
        
        For each pair of adjacent output columns (k, k+1), we can use a
        rank-3 scheme to compute both C[:,k] and C[:,k+1] contributions
        from a given A-column pair, sharing intermediate sums.
        
        We split the 4 inner columns into 2 pairs: (j=0,1) and (j=2,3).
        For each inner pair and each output pair, use a 4-term scheme.
        For the last output column (k=6), use 4 direct terms per inner pair.
        
        Total: 2 inner pairs × (3 output pairs × 4 + 1 single × 4) = 2×16 = 32?
        Let me verify this more carefully.
        """
        rank = 42
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        col = 0
        
        # For the 2x2 @ 2x2 subproblem (A 2x2 block times B 2x2 block),
        # use the standard 4-term schoolbook or 7-term Strassen.
        # But we can do better by sharing across output columns.
        
        # Key idea: For a fixed inner-column pair (j0, j0+1) and output
        # column pair (k0, k0+1), the 4 products A[i,j0]*B[j0,k] and
        # A[i,j0+1]*B[j0+1,k] for i in {0,1}, k in {k0,k0+1} can be
        # computed with 4 rank-1 terms (schoolbook) or potentially fewer
        # with sharing.
        
        # Actually, let me use a different approach: 
        # Split into 4 inner columns, and for each inner column j,
        # compute A[:,j] @ B[j,:] which is a 2x1 @ 1x7 = 2x7 outer product.
        # This takes 14 rank-1 terms per inner column, total 56.
        # 
        # Better: group inner columns in pairs. For inner pair (j0,j0+1),
        # we compute A[:,j0:j0+2] @ B[j0:j0+2,:] which is 2x2 @ 2x7.
        # Split the 7 output cols into 3 pairs + 1.
        # For each 2x2x2 block: use 7 Strassen terms.
        # For each 2x2x1 block: use 4 schoolbook terms.
        # Per 2x2x7: 3*7 + 4 = 25 terms.
        # Two inner pairs: 50 terms. (This is what we had.)
        
        # To reduce, let me try a non-standard 2x2x7 decomposition.
        # The tensor for 2x2x7 matrix mult has rank at most 25 (from above).
        # Can we do better?
        
        # Let me try: for 2x2 @ 2x7, use a "sliding" approach.
        # Define for output columns k=0..6:
        #   C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
        # 
        # Define intermediate values:
        #   P0 = A[0,0] + A[1,1]  (shared across all k)
        #   P1 = A[1,0] + A[1,1]  (shared)
        #   P2 = A[0,0]           (shared)
        #   P3 = A[1,1]           (shared)
        #   Q_k = B[0,k] + B[1,k] for each k
        #   R_k = B[0,k] for each k
        #   S_k = B[1,k] for each k
        #
        # This doesn't obviously help. Let me just try to find the
        # matricization rank and see if we can hit it.
        
        # Build the full tensor
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    T[a, b, c] = 1.0
        
        # Compute matricization ranks for lower bound
        # Mode-1 matricization: (n*m) x (m*p * p*n)
        T_m1 = T.reshape(n*m, m*p * p*n)
        r1 = np.linalg.matrix_rank(T_m1)
        
        # Mode-2 matricization: (m*p) x (n*m * p*n)  
        T_m2 = T.reshape(m*p, n*m * p*n)
        r2 = np.linalg.matrix_rank(T_m2)
        
        # Mode-3 matricization: (p*n) x (n*m * m*p)
        T_m3 = T.reshape(p*n, n*m * m*p)
        r3 = np.linalg.matrix_rank(T_m3)
        
        lower_bound = max(r1, r2, r3)
        
        # For 2x4x7, let's see what the lower bound is
        # If lower_bound <= 42, we might be able to find a rank-42 decomp
        
        # Let me try a constructive approach with rank 42:
        # Use 2 Strassen blocks for the 2x2x2 core (covering 4 output cols)
        # and handle the remaining 3 output cols with optimized terms.
        
        # Actually, let me try the most direct approach:
        # 4 inner columns, 7 output columns.
        # For each inner column j (0..3) and output column k (0..6),
        # we need A[i,j]*B[j,k] for i=0,1.
        # 
        # Group by inner column pairs: (0,1) and (2,3)
        # For inner pair (0,1) and output pair (k,k+1):
        #   C[0,k]   += A[0,0]*B[0,k]   + A[0,1]*B[1,k]
        #   C[0,k+1] += A[0,0]*B[0,k+1] + A[0,1]*B[1,k+1]
        #   C[1,k]   += A[1,0]*B[0,k]   + A[1,1]*B[1,k]
        #   C[1,k+1] += A[1,0]*B[0,k+1] + A[1,1]*B[1,k+1]
        #
        # This is a 2x2 @ 2x2 subproblem: rank 4 (schoolbook) or 7 (Strassen).
        # Schoolbook is better here since we have many such blocks.
        
        # 2 inner pairs × 4 output groups (3 pairs + 1 single) 
        # = 2 × (3×4 + 1×4) = 2 × 16 = 32 terms with schoolbook.
        # Wait, that gives 32, but we need to verify correctness.
        
        # Let me just build the schoolbook decomposition and count:
        # For each i (0,1), j (0..3), k (0..6): 1 term
        # Total: 2×4×7 = 56 terms.
        
        # The Strassen approach gave 50. Can we do 42?
        
        # Let me try: use Strassen for 2x2x2 blocks but overlap them.
        # 
        # Alternative: use a 3-term scheme for 2x1x2 (multiply 2x1 by 1x2):
        # C[0,0] = A[0,0]*B[0,0]
        # C[0,1] = A[0,0]*B[0,1]  
        # C[1,0] = A[1,0]*B[0,0]
        # C[1,1] = A[1,0]*B[0,1]
        # This is 4 terms (outer product), can't be reduced below 4 for rank.
        
        # So for each inner column j, the 2x7 result needs at least 14 terms?
        # No, the rank of the 2x1x7 tensor is min(2*1, 2*7, 1*7) = 2? 
        # No, the tensor rank is different from matrix rank.
        
        # I think the safest bet is to stick with the rank-50 approach
        # and try to find linear dependencies to reduce it.
        
        # Let me build the rank-50 and then try to reduce.
        rank = 50
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        col = 0
        
        def emit_strassen(i0, j0, k0):
            nonlocal col
            # M1 = (a00+a11)(b00+b11)
            U[i0*m + j0, col] = 1.0
            U[(i0+1)*m + (j0+1), col] = 1.0
            V[j0*p + k0, col] = 1.0
            V[(j0+1)*p + (k0+1), col] = 1.0
            W[k0*n + i0, col] = 1.0
            W[(k0+1)*n + (i0+1), col] = 1.0
            col += 1
            
            # M2 = (a10+a11)b00
            U[(i0+1)*m + j0, col] = 1.0
            U[(i0+1)*m + (j0+1), col] = 1.0
            V[j0*p + k0, col] = 1.0
            W[k0*n + (i0+1), col] = 1.0
            W[(k0+1)*n + (i0+1), col] = -1.0
            col += 1
            
            # M3 = a00(b01-b11)
            U[i0*m + j0, col] = 1.0
            V[j0*p + (k0+1), col] = 1.0
            V[(j0+1)*p + (k0+1), col] = -1.0
            W[(k0+1)*n + i0, col] = 1.0
            W[(k0+1)*n + (i0+1), col] = 1.0
            col += 1
            
            # M4 = a11(b10-b00)
            U[(i0+1)*m + (j0+1), col] = 1.0
            V[(j0+1)*p + k0, col] = 1.0
            V[j0*p + k0, col] = -1.0
            W[k0*n + (i0+1), col] = 1.0
            W[k0*n + i0, col] = 1.0
            col += 1
            
            # M5 = (a00+a01)b11
            U[i0*m + j0, col] = 1.0
            U[i0*m + (j0+1), col] = 1.0
            V[(j0+1)*p + (k0+1), col] = 1.0
            W[k0*n + i0, col] = -1.0
            W[(k0+1)*n + i0, col] = 1.0
            col += 1
            
            # M6 = (a10-a00)(b00+b01)
            U[(i0+1)*m + j0, col] = 1.0
            U[i0*m + j0, col] = -1.0
            V[j0*p + k0, col] = 1.0
            V[j0*p + (k0+1), col] = 1.0
            W[(k0+1)*n + (i0+1), col] = 1.0
            col += 1
            
            # M7 = (a01-a11)(b10+b11)
            U[i0*m + (j0+1), col] = 1.0
            U[(i0+1)*m + (j0+1), col] = -1.0
            V[(j0+1)*p + k0, col] = 1.0
            V[(j0+1)*p + (k0+1), col] = 1.0
            W[k0*n + i0, col] = 1.0
            col += 1

        def emit_schoolbook_2x2x1(i0, j0, k0):
            nonlocal col
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj), col] = 1.0
                    V[(j0+dj)*p + k0, col] = 1.0
                    W[k0*n + (i0+di), col] = 1.0
                    col += 1

        # 6 Strassen blocks + 8 schoolbook terms = 50
        emit_strassen(0, 0, 0)
        emit_strassen(0, 0, 2)
        emit_strassen(0, 0, 4)
        emit_schoolbook_2x2x1(0, 0, 6)
        emit_strassen(0, 2, 0)
        emit_strassen(0, 2, 2)
        emit_strassen(0, 2, 4)
        emit_schoolbook_2x2x1(0, 2, 6)
        
        # Verify
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(T_recon, T, atol=1e-9):
            return None
        
        # Try to reduce rank by finding linear dependencies
        # The 50 rank-1 terms span a space. If some are linearly dependent
        # in a way that preserves the tensor, we can reduce.
        
        # Check if any columns of U, V, W are linearly dependent
        # in a way that allows removal.
        
        # For now, return rank 50
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    result = build_decomposition()
    return result
# EVOLVE-BLOCK-END
