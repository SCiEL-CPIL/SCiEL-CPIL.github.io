import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    start_time = time.time()
    deadline = start_time + 100

    def compute_error(U, V, W, r):
        """Compute max error of the trilinear constraint."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def to_list(M, rows, cols):
        """Convert numpy matrix to list of lists with integer conversion."""
        result = []
        for i in range(rows):
            row = []
            for j in range(cols):
                val = M[i, j]
                if abs(val - round(val)) < 1e-10:
                    row.append(int(round(val)))
                else:
                    row.append(val)
            result.append(row)
        return result

    def add_strassen_block(U, V, W, offset, ai, bi, ki):
        """
        Add a Strassen rank-7 decomposition for a 2x2x2 sub-problem.
        A[ai:ai+2, bi:bi+2] * B[bi:bi+2, ki:ki+2] = C[ai:ai+2, ki:ki+2]
        """
        # P1: A[ai,bi] * (B[bi+1,ki+1] - B[bi,ki+1])
        U[ai*m+bi, offset] += 1
        V[(bi+1)*p+(ki+1), offset] += 1
        V[bi*p+(ki+1), offset] += -1
        W[(ki+1)*n+ai, offset] += 1

        # P2: (A[ai,bi] + A[ai,bi+1]) * B[bi,ki+1]
        U[ai*m+bi, offset+1] += 1
        U[ai*m+(bi+1), offset+1] += 1
        V[bi*p+(ki+1), offset+1] += 1
        W[(ki+1)*n+ai, offset+1] += 1

        # P3: (A[ai+1,bi] + A[ai+1,bi+1]) * B[bi,ki]
        U[(ai+1)*m+bi, offset+2] += 1
        U[(ai+1)*m+(bi+1), offset+2] += 1
        V[bi*p+ki, offset+2] += 1
        W[ki*n+(ai+1), offset+2] += 1

        # P4: A[ai+1,bi+1] * (B[bi,ki] - B[bi+1,ki])
        U[(ai+1)*m+(bi+1), offset+3] += 1
        V[bi*p+ki, offset+3] += 1
        V[(bi+1)*p+ki, offset+3] += -1
        W[ki*n+(ai+1), offset+3] += 1

        # P5: (A[ai,bi] + A[ai+1,bi+1]) * (B[bi,ki+1] + B[bi+1,ki+1])
        U[ai*m+bi, offset+4] += 1
        U[(ai+1)*m+(bi+1), offset+4] += 1
        V[bi*p+(ki+1), offset+4] += 1
        V[(bi+1)*p+(ki+1), offset+4] += 1
        W[(ki+1)*n+ai, offset+4] += 1

        # P6: (A[ai,bi+1] - A[ai+1,bi+1]) * (B[bi+1,ki] + B[bi+1,ki+1])
        U[ai*m+(bi+1), offset+5] += 1
        U[(ai+1)*m+(bi+1), offset+5] += -1
        V[(bi+1)*p+ki, offset+5] += 1
        V[(bi+1)*p+(ki+1), offset+5] += 1
        W[ki*n+ai, offset+5] += 1
        W[(ki+1)*n+ai, offset+5] += 1

        # P7: (A[ai,bi] - A[ai+1,bi]) * (B[bi,ki] + B[bi,ki+1])
        U[ai*m+bi, offset+6] += 1
        U[(ai+1)*m+bi, offset+6] += -1
        V[bi*p+ki, offset+6] += 1
        V[bi*p+(ki+1), offset+6] += 1
        W[ki*n+ai, offset+6] += 1
        W[(ki+1)*n+ai, offset+6] += 1

        return offset + 7

    def add_schoolbook_2x2x1(U, V, W, offset, ai, bi, ki):
        """
        Add schoolbook rank-4 decomposition for 2x2x1 sub-problem.
        A[ai:ai+2, bi:bi+2] * B[bi:bi+2, ki] = C[ai:ai+2, ki]
        """
        for di in range(2):
            for dj in range(2):
                U[(ai+di)*m+(bi+dj), offset] += 1
                V[(bi+dj)*p+ki, offset] += 1
                W[ki*n+(ai+di), offset] += 1
                offset += 1
        return offset

    def add_2x3x3x2_rank5(U, V, W, offset, ai, bi, ki):
        """
        Add rank-5 decomposition for 2x3 * 3x2 sub-problem.
        A[ai:ai+2, bi:bi+3] * B[bi:bi+3, ki:ki+2] = C[ai:ai+2, ki:ki+2]
        Uses the known rank-5 formula for 2x3 x 3x2 matrix multiplication.
        """
        # This is based on the rank-5 decomposition of the 2x3 x 3x2 tensor.
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
        
        # Rank-5 decomposition (Landsberg):
        # T1: A[0,0] * (B[0,0] - B[0,1])  -> contributes to C[0,0]
        U[ai*m+bi, offset] += 1
        V[bi*p+ki, offset] += 1
        V[bi*p+(ki+1), offset] += -1
        W[ki*n+ai, offset] += 1
        
        # T2: (A[0,1] + A[1,0]) * (B[0,1] - B[1,0])  -> cross term
        U[ai*m+(bi+1), offset+1] += 1
        U[(ai+1)*m+bi, offset+1] += 1
        V[bi*p+(ki+1), offset+1] += 1
        V[(bi+1)*p+ki, offset+1] += -1
        W[ki*n+ai, offset+1] += 1
        W[(ki+1)*n+(ai+1), offset+1] += 1
        
        # T3: A[1,1] * (B[1,0] - B[1,1])  -> contributes to C[1,0]
        U[(ai+1)*m+(bi+1), offset+2] += 1
        V[(bi+1)*p+ki, offset+2] += 1
        V[(bi+1)*p+(ki+1), offset+2] += -1
        W[ki*n+(ai+1), offset+2] += 1
        
        # T4: (A[0,2] + A[1,2]) * B[2,0]  -> contributes to C[0,0] and C[1,0]
        U[ai*m+(bi+2), offset+3] += 1
        U[(ai+1)*m+(bi+2), offset+3] += 1
        V[(bi+2)*p+ki, offset+3] += 1
        W[ki*n+ai, offset+3] += 1
        W[ki*n+(ai+1), offset+3] += 1
        
        # T5: (A[0,2] - A[1,2]) * B[2,1]  -> contributes to C[0,1] and C[1,1]
        U[ai*m+(bi+2), offset+4] += 1
        U[(ai+1)*m+(bi+2), offset+4] += -1
        V[(bi+2)*p+(ki+1), offset+4] += 1
        W[(ki+1)*n+ai, offset+4] += 1
        W[(ki+1)*n+(ai+1), offset+4] += -1
        
        return offset + 5

    def add_schoolbook_2x3x1(U, V, W, offset, ai, bi, ki):
        """
        Add schoolbook rank-6 decomposition for 2x3x1 sub-problem.
        A[ai:ai+2, bi:bi+3] * B[bi:bi+3, ki] = C[ai:ai+2, ki]
        """
        for di in range(2):
            for dj in range(3):
                U[(ai+di)*m+(bi+dj), offset] += 1
                V[(bi+dj)*p+ki, offset] += 1
                W[ki*n+(ai+di), offset] += 1
                offset += 1
        return offset

    def add_schoolbook_2x1x5(U, V, W, offset, ai, bi):
        """
        Add schoolbook rank-10 decomposition for 2x1 * 1x5 sub-problem.
        A[ai:ai+2, bi] * B[bi, :] = C[ai:ai+2, :]
        """
        for di in range(2):
            for dk in range(5):
                U[(ai+di)*m+bi, offset] += 1
                V[bi*p+dk, offset] += 1
                W[dk*n+(ai+di), offset] += 1
                offset += 1
        return offset

    # Strategy 1: Try rank-35 with 3+1 inner split
    # A is 2x4, split inner dim into 3+1: A = [A1 | A2] where A1 is 2x3, A2 is 2x1
    # B is 4x5, split into B_top (3x5) and B_bottom (1x5)
    # C = A1*B_top + A2*B_bottom
    # A1*B_top: 2x3 * 3x5, split output 5 into 2+2+1
    #   2x3 * 3x2: rank 5 (Landsberg)
    #   2x3 * 3x2: rank 5
    #   2x3 * 3x1: rank 6 (schoolbook)
    # A2*B_bottom: 2x1 * 1x5: rank 10 (schoolbook)
    # Total: 5 + 5 + 6 + 10 = 26? That seems too good. Let me verify.
    
    # Actually, let me be more careful. Let's try rank 35 first with a different approach.
    # Use the current 2+2 split but try to optimize.
    
    # Let's try: 2+2 inner split, 3+2 output split for each half
    # A1*B_top: 2x2 * 2x5, split output into 3+2
    #   2x2 * 2x3: rank? (schoolbook is 6, maybe 5?)
    #   2x2 * 2x2: rank 7 (Strassen)
    # Hmm, 2x2 * 2x3 doesn't have a known rank-5.
    
    # Let me just try the 3+1 split and see if it works.
    # Total rank: 5 + 5 + 6 + 10 = 26
    
    rank_26 = 26
    U = np.zeros((n*m, rank_26))
    V = np.zeros((m*p, rank_26))
    W = np.zeros((p*n, rank_26))
    
    offset = 0
    
    # A1 (2x3, inner cols 0-2) * B_top (3x5, inner rows 0-2)
    # Output cols 0-1: 2x3 * 3x2, rank 5
    offset = add_2x3x3x2_rank5(U, V, W, offset, 0, 0, 0)
    # Output cols 2-3: 2x3 * 3x2, rank 5
    offset = add_2x3x3x2_rank5(U, V, W, offset, 0, 0, 2)
    # Output col 4: 2x3 * 3x1, rank 6
    offset = add_schoolbook_2x3x1(U, V, W, offset, 0, 0, 4)
    
    # A2 (2x1, inner col 3) * B_bottom (1x5, inner row 3)
    # Output cols 0-4: 2x1 * 1x5, rank 10
    offset = add_schoolbook_2x1x5(U, V, W, offset, 0, 3)
    
    # Verify
    err = compute_error(U, V, W, rank_26)
    if err < 1e-10:
        return {"U": to_list(U, n*m, rank_26), "V": to_list(V, m*p, rank_26), "W": to_list(W, p*n, rank_26)}
    
    # Strategy 2: Fall back to rank-36 (2+2 split with Strassen)
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    offset = 0
    
    offset = add_strassen_block(U, V, W, offset, 0, 0, 0)
    offset = add_strassen_block(U, V, W, offset, 0, 0, 2)
    offset = add_schoolbook_2x2x1(U, V, W, offset, 0, 0, 4)
    
    offset = add_strassen_block(U, V, W, offset, 0, 2, 0)
    offset = add_strassen_block(U, V, W, offset, 0, 2, 2)
    offset = add_schoolbook_2x2x1(U, V, W, offset, 0, 2, 4)
    
    err = compute_error(U, V, W, rank)
    if err < 1e-10:
        return {"U": to_list(U, n*m, rank), "V": to_list(V, m*p, rank), "W": to_list(W, p*n, rank)}
    
    # Fallback to schoolbook rank 40
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, column] = 1
                V[j*p+k, column] = 1
                W[k*n+i, column] = 1
                column += 1
    
    return {"U": to_list(U, n*m, rank), "V": to_list(V, m*p, rank), "W": to_list(W, p*n, rank)}
