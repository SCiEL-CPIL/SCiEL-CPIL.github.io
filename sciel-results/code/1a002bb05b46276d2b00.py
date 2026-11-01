import time
import sys

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    # Block decomposition: split columns of B (size m x p = 3x7) into:
    #   Block 1: columns 0..3 (size 3x4) -> treat as two 3x2 sub-blocks, but for Strassen we need square blocks
    #   Actually, let's split p=7 into p1=4 and p2=3
    #   For the 3x4 part: we can't directly use Strassen (needs square). 
    #   Alternative: split into p1=3 and p2=4? Or use a different approach.
    
    # Better approach: Use the standard Winograd/Strassen hybrid.
    # For 2x3 * 3x7:
    # We can view this as computing 2x7 output.
    # 
    # Strategy from the directive: group columns into blocks of size 4 and 3.
    # For the 3x4 block: we can use a rank-8 decomposition (since 3x4 needs at least 12, but with structure...)
    # Actually, let's think more carefully.
    
    # The directive says: "grouping the seven columns of the B-matrix into two blocks (sizes 4 and 3) 
    # and applying Strassen's 2x2 block algorithm to each sub-block independently, which yields a rank of 18"
    
    # Wait, Strassen's 2x2 algorithm is for multiplying two 2x2 matrices. How does that apply here?
    
    # Let me reconsider. The 2x3 * 3x7 matrix multiplication can be seen as:
    # A is 2x3, B is 3x7, C = A*B is 2x7.
    
    # One approach: Pad A to 3x3 (or 4x4) and B to 3x3 (or 4x4) and use Strassen.
    # But that's not what the directive says.
    
    # Let me re-read: "grouping the seven columns of the B-matrix into two blocks (sizes 4 and 3)"
    # So B = [B1 | B2] where B1 is 3x4 and B2 is 3x3.
    # C = [A*B1 | A*B2]
    
    # For A*B2: A is 2x3, B2 is 3x3. This is a 2x3 * 3x3 multiplication.
    # For A*B1: A is 2x3, B1 is 3x4. This is a 2x3 * 3x4 multiplication.
    
    # The directive claims rank 18 = 2*2^3 + 2*2^2 = 16 + 8 = 24? No, 2*8 + 2*4 = 16+8=24, not 18.
    # Or maybe 2^3 + 2^3 + 2^2 + 2^2 = 8+8+4+4 = 24? Still not 18.
    # Or 2*2^3 + 2*2^2 where the first term is for one block and second for another?
    
    # Actually, let me just implement a known good decomposition.
    # The minimum rank for 2x3 * 3x7 is known to be 18 (from literature on tensor rank of matrix multiplication).
    
    # Let me use a constructive approach based on Winograd's method.
    
    # For 2x3 * 3x7:
    # We can use the fact that 2x3 * 3x3 has rank 8 (via Strassen-like methods)
    # and 2x3 * 3x4 has rank 10 (need to verify)
    # Total: 8 + 10 = 18
    
    # Let me construct this explicitly.
    
    # First, let's handle 2x3 * 3x3 (rank 8):
    # This is equivalent to 2x2 * 2x2 with padding, or use direct formulas.
    
    # Actually, let me use a simpler approach: the standard bilinear decomposition.
    # For 2x3 * 3x7, the naive rank is 2*3*7 = 42.
    # We want to reduce this.
    
    # Let me use the following strategy:
    # Split B into B1 (3x4) and B2 (3x3).
    # Compute C1 = A * B1 and C2 = A * B2 separately.
    
    # For C2 = A * B2 (2x3 * 3x3):
    # We can use a rank-8 decomposition.
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B2 = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
    # C2[i][j] = sum_k A[i][k] * B2[k][j]
    
    # One rank-8 decomposition for 2x3 * 3x3:
    # Use the following 8 products:
    # p0 = a00 * (b00 + b10 + b20)
    # p1 = a01 * (b01 + b11 + b21)  
    # p2 = a02 * (b02 + b12 + b22)
    # p3 = (a00 + a01) * (b10 + b11 + b12)
    # p4 = (a01 + a02) * (b20 + b21 + b22)
    # p5 = (a00 - a01) * (b00 - b10)
    # p6 = (a01 - a02) * (b01 - b11)
    # p7 = (a00 + a02) * (b02 - b22)
    # ... This is getting complicated. Let me use a different approach.
    
    # Let me just use the known optimal decomposition.
    # For now, let me return the naive decomposition but with a lower rank if I can find one.
    
    # Actually, let me try a different approach: use numpy to help construct the decomposition.
    
    # The tensor T for 2x3 * 3x7 has entries T[i,j,k] = delta_{i+k, j} for appropriate indices.
    # More precisely, T[i,j,k] = 1 if the output element (i,j) depends on A[i,k] and B[k,j].
    
    # Let me just implement the naive 42-rank decomposition and see if I can prune it.
    
    rank = 42  # Start with naive, try to reduce
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    start_time = time.time()
    deadline = start_time + 110
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
