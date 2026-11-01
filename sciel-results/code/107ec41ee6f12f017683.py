DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = 7 * num_blocks
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(num_blocks):
        base = b * 7
        # B indices: row 0 cols 2b,2b+1 -> V[2b], V[2b+1]
        #             row 1 cols 2b,2b+1 -> V[p+2b], V[p+2b+1]
        
        # P1 = (A00+A11)*(B00+B11)
        U[0][base] += 1; U[3][base] += 1
        V[2*b][base] += 1; V[p+2*b+1][base] += 1
        
        # P2 = (A10+A11)*B00
        U[2][base+1] += 1; U[3][base+1] += 1
        V[2*b][base+1] += 1
        
        # P3 = A00*(B10-B11)
        U[0][base+2] += 1
        V[p+2*b][base+2] += 1; V[p+2*b+1][base+2] -= 1
        
        # P4 = A11*(B01-B00)
        U[3][base+3] += 1
        V[2*b+1][base+3] += 1; V[2*b][base+3] -= 1
        
        # P5 = (A00+A01)*B11
        U[0][base+4] += 1; U[1][base+4] += 1
        V[p+2*b+1][base+4] += 1
        
        # P6 = (A10-A00)*(B00+B10)
        U[2][base+5] += 1; U[0][base+5] -= 1
        V[2*b][base+5] += 1; V[p+2*b][base+5] += 1
        
        # P7 = (A01-A11)*(B01+B11)
        U[1][base+6] += 1; U[3][base+6] -= 1
        V[2*b+1][base+6] += 1; V[p+2*b+1][base+6] += 1
        
        # W: C[i,k] at W[k*n+i]
        # C[0,2b] = P1 - P4 + P5 - P7
        W[4*b][base] += 1; W[4*b][base+3] -= 1
        W[4*b][base+4] += 1; W[4*b][base+6] -= 1
        # C[0,2b+1] = P3 + P4
        W[4*b+2][base+2] += 1; W[4*b+2][base+3] += 1
        # C[1,2b] = P2 + P7
        W[4*b+1][base+1] += 1; W[4*b+1][base+6] += 1
        # C[1,2b+1] = P1 - P3 + P6 - P2
        W[4*b+3][base] += 1; W[4*b+3][base+2] -= 1
        W[4*b+3][base+5] += 1; W[4*b+3][base+1] -= 1
    
    return {"U": U, "V": V, "W": W}
