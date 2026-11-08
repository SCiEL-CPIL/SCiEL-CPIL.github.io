import time
import sys

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=9
    # Rank is 3 * 17 = 51
    rank = 51
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # We process 3 blocks of B, each 3x3.
    # Each block splits into 3x2 (rank 11) and 3x1 (rank 6).
    
    # Coefficients for the 11 multiplications of M_2,3,2
    # A is 2x3, B_sub is 3x2.
    # A indices: 0..5 (i*m+j)
    # B_sub indices: 0..5 (j*p_sub+k) where p_sub=2. But in full V, it's offset.
    
    # Let's define the terms for M_2,3,2 (A 2x3, B 3x2)
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B = [[b00, b01], [b10, b11], [b20, b21]]
    
    # Terms:
    # 1: (b00+b11)*(a00+a11)
    # 2: (b01+b11)*a00
    # 3: b00*(a01-a11)
    # 4: b11*(a10-a00)
    # 5: (b00+b10)*a11
    # 6: (b01-b00)*(a00+a10)
    # 7: (b10-b11)*(a01+a11)
    # 8: b02*a00
    # 9: b02*a10
    # 10: b12*a00
    # 11: b12*a10
    
    # Note: b indices in B_sub (3x2) are:
    # b00 -> B[0,0], b01 -> B[0,1]
    # b10 -> B[1,0], b11 -> B[1,1]
    # b20 -> B[2,0], b21 -> B[2,1]
    
    # In full V (3x9), for block r:
    # B_sub[0,0] is V[0, 3r]
    # B_sub[0,1] is V[0, 3r+1]
    # B_sub[1,0] is V[1, 3r]
    # B_sub[1,1] is V[1, 3r+1]
    # B_sub[2,0] is V[2, 3r]
    # B_sub[2,1] is V[2, 3r+1]
    
    # Output C (2x2) indices in W (9x2):
    # C[0,0] -> W[3r, 0]
    # C[0,1] -> W[3r+1, 0]
    # C[1,0] -> W[3r, 1]
    # C[1,1] -> W[3r+1, 1]
    
    # Terms for M_2,3,1 (A 2x3, B 3x1)
    # 12: a00*b00
    # 13: a01*b10
    # 14: a02*b20
    # 15: a10*b00
    # 16: a11*b10
    # 17: a12*b20
    
    # B_sub (3x1) indices in V:
    # b00 -> V[0, 3r+2]
    # b10 -> V[1, 3r+2]
    # b20 -> V[2, 3r+2]
    
    # Output C (2x1) indices in W:
    # C[0,0] -> W[3r+2, 0]
    # C[1,0] -> W[3r+2, 1]
    
    for r in range(3):
        # Part 1: Rank 11
        # A is constant, so U columns are same for all r.
        # V and W depend on r.
        
        # Term 1
        U[0][col_offset] += 1; U[1][col_offset] += 1; U[3][col_offset] += 1; U[4][col_offset] += 1
        V[0][col_offset] += 1; V[3][col_offset] += 1
        W[3*r][col_offset] += 1; W[3*r+1][col_offset] += 1; W[3*r+2][col_offset] += 1; W[3*r+3][col_offset] += 1
        # Wait, W index is k*n+i.
        # C[0,0] -> k=3r, i=0 -> W[3r*2 + 0] = W[6r]
        # C[0,1] -> k=3r+1, i=0 -> W[6r+2]
        # C[1,0] -> k=3r, i=1 -> W[6r+1]
        # C[1,1] -> k=3r+1, i=1 -> W[6r+3]
        
        # Let's re-verify W index.
        # W[k*n+i, r]. n=2.
        # k is column of C. i is row of C.
        
        # Term 1: (b00+b11)*(a00+a11)
        # a00=A[0,0]=U[0], a11=A[1,1]=U[3]
        # b00=V[0, 3r], b11=V[1, 3r+1]
        # Output:
        # C[0,0] += a00*b00 + a01*b10 + a02*b20. 
        # Strassen terms contribute to C[0,0], C[0,1], C[1,0], C[1,1].
        
        # Let's just add the contributions to W.
        # C[0,0] gets: a00*b00 + a01*b10 + a02*b20
        # C[0,1] gets: a00*b01 + a01*b11 + a02*b21
        # C[1,0] gets: a10*b00 + a11*b10 + a12*b20
        # C[1,1] gets: a10*b01 + a11*b11 + a12*b21
        
        # Term 1: (a00+a11)(b00+b11)
        # a00*b00 -> C[0,0]
        # a00*b11 -> C[0,1]
        # a11*b00 -> C[1,0]
        # a11*b11 -> C[1,1]
        
        # So W[6r] += 1, W[6r+2] += 1, W[6r+1] += 1, W[6r+3] += 1.
        
        # I will clear the previous manual addition and do it systematically.
        
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col_offset = 0
    
    for r in range(3):
        # Part 1: 11 terms
        # A indices: 0,1,2 (row 0), 3,4,5 (row 1)
        # B indices (relative to 3r):
        # b00: V[0, 3r]
        # b01: V[0, 3r+1]
        # b10: V[1, 3r]
        # b11: V[1, 3r+1]
        # b20: V[2, 3r]
        # b21: V[2, 3r+1]
        
        # W indices (relative to 6r):
        # C00: W[6r]
        # C10: W[6r+1]
        # C01: W[6r+2]
        # C11: W[6r+3]
        
        # Term 1: (a00+a11)(b00+b11)
        U[0][col_offset] += 1; U[3][col_offset] += 1
        V[0][col_offset] += 1; V[1][col_offset] += 1
        W[6*r][col_offset] += 1; W[6*r+1][col_offset] += 1; W[6*r+2][col_offset] += 1; W[6*r+3][col_offset] += 1
        col_offset += 1
        
        # Term 2: (a10+a11)b00
        U[3][col_offset] += 1; U[4][col_offset] += 1
        V[0][col_offset] += 1
        W[6*r+1][col_offset] += 1; W[6*r+2][col_offset] += 1
        col_offset += 1
        
        # Term 3: a00(b10-b11)
        U[0][col_offset] += 1
        V[1][col_offset] += 1; V[1][col_offset] -= 1 # b10 is V[1,3r], b11 is V[1,3r+1]
        # Wait, b10 is V[1, 3r], b11 is V[1, 3r+1].
        # So V[1][col_offset] += 1 is for b10? No, V index is j*p+k.
        # V[1, 3r] is index 1*9+3r.
        # V[1, 3r+1] is index 1*9+3r+1.
        # I am using V as a list of lists. V[j][k].
        # So V[1][3r] and V[1][3r+1].
        
        # Let's fix V indexing.
        # V is (m*p) x rank. m=3, p=9.
        # V[j*p+k][c].
        
        # Term 3: a00(b10-b11)
        U[0][col_offset] += 1
        V[1*9 + 3*r][col_offset] += 1
        V[1*9 + 3*r + 1][col_offset] -= 1
        # Output: a00*b10 -> C[0,1]? No.
        # a00*b10: i=0, j=1, k=0. C[0,0].
        # a00*b11: i=0, j=1, k=1. C[0,1].
        # So C[0,0] += 1, C[0,1] -= 1.
        W[6*r][col_offset] += 1
        W[6*r+2][col_offset] -= 1
        col_offset += 1
        
        # Term 4: a11(b01-b00)
        U[3][col_offset] += 1
        V[0*9 + 3*r + 1][col_offset] += 1
        V[0*9 + 3*r][col_offset] -= 1
        # a11*b01 -> C[1,1]. a11*b00 -> C[1,0].
        W[6*r+3][col_offset] += 1
        W[6*r+1][col_offset] -= 1
        col_offset += 1
        
        # Term 5: (a00+a01)b11
        U[0][col_offset] += 1; U[1][col_offset] += 1
        V[1*9 + 3*r + 1][col_offset] += 1
        # a00*b11 -> C[0,1]. a01*b11 -> C[0,1]?
        # a01 is A[0,1]. b11 is B[1,1].
        # C[0,1] += a00*b11 + a01*b11.
        # So C[0,1] += 2? No, the term is (a00+a01)*b11.
        # So C[0,1] += 1.
        W[6*r+2][col_offset] += 1
        col_offset += 1
        
        # Term 6: (a10-a00)(b00+b01)
        U[3][col_offset] += 1; U[0][col_offset] -= 1
        V[0*9 + 3*r][col_offset] += 1
        V[0*9 + 3*r + 1][col_offset] += 1
        # a10*b00 -> C[1,0]. a10*b01 -> C[1,1].
        # -a00*b00 -> -C[0,0]. -a00*b01 -> -C[0,1].
        W[6*r+1][col_offset] += 1
        W[6*r+3][col_offset] += 1
        W[6*r][col_offset] -= 1
        W[6*r+2][col_offset] -= 1
        col_offset += 1
        
        # Term 7: (a01-a11)(b10+b11)
        U[1][col_offset] += 1; U[3][col_offset] -= 1
        V[1*9 + 3*r][col_offset] += 1
        V[1*9 + 3*r + 1][col_offset] += 1
        # a01*b10 -> C[0,0]. a01*b11 -> C[0,1].
        # -a11*b10 -> -C[1,0]. -a11*b11 -> -C[1,1].
        W[6*r][col_offset] += 1
        W[6*r+2][col_offset] += 1
        W[6*r+1][col_offset] -= 1
        W[6*r+3][col_offset] -= 1
        col_offset += 1
        
        # Term 8: a02*b00
        U[2][col_offset] += 1
        V[0*9 + 3*r][col_offset] += 1
        W[6*r][col_offset] += 1
        col_offset += 1
        
        # Term 9: a02*b01
        U[2][col_offset] += 1
        V[0*9 + 3*r + 1][col_offset] += 1
        W[6*r+2][col_offset] += 1
        col_offset += 1
        
        # Term 10: a12*b00
        U[5][col_offset] += 1
        V[0*9 + 3*r][col_offset] += 1
        W[6*r+1][col_offset] += 1
        col_offset += 1
        
        # Term 11: a12*b01
        U[5][col_offset] += 1
        V[0*9 + 3*r + 1][col_offset] += 1
        W[6*r+3][col_offset] += 1
        col_offset += 1
        
        # Part 2: 6 terms (3x1)
        # B col is 3r+2.
        
        # Term 12: a00*b00
        U[0][col_offset] += 1
        V[0*9 + 3*r + 2][col_offset] += 1
        W[6*r+4][col_offset] += 1 # C[0, 3r+2] -> W[(3r+2)*2 + 0]
        col_offset += 1
        
        # Term 13: a01*b10
        U[1][col_offset] += 1
        V[1*9 + 3*r + 2][col_offset] += 1
        W[6*r+4][col_offset] += 1
        col_offset += 1
        
        # Term 14: a02*b20
        U[2][col_offset] += 1
        V[2*9 + 3*r + 2][col_offset] += 1
        W[6*r+4][col_offset] += 1
        col_offset += 1
        
        # Term 15: a10*b00
        U[3][col_offset] += 1
        V[0*9 + 3*r + 2][col_offset] += 1
        W[6*r+5][col_offset] += 1 # C[1, 3r+2] -> W[(3r+2)*2 + 1]
        col_offset += 1
        
        # Term 16: a11*b10
        U[4][col_offset] += 1
        V[1*9 + 3*r + 2][col_offset] += 1
        W[6*r+5][col_offset] += 1
        col_offset += 1
        
        # Term 17: a12*b20
        U[5][col_offset] += 1
        V[2*9 + 3*r + 2][col_offset] += 1
        W[6*r+5][col_offset] += 1
        col_offset += 1
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
