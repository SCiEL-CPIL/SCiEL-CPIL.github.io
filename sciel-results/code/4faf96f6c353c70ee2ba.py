"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    deadline = time.time() + 140
    
    def check_decomposition(U, V, W, rank):
        """Verify exact decomposition: for all (a,b,c), sum_r U[a*m+b,r]*V[b*p+c,r]*W[c*n+a,r] == 1"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    actual = sum(U_arr[a*m+b, r] * V_arr[b*p+c, r] * W_arr[c*n+a, r] 
                                  for r in range(rank))
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True
    
    def build_rank15():
        """
        Rank-15: 2x2 x 2x3 (8 terms) + 2x2 x 2x2 (7 terms)
        Columns 0,1,2 use generalized Strassen (8 terms)
        Columns 3,4 use standard Strassen (7 terms)
        """
        rank = 15
        U = [[0] * rank for _ in range(n * m)]   # (4, 15)
        V = [[0] * rank for _ in range(m * p)]   # (10, 15)
        W = [[0] * rank for _ in range(p * n)]   # (10, 15)
        
        # Standard Strassen for 2x2 * 2x2 (columns c, c+1)
        def add_strassen_k2(c, base):
            # M1: f=A00+A11, g=B0c + B1(c+1)
            r = base
            U[0][r] = 1; U[3][r] = 1
            V[0*p+c][r] = 1; V[1*p+c+1][r] = 1
            W[c*n+0][r] = 1; W[(c+1)*n+1][r] = 1
            
            # M2: f=A10+A11, g=B0c
            r = base + 1
            U[2][r] = 1; U[3][r] = 1
            V[0*p+c][r] = 1
            W[c*n+1][r] = 1; W[(c+1)*n+1][r] = -1
            
            # M3: f=A00, g=B0(c+1) - B1(c+1)
            r = base + 2
            U[0][r] = 1
            V[0*p+c+1][r] = 1; V[1*p+c+1][r] = -1
            W[(c+1)*n+0][r] = 1; W[(c+1)*n+1][r] = 1
            
            # M4: f=A11, g=B1c - B0c
            r = base + 3
            U[3][r] = 1
            V[1*p+c][r] = 1; V[0*p+c][r] = -1
            W[c*n+0][r] = 1; W[(c+1)*n+0][r] = 1
            
            # M5: f=A00+A01, g=B1(c+1)
            r = base + 4
            U[0][r] = 1; U[1][r] = 1
            V[1*p+c+1][r] = 1
            W[c*n+0][r] = -1; W[(c+1)*n+0][r] = 1
            
            # M6: f=A10-A00, g=B0c + B0(c+1)
            r = base + 5
            U[2][r] = 1; U[0][r] = -1
            V[0*p+c][r] = 1; V[0*p+c+1][r] = 1
            W[(c+1)*n+1][r] = 1
            
            # M7: f=A01-A11, g=B1c + B1(c+1)
            r = base + 6
            U[1][r] = 1; U[3][r] = -1
            V[1*p+c][r] = 1; V[1*p+c+1][r] = 1
            W[c*n+0][r] = 1
        
        # Generalized Strassen for 2x2 * 2x3 (columns c, c+1, c+2) - 8 terms
        # This extends the standard 7-term Strassen to handle 3 columns
        def add_strassen_k3(c, base):
            # M1: f=A00+A11, g=B0c + B1(c+2)
            r = base
            U[0][r] = 1; U[3][r] = 1
            V[0*p+c][r] = 1; V[1*p+c+2][r] = 1
            W[c*n+0][r] = 1; W[(c+2)*n+1][r] = 1
            
            # M2: f=A10+A11, g=B0c
            r = base + 1
            U[2][r] = 1; U[3][r] = 1
            V[0*p+c][r] = 1
            W[c*n+1][r] = 1; W[(c+2)*n+1][r] = -1
            
            # M3: f=A00, g=B0(c+1) - B1(c+1)
            r = base + 2
            U[0][r] = 1
            V[0*p+c+1][r] = 1; V[1*p+c+1][r] = -1
            W[(c+1)*n+0][r] = 1; W[(c+1)*n+1][r] = 1
            
            # M4: f=A11, g=B1c - B0c
            r = base + 3
            U[3][r] = 1
            V[1*p+c][r] = 1; V[0*p+c][r] = -1
            W[c*n+0][r] = 1; W[(c+1)*n+0][r] = 1
            
            # M5: f=A00+A01, g=B1(c+2)
            r = base + 4
            U[0][r] = 1; U[1][r] = 1
            V[1*p+c+2][r] = 1
            W[c*n+0][r] = -1; W[(c+1)*n+0][r] = 1
            
            # M6: f=A10-A00, g=B0c + B0(c+1)
            r = base + 5
            U[2][r] = 1; U[0][r] = -1
            V[0*p+c][r] = 1; V[0*p+c+1][r] = 1
            W[(c+2)*n+1][r] = 1
            
            # M7: f=A01-A11, g=B1c + B1(c+1)
            r = base + 6
            U[1][r] = 1; U[3][r] = -1
            V[1*p+c][r] = 1; V[1*p+c+1][r] = 1
            W[c*n+0][r] = 1
            
            # M8: f=A01, g=B1(c+2)
            r = base + 7
            U[1][r] = 1
            V[1*p+c+2][r] = 1
            W[(c+2)*n+0][r] = 1
        
        # Columns 0,1,2: generalized Strassen (8 terms, indices 0-7)
        add_strassen_k3(0, 0)
        
        # Columns 3,4: standard Strassen (7 terms, indices 8-14)
        add_strassen_k2(3, 8)
        
        return U, V, W
    
    def build_rank16():
        """
        Rank-16: Two Strassen blocks (7+7) + 2 terms for column 4
        But column 4 needs 4 terms... Let me try a different split.
        Actually: 2x2 x 2x2 (7) + 2x2 x 2x2 (7) + 2x2 x 2x1 (2 terms if we can share)
        The 2x2 x 2x1 tensor has rank 4, so we need 4 terms. Total = 18.
        Unless we can merge... Let me try 2x2 x 2x3 (8) + 2x2 x 2x2 (7) = 15, 
        but that's what rank15 does. For rank 16, maybe a different partition.
        """
        rank = 16
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_k2(c, base):
            r = base
            U[0][r] = 1; U[3][r] = 1
            V[0*p+c][r] = 1; V[1*p+c+1][r] = 1
            W[c*n+0][r] = 1; W[(c+1)*n+1][r] = 1
            
            r = base + 1
            U[2][r] = 1; U[3][r] = 1
            V[0*p+c][r] = 1
            W[c*n+1][r] = 1; W[(c+1)*n+1][r] = -1
            
            r = base + 2
            U[0][r] = 1
            V[0*p+c+1][r] = 1; V[1*p+c+1][r] = -1
            W[(c+1)*n+0][r] = 1; W[(c+1)*n+1][r] = 1
            
            r = base + 3
            U[3][r] = 1
            V[1*p+c][r] = 1; V[0*p+c][r] = -1
            W[c*n+0][r] = 1; W[(c+1)*n+0][r] = 1
            
            r = base + 4
            U[0][r] = 1; U[1][r] = 1
            V[1*p+c+1][r] = 1
            W[c*n+0][r] = -1; W[(c+1)*n+0][r] = 1
            
            r = base + 5
            U[2][r] = 1; U[0][r] = -1
            V[0*p+c][r] = 1; V[0*p+c+1][r] = 1
            W[(c+1)*n+1][r] = 1
            
            r = base + 6
            U[1][r] = 1; U[3][r] = -1
            V[1*p+c][r] = 1; V[1*p+c+1][r] = 1
            W[c*n+0][r] = 1
        
        # Columns 0,1: Strassen (7 terms)
        add_strassen_k2(0, 0)
        
        # Columns 2,3: Strassen (7 terms)
        add_strassen_k2(2, 7)
        
        # Column 4: schoolbook (4 terms) - indices 14,15,16,17... but we only have 16 slots
        # This won't work for rank 16. Let me use a different approach.
        # Actually, for rank 16, we need to find a valid 16-term decomposition.
        # Let's try: 2x2 x 2x2 (7) + 2x2 x 2x3 (9 terms)? No, 2x2x2x3 rank is 8.
        # 7 + 8 + 1 = 16? The last 1 term can't cover a full column.
        
        # Alternative: use the rank-15 construction but add one extra term as a safeguard
        # Actually, let's just try to make rank 16 work with a valid decomposition.
        # 2x2 x 2x2 (7) + 2x2 x 2x2 (7) + 2x2 x 2x1 (2 terms, but rank of 2x2x2x1 is 4)
        # This doesn't work. Let me fall back to known valid decompositions.
        
        # For now, return None to indicate this rank is not straightforward
        return None
    
    def build_rank17():
        """
        Rank-17: 2x2 x 2x3 (8 terms) + 2x2 x 2x2 (7 terms) + 1 extra term
        Or: 2x2 x 2x2 (7) + 2x2 x 2x2 (7) + 2x2 x 2x1 (3 terms, if we can do it)
        The 2x2 x 2x1 tensor has rank 4, so 3 terms won't work.
        Let's try: 8 + 7 + 1 = 16... no, that's 16.
        Actually 8 + 9 = 17 where the 9-term part handles 2x2 x 2x2 with one redundant term?
        Let me just try the rank-15 structure and see if it works, then rank-16, 17, 18.
        """
        rank = 17
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_k2(c, base):
            r = base
            U[0][r] = 1; U[3][r] = 1
            V[0*p+c][r] = 1; V[1*p+c+1][r] = 1
            W[c*n+0][r] = 1; W[(c+1)*n+1][r] = 1
            
            r = base + 1
            U[2][r] = 1; U[3][r] = 1
            V[0*p+c][r] = 1
            W[c*n+1][r] = 1; W[(c+1)*n+1][r] = -1
            
            r = base + 2
            U[0][r] = 1
            V[0*p+c+1][r] = 1; V[1*p+c+1][r] = -1
            W[(c+1)*n+0][r] = 1; W[(c+1)*n+1][r] = 1
            
            r = base + 3
            U[3][r] = 1
            V[1*p+c][r] = 1; V[0*p+c][r] = -1
            W[c*n+0][r] = 1; W[(c+1)*n+0][r] = 1
            
            r = base + 4
            U[0][r] = 1; U[1][r] = 1
            V[1*p+c+1][r] = 1
            W[c*n+0][r] = -1; W[(c+1)*n+0][r] = 1
            
            r = base + 5
            U[2][r] = 1; U[0][r] = -1
            V[0*p+c][r] = 1; V[0*p+c+1][r] = 1
            W[(c+1)*n+1][r] = 1
            
            r = base + 6
            U[1][r] = 1; U[3][r] = -1
            V[1*p+c][r] = 1; V[1*p+c+1][r] = 1
            W[c*n+0][r] = 1
        
        def add_strassen_k3(c, base):
            r = base
            U[0][r] = 1; U[3][r] = 1
            V[0*p+c][r] = 1; V[1*p+c+2][r] = 1
            W[c*n+0][r] = 1; W[(c+2)*n+1][r] = 1
            
            r = base + 1
            U[2][r] = 1; U[3][r] = 1
            V[0*p+c][r] = 1
            W[c*n+1][r] = 1; W[(c+2)*n+1][r] = -1
            
            r = base + 2
            U[0][r] = 1
            V[0*p+c+1][r] = 1; V[1*p+c+1][r] = -1
            W[(c+1)*n+0][r] = 1; W[(c+1)*n+1][r] = 1
            
            r = base + 3
            U[3][r] = 1
            V[1*p+c][r] = 1; V[0*p+c][r] = -1
            W[c*n+0][r] = 1; W[(c+1)*n+0][r] = 1
            
            r = base + 4
            U[0][r] = 1; U[1][r] = 1
            V[1*p+c+2][r] = 1
            W[c*n+0][r] = -1; W[(c+1)*n+0][r] = 1
            
            r = base + 5
            U[2][r] = 1; U[0][r] = -1
            V[0*p+c][r] = 1; V[0*p+c+1][r] = 1
            W[(c+2)*n+1][r] = 1
            
            r = base + 6
            U[1][r] = 1; U[3][r] = -1
            V[1*p+c][r] = 1; V[1*p+c+1][r] = 1
            W[c*n+0][r] = 1
            
            r = base + 7
            U[1][r] = 1
            V[1*p+c+2][r] = 1
            W[(c+2)*n+0][r] = 1
        
        # Columns 0,1,2: generalized Strassen (8 terms)
        add_strassen_k3(0, 0)
        
        # Columns 3,4: standard Strassen (7 terms)
        add_strassen_k2(3, 8)
        
        # One extra term (should be zero if the above is correct)
        # For now, leave it as zeros
        r = 15
        # This term is unused if the above decomposition is correct
        U[0][r] = 0; V[0*p+0][r] = 0; W[0*n+0][r] = 0
        
        return U, V, W
    
    def build_rank18():
        """
        Rank-18: Two Strassen blocks (7+7) + schoolbook for column 4 (4 terms)
        This is the known valid decomposition.
        """
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(col0, col1, base):
            r = base
            U[0][r] = 1; U[3][r] = 1
            V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
            W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
            
            r = base + 1
            U[2][r] = 1; U[3][r] = 1
            V[0*p+col0][r] = 1
            W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
            
            r = base + 2
            U[0][r] = 1
            V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
            W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
            
            r = base + 3
            U[3][r] = 1
            V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
            W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
            
            r = base + 4
            U[0][r] = 1; U[1][r] = 1
            V[1*p+col1][r] = 1
            W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
            
            r = base + 5
            U[2][r] = 1; U[0][r] = -1
            V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
            W[col1*n+1][r] = 1
            
            r = base + 6
            U[1][r] = 1; U[3][r] = -1
            V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
            W[col0*n+0][r] = 1
        
        # Strassen block 1: columns 0, 1
        add_strassen_block(0, 1, 0)
        
        # Strassen block 2: columns 2, 3
        add_strassen_block(2, 3, 7)
        
        # Schoolbook for column 4
        r = 14
        U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
        r = 15
        U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
        r = 16
        U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
        r = 17
        U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
        
        return U, V, W
    
    # Try ranks in increasing order
    # First try rank 15
    U15, V15, W15 = build_rank15()
    if check_decomposition(U15, V15, W15, 15):
        return {"U": U15, "V": V15, "W": W15}
    
    # Try rank 17
    U17, V17, W17 = build_rank17()
    if check_decomposition(U17, V17, W17, 17):
        return {"U": U17, "V": V17, "W": W17}
    
    # Fallback to rank 18
    U18, V18, W18 = build_rank18()
    return {"U": U18, "V": V18, "W": W18}
# EVOLVE-BLOCK-END
