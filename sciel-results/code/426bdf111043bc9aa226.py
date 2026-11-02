import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = 3, 3, 6
    
    def make_schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank

    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def laderman_23_terms():
        """
        Returns 23 triples (u, v, w) where each is a 3x3 matrix as 9-element list (row-major).
        These satisfy: for all i,j,k in {0,1,2}:
        sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1
        """
        terms = []
        
        def make_mat(coefs):
            """Create 3x3 matrix from list of (i,j,val) tuples"""
            mat = [0]*9
            for i, j, val in coefs:
                mat[i*3+j] = val
            return mat

        # 9 trivial terms
        terms.append((make_mat([(0,0,1)]), make_mat([(0,0,1)]), make_mat([(0,0,1)])))
        terms.append((make_mat([(0,1,1)]), make_mat([(1,0,1)]), make_mat([(1,0,1)]))) # c01 -> W[1,0]
        terms.append((make_mat([(0,2,1)]), make_mat([(2,0,1)]), make_mat([(2,0,1)]))) # c02 -> W[2,0]
        terms.append((make_mat([(1,0,1)]), make_mat([(0,1,1)]), make_mat([(0,1,1)]))) # c10 -> W[0,1]
        terms.append((make_mat([(1,1,1)]), make_mat([(1,1,1)]), make_mat([(1,1,1)]))) # c11 -> W[1,1]
        terms.append((make_mat([(1,2,1)]), make_mat([(2,1,1)]), make_mat([(2,1,1)]))) # c12 -> W[2,1]
        terms.append((make_mat([(2,0,1)]), make_mat([(0,2,1)]), make_mat([(0,2,1)]))) # c20 -> W[0,2]
        terms.append((make_mat([(2,1,1)]), make_mat([(1,2,1)]), make_mat([(1,2,1)]))) # c21 -> W[1,2]
        terms.append((make_mat([(2,2,1)]), make_mat([(2,2,1)]), make_mat([(2,2,1)]))) # c22 -> W[2,2]

        # 14 non-trivial terms
        # 10
        terms.append((
            make_mat([(0,0,1), (0,1,-1)]),
            make_mat([(1,0,1), (1,1,-1)]),
            make_mat([(1,0,1), (0,0,-1), (1,1,-1), (0,1,1)]) # W[1,0]=1, W[0,0]=-1, W[1,1]=-1, W[0,1]=1
        ))
        # 11
        terms.append((
            make_mat([(0,1,1), (0,2,-1)]),
            make_mat([(2,0,1), (2,1,-1)]),
            make_mat([(2,0,1), (1,0,-1), (2,1,-1), (1,1,1)]) # W[2,0]=1, W[1,0]=-1, W[2,1]=-1, W[1,1]=1
        ))
        # 12
        terms.append((
            make_mat([(1,0,1), (1,1,-1)]),
            make_mat([(0,1,1), (0,2,-1)]),
            make_mat([(0,1,1), (1,1,-1), (1,2,-1), (0,2,1)]) # W[0,1]=1, W[1,1]=-1, W[1,2]=-1, W[0,2]=1
        ))
        # 13
        terms.append((
            make_mat([(1,1,1), (1,2,-1)]),
            make_mat([(1,2,1), (1,0,-1)]),
            make_mat([(2,1,1), (1,1,-1), (2,2,-1), (1,2,1)]) # W[2,1]=1, W[1,1]=-1, W[2,2]=-1, W[1,2]=1
        ))
        # 14
        terms.append((
            make_mat([(2,0,1), (2,1,-1)]),
            make_mat([(0,2,1), (0,0,-1)]),
            make_mat([(0,2,1), (1,2,-1), (0,0,-1), (2,0,1)]) # W[0,2]=1, W[1,2]=-1, W[0,0]=-1, W[2,0]=1
        ))
        # 15
        terms.append((
            make_mat([(2,1,1), (2,2,-1)]),
            make_mat([(1,0,1), (1,1,-1)]),
            make_mat([(1,2,1), (2,2,-1), (1,0,-1), (2,0,1)]) # W[1,2]=1, W[2,2]=-1, W[1,0]=-1, W[2,0]=1
        ))
        # 16
        terms.append((
            make_mat([(0,0,1), (2,2,1)]),
            make_mat([(2,2,1), (1,1,-1)]),
            make_mat([(2,2,1), (0,0,1), (2,0,-1), (0,2,-1)]) # W[2,2]=1, W[0,0]=1, W[2,0]=-1, W[0,2]=-1
        ))
        # 17
        terms.append((
            make_mat([(0,2,1), (2,0,1)]),
            make_mat([(1,1,1), (2,2,-1)]),
            make_mat([(2,0,1), (0,2,1), (0,0,-1), (2,2,-1)]) # W[2,0]=1, W[0,2]=1, W[0,0]=-1, W[2,2]=-1
        ))
        # 18
        terms.append((
            make_mat([(1,0,1), (2,1,1)]),
            make_mat([(1,2,1), (0,1,-1)]),
            make_mat([(2,1,1), (1,2,1), (1,1,-1), (0,2,-1)]) # W[2,1]=1, W[1,2]=1, W[1,1]=-1, W[0,2]=-1
        ))
        # 19
        terms.append((
            make_mat([(1,2,1), (2,0,1)]),
            make_mat([(0,2,1), (1,1,-1)]),
            make_mat([(2,1,1), (0,2,1), (0,1,-1), (2,2,-1)]) # W[2,1]=1, W[0,2]=1, W[0,1]=-1, W[2,2]=-1
        ))
        # 20
        terms.append((
            make_mat([(0,1,1), (1,2,1)]),
            make_mat([(2,1,1), (0,2,-1)]),
            make_mat([(2,0,1), (2,1,1), (1,0,-1), (2,2,-1)]) # W[2,0]=1, W[2,1]=1, W[1,0]=-1, W[2,2]=-1
        ))
        # 21
        terms.append((
            make_mat([(1,0,1), (0,1,1)]),
            make_mat([(1,1,1), (2,0,-1)]),
            make_mat([(1,1,1), (1,0,1), (0,1,-1), (0,2,-1)]) # W[1,1]=1, W[1,0]=1, W[0,1]=-1, W[0,2]=-1
        ))
        # 22
        terms.append((
            make_mat([(2,0,1), (1,1,1)]),
            make_mat([(0,2,1), (1,0,-1)]),
            make_mat([(0,2,1), (1,1,1), (2,2,-1), (0,1,-1)]) # W[0,2]=1, W[1,1]=1, W[2,2]=-1, W[0,1]=-1
        ))
        # 23
        terms.append((
            make_mat([(2,1,1), (0,2,1)]),
            make_mat([(2,2,1), (0,0,-1)]),
            make_mat([(1,2,1), (2,0,1), (2,2,-1), (0,0,-1)]) # W[1,2]=1, W[2,0]=1, W[2,2]=-1, W[0,0]=-1
        ))
        
        return terms

    # Try to build rank-46 decomposition
    def try_laderman_46():
        laderman_terms = laderman_23_terms()
        
        rank = 46
        U = [[0] * rank for _ in range(n * m)]  # 9 x 46
        V = [[0] * rank for _ in range(m * p)]  # 18 x 46
        W = [[0] * rank for _ in range(p * n)]  # 18 x 46
        
        for block in range(2):
            for t in range(23):
                col = block * 23 + t
                u_mat, v_mat, w_mat = laderman_terms[t]
                
                for i in range(3):
                    for j in range(3):
                        U[i*3+j][col] = u_mat[i*3+j]
                
                for j in range(3):
                    for k in range(3):
                        V[j*6 + block*3 + k][col] = v_mat[j*3+k]
                
                for k in range(3):
                    for i in range(3):
                        W[(block*3+k)*3+i][col] = w_mat[k*3+i]
        
        return U, V, W, rank

    # Fallback
    U_school, V_school, W_school, rank_school = make_schoolbook(n, m, p)
    
    result_46 = try_laderman_46()
    
    if result_46 is not None:
        U, V, W, rank = result_46
        if verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    U, V, W, rank = U_school, V_school, W_school, rank_school
    return {"U": U, "V": V, "W": W}
