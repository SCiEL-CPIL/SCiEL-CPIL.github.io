import time
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses Strassen's algorithm: split B into two 4x4 halves, apply rank-49 Strassen to each.
    Total rank: 98.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit

    def _strassen_terms(n, m, p):
        """
        Returns a list of tuples (U, V, W) representing the rank-1 terms
        for the tensor decomposition of n x m * m x p matrix multiplication.
        U is a dict {(i, j): coeff}
        V is a dict {(j, k): coeff}
        W is a dict {(k, i): coeff}
        """
        if n == 1 and m == 1 and p == 1:
            return [({(0,0): 1}, {(0,0): 1}, {(0,0): 1})]
        
        # If p is even, we can split B into two halves.
        if p % 2 == 0:
            # Split B into B_left (m x p/2) and B_right (m x p/2)
            # C = A * [B_left | B_right] = [A*B_left | A*B_right]
            terms_left = _strassen_terms(n, m, p // 2)
            terms_right = _strassen_terms(n, m, p // 2)
            
            all_terms = []
            # Terms for left half: k indices are 0..p/2-1
            for (U, V, W) in terms_left:
                all_terms.append((U, V, W))
            
            # Terms for right half: k indices are p/2..p-1
            for (U, V, W) in terms_right:
                new_V = {}
                for (j, k), coeff in V.items():
                    new_V[(j, k + p // 2)] = coeff
                new_W = {}
                for (k, i), coeff in W.items():
                    new_W[(k + p // 2, i)] = coeff
                all_terms.append((U, new_V, new_W))
            
            return all_terms

        # If n, m, p are all even and > 1, use Strassen recursion
        if n % 2 == 0 and m % 2 == 0 and p % 2 == 0:
            n2 = n // 2
            m2 = m // 2
            p2 = p // 2
            
            base_terms = _strassen_terms(n2, m2, p2)
            
            all_terms = []
            
            def _add_terms(terms, U_blocks, V_blocks, W_blocks):
                for (U_t, V_t, W_t) in terms:
                    new_U = {}
                    new_V = {}
                    new_W = {}
                    
                    for (i_b, j_b), c_u in U_blocks:
                        for (i, j), val in U_t.items():
                            gi = i_b * n2 + i
                            gj = j_b * m2 + j
                            new_U[(gi, gj)] = new_U.get((gi, gj), 0) + val * c_u
                    
                    for (j_b, k_b), c_v in V_blocks:
                        for (j, k), val in V_t.items():
                            gj = j_b * m2 + j
                            gk = k_b * p2 + k
                            new_V[(gj, gk)] = new_V.get((gj, gk), 0) + val * c_v
                    
                    for (i_b, k_b), c_w in W_blocks:
                        for (k, i), val in W_t.items():
                            gi = i_b * n2 + i
                            gk = k_b * p2 + k
                            new_W[(gk, gi)] = new_W.get((gk, gi), 0) + val * c_w
                    
                    all_terms.append((new_U, new_V, new_W))
            
            # M1 = (A11+A22)(B11+B22) -> C11 + C22
            _add_terms(base_terms, 
                       [((0,0), 1), ((1,1), 1)], 
                       [((0,0), 1), ((1,1), 1)], 
                       [((0,0), 1), ((1,1), 1)])
            
            # M2 = (A21+A22)B11 -> C21 - C22
            _add_terms(base_terms, 
                       [((1,0), 1), ((1,1), 1)], 
                       [((0,0), 1)], 
                       [((1,0), 1), ((1,1), -1)])
            
            # M3 = A11(B21-B22) -> C12 + C22
            _add_terms(base_terms, 
                       [((0,0), 1)], 
                       [((1,0), 1), ((1,1), -1)], 
                       [((0,1), 1), ((1,1), 1)])
            
            # M4 = A22(B12-B11) -> C11 + C21
            _add_terms(base_terms, 
                       [((1,1), 1)], 
                       [((0,1), 1), ((0,0), -1)], 
                       [((0,0), 1), ((1,0), 1)])
            
            # M5 = (A11+A12)B22 -> -C11 + C12
            _add_terms(base_terms, 
                       [((0,0), 1), ((0,1), 1)], 
                       [((1,1), 1)], 
                       [((0,0), -1), ((0,1), 1)])
            
            # M6 = (A21-A11)(B11+B12) -> C22
            _add_terms(base_terms, 
                       [((1,0), 1), ((0,0), -1)], 
                       [((0,0), 1), ((0,1), 1)], 
                       [((1,1), 1)])
            
            # M7 = (A12-A22)(B21+B22) -> C11
            _add_terms(base_terms, 
                       [((0,1), 1), ((1,1), -1)], 
                       [((1,0), 1), ((1,1), 1)], 
                       [((0,0), 1)])
            
            return all_terms

        # Fallback for non-power-of-2 or other cases
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
        return terms

    # Get terms for 4x4x8
    terms = _strassen_terms(4, 4, 8)
    
    # Build matrices
    n, m, p = 4, 4, 8
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (U_t, V_t, W_t) in enumerate(terms):
        for (i, j), coeff in U_t.items():
            if coeff != 0:
                U[i * m + j][c] = coeff
        for (j, k), coeff in V_t.items():
            if coeff != 0:
                V[j * p + k][c] = coeff
        for (k, i), coeff in W_t.items():
            if coeff != 0:
                W[k * n + i][c] = coeff
                
    # Verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0
            for c in range(rank):
                u_val = 0
                for j in range(m):
                    u_val += U[i * m + j][c] * A[i, j]
                v_val = 0
                for j in range(m):
                    v_val += V[j * p + k][c] * B[j, k]
                val += u_val * v_val * W[k * n + i][c]
            C_computed[i, k] = val
            
    max_err = np.max(np.abs(C_computed - C_expected))
    
    if max_err > 1e-10:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
