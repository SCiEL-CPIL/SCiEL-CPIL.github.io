import time
import numpy as np
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 120.0
    n, m, p = 2, 4, 4
    
    # Fallback: The known Strassen-like decomposition of rank 28
    def make_strassen_solution():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def add_block(col_offset, X_rows, Y_rows, Z_rows):
            # M1
            U[X_rows[0]][col_offset + 0] += 1
            U[X_rows[3]][col_offset + 0] += 1
            V[Y_rows[0]][col_offset + 0] += 1
            V[Y_rows[3]][col_offset + 0] += 1
            W[Z_rows[0]][col_offset + 0] += 1
            W[Z_rows[3]][col_offset + 0] += 1

            # M2
            U[X_rows[2]][col_offset + 1] += 1
            U[X_rows[3]][col_offset + 1] += 1
            V[Y_rows[0]][col_offset + 1] += 1
            W[Z_rows[2]][col_offset + 1] += 1
            W[Z_rows[3]][col_offset + 1] -= 1

            # M3
            U[X_rows[0]][col_offset + 2] += 1
            V[Y_rows[1]][col_offset + 2] += 1
            V[Y_rows[3]][col_offset + 2] -= 1
            W[Z_rows[1]][col_offset + 2] += 1
            W[Z_rows[3]][col_offset + 2] += 1

            # M4
            U[X_rows[3]][col_offset + 3] += 1
            V[Y_rows[2]][col_offset + 3] += 1
            V[Y_rows[0]][col_offset + 3] -= 1
            W[Z_rows[0]][col_offset + 3] += 1
            W[Z_rows[2]][col_offset + 3] += 1

            # M5
            U[X_rows[0]][col_offset + 4] += 1
            U[X_rows[1]][col_offset + 4] += 1
            V[Y_rows[3]][col_offset + 4] += 1
            W[Z_rows[0]][col_offset + 4] -= 1
            W[Z_rows[1]][col_offset + 4] += 1

            # M6
            U[X_rows[2]][col_offset + 5] += 1
            U[X_rows[0]][col_offset + 5] -= 1
            V[Y_rows[0]][col_offset + 5] += 1
            V[Y_rows[1]][col_offset + 5] += 1
            W[Z_rows[3]][col_offset + 5] += 1

            # M7
            U[X_rows[1]][col_offset + 6] += 1
            U[X_rows[3]][col_offset + 6] -= 1
            V[Y_rows[2]][col_offset + 6] += 1
            V[Y_rows[3]][col_offset + 6] += 1
            W[Z_rows[0]][col_offset + 6] += 1

        # Block 1: A_L B_TL -> C_LL
        X1 = [0, 1, 4, 5]
        Y1 = [0, 1, 4, 5]
        Z1 = [0, 2, 1, 3]
        add_block(0, X1, Y1, Z1)

        # Block 2: A_R B_BL -> C_LR
        X2 = [2, 3, 6, 7]
        Y2 = [8, 9, 12, 13]
        Z2 = [0, 2, 1, 3]
        add_block(7, X2, Y2, Z2)

        # Block 3: A_L B_TR -> C_RL
        X3 = [0, 1, 4, 5]
        Y3 = [2, 3, 6, 7]
        Z3 = [4, 6, 5, 7]
        add_block(14, X3, Y3, Z3)

        # Block 4: A_R B_BR -> C_RR
        X4 = [2, 3, 6, 7]
        Y4 = [10, 11, 14, 15]
        Z4 = [4, 6, 5, 7]
        add_block(21, X4, Y4, Z4)

        return {"U": U, "V": V, "W": W}

    # Greedy Search for lower rank
    def try_greedy_search(max_steps=27):
        # Build Tensor T (8, 16, 8)
        T = np.zeros((8, 16, 8), dtype=float)
        for j in range(4):
            for i in range(2):
                for k in range(4):
                    a = 4 * i + j
                    b = 4 * j + k
                    c = 2 * k + i
                    T[a, b, c] = 1.0
        
        R = T.copy()
        components = []
        
        for step in range(max_steps):
            if time.time() > deadline - 10:
                return None
            
            # 1. Find w (mode 3)
            R3 = R.reshape(8 * 16, 8)
            try:
                _, _, Vh = np.linalg.svd(R3)
                w = Vh[0] # Leading right singular vector
            except:
                return None
            
            # 2. Contract to get R' (8, 16)
            R_prime = np.einsum('abc,c->ab', R, w)
            
            # 3. Find v (mode 2)
            try:
                _, _, Vh2 = np.linalg.svd(R_prime)
                v = Vh2[0] # Leading right singular vector
            except:
                return None
            
            # 4. Contract to get R'' (8)
            R_double_prime = np.einsum('ab,b->a', R_prime, v)
            
            # 5. Find u (mode 1)
            u_norm = np.linalg.norm(R_double_prime)
            if u_norm < 1e-10:
                break
            u = R_double_prime / u_norm
            
            # 6. Calculate alpha
            # <R, u v w>
            inner = np.einsum('abc,a,b,c->', R, u, v, w)
            norm_sq = (u_norm**2) * (np.linalg.norm(v)**2) * (np.linalg.norm(w)**2)
            if norm_sq < 1e-10:
                break
            alpha = inner / norm_sq
            
            # 7. Update R
            R -= alpha * np.einsum('a,b,c->abc', u, v, w)
            
            # Store component
            components.append((u, v, w, alpha))
            
            # Check residual
            if np.linalg.norm(R) < 1e-8:
                # Found a decomposition of rank step+1
                # Attempt rational reconstruction
                return reconstruct_integer_solution(components)
                
        return None

    def reconstruct_integer_solution(components):
        # This is a simplified reconstruction. 
        # We assume the coefficients are integers or half-integers.
        # We will try to scale each component to integers.
        U = []
        V = []
        W = []
        
        for u, v, w, alpha in components:
            # We want to represent alpha * u * v * w as integers.
            # Let's try to find a common scale.
            # Since T is integer, the sum of products must be integer.
            # We can just store the floats and hope the evaluator handles it?
            # No, "integer/half-integer entries".
            
            # Try to convert to Fraction
            try:
                alpha_f = Fraction(alpha).limit_denominator(100)
                u_list = [Fraction(x).limit_denominator(100) for x in u]
                v_list = [Fraction(x).limit_denominator(100) for x in v]
                w_list = [Fraction(x).limit_denominator(100) for x in w]
                
                # Check if they are "close" to integers/half-integers
                # If not, this component is likely noise or non-rational
                # For now, if we get here, we assume it's valid.
                
                # We need to output matrices.
                # U, V, W will be lists of lists.
                
                # We append the column for this rank-1 component.
                # But we need to ensure they are integers.
                # If alpha is 0.5, and u is 1, then U=1, V=1, W=0.5?
                # The problem allows half-integers.
                
                # We will store the values.
                U.append([float(x) for x in u_list])
                V.append([float(x) for x in v_list])
                W.append([float(alpha_f * x) for x in w_list])
                
            except:
                return None

        # Transpose to get (n*m, rank) etc.
        if not U:
            return None
            
        U_mat = np.array(U).T
        V_mat = np.array(V).T
        W_mat = np.array(W).T
        
        # Verify
        # Check if U_mat @ V_mat @ W_mat approximates T
        # T is 8x16x8.
        # Product is sum of U[:,r] V[:,r] W[:,r]
        # Reconstruct T_check
        T_check = np.zeros((8, 16, 8))
        for r in range(U_mat.shape[1]):
            T_check += np.einsum('a,b,c->abc', U_mat[:,r], V_mat[:,r], W_mat[:,r])
            
        if not np.allclose(T_check, T, atol=1e-5):
            return None
            
        # Convert to lists of ints/floats
        # The evaluator expects integer/half-integer.
        # We will round to nearest 0.5
        def round_half(x):
            return round(x * 2) / 2.0
            
        U_out = [[round_half(x) for x in row] for row in U_mat]
        V_out = [[round_half(x) for x in row] for row in V_mat]
        W_out = [[round_half(x) for x in row] for row in W_mat]
        
        return {"U": U_out, "V": V_out, "W": W_out}

    # Try greedy search
    result = try_greedy_search()
    
    if result is not None:
        return result
    else:
        return make_strassen_solution()
