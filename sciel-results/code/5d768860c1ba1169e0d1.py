"""Rewritten program for 4x5x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    def strassen2_decomp():
        """2x2x2 Strassen, rank 7. Returns U(4x7), V(4x7), W(4x7)"""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        U[2][1]=1; U[3][1]=1; V[2][1]=1; W[2][1]=1
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1; W[3][6]=1
        return U, V, W

    def get_decomp(ni, mi, pi):
        """
        Returns U, V, W for ni x mi x pi block.
        Uses Strassen for 2x2x2.
        Uses optimized schoolbook-like for others based on rank analysis.
        """
        if ni == 2 and mi == 2 and pi == 2:
            return strassen2_decomp()
        
        # For other sizes, we can use a simple construction that achieves the theoretical lower bound for these small cases.
        # Rank(ni, 1, pi) = ni
        # Rank(1, mi, pi) = pi
        # Rank(ni, mi, 1) = mi
        # Rank(1, 1, pi) = 1
        # Rank(ni, 1, 1) = 1
        # Rank(1, mi, 1) = 1
        
        # General schoolbook is ni*mi*pi.
        # But we can do better if one dim is 1.
        
        rank = ni * mi * pi
        # Check specific cases
        if ni == 1 and mi == 1:
            rank = 1
        elif mi == 1:
            rank = ni
        elif pi == 1:
            rank = mi
        elif ni == 1:
            rank = pi
        # Note: 1x2x2 rank is 4? Or 2?
        # If ni=1, mi=2, pi=2. Rank is 4?
        # Actually, 1x2x2 is equivalent to 2x2x1 by symmetry? No.
        # Let's stick to safe ranks. If not 2x2x2, use schoolbook unless one dim is 1.
        # For 1x2x2, schoolbook is 4. Is rank 2 possible?
        # C = A B. A is 1x2, B is 2x2.
        # C0 = A0 B00 + A1 B10
        # C1 = A0 B01 + A1 B11
        # This is 4 multiplications.
        # Can we do it in 2?
        # M1 = A0 (B00+B11), M2 = A1 (B01+B10)?
        # C0 = M1 + A0 B10 + A1 B00? No.
        # It is known that rank(1, m, p) = m*p for m,p >= 2?
        # Actually, rank(1, 2, 2) is 4.
        # So we only improve if ni=1 AND mi=1, or mi=1, or pi=1.
        
        U = [[0]*rank for _ in range(ni*mi)]
        V = [[0]*rank for _ in range(mi*pi)]
        W = [[0]*rank for _ in range(pi*ni)]
        
        if ni == 1 and mi == 1:
            # Rank 1
            U[0][0] = 1
            V[0][0] = 1
            for k in range(pi):
                W[k*ni][0] = 1
            return U, V, W, rank

        if mi == 1:
            # Rank ni
            # C[i, k] = A[i, 0] B[0, k]
            # Term i: U[i, i]=1, V[0, i]=1, W[k*ni+i, i]=1 for all k?
            # Sum over r. r is i.
            # U[i, i]=1. V[0, i]=1.
            # W[k*ni+i, i] should be 1 for all k?
            # Then sum_r U[i,r] V[0,r] W[k*ni+i,r] = U[i,i] V[0,i] W[k*ni+i,i] = 1*1*1 = 1.
            # Correct.
            for i in range(ni):
                U[i][i] = 1
                V[0][i] = 1
                for k in range(pi):
                    W[k*ni + i][i] = 1
            return U, V, W, rank

        if pi == 1:
            # Rank mi
            # C[i, 0] = sum_j A[i, j] B[j, 0]
            # Term j: U[i*mi+j, j]=1, V[j*pi+0, j]=1, W[0*ni+i, j]=1?
            # W index is k*ni+i. k=0. W[i, j].
            for j in range(mi):
                for i in range(ni):
                    U[i*mi+j][j] = 1
                V[j][j] = 1
                for i in range(ni):
                    W[i][j] = 1
            return U, V, W, rank

        if ni == 1:
            # Rank pi
            # C[0, k] = sum_j A[0, j] B[j, k]
            # Term k: U[0*mi+j, k]? No.
            # We need sum_j A[0,j] B[j,k] = C[0,k].
            # If we use rank pi, we can group by k?
            # No, this is not straightforward.
            # Use schoolbook for ni=1, mi>1, pi>1.
            pass

        # Schoolbook
        col = 0
        for i in range(ni):
            for j in range(mi):
                for k in range(pi):
                    U[i*mi+j][col] = 1
                    V[j*pi+k][col] = 1
                    W[k*ni+i][col] = 1
                    col += 1
        return U, V, W, rank

    def build_partition(n_parts, m_parts, p_parts):
        total_rank = 0
        for ni in n_parts:
            for mi in m_parts:
                for pi in p_parts:
                    total_rank += get_decomp(ni, mi, pi)[3]
        
        U_full = [[0] * total_rank for _ in range(n*m)]
        V_full = [[0] * total_rank for _ in range(m*p)]
        W_full = [[0] * total_rank for _ in range(p*n)]
        
        col = 0
        i_off = 0
        for ni in n_parts:
            j_off = 0
            for mi in m_parts:
                k_off = 0
                for pi in p_parts:
                    U_b, V_b, W_b, rank_b = get_decomp(ni, mi, pi)
                    for r in range(rank_b):
                        for i in range(ni):
                            for j in range(mi):
                                U_full[(i_off+i)*m + (j_off+j)][col+r] = U_b[i*mi+j][r]
                        for j in range(mi):
                            for k in range(pi):
                                V_full[(j_off+j)*p + (k_off+k)][col+r] = V_b[j*pi+k][r]
                        for k in range(pi):
                            for i in range(ni):
                                W_full[(k_off+k)*n + (i_off+i)][col+r] = W_b[k*ni+i][r]
                    col += rank_b
                    k_off += pi
                j_off += mi
            i_off += ni
        
        return U_full, V_full, W_full, total_rank

    def verify(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i]
                    s = np.sum(prod)
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True

    # Generate partitions
    def gen_parts(total, max_size=2):
        parts = []
        def backtrack(remaining, current):
            if remaining == 0:
                parts.append(current[:])
                return
            for size in range(1, min(max_size, remaining) + 1):
                current.append(size)
                backtrack(remaining - size, current)
                current.pop()
        backtrack(total, [])
        return parts

    n_parts_list = gen_parts(n)
    m_parts_list = gen_parts(m)
    p_parts_list = gen_parts(p)

    best_rank_local = best_rank
    best_result_local = best_result

    for n_p in n_parts_list:
        for m_p in m_parts_list:
            for p_p in p_parts_list:
                if time.time() > deadline - 5:
                    break
                U_t, V_t, W_t, rank_t = build_partition(n_p, m_p, p_p)
                if rank_t < best_rank_local:
                    if verify(U_t, V_t, W_t, rank_t):
                        best_result_local = {"U": U_t, "V": V_t, "W": W_t}
                        best_rank_local = rank_t

    return best_result_local
# EVOLVE-BLOCK-END
