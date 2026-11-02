"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    start_time = time.time()
    deadline = start_time + 140.0

    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def strassen_2x2x2():
        """Strassen's 2x2x2 rank-7 decomposition."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1: (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1
        W[3, 0] = 1
        
        # M2: (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        W[3, 1] = -1
        
        # M3: A11(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        W[3, 2] = 1
        
        # M4: A22(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1
        W[2, 3] = 1
        
        # M5: (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1
        W[1, 4] = 1
        
        # M6: (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7: (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def get_block_decomp(n, m, p):
        """Get a decomposition for n x m x p."""
        if n == 1 and m == 1 and p == 1:
            return np.ones((1,1)), np.ones((1,1)), np.ones((1,1))
        if n == 1 or m == 1 or p == 1:
            return schoolbook(n, m, p)
        if n == 2 and m == 2 and p == 2:
            return strassen_2x2x2()
        return schoolbook(n, m, p)

    def block_rank(n, m, p):
        """Get the rank of the best known decomposition for n x m x p."""
        if n == 2 and m == 2 and p == 2:
            return 7
        return n * m * p

    def verify(U, V, W, n, m, p):
        """Verify the decomposition."""
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_decomp = np.zeros((n, p))
        for r in range(U.shape[1]):
            M_r = np.zeros(n)
            N_r = np.zeros(p)
            for i in range(n):
                for j in range(m):
                    M_r[i] += U[i*m+j, r] * A[i, j]
            for k in range(p):
                for j in range(m):
                    N_r[k] += V[j*p+k, r] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_decomp[i, k] += W[k*n+i, r] * M_r[i] * N_r[k]
        
        return np.allclose(C_decomp, C_expected, atol=1e-10)

    def build_decomposition(n_s, m_s, p_s):
        """Build decomposition given partitions of n, m, p."""
        U_list = []
        V_list = []
        W_list = []
        
        # Compute offsets
        n_off = [0]
        for s in n_s[:-1]:
            n_off.append(n_off[-1] + s)
        m_off = [0]
        for s in m_s[:-1]:
            m_off.append(m_off[-1] + s)
        p_off = [0]
        for s in p_s[:-1]:
            p_off.append(p_off[-1] + s)
        
        for i in range(len(n_s)):
            ni = n_s[i]
            no = n_off[i]
            for j in range(len(m_s)):
                mj = m_s[j]
                mo = m_off[j]
                for k in range(len(p_s)):
                    pk = p_s[k]
                    po = p_off[k]
                    
                    U_blk, V_blk, W_blk = get_block_decomp(ni, mj, pk)
                    r_blk = U_blk.shape[1]
                    
                    # Expand to global
                    U_exp = np.zeros((n * m, r_blk))
                    V_exp = np.zeros((m * p, r_blk))
                    W_exp = np.zeros((p * n, r_blk))
                    
                    for li in range(ni):
                        for lj in range(mj):
                            U_exp[(no + li) * m + (mo + lj), :] = U_blk[li * mj + lj, :]
                    
                    for lj in range(mj):
                        for lk in range(pk):
                            V_exp[(mo + lj) * p + (po + lk), :] = V_blk[lj * pk + lk, :]
                    
                    for li in range(ni):
                        for lk in range(pk):
                            W_exp[(po + lk) * n + (no + li), :] = W_blk[lk * ni + li, :]
                    
                    U_list.append(U_exp)
                    V_list.append(V_exp)
                    W_list.append(W_exp)
        
        U_final = np.hstack(U_list)
        V_final = np.hstack(V_list)
        W_final = np.hstack(W_list)
        return U_final, V_final, W_final

    def compute_total_rank(n_s, m_s, p_s):
        total = 0
        for ni in n_s:
            for mj in m_s:
                for pk in p_s:
                    total += block_rank(ni, mj, pk)
        return total

    # Find optimal partition
    # Generate all compositions of n, m, p
    def compositions(total, max_parts=3):
        """Generate all compositions of total into at most max_parts positive integers."""
        results = []
        def helper(remaining, parts, current):
            if not parts:
                if remaining > 0:
                    results.append(current + [remaining])
                return
            for x in range(1, remaining + 1):
                helper(remaining - x, parts - 1, current + [x])
        helper(total, max_parts, [])
        return results

    n_comps = compositions(3, 3)
    m_comps = compositions(5, 3)
    p_comps = compositions(6, 3)
    
    best_rank = 90  # schoolbook
    best_partition = None
    
    for nc in n_comps:
        for mc in m_comps:
            for pc in p_comps:
                r = compute_total_rank(nc, mc, pc)
                if r < best_rank:
                    best_rank = r
                    best_partition = (nc, mc, pc)
    
    if best_partition is None:
        best_partition = ([3], [5], [6])
    
    nc, mc, pc = best_partition
    
    # Build the decomposition
    U, V, W = build_decomposition(nc, mc, pc)
    
    # Verify
    if not verify(U, V, W, 3, 5, 6):
        # Fallback to schoolbook
        U, V, W = schoolbook(3, 5, 6)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
