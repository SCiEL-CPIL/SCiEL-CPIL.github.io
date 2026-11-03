import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 135
    n, m, p = DIMENSIONS  # 3, 5, 7
    nm, mp, pn = n * m, m * p, p * n

    def schoolbook():
        rank = nm * mp // m
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()

    def verify(U, V, W):
        R = U.shape[1]
        Ur = U.reshape(n, m, R)
        Vr = V.reshape(m, p, R)
        Wr = W.reshape(p, n, R)
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        return np.allclose(T, 1.0, atol=1e-10)

    def strassen_block_terms(i0, i1, j0, j1, k0, k1):
        terms = []
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j0] = 1.0; u[i1*m+j1] = 1.0
        v[j0*p+k0] = 1.0; v[j1*p+k1] = 1.0
        w[k0*n+i0] = 1.0; w[k1*n+i1] = 1.0
        terms.append((u, v, w))
        
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i1*m+j0] = 1.0; u[i1*m+j1] = 1.0
        v[j0*p+k0] = 1.0
        w[k0*n+i1] = 1.0; w[k1*n+i1] = -1.0
        terms.append((u, v, w))
        
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j0] = 1.0
        v[j0*p+k1] = 1.0; v[j1*p+k1] = -1.0
        w[k1*n+i0] = 1.0; w[k1*n+i1] = 1.0
        terms.append((u, v, w))
        
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i1*m+j1] = 1.0
        v[j1*p+k0] = 1.0; v[j0*p+k0] = -1.0
        w[k0*n+i0] = 1.0; w[k0*n+i1] = 1.0
        terms.append((u, v, w))
        
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j0] = 1.0; u[i0*m+j1] = 1.0
        v[j1*p+k1] = 1.0
        w[k0*n+i0] = -1.0; w[k1*n+i0] = 1.0
        terms.append((u, v, w))
        
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i1*m+j0] = 1.0; u[i0*m+j0] = -1.0
        v[j0*p+k0] = 1.0; v[j0*p+k1] = 1.0
        w[k1*n+i1] = 1.0
        terms.append((u, v, w))
        
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j1] = 1.0; u[i1*m+j1] = -1.0
        v[j1*p+k0] = 1.0; v[j1*p+k1] = 1.0
        w[k0*n+i0] = 1.0
        terms.append((u, v, w))
        
        return terms

    def build_decomposition(blocks):
        U_list, V_list, W_list = [], [], []
        covered = set()
        
        for I, J, K in blocks:
            i0, i1 = sorted(I)
            j0, j1 = sorted(J)
            k0, k1 = sorted(K)
            terms = strassen_block_terms(i0, i1, j0, j1, k0, k1)
            for u, v, w in terms:
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
            for i in I:
                for j in J:
                    for k in K:
                        covered.add((i, j, k))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
                        u[i*m+j] = 1.0
                        v[j*p+k] = 1.0
                        w[k*n+i] = 1.0
                        U_list.append(u)
                        V_list.append(v)
                        W_list.append(w)
        
        U = np.array(U_list).T
        V = np.array(V_list).T
        W = np.array(W_list).T
        return U, V, W

    def is_valid_block_set(blocks):
        seen = set()
        for I, J, K in blocks:
            for i in I:
                for j in J:
                    for k in K:
                        if (i, j, k) in seen:
                            return False
                        seen.add((i, j, k))
        return True

    # Known good configuration: 6 blocks
    # I: {0,1}, {1,2}
    # J: {0,1}, {2,3}, {3,4}
    # K: {0,1}, {2,3}, {4,5}, {5,6}
    
    # We will search for the best set of up to 8 blocks
    # to see if we can beat 99 (which is 105 - 6*1 = 99)
    # Rank = 105 - (#blocks)
    # So we want to maximize #blocks.
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Generate candidate blocks
    I_sets = [frozenset({i, i+1}) for i in range(n-1)]
    J_sets = [frozenset({j, j+1}) for j in range(m-1)]
    K_sets = [frozenset({k, k+1}) for k in range(p-1)]
    
    all_blocks = []
    for I in I_sets:
        for J in J_sets:
            for K in K_sets:
                all_blocks.append((I, J, K))
                
    # Greedy search for max disjoint blocks
    # This is a set packing problem.
    # We can use a simple backtracking or greedy with random restarts.
    
    def greedy_packing(max_blocks=10):
        best_count = 0
        best_set = []
        for _ in range(50): # 50 restarts
            current_set = []
            remaining_blocks = list(all_blocks)
            import random
            random.shuffle(remaining_blocks)
            for b in remaining_blocks:
                if len(current_set) >= max_blocks:
                    break
                # Check if b is disjoint from all in current_set
                ok = True
                for cb in current_set:
                    # Check intersection
                    # (I1 x J1 x K1) intersect (I2 x J2 x K2) is empty?
                    # Equivalent to: I1 cap I2 = empty OR J1 cap J2 = empty OR K1 cap K2 = empty
                    I1, J1, K1 = b
                    I2, J2, K2 = cb
                    if I1 & I2 and J1 & J2 and K1 & K2:
                        ok = False
                        break
                if ok:
                    current_set.append(b)
            if len(current_set) > best_count:
                best_count = len(current_set)
                best_set = current_set
        return best_set

    if time.time() < deadline - 20:
        found_blocks = greedy_packing(10)
        if is_valid_block_set(found_blocks):
            U, V, W = build_decomposition(found_blocks)
            if verify(U, V, W):
                rank = U.shape[1]
                if rank < best_rank:
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W

    # Remove redundant terms
    if best_U is not None:
        U, V, W = best_U, best_V, best_W
        R = U.shape[1]
        Ur = U.reshape(n, m, R)
        Vr = V.reshape(m, p, R)
        Wr = W.reshape(p, n, R)
        
        removed_any = True
        while removed_any and time.time() < deadline - 10:
            removed_any = False
            for c in range(R):
                if time.time() >= deadline - 10:
                    break
                ij_sup = set((i,j) for i in range(n) for j in range(m) if Ur[i,j,c]!=0)
                jk_sup = set((j,k) for j in range(m) for k in range(p) if Vr[j,k,c]!=0)
                ki_sup = set((k,i) for k in range(p) for i in range(n) if Wr[k,i,c]!=0)
                
                if not ij_sup or not jk_sup or not ki_sup:
                    U = np.delete(U, c, axis=1)
                    V = np.delete(V, c, axis=1)
                    W = np.delete(W, c, axis=1)
                    R -= 1
                    Ur = U.reshape(n, m, R)
                    Vr = V.reshape(m, p, R)
                    Wr = W.reshape(p, n, R)
                    removed_any = True
                    break
                    
                # Check intersection
                found = False
                for (i,j) in ij_sup:
                    for (j2,k) in jk_sup:
                        if j==j2:
                            for (k2,i2) in ki_sup:
                                if k==k2 and i==i2:
                                    found = True
                                    break
                            if found: break
                    if found: break
                if not found:
                    U = np.delete(U, c, axis=1)
                    V = np.delete(V, c, axis=1)
                    W = np.delete(W, c, axis=1)
                    R -= 1
                    Ur = U.reshape(n, m, R)
                    Vr = V.reshape(m, p, R)
                    Wr = W.reshape(p, n, R)
                    removed_any = True
                    break

        if verify(U, V, W):
            best_U, best_V, best_W = U, V, W
            best_rank = U.shape[1]

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
