import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 115

    n, m, p = 4, 4, 4

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    # Strassen 2x2 block terms: (A_comb, B_comb, C_comb)
    # Each term is a list of ((block_i, block_j), coeff)
    block_terms = [
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1), ((1,1), 1)]),
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((1,0), 1)]),
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((0,1), 1)]),
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
    ]

    sub_terms = block_terms

    U = np.zeros((n * m, 49), dtype=float)
    V = np.zeros((m * p, 49), dtype=float)
    W = np.zeros((p * n, 49), dtype=float)

    rank_idx = 0
    for A_comb, B_comb, C_comb in block_terms:
        for k in range(7):
            A_loc, B_loc, C_loc = sub_terms[k]

            U_elems = {}
            for (bi, bj), coeff_block in A_comb:
                for (lr, lc), coeff_loc in A_loc:
                    gi = 2 * bi + lr
                    gj = 2 * bj + lc
                    idx = u_idx(gi, gj)
                    U_elems[idx] = U_elems.get(idx, 0) + coeff_block * coeff_loc

            V_elems = {}
            for (bi, bj), coeff_block in B_comb:
                for (lr, lc), coeff_loc in B_loc:
                    gj = 2 * bi + lr
                    gk = 2 * bj + lc
                    idx = v_idx(gj, gk)
                    V_elems[idx] = V_elems.get(idx, 0) + coeff_block * coeff_loc

            W_elems = {}
            for (ci, cj), coeff_block in C_comb:
                for (lr, lc), coeff_loc in C_loc:
                    gi = 2 * ci + lr
                    gj = 2 * cj + lc
                    idx = w_idx(gj, gi)
                    W_elems[idx] = W_elems.get(idx, 0) + coeff_block * coeff_loc

            for idx, val in U_elems.items():
                U[idx, rank_idx] += val
            for idx, val in V_elems.items():
                V[idx, rank_idx] += val
            for idx, val in W_elems.items():
                W[idx, rank_idx] += val

            rank_idx += 1

    def verify_exact(U_mat, V_mat, W_mat):
        """Verify decomposition using exact Fraction arithmetic."""
        try:
            U_f = [[Fraction(int(round(val * 2))) / 2 for val in row] for row in U_mat]
            V_f = [[Fraction(int(round(val * 2))) / 2 for val in row] for row in V_mat]
            W_f = [[Fraction(int(round(val * 2))) / 2 for val in row] for row in W_mat]
            
            r = len(U_f[0])
            T = np.zeros((n, m, p), dtype=object)
            
            # Efficient exact check: sum over k of U[i,j,k] * V[j,l,k] * W[l,i,k]
            # We only need to check if T[i,j,l] == 1 for all i,j,l
            # But we can just compute the tensor and check.
            for i in range(n):
                for j in range(m):
                    for l in range(p):
                        total = Fraction(0)
                        for k in range(r):
                            total += U_f[i*m+j][k] * V_f[j*p+l][k] * W_f[l*n+i][k]
                        if total != 1:
                            return False
            return True
        except Exception:
            return False

    if time.time() - start_time > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt rank reduction to 48
    try:
        # Build matricization
        terms_flat = np.zeros((64, 49), dtype=float)
        for k in range(49):
            for i in range(n):
                for j in range(m):
                    for l in range(p):
                        terms_flat[i * 16 + j * 4 + l, k] = (
                            U[i * m + j, k] * V[j * p + l, k] * W[l * n + i, k]
                        )

        U_svd, s, Vt_svd = np.linalg.svd(terms_flat, full_matrices=True)
        tol_sv = max(terms_flat.shape) * np.finfo(float).eps * (s[0] if s[0] > 0 else 1.0)
        num_sv = len(s)
        null_dim = sum(1 for idx in range(num_sv) if s[idx] < tol_sv)

        if null_dim > 0:
            c = Vt_svd[num_sv - null_dim, :].copy()
            k_elim = int(np.argmax(np.abs(c)))
            
            # Snap c to rationals with small denominators
            c_rational = []
            valid_snap = True
            for val in c:
                # Try denominators 1, 2, 4, 8
                found = False
                for den in [1, 2, 4, 8]:
                    num = round(val * den)
                    if abs(val - num/den) < 1e-6:
                        c_rational.append(Fraction(num, den))
                        found = True
                        break
                if not found:
                    valid_snap = False
                    break
            
            if valid_snap:
                c_rational = [x / c_rational[k_elim] for x in c_rational]
                
                # Apply reduction
                U48 = U.copy()
                V48 = V.copy()
                W48 = W.copy()
                
                for j in range(49):
                    if j != k_elim:
                        factor = 1 - c_rational[j]
                        U48[:, j] *= float(factor)
                        # We can scale V or W instead, but U is fine.
                        # Actually, to keep integers/half-integers, we should be careful.
                        # The term is U_k V_k W_k.
                        # New term k is (U_k - c_k U_elim) V_k W_k = U_k V_k W_k - c_k U_elim V_k W_k.
                        # This changes the structure.
                        # Standard elimination:
                        # Sum_{k != elim} (U_k - c_k/c_elim U_elim) V_k W_k + 0 * U_elim V_elim W_elim
                        # This is correct.
                        
                # Remove column k_elim
                keep = [j for j in range(49) if j != k_elim]
                U48 = U48[:, keep]
                V48 = V48[:, keep]
                W48 = W48[:, keep]
                
                if verify_exact(U48, V48, W48):
                    return {"U": U48.tolist(), "V": V48.tolist(), "W": W48.tolist()}
                    
    except Exception:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
