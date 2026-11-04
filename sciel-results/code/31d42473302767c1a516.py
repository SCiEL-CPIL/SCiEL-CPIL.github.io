"""Systematic rank-1 peeling for 2x3x5 matrix multiplication tensor."""
import numpy as np
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 30
    
    # Build the tensor as a dict of non-zero entries (exact arithmetic)
    # T[(i*m+j, j*p+k, k*n+i)] = 1
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i*m+j, j*p+k, k*n+i)] = Fraction(1)
    
    # Try rank-1 peeling with exact arithmetic
    residual = dict(T)
    U_list = []
    V_list = []
    W_list = []
    
    for step in range(schoolbook_rank):
        if not residual:
            break
        
        support = set(residual.keys())
        found = False
        
        # Try rank-1 tensors with small support
        # u supported on Su, v on Sv, w on Sw
        # Need Su x Sv x Sw subset of support
        for su_size in range(1, 3):
            if found:
                break
            for sv_size in range(1, 3):
                if found:
                    break
                for sw_size in range(1, 3):
                    if found:
                        break
                    if su_size * sv_size * sw_size > len(support):
                        continue
                    
                    all_u = list(range(n*m))
                    all_v = list(range(m*p))
                    all_w = list(range(p*n))
                    
                    for Su in combinations(all_u, su_size):
                        if found:
                            break
                        for Sv in combinations(all_v, sv_size):
                            if found:
                                break
                            for Sw in combinations(all_w, sw_size):
                                # Check Su x Sv x Sw subset of support
                                valid = all((a, b, c) in support for a in Su for b in Sv for c in Sw)
                                if not valid:
                                    continue
                                
                                # Solve u[a]*v[b]*w[c] = residual[(a,b,c)]
                                # Fix w[Sw[0]] = 1, solve for u, v from residual[(a,b,Sw[0])]
                                c0 = Sw[0]
                                M = np.array([[float(residual[(a, b, c0)]) for b in Sv] for a in Su])
                                
                                if np.linalg.matrix_rank(M) > 1:
                                    continue
                                
                                # Extract u, v
                                if M[0, 0] != 0:
                                    u_vals = [Fraction(1)] + [Fraction(residual[(a, Sv[0], c0)]) / Fraction(residual[(Su[0], Sv[0], c0)]) for a in Su[1:]]
                                    v_vals = [Fraction(residual[(Su[0], b, c0)]) for b in Sv]
                                else:
                                    continue
                                
                                # Verify u*v matches
                                ok = all(u_vals[ai] * v_vals[bi] == residual[(Su[ai], Sv[bi], c0)] 
                                        for ai in range(su_size) for bi in range(sv_size))
                                if not ok:
                                    continue
                                
                                # Find w for other c
                                w_vals = [Fraction(1)] * sw_size
                                ok = True
                                for ci in range(1, sw_size):
                                    c = Sw[ci]
                                    if u_vals[0] != 0 and v_vals[0] != 0:
                                        w_vals[ci] = residual[(Su[0], Sv[0], c)] / (u_vals[0] * v_vals[0])
                                    else:
                                        ok = False
                                        break
                                    for ai in range(su_size):
                                        for bi in range(sv_size):
                                            if u_vals[ai] * v_vals[bi] * w_vals[ci] != residual[(Su[ai], Sv[bi], c)]:
                                                ok = False
                                                break
                                        if not ok:
                                            break
                                    if not ok:
                                        break
                                if not ok:
                                    continue
                                
                                # Build full vectors
                                u_full = [Fraction(0)] * (n*m)
                                v_full = [Fraction(0)] * (m*p)
                                w_full = [Fraction(0)] * (p*n)
                                for idx, a in enumerate(Su):
                                    u_full[a] = u_vals[idx]
                                for idx, b in enumerate(Sv):
                                    v_full[b] = v_vals[idx]
                                for idx, c in enumerate(Sw):
                                    w_full[c] = w_vals[idx]
                                
                                # Subtract from residual
                                for a in Su:
                                    for b in Sv:
                                        for c in Sw:
                                            key = (a, b, c)
                                            residual[key] -= u_full[a] * v_full[b] * w_full[c]
                                            if residual[key] == 0:
                                                del residual[key]
                                
                                U_list.append(u_full)
                                V_list.append(v_full)
                                W_list.append(w_full)
                                found = True
                                break
        
        if not found:
            break
    
    final_rank = len(U_list)
    
    # If we didn't find a complete decomposition or it's not better, use schoolbook
    if final_rank == 0 or final_rank >= schoolbook_rank or residual:
        U = [[0] * schoolbook_rank for _ in range(n*m)]
        V = [[0] * schoolbook_rank for _ in range(m*p)]
        W = [[0] * schoolbook_rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to float lists, transposed to (dim, rank)
    U_mat = [[float(U_list[c][i]) for c in range(final_rank)] for i in range(n*m)]
    V_mat = [[float(V_list[c][i]) for c in range(final_rank)] for i in range(m*p)]
    W_mat = [[float(W_list[c][i]) for c in range(final_rank)] for i in range(p*n)]
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
