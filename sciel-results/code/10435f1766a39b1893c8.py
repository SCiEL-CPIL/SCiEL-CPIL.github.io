"""Matrix multiplication 3x5x6 tensor decomposition."""
import time
import itertools
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify decomposition."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    def build_decomposition(n_parts, m_parts, p_parts, use_strassen=True):
        """Build decomposition from block partition."""
        n_offsets = [0]
        for i in range(len(n_parts) - 1):
            n_offsets.append(n_offsets[-1] + n_parts[i])
        m_offsets = [0]
        for i in range(len(m_parts) - 1):
            m_offsets.append(m_offsets[-1] + m_parts[i])
        p_offsets = [0]
        for i in range(len(p_parts) - 1):
            p_offsets.append(p_offsets[-1] + p_parts[i])
        
        terms = []
        
        def add_strassen(a_rows, c_cols, d_cols):
            a0, a1 = a_rows
            c0, c1 = c_cols
            d0, d1 = d_cols
            # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
            terms.append(([(a0*m+c0,1),(a1*m+c1,1)], [(c0*p+d0,1),(c1*p+d1,1)], [(d0*n+a0,1),(d1*n+a1,1)]))
            # M2 = (A10+A11)B00 -> C10+=1, C11+=-1
            terms.append(([(a1*m+c0,1),(a1*m+c1,1)], [(c0*p+d0,1)], [(d0*n+a1,1),(d1*n+a1,-1)]))
            # M3 = A00(B01-B11) -> C01+=1, C11+=1
            terms.append(([(a0*m+c0,1)], [(c0*p+d1,1),(c1*p+d1,-1)], [(d1*n+a0,1),(d1*n+a1,1)]))
            # M4 = A11(B10-B00) -> C00+=1, C10+=1
            terms.append(([(a1*m+c1,1)], [(c1*p+d0,1),(c0*p+d0,-1)], [(d0*n+a0,1),(d0*n+a1,1)]))
            # M5 = (A00+A01)B11 -> C00+=-1, C01+=1
            terms.append(([(a0*m+c0,1),(a0*m+c1,1)], [(c1*p+d1,1)], [(d0*n+a0,-1),(d1*n+a0,1)]))
            # M6 = (A10-A00)(B00+B01) -> C11+=1
            terms.append(([(a1*m+c0,1),(a0*m+c0,-1)], [(c0*p+d0,1),(c0*p+d1,1)], [(d1*n+a1,1)]))
            # M7 = (A01-A11)(B10+B11) -> C00+=1
            terms.append(([(a0*m+c1,1),(a1*m+c1,-1)], [(c1*p+d0,1),(c1*p+d1,1)], [(d0*n+a0,1)]))
        
        def add_schoolbook(a_rows, c_cols, d_cols):
            for la in range(len(a_rows)):
                for lc in range(len(c_cols)):
                    for ld in range(len(d_cols)):
                        terms.append(([(a_rows[la]*m + c_cols[lc], 1)],
                                      [(c_cols[lc]*p + d_cols[ld], 1)],
                                      [(d_cols[ld]*n + a_rows[la], 1)]))
        
        for ni in range(len(n_parts)):
            for mi in range(len(m_parts)):
                for pi in range(len(p_parts)):
                    a_rows = list(range(n_offsets[ni], n_offsets[ni] + n_parts[ni]))
                    c_cols = list(range(m_offsets[mi], m_offsets[mi] + m_parts[mi]))
                    d_cols = list(range(p_offsets[pi], p_offsets[pi] + p_parts[pi]))
                    if use_strassen and n_parts[ni] == 2 and m_parts[mi] == 2 and p_parts[pi] == 2:
                        add_strassen(a_rows, c_cols, d_cols)
                    else:
                        add_schoolbook(a_rows, c_cols, d_cols)
        
        total_rank = len(terms)
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        for r, (u_entries, v_entries, w_entries) in enumerate(terms):
            for (row, coeff) in u_entries:
                U[row][r] = coeff
            for (row, coeff) in v_entries:
                V[row][r] = coeff
            for (row, coeff) in w_entries:
                W[row][r] = coeff
        
        return U, V, W, total_rank
    
    # Try multiple block partitions to find best rank
    best_U, best_V, best_W, best_rank = None, None, None, n*m*p  # 90 schoolbook
    
    # Candidate partitions (n=3, m=5, p=6)
    partitions = [
        ([2,1], [2,2,1], [2,2,2]),  # 84
        ([2,1], [2,1,2], [2,2,2]),
        ([2,1], [1,2,2], [2,2,2]),
        ([2,1], [2,2,1], [2,1,3]),
        ([2,1], [2,2,1], [1,2,3]),
        ([2,1], [2,2,1], [3,2,1]),
        ([2,1], [2,2,1], [2,3,1]),
        ([2,1], [2,2,1], [1,3,2]),
        ([2,1], [2,2,1], [3,1,2]),
        ([1,2], [2,2,1], [2,2,2]),
        ([2,1], [2,1,2], [2,1,3]),
        ([2,1], [2,1,2], [1,2,3]),
        ([2,1], [1,2,2], [2,1,3]),
        ([2,1], [1,2,2], [1,2,3]),
        ([2,1], [2,2,1], [2,2,1,1]),
        ([2,1], [2,2,1], [2,1,1,2]),
        ([2,1], [2,1,2], [2,2,2]),
        ([2,1], [1,1,3], [2,2,2]),
        ([2,1], [3,2], [2,2,2]),
        ([2,1], [2,3], [2,2,2]),
        ([1,2], [2,3], [2,2,2]),
        ([2,1], [2,2,1], [1,1,4]),
        ([2,1], [2,2,1], [4,1,1]),
    ]
    
    for np_, mp_, pp_ in partitions:
        if sum(np_) != n or sum(mp_) != m or sum(pp_) != p:
            continue
        if time.time() > deadline - 20:
            break
        try:
            U, V, W, rank = build_decomposition(np_, mp_, pp_, use_strassen=True)
            if _verify(U, V, W, n, m, p):
                if rank < best_rank:
                    best_U, best_V, best_W, best_rank = U, V, W, rank
        except Exception:
            continue
    
    # Also try without Strassen for comparison
    for np_, mp_, pp_ in partitions:
        if sum(np_) != n or sum(mp_) != m or sum(pp_) != p:
            continue
        if time.time() > deadline - 20:
            break
        try:
            U, V, W, rank = build_decomposition(np_, mp_, pp_, use_strassen=False)
            if _verify(U, V, W, n, m, p):
                if rank < best_rank:
                    best_U, best_V, best_W, best_rank = U, V, W, rank
        except Exception:
            continue
    
    if best_U is None:
        # Fallback to pure schoolbook
        best_rank = n*m*p
        best_U = [[0]*best_rank for _ in range(n*m)]
        best_V = [[0]*best_rank for _ in range(m*p)]
        best_W = [[0]*best_rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i*m+j][col] = 1
                    best_V[j*p+k][col] = 1
                    best_W[k*n+i][col] = 1
                    col += 1
    
    # Try to reduce rank further via structured search
    # Strategy: look for linear dependencies in the tensor representation
    # Use the fact that we can try to eliminate terms by finding
    # linear combinations that preserve the tensor
    
    def try_eliminate_terms(U, V, W, n, m, p, max_time=60):
        """Try to eliminate redundant terms."""
        rank = len(U[0])
        nm, mp, pn = n*m, m*p, p*n
        
        # Convert to numpy for speed
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Try to find if any term can be expressed as combination of others
        # For term r, check if removing it and adjusting others preserves tensor
        # This is complex, so try a simpler heuristic:
        # Check for duplicate or proportional terms
        
        eliminated = set()
        new_terms = []
        
        for r in range(rank):
            if r in eliminated:
                continue
            # Check if term r is a scalar multiple of an earlier kept term
            is_dup = False
            for r2 in new_terms:
                # Check if U[:,r] is proportional to U[:,r2]
                # and V[:,r] proportional to V[:,r2] and W[:,r] proportional to W[:,r2]
                u_r = U_np[:, r]
                u_r2 = U_np[:, r2]
                v_r = V_np[:, r]
                v_r2 = V_np[:, r2]
                w_r = W_np[:, r]
                w_r2 = W_np[:, r2]
                
                # Find scaling factors
                nonzero_u = np.nonzero(u_r)[0]
                if len(nonzero_u) == 0:
                    continue
                    
                # Check proportionality
                scale_u = None
                for idx in nonzero_u:
                    if u_r2[idx] != 0:
                        scale_u = u_r[idx] / u_r2[idx]
                        break
                    elif u_r[idx] != 0:
                        break
                
                if scale_u is None:
                    continue
                    
                # Verify all components
                if np.allclose(u_r, scale_u * u_r2, atol=1e-10):
                    if np.allclose(v_r, scale_u * v_r2, atol=1e-10):
                        if np.allclose(w_r, scale_u * w_r2, atol=1e-10):
                            is_dup = True
                            break
            
            if is_dup:
                eliminated.add(r)
            else:
                new_terms.append(r)
        
        if len(eliminated) > 0:
            new_rank = len(new_terms)
            new_U = [[U[i][r] for r in new_terms] for i in range(nm)]
            new_V = [[V[i][r] for r in new_terms] for i in range(mp)]
            new_W = [[W[i][r] for r in new_terms] for i in range(pn)]
            
            if _verify(new_U, new_V, new_W, n, m, p):
                return new_U, new_V, new_W, new_rank
        
        return U, V, W, rank
    
    if time.time() < deadline - 10:
        try:
            best_U, best_V, best_W, best_rank = try_eliminate_terms(
                best_U, best_V, best_W, n, m, p
            )
        except Exception:
            pass
    
    # Try alternating least squares for rank reduction
    if time.time() < deadline - 15:
        def solve_for_U(V_fixed, W_fixed, r, n, m, p):
            U_sol = np.zeros((n*m, r))
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V_fixed[j*p+k, :] * W_fixed[k*n+i, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U_sol[i*m+j, :] = x
                    except:
                        pass
            return U_sol
        
        def solve_for_V(U_fixed, W_fixed, r, n, m, p):
            V_sol = np.zeros((m*p, r))
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U_fixed[i*m+j, :] * W_fixed[k*n+i, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V_sol[j*p+k, :] = x
                    except:
                        pass
            return V_sol
        
        def solve_for_W(U_fixed, V_fixed, r, n, m, p):
            W_sol = np.zeros((p*n, r))
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U_fixed[i*m+j, :] * V_fixed[j*p+k, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W_sol[k*n+i, :] = x
                    except:
                        pass
            return W_sol
        
        def check_exact_float(U_c, V_c, W_c, n, m, p, tol=1e-8):
            r = len(U_c[0])
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for s_idx in range(r):
                            s += U_c[i*m+j][s_idx] * V_c[j*p+k][s_idx] * W_c[k*n+i][s_idx]
                        if abs(s - 1.0) > tol:
                            return False
            return True
        
        U_np = np.array(best_U, dtype=float)
        V_np = np.array(best_V, dtype=float)
        W_np = np.array(best_W, dtype=float)
        
        random.seed(42)
        target_ranks = list(range(best_rank - 1, max(20, best_rank - 10), -1))
        
        for target_rank in target_ranks:
            if time.time() > deadline - 15:
                break
            
            # Try a few random restarts
            for restart in range(3):
                if time.time() > deadline - 15:
                    break
                
                # Initialize from truncated original + random noise
                U_r = U_np[:, :min(target_rank, best_rank)].copy()
                V_r = V_np[:, :min(target_rank, best_rank)].copy()
                W_r = W_np[:, :min(target_rank, best_rank)].copy()
                
                if target_rank < best_rank:
                    # Pad with random
                    U_r = np.hstack([U_r, np.random.randn(n*m, target_rank - min(target_rank, best_rank)) * 0.1])
                    V_r = np.hstack([V_r, np.random.randn(m*p, target_rank - min(target_rank, best_rank)) * 0.1])
                    W_r = np.hstack([W_r, np.random.randn(p*n, target_rank - min(target_rank, best_rank)) * 0.1])
                
                # Alternating least squares
                for iteration in range(100):
                    if time.time() > deadline - 15:
                        break
                    U_r = solve_for_U(V_r, W_r, target_rank, n, m, p)
                    V_r = solve_for_V(U_r, W_r, target_rank, n, m, p)
                    W_r = solve_for_W(U_r, V_r, target_rank, n, m, p)
                
                # Check exactness
                if check_exact_float(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol=1e-7):
                    # Try to snap to rational values
                    U_snap = [[round(x, 6) for x in row] for row in U_r.tolist()]
                    V_snap = [[round(x, 6) for x in row] for row in V_r.tolist()]
                    W_snap = [[round(x, 6) for x in row] for row in W_r.tolist()]
                    
                    if _verify(U_snap, V_snap, W_snap, n, m, p):
                        best_U, best_V, best_W, best_rank = U_snap, V_snap, W_snap, target_rank
                        break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
