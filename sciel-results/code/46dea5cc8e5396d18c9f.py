"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy import linalg

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for 150s limit

    # Initialize with the known rank 36 decomposition
    U = [[0.0] * 36 for _ in range(n * m)]
    V = [[0.0] * 36 for _ in range(m * p)]
    W = [[0.0] * 36 for _ in range(p * n)]

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    col = 0

    def strassen_block(y, z, w):
        nonlocal col
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z
        w00, w01, w10, w11 = w
        # M1
        U[u_idx(*y00)][col] += 1; U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w11)][col] += 1
        col += 1
        # M2
        U[u_idx(*y10)][col] += 1; U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        W[w_idx(*w10)][col] += 1; W[w_idx(*w11)][col] += -1
        col += 1
        # M3
        U[u_idx(*y00)][col] += 1
        V[v_idx(*z01)][col] += 1; V[v_idx(*z11)][col] += -1
        W[w_idx(*w01)][col] += 1; W[w_idx(*w11)][col] += 1
        col += 1
        # M4
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z10)][col] += 1; V[v_idx(*z00)][col] += -1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w10)][col] += 1
        col += 1
        # M5
        U[u_idx(*y00)][col] += 1; U[u_idx(*y01)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += -1; W[w_idx(*w01)][col] += 1
        col += 1
        # M6
        U[u_idx(*y10)][col] += 1; U[u_idx(*y00)][col] += -1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z01)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1
        # M7
        U[u_idx(*y01)][col] += 1; U[u_idx(*y11)][col] += -1
        V[v_idx(*z10)][col] += 1; V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1
        col += 1

    AL = [(0,0),(0,1),(1,0),(1,1)]
    AR = [(0,2),(0,3),(1,2),(1,3)]
    B11 = [(0,0),(0,1),(1,0),(1,1)]
    B12 = [(0,2),(0,3),(1,2),(1,3)]
    B21 = [(2,0),(2,1),(3,0),(3,1)]
    B22 = [(2,2),(2,3),(3,2),(3,3)]
    C11 = [(0,0),(1,0),(0,1),(1,1)]
    C12 = [(2,0),(3,0),(2,1),(3,1)]

    strassen_block(AL, B11, C11)
    strassen_block(AR, B21, C11)
    strassen_block(AL, B12, C12)
    strassen_block(AR, B22, C12)

    for (ai, aj) in [(0,0),(0,1),(1,0),(1,1)]:
        U[u_idx(ai, aj)][col] += 1
        V[v_idx(aj, 4)][col] += 1
        W[w_idx(4, ai)][col] += 1
        col += 1
    for (ai, aj) in [(0,2),(0,3),(1,2),(1,3)]:
        U[u_idx(ai, aj)][col] += 1
        V[v_idx(aj, 4)][col] += 1
        W[w_idx(4, ai)][col] += 1
        col += 1

    assert col == 36, col

    def verify(Ul, Vl, Wl, R):
        Ua = np.array(Ul, dtype=float)
        Va = np.array(Vl, dtype=float)
        Wa = np.array(Wl, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-7:
                        return False
        return True

    def verify_fast(Ua, Va, Wa):
        """Vectorized verification"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-7:
                        return False
        return True

    base_U, base_V, base_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best = (36, base_U, base_V, base_W)

    if not verify(base_U, base_V, base_W, 36):
        # Fallback: return schoolbook rank 40
        U_sch = [[0.0] * 40 for _ in range(n * m)]
        V_sch = [[0.0] * 40 for _ in range(m * p)]
        W_sch = [[0.0] * 40 for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sch[i*m+j][c] = 1.0
                    V_sch[j*p+k][c] = 1.0
                    W_sch[k*n+i][c] = 1.0
                    c += 1
        return {"U": U_sch, "V": V_sch, "W": W_sch}

    # Convert to numpy for efficient operations
    U_np = np.array(base_U, dtype=float)
    V_np = np.array(base_V, dtype=float)
    W_np = np.array(base_W, dtype=float)

    # Try to reduce rank by finding linear dependencies
    # A column c can be eliminated if its rank-1 tensor is in the span of others
    # i.e., if there exist alpha, beta, gamma such that:
    # U[:,c] = alpha * U[:,c1] for some c1, OR more generally,
    # the tensor T_c = U[:,c] ⊗ V[:,c] ⊗ W[:,c] is in the span of {T_r : r != c}
    
    # Better approach: try to express column c as a linear combination
    # For each pair (c, c1), check if we can eliminate c by adjusting c1
    
    def try_eliminate_column_linear(c, targets):
        """Try to eliminate column c by expressing its tensor as a linear combo of target columns.
        
        We need: U[:,c]⊗V[:,c]⊗W[:,c] = sum_{t in targets} alpha_t * U[:,t]⊗V[:,t]⊗W[:,t]
        
        This is a nonlinear constraint. A practical approach:
        For each target t, try alpha = 1 (or -1) and see if the residual is zero.
        For a single target, the condition is:
          U[:,c]⊗V[:,c]⊗W[:,c] = alpha * U[:,t]⊗V[:,t]⊗W[:,t]
        which means U[:,c] = a*U[:,t], V[:,c] = b*V[:,t], W[:,c] = c*W[:,t] with a*b*c = alpha.
        
        For multiple targets, we solve a system.
        """
        if not targets:
            return None
        
        # Try single-target elimination first (fastest)
        for t in targets:
            if t == c:
                continue
            # Check if U[:,c] is proportional to U[:,t]
            Ut = U_np[:, t]
            Uc = U_np[:, c]
            Vt = V_np[:, t]
            Vc = V_np[:, c]
            Wt = W_np[:, t]
            Wc = W_np[:, c]
            
            # Find if Uc = a * Ut
            nonzero_u = np.nonzero(Ut)[0]
            if len(nonzero_u) == 0:
                if np.allclose(Uc, 0):
                    # U part is zero, check V and W
                    nonzero_v = np.nonzero(Vt)[0]
                    if len(nonzero_v) == 0:
                        continue
                    a = 0.0
                    b = Vc[nonzero_v[0]] / Vt[nonzero_v[0]]
                    if np.allclose(Vc, b * Vt):
                        c_coeff = Wc[0] / Wt[0] if Wt[0] != 0 else (Wc[0] if Wt[0] == 0 and Wc[0] != 0 else None)
                        if c_coeff is not None and np.allclose(Wc, c_coeff * Wt):
                            # Scale: a*b*c_coeff should be 1 for addition
                            pass
                continue
            
            a = Uc[nonzero_u[0]] / Ut[nonzero_u[0]]
            if not np.allclose(Uc, a * Ut, atol=1e-10):
                continue
            
            nonzero_v = np.nonzero(Vt)[0]
            if len(nonzero_v) == 0:
                if np.allclose(Vc, 0):
                    nonzero_w = np.nonzero(Wt)[0]
                    if len(nonzero_w) > 0:
                        b = 0.0
                        c_coeff = Wc[nonzero_w[0]] / Wt[nonzero_w[0]]
                        if np.allclose(Wc, c_coeff * Wt):
                            # Try adding
                            U_new = U_np.copy()
                            V_new = V_np.copy()
                            W_new = W_np.copy()
                            U_new[:, t] += U_new[:, c]
                            V_new[:, t] += V_new[:, c]
                            W_new[:, t] += W_new[:, c]
                            U_new = np.delete(U_new, c, axis=1)
                            V_new = np.delete(V_new, c, axis=1)
                            W_new = np.delete(W_new, c, axis=1)
                            if verify_fast(U_new, V_new, W_new):
                                return (U_new, V_new, W_new)
                continue
            
            b = Vc[nonzero_v[0]] / Vt[nonzero_v[0]]
            if not np.allclose(Vc, b * Vt, atol=1e-10):
                continue
            
            nonzero_w = np.nonzero(Wt)[0]
            if len(nonzero_w) == 0:
                if np.allclose(Wc, 0):
                    U_new = U_np.copy()
                    V_new = V_np.copy()
                    W_new = W_np.copy()
                    U_new[:, t] += U_new[:, c]
                    V_new[:, t] += V_new[:, c]
                    W_new[:, t] += W_new[:, c]
                    U_new = np.delete(U_new, c, axis=1)
                    V_new = np.delete(V_new, c, axis=1)
                    W_new = np.delete(W_new, c, axis=1)
                    if verify_fast(U_new, V_new, W_new):
                        return (U_new, V_new, W_new)
                continue
            
            c_coeff = Wc[nonzero_w[0]] / Wt[nonzero_w[0]]
            if not np.allclose(Wc, c_coeff * Wt, atol=1e-10):
                continue
            
            # All three are proportional. Try adding column c to column t
            U_new = U_np.copy()
            V_new = V_np.copy()
            W_new = W_np.copy()
            U_new[:, t] += U_new[:, c]
            V_new[:, t] += V_new[:, c]
            W_new[:, t] += W_new[:, c]
            U_new = np.delete(U_new, c, axis=1)
            V_new = np.delete(V_new, c, axis=1)
            W_new = np.delete(W_new, c, axis=1)
            if verify_fast(U_new, V_new, W_new):
                return (U_new, V_new, W_new)
            
            # Try subtracting
            U_new = U_np.copy()
            V_new = V_np.copy()
            W_new = W_np.copy()
            U_new[:, t] -= U_new[:, c]
            V_new[:, t] -= V_new[:, c]
            W_new[:, t] -= W_new[:, c]
            U_new = np.delete(U_new, c, axis=1)
            V_new = np.delete(V_new, c, axis=1)
            W_new = np.delete(W_new, c, axis=1)
            if verify_fast(U_new, V_new, W_new):
                return (U_new, V_new, W_new)
        
        # Try two-target elimination: express col c as combo of cols t1, t2
        # This requires solving a system. For small problems, try all pairs.
        if len(targets) >= 2:
            target_list = [t for t in targets if t != c]
            for i in range(len(target_list)):
                if time.time() > deadline - 15:
                    return None
                t1 = target_list[i]
                for j in range(i+1, len(target_list)):
                    if time.time() > deadline - 15:
                        return None
                    t2 = target_list[j]
                    
                    # We want: U[:,c]⊗V[:,c]⊗W[:,c] = alpha1 * U[:,t1]⊗V[:,t1]⊗W[:,t1] + alpha2 * U[:,t2]⊗V[:,t2]⊗W[:,t2]
                    # This is a system of n*m*p equations in 2 unknowns (alpha1, alpha2)
                    # But it's nonlinear... Actually, let's think differently.
                    # After adding alpha1*col_t1 + alpha2*col_t2 to col_c and removing col_c,
                    # we need the resulting decomposition to still work.
                    # 
                    # Alternative: try alpha1=1, alpha2=0 (single target), alpha1=0, alpha2=1, 
                    # alpha1=1, alpha2=1, alpha1=1, alpha2=-1, etc.
                    for a1, a2 in [(1,1), (1,-1), (-1,1), (-1,-1), (1,0), (0,1), (-1,0), (0,-1)]:
                        if a1 == 0 and a2 == 0:
                            continue
                        U_new = U_np.copy()
                        V_new = V_np.copy()
                        W_new = W_np.copy()
                        
                        if a1 != 0:
                            U_new[:, t1] += a1 * U_new[:, c]
                            V_new[:, t1] += a1 * V_new[:, c]
                            W_new[:, t1] += a1 * W_new[:, c]
                        if a2 != 0:
                            U_new[:, t2] += a2 * U_new[:, c]
                            V_new[:, t2] += a2 * V_new[:, c]
                            W_new[:, t2] += a2 * W_new[:, c]
                        
                        U_new = np.delete(U_new, c, axis=1)
                        V_new = np.delete(V_new, c, axis=1)
                        W_new = np.delete(W_new, c, axis=1)
                        
                        if verify_fast(U_new, V_new, W_new):
                            return (U_new, V_new, W_new)
        
        return None

    # Systematic search: try all columns, greedily eliminate
    found = False
    current_rank = 36
    current_U, current_V, current_W = U_np.copy(), V_np.copy(), W_np.copy()
    
    # Try eliminating one column at a time, greedily
    max_rounds = 20
    for round_num in range(max_rounds):
        if time.time() > deadline - 15:
            break
        
        eliminated = False
        for c in range(current_rank):
            if time.time() > deadline - 15:
                break
            
            # Try eliminating column c using all other columns as targets
            targets = [t for t in range(current_rank) if t != c]
            result = try_eliminate_column_linear(c, targets)
            if result is not None:
                current_U, current_V, current_W = result
                current_rank -= 1
                eliminated = True
                found = True
                break
        
        if not eliminated:
            break
    
    if found:
        best = (current_rank, 
                current_U.tolist(), 
                current_V.tolist(), 
                current_W.tolist())
    else:
        best = (36, base_U, base_V, base_W)

    rank, bU, bV, bW = best
    return {
        "U": bU,
        "V": bV,
        "W": bW
    }
# EVOLVE-BLOCK-END
