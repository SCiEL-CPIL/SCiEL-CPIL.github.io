"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = 110.0  # safety margin before 120s kill

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _verify(U, V, W, n=4, m=4, p=5):
    """Exact integer verification of the 4x4x5 MM tensor.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Check: for all i,j,k: sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] == delta(i,k)
    (C[k*n+i] = sum_j A[i,j]*B[j,k]; coefficient of A[i,j]*B[j,k] is 1 iff i==k)
    """
    U = np.asarray(U, dtype=object)
    V = np.asarray(V, dtype=object)
    W = np.asarray(W, dtype=object)
    r = U.shape[1]
    # Build for each (i,j,k) the sum over t
    # We'll iterate over t and accumulate into a 3D array T[i,j,k]
    T = np.zeros((n, m, p), dtype=object)
    for t in range(r):
        u = U[:, t]
        v = V[:, t]
        w = W[:, t]
        # outer product u (n*m) x v (m*p) -> (n*m, m*p); then contract with w
        # T[i,j,k] += u[i,j]*v[j,k]*w[k,i]
        # do it by loops (small)
        for i in range(n):
            for j in range(m):
                uij = u[i*m + j]
                if uij == 0:
                    continue
                for k in range(p):
                    vjk = v[j*p + k]
                    if vjk == 0:
                        continue
                    wki = w[k*n + i]
                    if wki == 0:
                        continue
                    T[i, j, k] += uij * vjk * wki
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1 if i == k else 0
                if T[i, j, k] != expected:
                    return False
    return True

def _build_rank65():
    # Guaranteed-correct: Strassen 4x4x4 (49) + schoolbook 5th col (16) = 65
    n, m, p = 4, 4, 5
    rank = 65
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0: U[i*m + j][col] += c
        for j, k, c in v_list:
            if c != 0: V[j*p + k][col] += c
        for k, i, c in w_list:
            if c != 0: W[k*n + i][col] += c
    col = 0
    def get_indices(s, t):
        u_list=[]; v_list=[]; w_list=[]
        if s==0: p_blocks=[1,0,0,1]
        elif s==1: p_blocks=[0,0,1,1]
        elif s==2: p_blocks=[1,0,0,0]
        elif s==3: p_blocks=[0,0,0,1]
        elif s==4: p_blocks=[1,1,0,0]
        elif s==5: p_blocks=[-1,0,1,0]
        else: p_blocks=[0,1,0,-1]
        if s==0: q_blocks=[1,0,0,1]
        elif s==1: q_blocks=[1,0,0,0]
        elif s==2: q_blocks=[0,1,0,-1]
        elif s==3: q_blocks=[-1,0,1,0]
        elif s==4: q_blocks=[0,0,0,1]
        elif s==5: q_blocks=[1,1,0,0]
        else: q_blocks=[0,0,1,1]
        if t==0:
            p_elems=[(0,1),(3,1)]; q_elems=[(0,1),(3,1)]; s_contrib=[1,0,0,1]
        elif t==1:
            p_elems=[(2,1),(3,1)]; q_elems=[(0,1)]; s_contrib=[0,0,1,-1]
        elif t==2:
            p_elems=[(0,1)]; q_elems=[(1,1),(3,-1)]; s_contrib=[0,1,0,1]
        elif t==3:
            p_elems=[(3,1)]; q_elems=[(2,1),(0,-1)]; s_contrib=[1,0,1,0]
        elif t==4:
            p_elems=[(0,1),(1,1)]; q_elems=[(3,1)]; s_contrib=[-1,1,0,0]
        elif t==5:
            p_elems=[(2,1),(0,-1)]; q_elems=[(0,1),(1,1)]; s_contrib=[0,0,0,1]
        else:
            p_elems=[(1,1),(3,-1)]; q_elems=[(2,1),(3,1)]; s_contrib=[1,0,0,0]
        for p_idx, p_coeff in p_elems:
            r_ = p_idx//2; c_ = p_idx%2
            for b_idx, coeff in enumerate(p_blocks):
                if coeff==0: continue
                if b_idx==0: gi,gj = r_,c_
                elif b_idx==1: gi,gj = r_,2+c_
                elif b_idx==2: gi,gj = 2+r_,c_
                else: gi,gj = 2+r_,2+c_
                u_list.append((gi,gj,coeff*p_coeff))
        for q_idx, q_coeff in q_elems:
            r_ = q_idx//2; c_ = q_idx%2
            for b_idx, coeff in enumerate(q_blocks):
                if coeff==0: continue
                if b_idx==0: gj,gk = r_,c_
                elif b_idx==1: gj,gk = r_,2+c_
                elif b_idx==2: gj,gk = 2+r_,c_
                else: gj,gk = 2+r_,2+c_
                v_list.append((gj,gk,coeff*q_coeff))
        if s==0: c_block_contrib=[1,0,0,1]
        elif s==1: c_block_contrib=[0,0,1,-1]
        elif s==2: c_block_contrib=[0,1,0,1]
        elif s==3: c_block_contrib=[1,0,1,0]
        elif s==4: c_block_contrib=[-1,1,0,0]
        elif s==5: c_block_contrib=[0,0,0,1]
        else: c_block_contrib=[1,0,0,0]
        for s_pos in range(4):
            s_coeff = s_contrib[s_pos]
            if s_coeff==0: continue
            s_r = s_pos//2; s_c = s_pos%2
            for c_b in range(4):
                c_coeff = c_block_contrib[c_b]
                if c_coeff==0: continue
                total = s_coeff*c_coeff
                if total==0: continue
                if c_b==0: gi,gk = s_r,s_c
                elif c_b==1: gi,gk = s_r,2+s_c
                elif c_b==2: gi,gk = 2+s_r,s_c
                else: gi,gk = 2+s_r,2+s_c
                w_list.append((gk,gi,total))
        return u_list, v_list, w_list
    for s in range(7):
        for t in range(7):
            u_list, v_list, w_list = get_indices(s, t)
            add_term(col, u_list, v_list, w_list); col += 1
    for i in range(4):
        for j in range(4):
            add_term(col, [(i,j,1)], [(j,4,1)], [(4,i,1)]); col += 1
    return {"U":U, "V":V, "W":W}

def _als_cp(T, r, n_iter=200, seed=0):
    """ALS CP decomposition of a 3D array T (n,m,p) into r components.
    Returns U(n*m,r), V(m*p,r), W(p*n,r) as float arrays (un-normalized)."""
    rng = np.random.default_rng(seed)
    n, m, p = T.shape
    U = rng.standard_normal((n*m, r))
    V = rng.standard_normal((m*p, r))
    W = rng.standard_normal((p*n, r))
    # Normalize columns to avoid scale drift
    for _ in range(n_iter):
        # Update U: for each component t, solve for U[:,t]
        # T[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t]
        # Fix V,W: for each t, U[:,t] minimizes sum_{i,j,k} (T[i,j,k] - U[i,j,t]*(V[j,k,t]*W[k,i,t]))^2 + (rest)
        # Equivalent: U[:,t] = argmin ||T - sum_t U_t * (V_t outer W_t)||
        # Build for each t the "residual-weighted" system.
        # Simpler: treat as linear in U[:,t]:
        # Let M_t[i,j] = sum_k V[j,k,t]*W[k,i,t]  (this is (V_t reshaped m*p) @ (W_t reshaped p*n) -> (m,n) transposed)
        # V_t is (m*p) indexed (j,k). W_t is (p*n) indexed (k,i).
        # M_t[i,j] = sum_k V[j,k,t]*W[k,i,t] = (V_t as (m,p)) @ (W_t as (p,n))
        Vm = V.reshape(m, p)
        Wp = W.reshape(p, n)
        for t in range(r):
            Mt = Vm[:, t] @ Wp[t, :]   # (m,n): Mt[j,i] = sum_k V[j,k,t]*W[k,i,t]
            # We need T[i,j,k] = sum_t U[i,j,t] * (sum_k V[j,k,t]W[k,i,t])? No.
            # Actually T[i,j,k] = sum_t U[i,j,t]*V[j,k,t]*W[k,i,t].
            # For fixed t, the contribution to T from component t is U_t (n*m) * (V_t (m*p) with W_t (p*n)) -> a 3D array A_t[i,j,k]=U[i,j,t]*V[j,k,t]*W[k,i,t].
            # Linear in U[:,t]: minimize ||T - sum_t A_t||_F^2.
            # d/dU[i,j,t]: -2 sum_k (T[i,j,k]-...)*V[j,k,t]*W[k,i,t] = 0
            # => U[i,j,t] = sum_k (T[i,j,k] - sum_{t'!=t} U[i,j,t']V[j,k,t']W[k,i,t']) * V[j,k,t]*W[k,i,t] / sum_k (V[j,k,t]*W[k,i,t])^2
            # Compute residual R = T - sum_{t'} A_{t'} (with current U).
            pass
        # Do a proper residual-based update
        # Compute current tensor estimate
        Est = np.zeros((n,m,p))
        for t in range(r):
            Vm = V.reshape(m,p)[:, t]   # (m,)
            Wp = W.reshape(p,n)[:, t]   # (n,)  Wp[i] = W[k*n+i, t] with k summed? No.
            # W is (p*n) indexed (k,i). W[:,t] is (p*n). Reshape to (p,n): Wpn[k,i]=W[k*n+i,t]
            pass
        # This is getting complicated; use a direct least-squares per component via matricization.
        # Use the standard CP-ALS with matricizations.
        # T_(0) is (n, m*p): T_(0)[i, j*p+k] = T[i,j,k]
        # T_(1) is (m, n*p): T_(1)[j, i*p+k] = T[i,j,k]
        # T_(2) is (p, n*m): T_(2)[k, i*m+j] = T[i,j,k]
        T0 = np.zeros((n, m*p)); T1 = np.zeros((m, n*p)); T2 = np.zeros((p, n*m))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T0[i, j*p+k] = T[i,j,k]
                    T1[j, i*p+k] = T[i,j,k]
                    T2[k, i*m+j] = T[i,j,k]
        for it in range(n_iter):
            # Update U (mode-0): U = T0 @ (W_pn^T ⊗ V_mp^T) ... standard: U = T0 @ ( (W ⊙ V) )
            # (W ⊙ V): for component t, column = W[:,t] (p*n) ⊗ V[:,t] (m*p) -> (p*n * m*p)
            # T0 (n, m*p). We need T0 (n, m*p) @ something (m*p, r).
            # Standard CP-ALS: U = T0 @ (V_pn ⊙ W_mp) where the Khatri-Rao is over the *other two* modes.
            # Let me define: for mode-0 update, the Khatri-Rao of (W reshaped to (p,n) per comp -> but W is (p*n)) and V (m*p).
            # The mode-0 matricization T0[i, (j,k)] = sum_t U[i,t] V[(j,k),t] W[(k,i),t].
            # Note W depends on i! This is the non-standard part: the 3rd mode index of W is (k,i), which couples k and i.
            # So T0[i, j*p+k] = sum_t U[i*m? no U[i,j? ...
            pass
        return U, V, W
    return U, V, W

def _try_find_better():
    """Search for a 4x4x4 decomposition of rank < 49 via CP-ALS + rounding + exact verify.
    Returns (rank44, U44, V44, W44) or None."""
    n=m=p=4
    T = np.zeros((n,m,p))
    for i in range(n):
        for k in range(p):
            if i==k:
                for j in range(m):
                    T[i,j,k]=1
    best = None
    # Try ranks from 19 upward; for each, a few restarts; time-bounded.
    for r in range(19, 49):
        if _time_left() < 12:
            break
        max_restarts = 8
        for seed in range(max_restarts):
            if _time_left() < 6:
                break
            U, V, W = _cp_als(T, r, seed)
            # Round to integers and halves; try a few rounding grids
            found = _round_and_verify_44(U, V, W, r)
            if found is not None:
                if best is None or r < best[0]:
                    best = (r,) + found
                break  # found a valid one at this rank; move to next (smaller? no, r increases) -> actually we want smallest, so break outer too
        if best is not None and best[0] == r:
            break
    return best

def _cp_als(T, r, seed, n_iter=300):
    # Proper CP-ALS handling the W(k,i) coupling.
    n,m,p = T.shape
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n*m, r))
    V = rng.standard_normal((m*p, r))
    W = rng.standard_normal((p*n, r))
    # Normalize
    for t in range(r):
        nU=np.linalg.norm(U[:,t]); nV=np.linalg.norm(V[:,t]); nW=np.linalg.norm(W[:,t])
        if nU>1e-12: U[:,t]/=nU
        if nV>1e-12: V[:,t]/=nV
        if nW>1e-12: W[:,t]/=nW
    for it in range(n_iter):
        # Update U: for each i,j, U[i*m+j, t] = sum_k (T[i,j,k] - sum_{t'}U[i*m+j,t']*V[j*p+k,t']*W[k*n+i,t']) * V[j*p+k,t]*W[k*n+i,t] / denom
        # Precompute for speed: build A_t[i,j,k] = V[j*p+k,t]*W[k*n+i,t]
        # Residual R[i,j,k] = T[i,j,k] - sum_t U[i*m+j,t]*A_t[i,j,k]
        # Then U[:,t] = sum_k R[i,j,k]*A_t[i,j,k] / sum_k A_t[i,j,k]^2
        # This is O(r * n*m*p) per iteration = 49*64=3136, fine.
        R = T.copy()
        for t in range(r):
            R -= U[:,t].reshape(n,m)[:,:,None] * (V[:,t].reshape(m,p)[:, :, None] * W[:,t].reshape(p,n)[None, :, :])
        # Wait: V[:,t].reshape(m,p)[j,k]=V[j*p+k,t]; W[:,t].reshape(p,n)[k,i]=W[k*n+i,t].
        # A_t[i,j,k] = V[j,k,t]*W[k,i,t]
        for t in range(r):
            Vm = V[:,t].reshape(m,p)  # (m,p)
            Wp = W[:,t].reshape(p,n)  # (p,n)
            At = Vm[:, :, None] * Wp[None, :, :]  # (m,p,n): At[j,k,i]=V[j,k,t]*W[k,i,t]
            # U[i,j,t] = sum_k R[i,j,k]*At[j,k,i] / sum_k At[j,k,i]^2
            num = np.einsum('ijk,jki->ij', R, At)  # (n,m)
            den = np.einsum('jki,jki->jk', At, At)  # (m,p) -> but we need per (i,j): den depends on j,k -> sum_k At[j,k,i]^2 depends on i too!
            # den[i,j] = sum_k At[j,k,i]^2
            den = np.einsum('jki,jki->ji', At, At)  # (n,m): den[i,j]=sum_k At[j,k,i]^2
            U[:,t] = (num / np.maximum(den, 1e-12)).reshape(-1)
        # Update V similarly
        R = T.copy()
        for t in range(r):
            R -= U[:,t].reshape(n,m)[:,:,None] * (V[:,t].reshape(m,p)[:, :, None] * W[:,t].reshape(p,n)[None, :, :])
        for t in range(r):
            Um = U[:,t].reshape(n,m)  # (n,m)
            Wp = W[:,t].reshape(p,n)  # (p,n)
            At = Um[:, None, :] * Wp[None, :, :]  # (n,m,p): At[i,j,k]=U[i,j,t]*W[k,i,t]
            # V[j,k,t] = sum_i R[i,j,k]*At[i,j,k] / sum_i At[i,j,k]^2
            num = np.einsum('ijk,ijk->jk', R, At)  # (m,p)
            den = np.einsum('ijk,ijk->jk', At, At)  # (m,p)
            V[:,t] = (num / np.maximum(den, 1e-12)).reshape(-1)
        # Update W
        R = T.copy()
        for t in range(r):
            R -= U[:,t].reshape(n,m)[:,:,None] * (V[:,t].reshape(m,p)[:, :, None] * W[:,t].reshape(p,n)[None, :, :])
        for t in range(r):
            Um = U[:,t].reshape(n,m)  # (n,m)
            Vm = V[:,t].reshape(m,p)  # (m,p)
            At = Um[:, :, None] * Vm[:, :, None]  # (n,m,p): At[i,j,k]=U[i,j,t]*V[j,k,t]
            # W[k,i,t] = sum_j R[i,j,k]*At[i,j,k] / sum_j At[i,j,k]^2
            num = np.einsum('ijk,ijk->ki', R, At)  # (p,n)
            den = np.einsum('ijk,ijk->ki', At, At)  # (p,n)
            W[:,t] = (num / np.maximum(den, 1e-12)).reshape(-1)
        # Renormalize to control scale
        for t in range(r):
            nU=np.linalg.norm(U[:,t]); nV=np.linalg.norm(V[:,t]); nW=np.linalg.norm(W[:,t])
            if nU>1e-12: U[:,t]/=nU
            if nV>1e-12: V[:,t]/=nV
            if nW>1e-12: W[:,t]/=nW
    return U, V, W

def _round_and_verify_44(U, V, W, r):
    # Try rounding each of U,V,W to nearest integer and to nearest half-integer,
    # and combinations; verify the 4x4x4 tensor exactly.
    # To keep it tractable, round all three to nearest integer first; if fails, try halves.
    def to_int_round(M):
        return np.rint(M).astype(object)
    def to_half_round(M):
        return np.rint(M*2).astype(object)/2
    candidates = [
        (to_int_round, to_int_round, to_int_round),
        (to_half_round, to_int_round, to_int_round),
        (to_int_round, to_half_round, to_int_round),
        (to_int_round, to_int_round, to_half_round),
        (to_half_round, to_half_round, to_int_round),
        (to_half_round, to_int_round, to_half_round),
        (to_int_round, to_half_round, to_half_round),
        (to_half_round, to_half_round, to_half_round),
    ]
    n=m=p=4
    for ru, rv, rw in candidates:
        Ui = ru(U); Vi = rv(V); Wi = rw(W)
        # Build 4x4x4 tensor
        T = np.zeros((n,m,p), dtype=object)
        for t in range(r):
            u = Ui[:,t]; v = Vi[:,t]; w = Wi[:,t]
            for i in range(n):
                for j in range(m):
                    uij = u[i*m+j]
                    if uij==0: continue
                    for k in range(p):
                        vjk = v[j*p+k]
                        if vjk==0: continue
                        wki = w[k*n+i]
                        if wki==0: continue
                        T[i,j,k] += uij*vjk*wki
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    exp = 1 if i==k else 0
                    if T[i,j,k] != exp:
                        ok=False; break
                if not ok: break
            if not ok: break
        if ok:
            return (Ui.tolist(), Vi.tolist(), Wi.tolist())
    return None

def construct():
    # Guaranteed fallback
    fallback = _build_rank65()
    # Attempt to find a better 4x4x4 decomposition
    best = _try_find_better()
    if best is not None:
        r44, U44, V44, W44 = best
        # Build full 4x4x5: first 4 cols use 4x4x4 decomp, 5th col schoolbook (16)
        n, m, p = 4, 4, 5
        rank = r44 + 16
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for t in range(r44):
            for i in range(n*m):
                U[i][t] = U44[i][t]
            for j in range(m*p):
                V[j][t] = V44[j][t]
            for k in range(p*n):
                W[k][t] = W44[k][t]
        col = r44
        for i in range(4):
            for j in range(4):
                U[i*m+j][col] += 1
                V[j*p+4][col] += 1
                W[4*n+i][col] += 1
                col += 1
        result = {"U":U, "V":V, "W":W}
        if _verify(U, V, W):
            return result
    return fallback
# EVOLVE-BLOCK-END
