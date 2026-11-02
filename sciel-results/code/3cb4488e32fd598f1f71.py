"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    t_start = time.time()
    deadline = t_start + 125  # safety margin before 150s kill

    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T = build_tensor()

    def verify(U, V, W, tol=1e-8):
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def to_lists(U, V, W, r):
        U_list = [[float(U[i, t]) for t in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, t]) for t in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, t]) for t in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    def strassen_2x2_decomp():
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        U[0,0]=1.0; U[3,0]=1.0; V[0,0]=1.0; V[3,0]=1.0; W[0,0]=1.0; W[3,0]=1.0
        U[2,1]=1.0; U[3,1]=1.0; V[0,1]=1.0; W[1,1]=1.0; W[3,1]=-1.0
        U[0,2]=1.0; V[1,2]=1.0; V[3,2]=-1.0; W[2,2]=1.0; W[3,2]=1.0
        U[3,3]=1.0; V[2,3]=1.0; V[0,3]=-1.0; W[0,3]=1.0; W[1,3]=1.0
        U[0,4]=1.0; U[1,4]=1.0; V[3,4]=1.0; W[0,4]=-1.0; W[2,4]=1.0
        U[2,5]=1.0; U[0,5]=-1.0; V[0,5]=1.0; V[1,5]=1.0; W[3,5]=1.0
        U[1,6]=1.0; U[3,6]=-1.0; V[2,6]=1.0; V[3,6]=1.0; W[0,6]=1.0
        return U, V, W

    def build_strassen28():
        U_s, V_s, W_s = strassen_2x2_decomp()
        full_rank = 28
        U_full = np.zeros((n * m, full_rank))
        V_full = np.zeros((m * p, full_rank))
        W_full = np.zeros((p * n, full_rank))
        blocks = [
            (lambda li,lj: li*4+lj,          lambda lj,lk: lj*4+lk,          lambda lk,li: lk*2+li),
            (lambda li,lj: li*4+(lj+2),      lambda lj,lk: (lj+2)*4+lk,      lambda lk,li: lk*2+li),
            (lambda li,lj: li*4+lj,          lambda lj,lk: lj*4+(lk+2),      lambda lk,li: (lk+2)*2+li),
            (lambda li,lj: li*4+(lj+2),      lambda lj,lk: (lj+2)*4+(lk+2),  lambda lk,li: (lk+2)*2+li),
        ]
        for bi,(ga,gb,gc) in enumerate(blocks):
            for t in range(7):
                col = bi*7+t
                for li in range(2):
                    for lj in range(2):
                        U_full[ga(li,lj), col] = U_s[li*2+lj, t]
                for lj in range(2):
                    for lk in range(2):
                        V_full[gb(lj,lk), col] = V_s[lj*2+lk, t]
                for lk in range(2):
                    for li in range(2):
                        W_full[gc(lk,li), col] = W_s[lk*2+li, t]
        return U_full, V_full, W_full

    U28, V28, W28 = build_strassen28()

    # Verify rank-28 works
    if not verify(U28, V28, W28, tol=1e-8):
        # fallback to schoolbook
        U = np.zeros((n*m, 32)); V = np.zeros((m*p, 32)); W = np.zeros((p*n, 32))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col]=1.0; V[j*p+k, col]=1.0; W[k*n+i, col]=1.0; col+=1
        return to_lists(U, V, W, 32)

    # === Tensor splitting: decompose into row-block subproblems ===
    # The 2x4x4 tensor can be split by rows of A into two 1x4x4 tensors.
    # For 1x4x4 (n=1, m=4, p=4), the tensor is T1[a,b,c] where a=j, b=j*p+k, c=k.
    # This is the standard 1x4x4 matrix multiplication tensor.
    # 
    # Strategy: Use a known efficient decomposition for each half.
    # For 1x4x4, we can use a 4x4 decomposition. The rank of 1x4x4 is at most 16 (schoolbook).
    # But we can do better using Strassen-like methods.
    # 
    # Alternative: Use the fact that the 2x4x4 tensor has a specific structure.
    # Let's try a direct optimization approach for rank 27.
    
    def try_rank_r_gradient(r, max_iters=20000, seed=42):
        """Try to find rank-r via gradient descent from perturbed Strassen init."""
        rng = np.random.RandomState(seed)
        
        best_U, best_V, best_W = None, None, None
        best_err = np.inf
        
        # Build initial candidates: drop (28-r) columns from Strassen-28
        drop_count = 28 - r
        if drop_count <= 0:
            return None
        
        # Generate all combinations of columns to drop
        from itertools import combinations
        drop_combos = list(combinations(range(28), drop_count))
        
        # Also try random perturbations of the full 28-rank then project
        for combo in drop_combos:
            if time.time() > deadline:
                break
            
            keep = [i for i in range(28) if i not in set(combo)]
            U0 = U28[:, keep].copy()
            V0 = V28[:, keep].copy()
            W0 = W28[:, keep].copy()
            
            # Add small perturbation to break symmetry
            U0 += rng.randn(n*m, r) * 0.05
            V0 += rng.randn(m*p, r) * 0.05
            W0 += rng.randn(p*n, r) * 0.05
            
            # Gradient descent with adaptive learning rate
            lr = 0.1
            prev_err = np.inf
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                recon = np.einsum('at,bt,ct->abc', U0, V0, W0)
                F = recon - T
                err = float(np.sum(F*F))
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
                
                if err < 1e-14:
                    break
                
                gU = np.einsum('abc,bt,ct->at', F, V0, W0)
                gV = np.einsum('abc,at,ct->bt', F, U0, W0)
                gW = np.einsum('abc,at,bt->ct', F, U0, V0)
                
                # Adaptive learning rate: decrease if error not decreasing
                if err > prev_err * 1.1:
                    lr *= 0.5
                elif err < prev_err * 0.9:
                    lr = min(lr * 1.1, 0.5)
                prev_err = err
                
                U0 -= lr * gU
                V0 -= lr * gV
                W0 -= lr * gW
        
        if best_err < 1e-8:
            # Try to snap to half-integers
            U_snap = np.round(best_U * 2) / 2
            V_snap = np.round(best_V * 2) / 2
            W_snap = np.round(best_W * 2) / 2
            if verify(U_snap, V_snap, W_snap, tol=1e-8):
                return (U_snap, V_snap, W_snap)
            # Try integer snap
            U_snap2 = np.round(best_U)
            V_snap2 = np.round(best_V)
            W_snap2 = np.round(best_W)
            if verify(U_snap2, V_snap2, W_snap2, tol=1e-8):
                return (U_snap2, V_snap2, W_snap2)
            # Return the continuous solution if close enough
            if best_err < 1e-10:
                return (best_U, best_V, best_W)
        
        return None

    def try_rank27_gradient():
        """Try to find rank-27 via gradient descent."""
        result = try_rank_r_gradient(27, max_iters=15000, seed=42)
        if result is not None:
            return result
        # Try with different seed
        result = try_rank_r_gradient(27, max_iters=5000, seed=123)
        return result

    result = try_rank27_gradient()
    if result is not None:
        Uf, Vf, Wf = result
        return to_lists(Uf, Vf, Wf, 27)

    # Fallback: verified rank-28 Strassen
    return to_lists(U28, V28, W28, 28)
# EVOLVE-BLOCK-END
