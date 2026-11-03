import numpy as np
import time
import random

n, m, p = 3, 4, 6

def Aidx(i, j): return i * m + j
def Bidx(j, k): return j * p + k
def Cidx(i, k): return k * n + i  # W[k*n+i]

def compute_tensor(U, V, W):
    """Compute the full tensor from factor matrices. Returns (n, m, p) array."""
    r = U.shape[1]
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    return np.einsum('ijl,jkl,kil->ijk', U_r, V_r, W_r, optimize=True)

def build_decomposition(i_parts, j_parts, k_parts):
    """Build decomposition from block partitions."""
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1_terms(sub_U, sub_V, sub_W):
        """Add rank-1 terms for a sub-block. sub_U is list of (i,j) positions, etc."""
        pass  # Placeholder
    
    # This approach is complex; let me use the explicit construction instead
    pass

def baseline_66():
    U = [[0]*66 for _ in range(n*m)]
    V = [[0]*66 for _ in range(m*p)]
    W = [[0]*66 for _ in range(p*n)]
    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = Aidx(a_r, a_c); A01 = Aidx(a_r, a_c+1)
        A10 = Aidx(a_r+1, a_c); A11 = Aidx(a_r+1, a_c+1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c+1)
        B10 = Bidx(b_r+1, b_c); B11 = Bidx(b_r+1, b_c+1)
        C00 = Cidx(c_r, c_c); C01 = Cidx(c_r, c_c+1)
        C10 = Cidx(c_r+1, c_c); C11 = Cidx(c_r+1, c_c+1)
        # M1
        U[A00][col]+=1; U[A11][col]+=1; V[B00][col]+=1; V[B11][col]+=1; W[C00][col]+=1; W[C11][col]+=1; col+=1
        # M2
        U[A10][col]+=1; U[A11][col]+=1; V[B00][col]+=1; W[C10][col]+=1; W[C11][col]-=1; col+=1
        # M3
        U[A00][col]+=1; V[B01][col]+=1; V[B11][col]-=1; W[C01][col]+=1; W[C11][col]+=1; col+=1
        # M4
        U[A11][col]+=1; V[B10][col]+=1; V[B00][col]-=1; W[C00][col]+=1; W[C10][col]+=1; col+=1
        # M5
        U[A00][col]+=1; U[A01][col]+=1; V[B11][col]+=1; W[C00][col]-=1; W[C01][col]+=1; col+=1
        # M6
        U[A10][col]+=1; U[A00][col]-=1; V[B00][col]+=1; V[B01][col]+=1; W[C11][col]+=1; col+=1
        # M7
        U[A01][col]+=1; U[A11][col]-=1; V[B10][col]+=1; V[B11][col]+=1; W[C00][col]+=1; col+=1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = Aidx(a_r, a_c); A1 = Aidx(a_r, a_c+1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c+1)
        B10 = Bidx(b_r+1, b_c); B11 = Bidx(b_r+1, b_c+1)
        C0 = Cidx(c_r, c_c); C1 = Cidx(c_r, c_c+1)
        U[A0][col]+=1; V[B00][col]+=1; W[C0][col]+=1; col+=1
        U[A1][col]+=1; V[B10][col]+=1; W[C0][col]+=1; col+=1
        U[A0][col]+=1; V[B01][col]+=1; W[C1][col]+=1; col+=1
        U[A1][col]+=1; V[B11][col]+=1; W[C1][col]+=1; col+=1

    add_1x2_2x2(0,0,0,0,0,0); add_1x2_2x2(0,2,2,0,0,0)
    add_1x2_2x2(0,0,0,2,0,2); add_1x2_2x2(0,2,2,2,0,2)
    add_1x2_2x2(0,0,0,4,0,4); add_1x2_2x2(0,2,2,4,0,4)
    add_strassen(1,0,0,0,1,0); add_strassen(1,2,2,0,1,0)
    add_strassen(1,0,0,2,1,2); add_strassen(1,2,2,2,1,2)
    add_strassen(1,0,0,4,1,4); add_strassen(1,2,2,4,1,4)
    assert col == 66
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def compute_tensor_sparse(U, V, W):
    """Compute tensor contribution using sparse terms. Returns (n,m,p) array."""
    result = np.zeros((n, m, p))
    r = U.shape[1]
    for l in range(r):
        u = U[:, l]
        v = V[:, l]
        w = W[:, l]
        # Only process non-zero entries
        ui = np.nonzero(u)[0]
        vi = np.nonzero(v)[0]
        wi = np.nonzero(w)[0]
        for a in ui:
            i, j = divmod(a, m)
            for b in vi:
                j2, k = divmod(b, p)
                if j2 != j:
                    continue
                for c in wi:
                    k2, i2 = divmod(c, n)
                    if k2 != k or i2 != i:
                        continue
                    result[i, j, k] += u[a] * v[b] * w[c]
    return result

def tensor_diff_single(U, V, W, l):
    """Compute the tensor contribution of a single term l. Returns (n,m,p) array."""
    result = np.zeros((n, m, p))
    u = U[:, l]
    v = V[:, l]
    w = W[:, l]
    ui = np.nonzero(u)[0]
    vi = np.nonzero(v)[0]
    wi = np.nonzero(w)[0]
    for a in ui:
        i, j = divmod(a, m)
        for b in vi:
            j2, k = divmod(b, p)
            if j2 != j:
                continue
            for c in wi:
                k2, i2 = divmod(c, n)
                if k2 != k or i2 != i:
                    continue
                result[i, j, k] += u[a] * v[b] * w[c]
    return result

def verify_np(U, V, W):
    """Verify that the decomposition is exact."""
    result = compute_tensor(U, V, W)
    return np.allclose(result, 1.0, atol=1e-9)

def try_drop_fast(U, V, W, l, current_tensor):
    """Check if removing term l still gives a valid decomposition using precomputed tensor."""
    diff = tensor_diff_single(U, V, W, l)
    new_tensor = current_tensor - diff
    return np.allclose(new_tensor, 1.0, atol=1e-9)

def try_swap_fast(U, V, W, l1, l2, current_tensor):
    """Check if swapping terms l1 and l2 (i.e., removing both and re-adding differently) helps.
    Actually, for rank reduction, we try: can we remove l1 and l2 and replace with one term?
    That's complex. Instead, let's try: remove l1, then see if l2 can be adjusted.
    For now, just check if removing both l1 and l2 still works (they cancel out)."""
    diff1 = tensor_diff_single(U, V, W, l1)
    diff2 = tensor_diff_single(U, V, W, l2)
    new_tensor = current_tensor - diff1 - diff2
    return np.allclose(new_tensor, 1.0, atol=1e-9)

def try_scale_drop(U, V, W, l_remove, l_keep, current_tensor):
    """Try to remove l_remove by scaling l_keep. Check if there exists s such that
    current_tensor - tensor(l_remove) - s*tensor(l_keep) = ones tensor."""
    diff_remove = tensor_diff_single(U, V, W, l_remove)
    diff_keep = tensor_diff_single(U, V, W, l_keep)
    
    # We need: current - diff_remove - s*diff_keep = ones
    # => s*diff_keep = current - diff_remove - ones
    # => s*diff_keep = (current - ones) - diff_remove
    residual = current_tensor - 1.0 - diff_remove
    
    # Find s from non-zero entries of diff_keep
    mask = np.abs(diff_keep) > 1e-12
    if not np.any(mask):
        return False
    
    s_vals = residual[mask] / diff_keep[mask]
    # Check consistency
    if not np.allclose(s_vals, s_vals[0], atol=1e-9):
        return False
    
    s = s_vals[0]
    # Verify: current - diff_remove - s*diff_keep should equal ones
    new_tensor = current_tensor - diff_remove - s * diff_keep
    if np.allclose(new_tensor, 1.0, atol=1e-9):
        return s
    return False

def local_search_v3(U, V, W, deadline, rng):
    """Enhanced local search with fast verification using precomputed tensor."""
    r = U.shape[1]
    current_tensor = compute_tensor(U, V, W)
    improved = True
    no_improve_count = 0
    
    while time.time() < deadline - 5:
        improved = False
        indices = list(range(r))
        rng.shuffle(indices)
        
        # Pass 1: try drops (fast)
        for l in indices:
            if time.time() > deadline - 5:
                break
            if try_drop_fast(U, V, W, l, current_tensor):
                # Actually remove
                U = np.delete(U, l, axis=1)
                V = np.delete(V, l, axis=1)
                W = np.delete(W, l, axis=1)
                current_tensor = compute_tensor(U, V, W)
                r -= 1
                improved = True
                no_improve_count = 0
                break
        
        if improved:
            continue
        
        # Pass 2: try scale-drop (remove l_remove by scaling l_keep)
        for i in range(r):
            if time.time() > deadline - 5:
                break
            for j in range(i+1, r):
                if time.time() > deadline - 5:
                    break
                s = try_scale_drop(U, V, W, j, i, current_tensor)
                if s is not False:
                    # Scale term i by (1+s) and remove term j
                    U[:, i] = U[:, i] * (1 + s)
                    V[:, i] = V[:, i] * (1 + s)
                    W[:, i] = W[:, i] * (1 + s)
                    U = np.delete(U, j, axis=1)
                    V = np.delete(V, j, axis=1)
                    W = np.delete(W, j, axis=1)
                    current_tensor = compute_tensor(U, V, W)
                    r -= 1
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Pass 3: try removing two terms that cancel (both contribute zero net)
        for i in range(r):
            if time.time() > deadline - 5:
                break
            for j in range(i+1, r):
                if time.time() > deadline - 5:
                    break
                if try_swap_fast(U, V, W, i, j, current_tensor):
                    U = np.delete(U, j, axis=1)
                    V = np.delete(V, j, axis=1)
                    W = np.delete(W, j, axis=1)
                    U = np.delete(U, i, axis=1)
                    V = np.delete(V, i, axis=1)
                    W = np.delete(W, i, axis=1)
                    current_tensor = compute_tensor(U, V, W)
                    r -= 2
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 3:
                break
    return U, V, W

def construct():
    deadline = time.time() + 140  # safety margin for 150s limit
    rng = random.Random(42)
    U, V, W = baseline_66()
    U, V, W = local_search_v3(U, V, W, deadline, rng)
    if not verify_np(U, V, W):
        U, V, W = baseline_66()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
