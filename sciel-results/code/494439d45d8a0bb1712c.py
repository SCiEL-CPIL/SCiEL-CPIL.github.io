import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    random.seed(42)
    np.random.seed(42)

    def verify(U, V, W, rank):
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # ---- Strassen 2x2 * 2x2 (rank 7) ----
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[2,1]=1; W2[3,1]=1
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[0,2]=1; W2[1,2]=1
    U2[0,3]=1; U2[3,3]=-1; V2[1,3]=1; V2[3,3]=1; W2[0,3]=1; W2[3,3]=1
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[1,4]=1; W2[3,4]=1
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[2,5]=1; W2[0,5]=1
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[1,6]=1; W2[2,6]=1

    # ---- 4x4 * 4x4 via Kronecker Strassen (rank 49) ----
    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    for t1 in range(7):
        for t2 in range(7):
            col = t1*7 + t2
            for i1 in range(2):
                if U2[i1,t1]!=0:
                    for i2 in range(2):
                        if U2[i2,t2]!=0:
                            U44[i1*2+i2, col] = U2[i1,t1]*U2[i2,t2]
            for i1 in range(2):
                if V2[i1,t1]!=0:
                    for i2 in range(2):
                        if V2[i2,t2]!=0:
                            V44[i1*2+i2, col] = V2[i1,t1]*V2[i2,t2]
            for i1 in range(2):
                if W2[i1,t1]!=0:
                    for i2 in range(2):
                        if W2[i2,t2]!=0:
                            W44[i1*2+i2, col] = W2[i1,t1]*W2[i2,t2]

    # ---- 4x4 * 4x1 (rank 4) ----
    U41 = np.zeros((16, 4))
    V41 = np.zeros((20, 4))
    W41 = np.zeros((20, 4))
    for r in range(4):
        for i in range(4):
            for j in range(4):
                if i == r:
                    U41[i*4+j, r] = 1.0
                    W41[16+i, r] = 1.0
        for j in range(4):
            V41[j*5+4, r] = 1.0

    # ---- Combine into 4x4 * 4x5 (rank 53) ----
    rank = 53
    U = np.zeros((16, rank))
    V = np.zeros((20, rank))
    W = np.zeros((20, rank))
    U[:, 0:49] = U44
    for j in range(4):
        for k in range(4):
            V[j*5+k, 0:49] = V44[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W44[k*4+i, 0:49]
    U[:, 49:53] = U41
    V[:, 49:53] = V41
    W[:, 49:53] = W41

    if not verify(U, V, W, rank):
        rank = n*m*p
        U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j,col]=1.0; V[j*p+k,col]=1.0; W[k*n+i,col]=1.0; col+=1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ---- Refinement: try to reduce rank ----
    def residual(U, V, W, rank):
        # smooth surrogate: sum of squared residuals over all (i,j,k)
        r = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    r += (val - 1.0)**2
        return r

    # Try to drop columns that are redundant.
    # For a column c, its contribution to tensor is outer(U[:,c], V[:,c], W[:,c]).
    # If removing it still yields exact tensor, drop it.
    def try_drop_columns(U, V, W, rank):
        # iterate: attempt to drop each column
        changed = True
        while changed:
            changed = False
            for c in range(rank-1, -1, -1):
                if time.time() > deadline:
                    return U, V, W, rank
                # build without column c
                mask = np.ones(rank, dtype=bool); mask[c] = False
                Uc = U[:, mask]; Vc = V[:, mask]; Wc = W[:, mask]
                if verify(Uc, Vc, Wc, rank-1):
                    U, V, W = Uc, Vc, Wc
                    rank -= 1
                    changed = True
        return U, V, W, rank

    # Check if column c is a linear combination of other columns in a structured way
    def is_redundant_column(U, V, W, rank, c):
        """Check if column c's tensor contribution is zero or linearly dependent"""
        # Compute the tensor contribution of column c
        # T_c[i,j,k] = U[i,j,c] * V[j,k,c] * W[k,i,c]
        # If this is zero for all (i,j,k), column is redundant
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if U[i*m+j, c] != 0 and V[j*p+k, c] != 0 and W[k*n+i, c] != 0:
                        return False
        return True

    # Try to find a set of columns that can be removed simultaneously
    def try_drop_multiple(U, V, W, rank, max_drop=3):
        """Try to drop up to max_drop columns at once"""
        if time.time() > deadline:
            return U, V, W, rank
        best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
        
        # Try dropping 1 column
        for c in range(rank):
            mask = np.ones(rank, dtype=bool); mask[c] = False
            Uc = U[:, mask]; Vc = V[:, mask]; Wc = W[:, mask]
            if verify(Uc, Vc, Wc, rank-1):
                return Uc, Vc, Wc, rank-1
        
        # Try dropping 2 columns
        if rank >= 2:
            for c1 in range(rank):
                for c2 in range(c1+1, rank):
                    if time.time() > deadline:
                        return best_U, best_V, best_W, best_rank
                    mask = np.ones(rank, dtype=bool)
                    mask[c1] = False; mask[c2] = False
                    Uc = U[:, mask]; Vc = V[:, mask]; Wc = W[:, mask]
                    if verify(Uc, Vc, Wc, rank-2):
                        return Uc, Vc, Wc, rank-2
        
        return best_U, best_V, best_W, best_rank

    U, V, W, rank = try_drop_columns(U, V, W, rank)
    U, V, W, rank = try_drop_multiple(U, V, W, rank)

    # Structured local search: target specific patterns
    entry_set = [-1.0, -0.5, 0.0, 0.5, 1.0]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    def local_search_structured(U, V, W, rank, iters):
        """Structured local search that targets specific column patterns"""
        for it in range(iters):
            if time.time() > deadline:
                break
            
            # Pick a target column to eliminate
            target = np.random.randint(0, rank)
            
            # Strategy 1: Perturb entries in the target column to make it zero
            # This works if the column's contribution is already zero or can be zeroed
            n_perturb = np.random.randint(1, 4)
            orig = []
            
            # Perturb entries in the target column across U, V, W
            for _ in range(n_perturb):
                which = np.random.randint(0, 3)
                mat = U if which==0 else (V if which==1 else W)
                rows = mat.shape[0]
                r = np.random.randint(0, rows)
                c = target
                orig.append((which, r, c, mat[r,c]))
                mat[r,c] = np.random.choice(entry_set)
            
            # Check if target column is now redundant
            mask = np.ones(rank, dtype=bool); mask[target] = False
            Uc = U[:, mask]; Vc = V[:, mask]; Wc = W[:, mask]
            if verify(Uc, Vc, Wc, rank-1):
                return Uc, Vc, Wc, rank-1
            
            # Revert
            for which, r, c, val in orig:
                mat = U if which==0 else (V if which==1 else W)
                mat[r,c] = val
            
            # Strategy 2: Perturb a mix of target and non-target columns
            n_perturb = np.random.randint(2, 5)
            orig = []
            for _ in range(n_perturb):
                which = np.random.randint(0, 3)
                mat = U if which==0 else (V if which==1 else W)
                rows = mat.shape[0]
                r = np.random.randint(0, rows)
                # 50% chance to perturb target column, 50% other
                if np.random.random() < 0.5:
                    c = target
                else:
                    c = np.random.randint(0, rank)
                    while c == target and rank > 1:
                        c = np.random.randint(0, rank)
                orig.append((which, r, c, mat[r,c]))
                mat[r,c] = np.random.choice(entry_set)
            
            mask = np.ones(rank, dtype=bool); mask[target] = False
            Uc = U[:, mask]; Vc = V[:, mask]; Wc = W[:, mask]
            if verify(Uc, Vc, Wc, rank-1):
                return Uc, Vc, Wc, rank-1
            
            for which, r, c, val in orig:
                mat = U if which==0 else (V if which==1 else W)
                mat[r,c] = val
        
        return U, V, W, rank

    # Run structured local search
    iters = 3000
    U, V, W, rank = local_search_structured(U, V, W, rank, iters)
    
    # Final drop pass
    U, V, W, rank = try_drop_columns(U, V, W, rank)
    U, V, W, rank = try_drop_multiple(U, V, W, rank)

    if rank >= best_rank:
        U, V, W, rank = best_U, best_V, best_W, best_rank

    if not verify(U, V, W, rank):
        rank = n*m*p
        U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j,col]=1.0; V[j*p+k,col]=1.0; W[k*n+i,col]=1.0; col+=1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
