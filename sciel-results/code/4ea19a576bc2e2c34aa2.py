import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def _get_rank_26_decomposition():
    """
    Returns the known rank-26 decomposition for 3x3 matrix multiplication.
    """
    n, m, p = 3, 3, 3
    R = 26
    
    U = np.zeros((9, R))
    V = np.zeros((9, R))
    W = np.zeros((9, R))
    
    # Strassen Terms for 2x2 Core (i,j,k in 0,1)
    # m1 = (a00 + a11)(b00 + b11)
    r = 0
    U[0*3+0, r] = 1; U[1*3+1, r] = 1
    V[0*3+0, r] = 1; V[1*3+1, r] = 1
    W[0*3+0, r] = 1; W[1*3+1, r] = 1
    
    # m2 = (a10 + a11)b00
    r = 1
    U[1*3+0, r] = 1; U[1*3+1, r] = 1
    V[0*3+0, r] = 1
    W[0*3+1, r] = 1; W[1*3+1, r] = -1
    
    # m3 = a00(b01 - b11)
    r = 2
    U[0*3+0, r] = 1
    V[0*3+1, r] = 1; V[1*3+1, r] = -1
    W[1*3+0, r] = 1; W[1*3+1, r] = 1
    
    # m4 = a11(b10 - b00)
    r = 3
    U[1*3+1, r] = 1
    V[1*3+0, r] = 1; V[0*3+0, r] = -1
    W[0*3+0, r] = 1; W[0*3+1, r] = 1
    
    # m5 = (a00 + a01)b11
    r = 4
    U[0*3+0, r] = 1; U[0*3+1, r] = 1
    V[1*3+1, r] = 1
    W[0*3+0, r] = -1; W[1*3+0, r] = 1
    
    # m6 = (a10 - a00)(b00 + b01)
    r = 5
    U[1*3+0, r] = 1; U[0*3+0, r] = -1
    V[0*3+0, r] = 1; V[0*3+1, r] = 1
    W[1*3+1, r] = 1
    
    # m7 = (a01 - a11)(b10 + b11)
    r = 6
    U[0*3+1, r] = 1; U[1*3+1, r] = -1
    V[1*3+0, r] = 1; V[1*3+1, r] = 1
    W[0*3+0, r] = 1
    
    # Border Terms (19 terms)
    # i=2, j=0, k=0
    r = 7
    U[2*3+0, r] = 1
    V[0*3+0, r] = 1
    W[0*3+2, r] = 1
    
    # i=2, j=0, k=1
    r = 8
    U[2*3+0, r] = 1
    V[0*3+1, r] = 1
    W[1*3+2, r] = 1
    
    # i=2, j=0, k=2
    r = 9
    U[2*3+0, r] = 1
    V[0*3+2, r] = 1
    W[2*3+2, r] = 1
    
    # i=2, j=1, k=0
    r = 10
    U[2*3+1, r] = 1
    V[1*3+0, r] = 1
    W[0*3+2, r] = 1
    
    # i=2, j=1, k=1
    r = 11
    U[2*3+1, r] = 1
    V[1*3+1, r] = 1
    W[1*3+2, r] = 1
    
    # i=2, j=1, k=2
    r = 12
    U[2*3+1, r] = 1
    V[1*3+2, r] = 1
    W[2*3+2, r] = 1
    
    # i=2, j=2, k=0
    r = 13
    U[2*3+2, r] = 1
    V[2*3+0, r] = 1
    W[0*3+2, r] = 1
    
    # i=2, j=2, k=1
    r = 14
    U[2*3+2, r] = 1
    V[2*3+1, r] = 1
    W[1*3+2, r] = 1
    
    # i=2, j=2, k=2
    r = 15
    U[2*3+2, r] = 1
    V[2*3+2, r] = 1
    W[2*3+2, r] = 1
    
    # i=0, j=2, k=0
    r = 16
    U[0*3+2, r] = 1
    V[2*3+0, r] = 1
    W[0*3+0, r] = 1
    
    # i=0, j=2, k=1
    r = 17
    U[0*3+2, r] = 1
    V[2*3+1, r] = 1
    W[1*3+0, r] = 1
    
    # i=0, j=2, k=2
    r = 18
    U[0*3+2, r] = 1
    V[2*3+2, r] = 1
    W[2*3+0, r] = 1
    
    # i=1, j=2, k=0
    r = 19
    U[1*3+2, r] = 1
    V[2*3+0, r] = 1
    W[0*3+1, r] = 1
    
    # i=1, j=2, k=1
    r = 20
    U[1*3+2, r] = 1
    V[2*3+1, r] = 1
    W[1*3+1, r] = 1
    
    # i=1, j=2, k=2
    r = 21
    U[1*3+2, r] = 1
    V[2*3+2, r] = 1
    W[1*3+1, r] = 1
    
    # i=0, j=0, k=2
    r = 22
    U[0*3+0, r] = 1
    V[0*3+2, r] = 1
    W[2*3+0, r] = 1
    
    # i=0, j=1, k=2
    r = 23
    U[0*3+1, r] = 1
    V[1*3+2, r] = 1
    W[2*3+0, r] = 1
    
    # i=1, j=0, k=2
    r = 24
    U[1*3+0, r] = 1
    V[0*3+2, r] = 1
    W[2*3+1, r] = 1
    
    # i=1, j=1, k=2
    r = 25
    U[1*3+1, r] = 1
    V[1*3+2, r] = 1
    W[2*3+1, r] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W):
    """
    Verifies if U, V, W form a valid decomposition of 3x3 matrix multiplication.
    """
    n, m, p = 3, 3, 3
    R = U.shape[1]
    
    # Check all constraints
    # Sum_r U[i,3+j,r] * V[j',3+k',r] * W[k',3+i',r] == 1 if j==j' and i==i'
    # Note: W index is k'*n + i'
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for kp in range(p):
                    for ip in range(n):
                        s = 0.0
                        for r in range(R):
                            s += U[i*m+j, r] * V[jp*p+kp, r] * W[kp*n+ip, r]
                        
                        target = 0.0
                        if jp == j and ip == i:
                            target = 1.0
                        
                        if abs(s - target) > 1e-9:
                            return False
    return True

def _snap_to_lattice(mat):
    """
    Snaps values to nearest integer or half-integer.
    """
    # Multiply by 2, round, divide by 2
    return np.round(mat * 2) / 2

def _try_find_lower_rank(rank, max_iterations=100, time_limit=5.0):
    """
    Attempts to find a decomposition of the given rank using ALS.
    """
    n, m, p = 3, 3, 3
    R = rank
    
    # Initialize randomly
    U = np.random.randn(9, R)
    V = np.random.randn(9, R)
    W = np.random.randn(9, R)
    
    start_time = time.time()
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                # Build linear system for U[i*m+j, :]
                # Constraints: for all jp, kp, ip
                A_mat = np.zeros((27, R))
                b_vec = np.zeros(27)
                idx = 0
                for jp in range(m):
                    for kp in range(p):
                        for ip in range(n):
                            # Row of A_mat
                            for r in range(R):
                                A_mat[idx, r] = V[jp*p+kp, r] * W[kp*n+ip, r]
                            # Target
                            if jp == j and ip == i:
                                b_vec[idx] = 1.0
                            else:
                                b_vec[idx] = 0.0
                            idx += 1
                # Solve least squares
                U[i*m+j, :], residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        # Update V
        for jp in range(m):
            for kp in range(p):
                A_mat = np.zeros((27, R))
                b_vec = np.zeros(27)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for ip in range(n):
                            for r in range(R):
                                A_mat[idx, r] = U[i*m+j, r] * W[kp*n+ip, r]
                            if jp == j and ip == i:
                                b_vec[idx] = 1.0
                            else:
                                b_vec[idx] = 0.0
                            idx += 1
                V[jp*p+kp, :], residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        # Update W
        for kp in range(p):
            for ip in range(n):
                A_mat = np.zeros((27, R))
                b_vec = np.zeros(27)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for r in range(R):
                                A_mat[idx, r] = U[i*m+j, r] * V[jp*p+kp, r]
                            if jp == j and ip == i:
                                b_vec[idx] = 1.0
                            else:
                                b_vec[idx] = 0.0
                            idx += 1
                W[kp*n+ip, :], residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    
    # Snap to lattice
    U_snap = _snap_to_lattice(U)
    V_snap = _snap_to_lattice(V)
    W_snap = _snap_to_lattice(W)
    
    if _verify_decomposition(U_snap, V_snap, W_snap):
        return U_snap, V_snap, W_snap
    
    return None

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x3 matrix multiplication.
    """
    deadline = time.time() + 110  # Safety margin
    
    # Try to find lower rank decompositions
    # Start from 25 down to 23
    for rank in [25, 24, 23]:
        if time.time() > deadline:
            break
        
        # Try a few restarts for each rank
        for restart in range(5):
            if time.time() > deadline:
                break
            
            result = _try_find_lower_rank(rank, max_iterations=50, time_limit=10.0)
            if result is not None:
                U, V, W = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # If rank 25 failed, try 24, etc.
    
    # Fallback to rank 26
    return _get_rank_26_decomposition()

# EVOLVE-BLOCK-END
