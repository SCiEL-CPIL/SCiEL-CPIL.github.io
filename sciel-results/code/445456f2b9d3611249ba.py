"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import numpy as np
import time

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return the 7 rank-1 terms (U, V, W) for the 2x2x2 Strassen decomposition.
    Each term is a tuple of three 2x2 numpy arrays (U[i,j], V[j,k], W[k,i])."""
    terms = []
    # P1 = (A00+A11)(B00+B11)
    U1 = np.array([[1,0],[0,1]], dtype=float)
    V1 = np.array([[1,0],[0,1]], dtype=float)
    W1 = np.array([[1,0],[0,1]], dtype=float)  # W[k,i]: W[0,0]=1, W[1,1]=1
    terms.append((U1, V1, W1))
    # P2 = (A10+A11)B00
    U2 = np.array([[0,0],[1,1]], dtype=float)
    V2 = np.array([[1,0],[0,0]], dtype=float)
    W2 = np.array([[0,1],[0,-1]], dtype=float)  # W[0,1]=1, W[1,1]=-1
    terms.append((U2, V2, W2))
    # P3 = A00(B01-B11)
    U3 = np.array([[1,0],[0,0]], dtype=float)
    V3 = np.array([[0,1],[0,-1]], dtype=float)
    W3 = np.array([[0,0],[1,1]], dtype=float)  # W[1,0]=1, W[1,1]=1
    terms.append((U3, V3, W3))
    # P4 = A11(B10-B00)
    U4 = np.array([[0,0],[0,1]], dtype=float)
    V4 = np.array([[-1,0],[1,0]], dtype=float)
    W4 = np.array([[1,1],[0,0]], dtype=float)  # W[0,0]=1, W[0,1]=1
    terms.append((U4, V4, W4))
    # P5 = (A00+A01)B11
    U5 = np.array([[1,1],[0,0]], dtype=float)
    V5 = np.array([[0,0],[0,1]], dtype=float)
    W5 = np.array([[-1,0],[1,0]], dtype=float)  # W[0,0]=-1, W[1,0]=1
    terms.append((U5, V5, W5))
    # P6 = (A10-A00)(B00+B01)
    U6 = np.array([[-1,0],[1,0]], dtype=float)
    V6 = np.array([[1,1],[0,0]], dtype=float)
    W6 = np.array([[0,0],[0,1]], dtype=float)  # W[1,1]=1
    terms.append((U6, V6, W6))
    # P7 = (A01-A11)(B10+B11)
    U7 = np.array([[0,1],[-1,0]], dtype=float)
    V7 = np.array([[0,0],[1,1]], dtype=float)
    W7 = np.array([[1,0],[0,0]], dtype=float)  # W[0,0]=1
    terms.append((U7, V7, W7))
    return terms

def _build_hybrid():
    """Build the 4-block Strassen + schoolbook decomposition of rank 56."""
    n, m, p = DIMENSIONS
    U = []
    V = []
    W = []
    
    # 4 disjoint Strassen blocks: (i0, j0, k0)
    blocks = [
        (0, 0, 0),  # i∈{0,1}, j∈{0,1}, k∈{0,1}
        (0, 2, 2),  # i∈{0,1}, j∈{2,3}, k∈{2,3}
        (1, 0, 2),  # i∈{1,2}, j∈{0,1}, k∈{2,3}
        (1, 2, 0),  # i∈{1,2}, j∈{2,3}, k∈{0,1}
    ]
    
    # Mark cells covered by Strassen blocks
    covered = set()
    for (i0, j0, k0) in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    # Add Strassen terms
    strassen_terms = _strassen_2x2x2()
    for (i0, j0, k0) in blocks:
        for (U2, V2, W2) in strassen_terms:
            # Expand to full size
            U_full = np.zeros((n, m), dtype=float)
            V_full = np.zeros((m, p), dtype=float)
            W_full = np.zeros((p, n), dtype=float)
            for di in range(2):
                for dj in range(2):
                    U_full[i0+di, j0+dj] = U2[di, dj]
            for dj in range(2):
                for dk in range(2):
                    V_full[j0+dj, k0+dk] = V2[dj, dk]
            for dk in range(2):
                for di in range(2):
                    W_full[k0+dk, i0+di] = W2[dk, di]
            U.append(U_full)
            V.append(V_full)
            W.append(W_full)
    
    # Add schoolbook terms for uncovered cells
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U_full = np.zeros((n, m), dtype=float)
                    V_full = np.zeros((m, p), dtype=float)
                    W_full = np.zeros((p, n), dtype=float)
                    U_full[i, j] = 1.0
                    V_full[j, k] = 1.0
                    W_full[k, i] = 1.0
                    U.append(U_full)
                    V.append(V_full)
                    W.append(W_full)
    
    U = np.array(U)  # shape (rank, n, m)
    V = np.array(V)  # shape (rank, m, p)
    W = np.array(W)  # shape (rank, p, n)
    return U, V, W

def _verify(U, V, W, tol=1e-9):
    """Verify the decomposition. U: (rank, n, m), V: (rank, m, p), W: (rank, p, n)."""
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[:, i, j] * V[:, j, k] * W[:, k, i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _to_interface(U, V, W):
    """Convert (rank, n, m), (rank, m, p), (rank, p, n) to interface format.
    Interface: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)."""
    n, m, p = DIMENSIONS
    rank = U.shape[0]
    U_out = np.zeros((n * m, rank), dtype=float)
    V_out = np.zeros((m * p, rank), dtype=float)
    W_out = np.zeros((p * n, rank), dtype=float)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                U_out[i * m + j, r] = U[r, i, j]
        for j in range(m):
            for k in range(p):
                V_out[j * p + k, r] = V[r, j, k]
        for k in range(p):
            for i in range(n):
                W_out[k * n + i, r] = W[r, k, i]
    return U_out.tolist(), V_out.tolist(), W_out.tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Build the 56-rank hybrid decomposition
    U, V, W = _build_hybrid()
    
    # Verify
    if not _verify(U, V, W):
        # Fallback to schoolbook
        U_sb = np.zeros((n * m, n * m * p), dtype=float)
        V_sb = np.zeros((m * p, n * m * p), dtype=float)
        W_sb = np.zeros((p * n, n * m * p), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1.0
                    V_sb[j * p + k, col] = 1.0
                    W_sb[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    U_list, V_list, W_list = _to_interface(U, V, W)
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
