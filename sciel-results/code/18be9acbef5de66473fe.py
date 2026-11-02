import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0

    n, m, p = 2, 5, 6

    # Generate candidate rank-1 tensors
    candidates = []
    
    # Add standard basis tensors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U = [0] * (n * m)
                V = [0] * (m * p)
                W = [0] * (p * n)
                U[i * m + j] = 1
                V[j * p + k] = 1
                W[k * n + i] = 1
                candidates.append((U, V, W))
    
    # Add Strassen patterns for all possible 2x2x2 blocks
    for j1 in range(m - 1):
        for j2 in range(j1 + 1, m):
            for k1 in range(p - 1):
                for k2 in range(k1 + 1, p):
                    # P1: (a00+a11)(b00+b11)
                    U = [0] * (n * m)
                    V = [0] * (m * p)
                    W = [0] * (p * n)
                    U[0 * m + j1] = 1
                    U[1 * m + j2] = 1
                    V[j1 * p + k1] = 1
                    V[j2 * p + k2] = 1
                    W[k1 * n + 0] = 1
                    W[k2 * n + 1] = 1
                    candidates.append((U, V, W))
                    
                    # P2: (a10+a11)*b00
                    U = [0] * (n * m)
                    V = [0] * (m * p)
                    W = [0] * (p * n)
                    U[1 * m + j1] = 1
                    U[1 * m + j2] = 1
                    V[j1 * p + k1] = 1
                    W[k1 * n + 1] = 1
                    W[k2 * n + 1] = -1
                    candidates.append((U, V, W))
                    
                    # P3: a00*(b01-b11)
                    U = [0] * (n * m)
                    V = [0] * (m * p)
                    W = [0] * (p * n)
                    U[0 * m + j1] = 1
                    V[j1 * p + k2] = 1
                    V[j2 * p + k2] = -1
                    W[k2 * n + 0] = 1
                    W[k2 * n + 1] = 1
                    candidates.append((U, V, W))
                    
                    # P4: a11*(b10-b00)
                    U = [0] * (n * m)
                    V = [0] * (m * p)
                    W = [0] * (p * n)
                    U[1 * m + j2] = 1
                    V[j2 * p + k1] = 1
                    V[j1 * p + k1] = -1
                    W[k1 * n + 0] = 1
                    W[k1 * n + 1] = 1
                    candidates.append((U, V, W))
                    
                    # P5: (a00+a01)*b11
                    U = [0] * (n * m)
                    V = [0] * (m * p)
                    W = [0] * (p * n)
                    U[0 * m + j1] = 1
                    U[0 * m + j2] = 1
                    V[j2 * p + k2] = 1
                    W[k1 * n + 0] = -1
                    W[k2 * n + 0] = 1
                    candidates.append((U, V, W))
                    
                    # P6: (a10-a00)*(b00+b01)
                    U = [0] * (n * m)
                    V = [0] * (m * p)
                    W = [0] * (p * n)
                    U[1 * m + j1] = 1
                    U[0 * m + j1] = -1
                    V[j1 * p + k1] = 1
                    V[j1 * p + k2] = 1
                    W[k2 * n + 1] = 1
                    candidates.append((U, V, W))
                    
                    # P7: (a01-a11)*(b10+b11)
                    U = [0] * (n * m)
                    V = [0] * (m * p)
                    W = [0] * (p * n)
                    U[0 * m + j2] = 1
                    U[1 * m + j2] = -1
                    V[j2 * p + k1] = 1
                    V[j2 * p + k2] = 1
                    W[k1 * n + 0] = 1
                    candidates.append((U, V, W))
    
    # Create target entries
    target_entries = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_entries.add((i, j, k))
    
    # For each candidate, compute which target entries it covers
    candidate_covers = []
    for U, V, W in candidates:
        covers = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U[i * m + j] * V[j * p + k] * W[k * n + i]
                    if val != 0:
                        covers.add((i, j, k))
        candidate_covers.append(covers)
    
    # Greedy set cover
    uncovered = set(target_entries)
    selected = []
    
    while uncovered and time.time() - start_time < deadline - 5:
        best_idx = -1
        best_count = 0
        for idx, covers in enumerate(candidate_covers):
            if idx in selected:
                continue
            count = len(covers & uncovered)
            if count > best_count:
                best_count = count
                best_idx = idx
        
        if best_idx == -1 or best_count == 0:
            break
        
        selected.append(best_idx)
        uncovered -= candidate_covers[best_idx]
    
    # Check if we've covered all entries
    if uncovered:
        # Fallback: use the standard 60-rank decomposition
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Solve for coefficients
    num_selected = len(selected)
    num_entries = n * m * p
    
    A = np.zeros((num_entries, num_selected))
    for l, idx in enumerate(selected):
        U, V, W = candidates[idx]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    e = i * m * p + j * p + k
                    A[e, l] = U[i * m + j] * V[j * p + k] * W[k * n + i]
    
    b = np.ones(num_entries)
    
    try:
        coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Build result
    U_result = [[0] * num_selected for _ in range(n * m)]
    V_result = [[0] * num_selected for _ in range(m * p)]
    W_result = [[0] * num_selected for _ in range(p * n)]
    
    for l, idx in enumerate(selected):
        U, V, W = candidates[idx]
        c = coeffs[l]
        for ii in range(n):
            for jj in range(m):
                U_result[ii * m + jj][l] = U[ii * m + jj] * c
        for jj in range(m):
            for kk in range(p):
                V_result[jj * p + kk][l] = V[jj * p + kk] * c
        for kk in range(p):
            for ii in range(n):
                W_result[kk * n + ii][l] = W[kk * n + ii] * c
    
    # Verify
    U_np = np.array(U_result)
    V_np = np.array(V_result)
    W_np = np.array(W_result)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                result = np.sum(U_np[i * m + j] * V_np[j * p + k] * W_np[k * n + i])
                if abs(result - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_result, "V": V_result, "W": W_result}
