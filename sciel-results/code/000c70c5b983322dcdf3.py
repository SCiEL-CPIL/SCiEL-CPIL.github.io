def synthesize(matrix, depth_limit):
    import numpy as np
    
    A = np.array(matrix, dtype=np.uint8)
    m, n = A.shape
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    # Perform Gaussian elimination over GF(2) to get RREF
    # Track the transformation matrix T such that RREF = T @ A
    M = A.copy().astype(np.int64)
    T = np.eye(m, dtype=np.int64)
    
    pivot_cols = []
    row_idx = 0
    
    for col in range(n):
        # Find a pivot row with 1 in this column
        pivot_row = -1
        for r in range(row_idx, m):
            if M[r, col] % 2 == 1:
                pivot_row = r
                break
        
        if pivot_row == -1:
            continue
        
        # Swap rows if needed
        if pivot_row != row_idx:
            M[[row_idx, pivot_row]] = M[[pivot_row, row_idx]]
            T[[row_idx, pivot_row]] = T[[pivot_row, row_idx]]
        
        # Eliminate this column from all other rows
        for r in range(m):
            if r != row_idx and M[r, col] % 2 == 1:
                M[r] = (M[r] + M[row_idx]) % 2
                T[r] = (T[r] + T[row_idx]) % 2
        
        pivot_cols.append((row_idx, col))
        row_idx += 1
        
        if row_idx == m:
            break
    
    # Now M is in RREF form
    # T @ A = M, so A = T^{-1} @ M
    
    # Invert T over GF(2)
    T_inv = np.eye(m, dtype=np.int64)
    for col in range(m):
        # Find pivot
        pivot_row = -1
        for r in range(col, m):
            if T[r, col] % 2 == 1:
                pivot_row = r
                break
        if pivot_row == -1:
            continue
        if pivot_row != col:
            T[[col, pivot_row]] = T[[pivot_row, col]]
            T_inv[[col, pivot_row]] = T_inv[[pivot_row, col]]
        for r in range(m):
            if r != col and T[r, col] % 2 == 1:
                T[r] = (T[r] + T[col]) % 2
                T_inv[r] = (T_inv[r] + T_inv[col]) % 2
    
    # Identify which rows of M are pivot rows (non-zero)
    pivot_row_indices = [r for r, c in pivot_cols]
    
    gates = []
    outputs = []
    next_wire = n + 1
    
    # Helper function to synthesize a linear combination of wires using a balanced tree
    def synthesize_xor(wires):
        nonlocal next_wire
        if len(wires) == 0:
            return n
        if len(wires) == 1:
            return wires[0]
        
        # Balanced split
        mid = len(wires) // 2
        left_wires = wires[:mid]
        right_wires = wires[mid:]
        
        left_out = synthesize_xor(left_wires)
        right_out = synthesize_xor(right_wires)
        
        gates.append([left_out, right_out])
        out = next_wire
        next_wire += 1
        return out

    # First, synthesize each pivot row of M
    pivot_row_wires = {}
    for r, c in pivot_cols:
        row = M[r]
        cols = np.where(row % 2 == 1)[0]
        
        if len(cols) == 0:
            pivot_row_wires[r] = n
        else:
            pivot_row_wires[r] = synthesize_xor([int(c) for c in cols])
    
    # Now synthesize each original row i as XOR of pivot rows where T_inv[i,j] == 1
    for i in range(m):
        # Find which pivot rows contribute to row i
        contributing = [r for r in pivot_row_indices if T_inv[i, r] % 2 == 1]
        
        if len(contributing) == 0:
            outputs.append(n)
        else:
            outputs.append(synthesize_xor([pivot_row_wires[r] for r in contributing]))
    
    return {'gates': gates, 'outputs': outputs}
