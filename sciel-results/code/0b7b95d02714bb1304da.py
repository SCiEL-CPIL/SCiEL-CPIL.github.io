"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def synthesize(matrix, depth_limit):
    import numpy as np
    
    # Parse input matrix
    try:
        A = np.array(matrix, dtype=int)
    except Exception:
        A = np.array(matrix)
        A = (A != 0).astype(int)
    
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    m, n = A.shape
    
    # Gaussian elimination to find RREF and pivot columns
    A_work = A.copy()
    pivot_cols = []
    row = 0
    for col in range(n):
        pivot_idx = -1
        for i in range(row, m):
            if A_work[i, col] == 1:
                pivot_idx = i
                break
        
        if pivot_idx == -1:
            continue
        
        if pivot_idx != row:
            A_work[[row, pivot_idx], :] = A_work[[pivot_idx, row], :]
        
        pivot_cols.append(col)
        
        for i in range(m):
            if i != row and A_work[i, col] == 1:
                A_work[i, :] = (A_work[i, :] + A_work[row, :]) % 2
        
        row += 1
        if row == m:
            break
    
    r = row
    B = A_work[:r, :]  # Basis vectors in RREF
    C = A[:, pivot_cols]  # Coefficient matrix
    
    gates = []
    outputs = []
    next_wire = n + 1
    
    # 1. Synthesize basis wires
    basis_wires = []
    for k in range(r):
        b_k = B[k]
        ones = np.where(b_k == 1)[0]
        
        if len(ones) == 0:
            basis_wires.append(n)
        elif len(ones) == 1:
            basis_wires.append(int(ones[0]))
        else:
            current_wire = int(ones[0])
            for j in range(1, len(ones)):
                gates.append([current_wire, int(ones[j])])
                current_wire = next_wire
                next_wire += 1
            basis_wires.append(current_wire)
    
    # 2. Synthesize outputs with caching to share XOR combinations
    xor_cache = {}  # frozenset of basis indices -> wire
    
    def get_xor_wire(indices):
        nonlocal next_wire
        key = frozenset(indices)
        if key in xor_cache:
            return xor_cache[key]
        if not indices:
            return n
        if len(indices) == 1:
            return basis_wires[list(indices)[0]]
        # Linear chain for the XOR
        idx_list = list(indices)
        current_wire = basis_wires[idx_list[0]]
        for idx in idx_list[1:]:
            gates.append([current_wire, basis_wires[idx]])
            current_wire = next_wire
            next_wire += 1
        xor_cache[key] = current_wire
        return current_wire
    
    for i in range(m):
        c_i = C[i]
        active_basis = np.where(c_i == 1)[0]
        active_list = [int(x) for x in active_basis]
        
        if not active_list:
            outputs.append(n)
        else:
            outputs.append(get_xor_wire(active_list))
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
