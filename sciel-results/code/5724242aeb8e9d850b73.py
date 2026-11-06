import numpy as np
import time

def _popcount(x):
    return int(bin(x).count('1'))

def synthesize(matrix, depth_limit):
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill
    matrix = np.array(matrix, dtype=np.uint8) % 2
    m, n = matrix.shape
    zero_wire = n

    # Convert rows to ints
    row_ints = []
    for i in range(m):
        v = 0
        for j in range(n):
            if matrix[i, j] & 1:
                v |= (1 << j)
        row_ints.append(v)

    # Gaussian elimination to get RREF and basis
    work = [(row_ints[i], 1 << i) for i in range(m)]
    basis_vecs = []
    basis_markers = []
    
    for col in range(n):
        bit = 1 << col
        idx = None
        for k in range(len(work)):
            v, mk = work[k]
            if mk == 0:
                continue
            if v & bit:
                idx = k
                break
        if idx is None:
            continue
        v, mk = work[idx]
        for k in range(len(work)):
            if k == idx:
                continue
            vk, mkk = work[k]
            if vk & bit:
                work[k] = (vk ^ v, mkk ^ mk)
        basis_vecs.append(v)
        basis_markers.append(mk)
    
    r = len(basis_vecs)
    
    if r == 0:
        return {'gates': [], 'outputs': [zero_wire] * m}

    # Solve for combos
    def solve_combo(row_int):
        v = row_int
        c = 0
        pcols = []
        for i in range(r):
            pv = basis_vecs[i]
            pcol = (pv & -pv).bit_length() - 1
            pcols.append(pcol)
        order = sorted(range(r), key=lambda i: pcols[i], reverse=True)
        for i in order:
            pv = basis_vecs[i]
            pcol = pcols[i]
            if v & (1 << pcol):
                v ^= pv
                c |= (1 << i)
        return c

    combo = [solve_combo(row_ints[j]) for j in range(m)]

    # Local Search to Improve Basis
    # Objective: Minimize Sum(popcount(basis_vecs)) + Sum(popcount(combo_vecs))
    def compute_combos(basis):
        combos = []
        for j in range(m):
            combos.append(solve_combo_with_basis(row_ints[j], basis))
        return combos

    def solve_combo_with_basis(row_int, basis):
        v = row_int
        c = 0
        pcols = []
        for i in range(r):
            pv = basis[i]
            pcol = (pv & -pv).bit_length() - 1
            pcols.append(pcol)
        order = sorted(range(r), key=lambda i: pcols[i], reverse=True)
        for i in order:
            pv = basis[i]
            pcol = pcols[i]
            if v & (1 << pcol):
                v ^= pv
                c |= (1 << i)
        return c

    def total_cost(basis):
        combos = compute_combos(basis)
        cost = 0
        for b in basis:
            cost += _popcount(b)
        for c in combos:
            cost += _popcount(c)
        return cost

    current_basis = basis_vecs[:]
    current_cost = total_cost(current_basis)
    
    # Local Search Loop
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(r):
            for j in range(r):
                if i == j:
                    continue
                new_bi = current_basis[i] ^ current_basis[j]
                if new_bi == 0:
                    continue
                old_bi = current_basis[i]
                current_basis[i] = new_bi
                new_cost = total_cost(current_basis)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    current_basis[i] = old_bi

    # Build Circuit with Intermediate Wire Sharing
    gates = []
    wire_count = n + 1
    vec_to_wire = {}
    
    def xor_gate(a, b):
        nonlocal wire_count
        w = wire_count
        wire_count += 1
        gates.append([a, b])
        return w

    def compute_vector(vec_int):
        if vec_int in vec_to_wire:
            return vec_to_wire[vec_int]
        
        bits = []
        for j in range(n):
            if vec_int & (1 << j):
                bits.append(j)
        
        if not bits:
            vec_to_wire[vec_int] = zero_wire
            return zero_wire
        if len(bits) == 1:
            vec_to_wire[vec_int] = bits[0]
            return bits[0]
        
        # Build tree to minimize depth, and try to share intermediate results
        # Simple approach: build left-to-right but check if intermediate XOR exists
        w = bits[0]
        for b in bits[1:]:
            # Check if XOR of current w and b is already computed
            # We need to track vectors for wires, but that's complex.
            # For now, just standard chain, but maybe optimize later.
            w = xor_gate(w, b)
        
        vec_to_wire[vec_int] = w
        return w

    # Compute basis wires
    basis_wires = []
    for i in range(r):
        basis_wires.append(compute_vector(current_basis[i]))
        
    # Compute output wires
    combos = compute_combos(current_basis)
    outputs = []
    for j in range(m):
        c = combos[j]
        idxs = [i for i in range(r) if c & (1 << i)]
        if not idxs:
            outputs.append(zero_wire)
        elif len(idxs) == 1:
            outputs.append(basis_wires[idxs[0]])
        else:
            # Combine basis wires
            # We can try to find a vector in vec_to_wire that matches the XOR of these basis wires
            # But that's expensive. Just chain them.
            w = basis_wires[idxs[0]]
            for i in idxs[1:]:
                w = xor_gate(w, basis_wires[i])
            outputs.append(w)
            
    # Verification
    wire_val = [0] * wire_count
    for j in range(n):
        wire_val[j] = 1 << j
    for k in range(len(gates)):
        a, b = gates[k]
        wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
        
    ok = True
    for j in range(m):
        if wire_val[outputs[j]] != row_ints[j]:
            ok = False
            break
            
    if not ok:
        # Fallback: Direct synthesis (no sharing)
        gates = []
        wire_count = n + 1
        def xor_gate2(a, b):
            nonlocal wire_count
            w = wire_count
            wire_count += 1
            gates.append([a, b])
            return w
            
        outputs = []
        for j in range(m):
            bits = [k for k in range(n) if row_ints[j] & (1 << k)]
            if not bits:
                outputs.append(zero_wire)
            elif len(bits) == 1:
                outputs.append(bits[0])
            else:
                w = bits[0]
                for b in bits[1:]:
                    w = xor_gate2(w, b)
                outputs.append(w)
                
    return {'gates': gates, 'outputs': outputs}
