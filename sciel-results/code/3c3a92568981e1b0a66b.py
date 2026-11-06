def synthesize(matrix, depth_limit):
    import numpy as np
    import time

    A = np.array(matrix, dtype=np.uint8)
    m, n = A.shape

    def compute_independent(A, n):
        gates = []
        next_wire = n + 1

        def xor_gate(a, b):
            nonlocal next_wire
            if a == b:
                return n
            if a == n:
                return b
            if b == n:
                return a
            gates.append([a, b])
            w = next_wire
            next_wire += 1
            return w

        outputs = []
        for i in range(A.shape[0]):
            ones = np.where(A[i] == 1)[0].tolist()
            if len(ones) == 0:
                outputs.append(n)
            elif len(ones) == 1:
                outputs.append(ones[0])
            else:
                wires = ones[:]
                while len(wires) > 1:
                    new_wires = []
                    for j in range(0, len(wires), 2):
                        if j + 1 < len(wires):
                            new_wires.append(xor_gate(wires[j], wires[j+1]))
                        else:
                            new_wires.append(wires[j])
                    wires = new_wires
                outputs.append(wires[0])

        return gates, outputs

    def compute_shared(A, n):
        A_rref = A.copy()
        m, n = A_rref.shape
        pivot_cols = []
        row_idx = 0
        for col in range(n):
            if row_idx >= m:
                break
            pivot = -1
            for r in range(row_idx, m):
                if A_rref[r, col] == 1:
                    pivot = r
                    break
            if pivot == -1:
                continue
            A_rref[[row_idx, pivot]] = A_rref[[pivot, row_idx]]
            for r in range(m):
                if r != row_idx and A_rref[r, col] == 1:
                    A_rref[r] = (A_rref[r] + A_rref[row_idx]) % 2
            pivot_cols.append(col)
            row_idx += 1

        rank = len(pivot_cols)

        gates = []
        next_wire = n + 1

        def xor_gate(a, b):
            nonlocal next_wire
            if a == b:
                return n
            if a == n:
                return b
            if b == n:
                return a
            gates.append([a, b])
            w = next_wire
            next_wire += 1
            return w

        basis_wires = []
        for j in range(rank):
            row = A_rref[j]
            ones = np.where(row == 1)[0].tolist()
            if len(ones) == 0:
                basis_wires.append(n)
            elif len(ones) == 1:
                basis_wires.append(ones[0])
            else:
                wires = ones[:]
                while len(wires) > 1:
                    new_wires = []
                    for k in range(0, len(wires), 2):
                        if k + 1 < len(wires):
                            new_wires.append(xor_gate(wires[k], wires[k+1]))
                        else:
                            new_wires.append(wires[k])
                    wires = new_wires
                basis_wires.append(wires[0])

        outputs = []
        for i in range(m):
            row = A_rref[i]
            basis_indices = [j for j in range(rank) if row[pivot_cols[j]] == 1]

            if len(basis_indices) == 0:
                outputs.append(n)
            elif len(basis_indices) == 1:
                outputs.append(basis_wires[basis_indices[0]])
            else:
                wires = [basis_wires[j] for j in basis_indices]
                while len(wires) > 1:
                    new_wires = []
                    for k in range(0, len(wires), 2):
                        if k + 1 < len(wires):
                            new_wires.append(xor_gate(wires[k], wires[k+1]))
                        else:
                            new_wires.append(wires[k])
                    wires = new_wires
                outputs.append(wires[0])

        return gates, outputs

    gates1, outputs1 = compute_independent(A, n)
    gates2, outputs2 = compute_shared(A, n)

    if len(gates1) <= len(gates2):
        return {'gates': gates1, 'outputs': outputs1}
    else:
        return {'gates': gates2, 'outputs': outputs2}
