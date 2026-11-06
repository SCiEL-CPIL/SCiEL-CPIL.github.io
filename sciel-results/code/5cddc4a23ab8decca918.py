import time
import numpy as np
from itertools import combinations

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 54.0  # 54s safety margin out of 60s

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    A = np.array(matrix, dtype=np.uint8) % 2

    def count_gates_for_basis(M, T_inv, rank):
        """Estimate gate count for a given basis without building actual gates."""
        # Count gates needed for each pivot row (XOR of input wires)
        total_gates = 0
        pivot_weights = []
        for i in range(rank):
            weight = int(np.sum(M[i, :]))
            pivot_weights.append(weight)
            if weight > 1:
                total_gates += weight - 1  # XOR of k wires needs k-1 gates
        
        # Count gates needed for each output (XOR of pivot wires)
        for j in range(m):
            weight = int(np.sum(T_inv[j, :rank]))
            if weight > 1:
                total_gates += weight - 1
        
        return total_gates

    def compute_rref_with_pivot_order(A, pivot_order):
        """Compute RREF using a specific pivot column order."""
        m = len(A)
        n = len(A[0]) if m > 0 else 0
        
        M = A.copy()
        T = np.eye(m, dtype=np.uint8)
        
        rank = 0
        pivot_cols_used = []
        
        for col in pivot_order:
            if rank == m:
                break
            if col >= n:
                continue
            pivot = -1
            for row in range(rank, m):
                if M[row, col] == 1:
                    pivot = row
                    break
            if pivot == -1:
                continue

            M[[rank, pivot]] = M[[pivot, rank]]
            T[[rank, pivot]] = T[[pivot, rank]]

            for row in range(m):
                if row != rank and M[row, col] == 1:
                    M[row] ^= M[rank]
                    T[row] ^= T[rank]

            pivot_cols_used.append(col)
            rank += 1
        
        # Compute T^{-1}
        aug = np.hstack([T, np.eye(m, dtype=np.uint8)])
        for col in range(m):
            pivot = -1
            for row in range(col, m):
                if aug[row, col] == 1:
                    pivot = row
                    break
            if pivot == -1:
                continue
            aug[[col, pivot]] = aug[[pivot, col]]
            for row in range(m):
                if row != col and aug[row, col] == 1:
                    aug[row] ^= aug[col]
        T_inv = aug[:, m:]
        
        return M, T_inv, rank

    def build_circuit(M, T_inv, rank, n):
        """Build the actual circuit from a basis."""
        gates = []
        wire_cache = {}

        def get_xor_wire(wire_set):
            if not wire_set:
                return n
            if len(wire_set) == 1:
                return wire_set[0] if isinstance(wire_set, list) else wire_set.pop()

            key = frozenset(wire_set)
            if key in wire_cache:
                return wire_cache[key]

            wire_list = sorted(wire_set) if isinstance(wire_set, set) else list(wire_set)

            if len(wire_list) <= 2:
                if len(wire_list) == 1:
                    result = wire_list[0]
                else:
                    g = len(gates)
                    gates.append([wire_list[0], wire_list[1]])
                    result = n + 1 + g
                wire_cache[key] = result
                return result

            mid = len(wire_list) // 2
            left_set = set(wire_list[:mid])
            right_set = set(wire_list[mid:])

            left = get_xor_wire(left_set)
            right = get_xor_wire(right_set)

            g = len(gates)
            gates.append([left, right])
            result = n + 1 + g
            wire_cache[key] = result
            return result

        # Build pivot wires
        pivot_wires = []
        for i in range(rank):
            ones = [j for j in range(n) if M[i, j] == 1]
            pivot_wires.append(get_xor_wire(ones))

        # Build outputs
        outputs = []
        for j in range(m):
            contributing = [i for i in range(rank) if T_inv[j, i] == 1]
            if not contributing:
                outputs.append(n)
            elif len(contributing) == 1:
                outputs.append(pivot_wires[contributing[0]])
            else:
                outputs.append(get_xor_wire([pivot_wires[i] for i in contributing]))

        return gates, outputs

    # First, compute standard RREF to get the rank
    M_std, T_inv_std, rank = compute_rref_with_pivot_order(A, list(range(n)))
    
    if rank == 0:
        return {'gates': [], 'outputs': [n] * m}

    # Generate candidate pivot orders
    # Strategy: try different orderings of columns
    candidate_orders = []
    
    # Standard left-to-right
    candidate_orders.append(list(range(n)))
    
    # Right-to-left
    candidate_orders.append(list(range(n-1, -1, -1)))
    
    # Weighted by column density (prefer columns with more 1s first)
    col_weights = np.sum(A, axis=0)
    weighted_order = sorted(range(n), key=lambda c: -col_weights[c])
    candidate_orders.append(weighted_order)
    
    # Weighted by column density (prefer columns with fewer 1s first)
    weighted_order_rev = sorted(range(n), key=lambda c: col_weights[c])
    candidate_orders.append(weighted_order_rev)
    
    # Random permutations (limited number)
    rng = np.random.default_rng(42)
    for _ in range(min(10, n)):  # Up to 10 random orders
        perm = rng.permutation(n)
        candidate_orders.append(list(perm))
    
    # Evaluate each candidate basis
    best_gates = None
    best_outputs = None
    best_gate_count = float('inf')
    
    for order in candidate_orders:
        if time.time() > deadline - 5.0:  # Leave time for building
            break
        
        M_cand, T_inv_cand, rank_cand = compute_rref_with_pivot_order(A, order)
        
        if rank_cand != rank:
            continue
        
        # Estimate gate count
        est_gates = count_gates_for_basis(M_cand, T_inv_cand, rank_cand)
        
        if est_gates < best_gate_count:
            # Build actual circuit to verify
            gates_cand, outputs_cand = build_circuit(M_cand, T_inv_cand, rank_cand, n)
            
            if len(gates_cand) < best_gate_count:
                best_gate_count = len(gates_cand)
                best_gates = gates_cand
                best_outputs = outputs_cand
    
    # Fallback to standard RREF if no better found
    if best_gates is None:
        best_gates, best_outputs = build_circuit(M_std, T_inv_std, rank, n)
    
    return {'gates': best_gates, 'outputs': best_outputs}
