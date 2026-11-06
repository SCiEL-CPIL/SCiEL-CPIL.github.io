import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Convert to numpy arrays for faster linear algebra
    # Matrix is m x n
    A = np.array(matrix, dtype=np.uint8)

    if not np.any(A):
        return {'gates': [], 'outputs': [n] * m}

    def _verify(circuit):
        gates = circuit['gates']
        outputs = circuit['outputs']
        num_wires = n + 1 + len(gates)
        # Use bitsets for speed in verification
        wire_val = [0] * num_wires
        for i in range(n):
            wire_val[i] = 1 << i
        wire_val[n] = 0

        for k, (a, b) in enumerate(gates):
            if a >= num_wires or b >= num_wires:
                return False
            wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]

        for i in range(m):
            # Reconstruct expected bitset for row i
            expected = 0
            for j in range(n):
                if A[i][j]:
                    expected |= (1 << j)
            if outputs[i] >= num_wires:
                return False
            if wire_val[outputs[i]] != expected:
                return False
        return True

    def _compute_depth(gates, outputs):
        if not gates:
            return 0
        num_wires = n + 1 + len(gates)
        wire_depth = [0] * num_wires
        for k, (a, b) in enumerate(gates):
            wire_depth[n + 1 + k] = max(wire_depth[a], wire_depth[b]) + 1
        max_depth = 0
        for w in outputs:
            if w < num_wires:
                max_depth = max(max_depth, wire_depth[w])
        return max_depth

    def _baseline_circuit():
        gates = []
        outputs = []
        next_wire = n + 1
        for i in range(m):
            row = A[i]
            if not np.any(row):
                outputs.append(n)
            else:
                indices = np.where(row == 1)[0].tolist()
                if len(indices) == 1:
                    outputs.append(indices[0])
                else:
                    prev = indices[0]
                    for idx in indices[1:]:
                        gates.append([prev, idx])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
        return {'gates': gates, 'outputs': outputs}

    # --- New Structural Algorithm: Minimum-Weight Basis Synthesis ---
    
    def _invert_gf2(M):
        """Invert a square matrix over GF(2). Returns None if singular."""
        n_m = len(M)
        Aug = [row[:] + [0]*n_m for row in M]
        for i in range(n_m):
            Aug[i][n_m + i] = 1
        
        for col in range(n_m):
            pivot_row = -1
            for row in range(col, n_m):
                if Aug[row][col] == 1:
                    pivot_row = row
                    break
            if pivot_row == -1:
                return None
            Aug[col], Aug[pivot_row] = Aug[pivot_row], Aug[col]
            for row in range(n_m):
                if row != col and Aug[row][col] == 1:
                    for c in range(2*n_m):
                        Aug[row][c] ^= Aug[col][c]
        return [row[n_m:] for row in Aug]

    def _solve_gf2(M, b):
        """Solve Mx = b over GF(2). M is k x n, b is m x n. Returns X (m x k) or None."""
        k = len(M)
        n = len(M[0]) if k > 0 else 0
        m = len(b)
        
        # We need to find X such that X @ M = B
        # This is equivalent to M^T @ X^T = B^T
        # Let's work with M^T (n x k) and B^T (n x m)
        MT = [[M[i][j] for i in range(k)] for j in range(n)]
        BT = [[b[i][j] for i in range(m)] for j in range(n)]
        
        # Augment MT with BT: [MT | BT] -> [I | X^T]
        # Dimensions: n x (k + m)
        Aug = []
        for i in range(n):
            row = MT[i][:] + BT[i][:]
            Aug.append(row)
            
        # Gauss-Jordan on MT part
        rank = 0
        for col in range(k):
            pivot_row = -1
            for row in range(rank, n):
                if Aug[row][col] == 1:
                    pivot_row = row
                    break
            if pivot_row == -1:
                continue
            Aug[rank], Aug[pivot_row] = Aug[pivot_row], Aug[rank]
            for row in range(n):
                if row != rank and Aug[row][col] == 1:
                    for c in range(k + m):
                        Aug[row][c] ^= Aug[rank][c]
            rank += 1
            
        if rank < k:
            return None
            
        # Check for consistency: if any row has 0s in MT part but 1s in BT part, no solution
        for row in range(rank, n):
            for c in range(k, k + m):
                if Aug[row][c] == 1:
                    return None
                    
        # Extract X^T from the right side of the pivot rows
        # X^T is m x k
        XT = [[0]*k for _ in range(m)]
        for i in range(k):
            for j in range(m):
                XT[j][i] = Aug[i][k + j]
                
        # Transpose XT to get X (m x k)
        X = [[XT[j][i] for j in range(m)] for i in range(k)]
        return X

    def _synthesize_with_basis(basis_rows, coeffs, time_budget):
        """
        Synthesizes circuit using a given basis and coefficients.
        Uses a greedy, weight-aware ordering for CSE.
        """
        k = len(basis_rows)
        if k == 0:
            return {'gates': [], 'outputs': [n]*m}, 0

        # 1. Synthesize Basis using Greedy CSE with Weight Heuristic
        # Represent basis rows as bitsets
        exprs = []
        for i in range(k):
            bits = 0
            for j in range(n):
                if basis_rows[i][j]:
                    bits |= (1 << j)
            exprs.append(bits)
        
        gates = []
        next_wire = n + 1
        pool = list(range(n + 1))
        wire_map = {} # Maps basis index to wire index
        
        # Iterative CSE:
        # Find pairs (a, b) in the current pool that appear together in many exprs.
        # Create a new wire g = a ^ b.
        # Replace (a, b) with g in exprs where both are present.
        # This reduces the number of terms in those exprs.
        
        # To be efficient, we limit iterations.
        max_iter = 50
        for _ in range(max_iter):
            if time.time() > time_budget:
                break
                
            # Calculate frequency of each wire in pool across all exprs
            freq = {}
            for w in pool:
                bit = 1 << w
                cnt = 0
                for e in exprs:
                    if e & bit:
                        cnt += 1
                if cnt >= 2:
                    freq[w] = cnt
            
            if not freq:
                break
                
            candidates = list(freq.keys())
            if len(candidates) < 2:
                break
                
            # Sort by frequency descending
            candidates.sort(key=lambda w: freq[w], reverse=True)
            top_k = min(len(candidates), 20)
            top_candidates = candidates[:top_k]
            
            best_pair = None
            best_benefit = 1 # Must save at least 1 gate to be worth it (2 gates -> 1 gate)
            
            # Evaluate pairs in top candidates
            for i in range(len(top_candidates)):
                a = top_candidates[i]
                a_bit = 1 << a
                for j in range(i + 1, len(top_candidates)):
                    b = top_candidates[j]
                    b_bit = 1 << b
                    benefit = 0
                    for e in exprs:
                        if (e & a_bit) and (e & b_bit):
                            benefit += 1
                    if benefit > best_benefit:
                        best_benefit = benefit
                        best_pair = (a, b)
                        
            if best_pair is None:
                break
                
            a, b = best_pair
            g = next_wire
            next_wire += 1
            gates.append([a, b])
            pool.append(g)
            
            a_bit = 1 << a
            b_bit = 1 << b
            g_bit = 1 << g
            
            for i in range(k):
                e = exprs[i]
                if (e & a_bit) and (e & b_bit):
                    e = (e & ~a_bit & ~b_bit) | g_bit
                    exprs[i] = e

        # 2. Synthesize remaining Basis Rows
        for i in range(k):
            e = exprs[i]
            if e == 0:
                wire_map[i] = n
            else:
                # Extract wires
                wires = []
                temp = e
                w = 0
                while temp:
                    if temp & 1:
                        wires.append(w)
                    temp >>= 1
                    w += 1
                if len(wires) == 1:
                    wire_map[i] = wires[0]
                else:
                    # XOR them sequentially
                    # Optimization: Sort wires by "cost" or just use as is?
                    # For now, sequential is fine, but we can try to balance.
                    # Let's just do sequential.
                    prev = wires[0]
                    for w in wires[1:]:
                        gates.append([prev, w])
                        prev = next_wire
                        next_wire += 1
                    wire_map[i] = prev
                    
        # 3. Synthesize Outputs
        gates_out = []
        outputs = []
        
        for i in range(m):
            c = coeffs[i]
            if not np.any(c):
                outputs.append(n)
            else:
                idxs = np.where(c == 1)[0].tolist()
                if len(idxs) == 1:
                    outputs.append(wire_map[idxs[0]])
                else:
                    prev = wire_map[idxs[0]]
                    for idx in idxs[1:]:
                        gates_out.append([prev, wire_map[idx]])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
        
        total_gates = len(gates) + len(gates_out)
        circuit = {'gates': gates + gates_out, 'outputs': outputs}
        return circuit, total_gates

    def _generate_basis_and_coeffs(A_mat, time_budget):
        """
        Generates a basis and coefficients for A_mat.
        Tries to find a basis with low total weight.
        """
        m_curr = A_mat.shape[0]
        n_curr = A_mat.shape[1]
        
        # Gaussian Elimination to find RREF and Rank
        B = A_mat.copy().astype(int)
        rank = 0
        pivot_cols = []
        
        for col in range(n_curr):
            # Find pivot
            pivot_row = -1
            for row in range(rank, m_curr):
                if B[row][col] == 1:
                    pivot_row = row
                    break
            if pivot_row == -1:
                continue
            
            # Swap
            B[rank], B[pivot_row] = B[pivot_row].copy(), B[rank].copy()
            pivot_cols.append(col)
            
            # Eliminate
            for row in range(m_curr):
                if row != rank and B[row][col] == 1:
                    B[row] = (B[row] + B[rank]) % 2
            
            rank += 1
            if rank == m_curr:
                break
        
        if rank == 0:
            return [], np.zeros((m_curr, 0), dtype=np.int8)
            
        basis_rows = B[0:rank].tolist()
        
        # Solve for coefficients
        # A = X @ B
        # We need X such that X @ B = A
        # B is rank x n_curr
        # A is m_curr x n_curr
        # X is m_curr x rank
        
        # We can solve this by picking 'rank' columns from B that are linearly independent.
        # The pivot columns of B are linearly independent.
        # Let P = pivot_cols.
        # B_P = B[:, P] (rank x rank)
        # A_P = A[:, P] (m_curr x rank)
        # X @ B_P = A_P
        # X = A_P @ B_P^{-1}
        
        B_P = B[0:rank, pivot_cols]
        A_P = A_mat[:, pivot_cols]
        
        # Invert B_P
        B_P_inv = _invert_gf2(B_P.tolist())
        if B_P_inv is None:
            return None, None
            
        # Compute X
        # X[i, j] = sum_k A_P[i, k] * B_P_inv[k, j]
        X = np.zeros((m_curr, rank), dtype=np.int8)
        for i in range(m_curr):
            for j in range(rank):
                s = 0
                for k in range(rank):
                    s ^= A_P[i, k] * B_P_inv[k][j]
                X[i, j] = s % 2
                
        return basis_rows, X

    # --- Main Logic ---
    
    best_result = _baseline_circuit()
    best_cost = len(best_result['gates'])
    
    # Strategy 1: Try a few random column permutations to find a "better" basis structure
    # The basis quality (total weight) depends on the column order in Gaussian Elimination.
    # Random permutations can yield lower-weight bases.
    
    num_tries = 10
    # Use a simple deterministic shuffle for reproducibility if needed, but random is fine
    # Let's use a fixed seed for stability in evaluation
    import random
    rng = random.Random(12345)
    
    for t in range(num_tries):
        if time.time() > deadline - 10.0:
            break
            
        # Random permutation of columns
        perm = list(range(n))
        rng.shuffle(perm)
        A_p = A[:, perm]
        
        # Generate basis in permuted space
        basis_p, coeffs = _generate_basis_and_coeffs(A_p, deadline - 10.0)
        if basis_p is None:
            continue
        
        # Map basis back to original column indices
        # basis_p is in permuted space.
        # If basis_p[i][j] = 1, it means column j in A_p is 1.
        # A_p[:, j] = A[:, perm[j]].
        # So the bit j in the permuted space corresponds to column perm[j] in the original space.
        adjusted_basis = []
        for row in basis_p:
            new_row = [0] * n
            for j, val in enumerate(row):
                if val:
                    new_row[perm[j]] = 1
            adjusted_basis.append(new_row)
            
        # Coeffs are the same because they are coefficients of the basis vectors, independent of column labeling.
        # Wait, coeffs X satisfies A = X @ B.
        # In permuted space: A_p = X @ B_p.
        # A_p = A @ P (where P is permutation matrix? No, A_p[:, j] = A[:, perm[j]]).
        # So A_p = A @ PermMatrix.
        # B_p = B @ PermMatrix.
        # A @ PermMatrix = X @ B @ PermMatrix.
        # A = X @ B.
        # So X is invariant.
        
        # Synthesize circuit
        circuit, cost = _synthesize_with_basis(adjusted_basis, coeffs, deadline - 5.0)
        
        if cost < best_cost:
            if _verify(circuit):
                d = _compute_depth(circuit['gates'], circuit['outputs'])
                if d <= depth_limit:
                    best_cost = cost
                    best_result = circuit

    return best_result
