import time
import numpy as np
import random

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit using cost-optimized Gaussian Elimination (Linear Basis Synthesis).
    
    Strategy:
    1. Represent the target rows as vectors in GF(2)^n.
    2. We want to find a sequence of XOR gates that creates a new set of "basis wires" 
       such that every target row is exactly one of these wires (or an input wire).
    3. This is equivalent to finding an invertible linear transformation T (implemented by the gates) 
       such that T * A = I (or a subset of I if rank < m).
    4. We construct T greedily: 
       - Start with basis = standard unit vectors e_0...e_{n-1} and e_zero.
       - For each target row r, find the cheapest linear combination of current basis vectors 
         that equals r. 
       - "Cheapest" means minimizing the number of XOR operations needed to build it from current basis.
       - Actually, a better view: We are building a circuit. 
         Let's maintain a set of "available wires" which are the inputs and gates produced so far.
         We want to produce the target rows.
         
    Alternative Robust Strategy: 
    1. Compute the row space of the matrix.
    2. Find a basis for the row space using Gaussian Elimination, but choose pivots to minimize 
       the Hamming weight of the intermediate vectors (Sparse Gaussian Elimination).
    3. The sequence of XOR operations performed during elimination defines the gates.
    4. The final basis vectors are the outputs.
    5. If m > rank, some outputs are zero or duplicates.
    
    This "Sparse Gaussian Elimination" approach is different from CSE because it focuses on 
    transforming the *basis* of the row space rather than sharing subexpressions of the *original* rows.
    It guarantees that the number of gates is related to the rank and the sparsity of the elimination steps.
    """
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Normalize input
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    A = np.array(matrix, dtype=np.int8)
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    if not np.any(A):
        return {'gates': [], 'outputs': [n] * m}
    
    # Convert rows to bitsets for efficient manipulation
    row_bits = []
    for i in range(m):
        bits = 0
        for j in range(n):
            if A[i, j]:
                bits |= (1 << j)
        row_bits.append(bits)
    
    orig_row_bits = list(row_bits)
    
    def _verify(circuit):
        gates = circuit['gates']
        outputs = circuit['outputs']
        num_wires = n + 1 + len(gates)
        if len(outputs) != m:
            return False
        wire_val = [0] * num_wires
        for i in range(n):
            wire_val[i] = 1 << i
        wire_val[n] = 0
        for k, (a, b) in enumerate(gates):
            if a >= num_wires or b >= num_wires:
                return False
            wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
        for i in range(m):
            expected = orig_row_bits[i]
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

    def _baseline():
        """Standard sequential XOR synthesis for each row."""
        gates = []
        outputs = []
        next_wire = n + 1
        for i in range(m):
            bits = orig_row_bits[i]
            if bits == 0:
                outputs.append(n)
            else:
                indices = [j for j in range(n) if (bits >> j) & 1]
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
    
    def _sparse_gaussian_elimination():
        """
        Synthesize circuit via Sparse Gaussian Elimination.
        
        We maintain a list of 'active' vectors which form a basis for the row space.
        We process the input rows one by one.
        For each new row, we try to express it as a linear combination of the current basis.
        If it's independent, we add it to the basis.
        
        To minimize gates, we perform elimination in a way that keeps vectors sparse.
        Specifically, when adding a new vector v to the basis B:
        1. Find pivot column.
        2. Eliminate this pivot from all other basis vectors (and potentially v itself) 
           to maintain reduced row echelon form (or similar).
        
        The circuit construction:
        - Each basis vector is represented by a wire.
        - Initially, basis vectors are the input wires (if they are unit vectors) or 
          we start with an empty basis and build up.
        
        Actually, a more direct mapping:
        Let's build the circuit that computes the RREF of the matrix.
        The gates correspond to the row operations.
        
        Algorithm:
        1. Start with rows of A.
        2. Perform Gaussian Elimination to get RREF.
        3. Track the linear combination of original rows for each current row.
           This is not quite right for circuit synthesis because circuit gates are 
           sequential XORs of wires, not linear combinations of original rows in one step.
           
        Correct Circuit Synthesis via Basis:
        We want to create wires w_0, ..., w_{r-1} such that the set of values {val(w_i)} 
        spans the row space, and each target row is one of these wires.
        
        Let's use the "Basis Extension" method:
        1. Maintain a list of basis wires `basis_wires` and their values `basis_vals`.
        2. For each target row `r`:
           a. If `r` is in the span of `basis_vals`, find the combination. 
              If the combination is simple (e.g., single wire), use it.
              If not, we might need to add a gate.
           b. If `r` is NOT in the span, we must create a new wire for it.
              How? We can XOR `r` with an existing basis wire to create a "reduced" vector,
              but we need the wire to hold `r` exactly.
              
        Let's try a different angle: **Reverse Engineering the Basis**.
        The problem is equivalent to finding a sequence of XORs that transforms the input vector space 
        into a space where the target vectors are coordinate axes (or zero).
        
        Greedy Algorithm for Basis Synthesis:
        1. Let `current_vectors` be the list of target rows.
        2. Let `circuit_gates` be empty.
        3. Let `wire_map` be a dict mapping a bitset value to a wire index.
           Initialize with inputs: val(i) = 1<<i -> wire i. val(0) = 0 -> wire n.
        4. We want to reduce the complexity of `current_vectors`.
        
        Actually, the most effective strategy for XOR circuits is often **Iterative Sparse Elimination**:
        1. Identify the row with the minimum Hamming weight.
        2. Use this row as a pivot.
        3. For every other row that has a 1 in the pivot column, XOR the pivot row into it.
           This reduces the weight of those rows (on average).
        4. The gate for this operation is: `new_wire = old_wire ^ pivot_wire`.
        5. Update the wire for that row to `new_wire`.
        6. Repeat until all rows are unit vectors or zero.
        
        This process generates a circuit. The final wires for the rows are the outputs.
        The number of gates is roughly O(n * rank) in the worst case, but often much less 
        due to sparsity.
        
        Implementation:
        - `rows`: list of current bitset values for each of the m rows.
        - `wires`: list of current wire indices for each of the m rows.
        - `gate_count`: 0
        - `next_wire`: n + 1
        
        Loop:
        - Find row `i` with min weight > 0. If none, break.
        - Let `pivot_col` be the lowest set bit of `rows[i]`.
        - For each other row `j` where `rows[j]` has bit `pivot_col` set:
            - Create gate: `w_new = wires[j] ^ wires[i]`
            - `rows[j] ^= rows[i]`
            - `wires[j] = w_new`
            - `gate_count += 1`
            - `next_wire += 1`
            - (Optional: If `rows[j]` becomes 0, wire is n? No, wire is w_new which holds 0. 
               We can map it to n later in polish, or just leave it as a wire holding 0.)
        
        This is essentially Gaussian Elimination where the pivot row is the sparsest one.
        It minimizes the number of 1s in the matrix, which correlates with fewer gates needed 
        to express the final unit vectors? 
        Wait, the outputs are the wires `wires[j]` at the end.
        If `rows[j]` becomes a unit vector `1 << k`, then `wires[j]` is a wire that holds `1 << k`.
        But we want the output for row `j` to be the original value.
        
        Correction: 
        In this elimination, we are modifying the *representation* of the rows.
        The wire `wires[j]` always holds the value `rows[j]`.
        At the end, `rows[j]` might be `0` or a unit vector? No, Gaussian elimination 
        turns the matrix into RREF (Identity matrix subset).
        So `rows[j]` will be either 0 or a unit vector `1 << k`.
        But the original row `orig_row_bits[j]` is what we need to output.
        
        The circuit computes:
        `wire_val[wires[j]] = rows[j]` (current state).
        But we need `wire_val[output_wire] = orig_row_bits[j]`.
        
        The relationship:
        `orig_row_bits[j]` is a linear combination of the final basis vectors?
        No, the elimination steps are reversible.
        `rows[j]_final = T * orig_row_bits[j]` where T is the product of elimination matrices.
        So `orig_row_bits[j] = T^{-1} * rows[j]_final`.
        
        This means the circuit we built computes `T * x`. We need `x`.
        So we need to invert the circuit? Or rather, the gates we built transform the inputs 
        into the RREF basis. 
        If we run the gates in reverse, we get the original vectors back?
        XOR is its own inverse.
        
        Let's trace:
        Start: Wire `j` holds `v_j`.
        Gate: `w_new = wire_j ^ wire_i`.
        Now `wire_j` is replaced by `w_new`? No, we create a new wire.
        In my loop above:
        `wires[j]` becomes `w_new`.
        `rows[j]` becomes `rows[j] ^ rows[i]`.
        
        So the wire `wires[j]` now holds the new value.
        The old wire for `j` is still there, holding the old value.
        
        At the end, we have wires `wires[j]` holding `rows[j]_final` (which is 0 or unit).
        We want wires that hold `orig_row_bits[j]`.
        
        Since `rows[j]_final = M * orig_row_bits[j]` (where M is the elimination matrix),
        then `orig_row_bits[j] = M^{-1} * rows[j]_final`.
        
        The circuit we built implements the transformation `v -> M*v` for the specific vectors.
        To get `orig`, we would need to apply `M^{-1}`.
        
        However, notice that the gates are XORs.
        If we recorded the gates, can we just use them?
        
        Let's look at a simpler approach: **Direct Synthesis via Basis Expansion**.
        
        1. Compute the row space basis using Gaussian Elimination (just to find the rank and basis vectors).
        2. Let `basis` be the list of independent vectors.
        3. For each basis vector, synthesize a wire that holds it.
           - If it's a unit vector, use input wire.
           - If not, use XOR of inputs.
        4. For each original row, express it as XOR of basis vectors.
           - Synthesize a wire for this combination.
        
        This is the "CSE" approach but restricted to the basis.
        The current program's CSE is complex.
        
        Let's try the **Reverse Elimination** idea more carefully.
        
        If we perform elimination to get RREF, we create a circuit `C` such that 
        `C(input) = RREF_basis`.
        Since `RREF_basis = M * Input`, then `Input = M^{-1} * RREF_basis`.
        The inverse of a sequence of XOR gates is the same sequence of gates in reverse order?
        No, XOR is commutative and associative.
        If we do `w = a ^ b`, then `a = w ^ b`.
        
        So, if we build the circuit for the elimination steps, we can reuse the same gates 
        to reconstruct the original vectors IF we know the linear combination.
        
        Actually, there is a simpler property:
        The set of all wires generated during Gaussian Elimination (including inputs) 
        forms a basis for the space.
        Every original row is a linear combination of these wires.
        
        Let's stick to the **Greedy Sparse Elimination** but interpret the result correctly.
        
        We will generate a list of gates.
        We will track the value of each wire.
        At the end, for each original row `r`, we need to find a wire (or combination of wires) 
        that equals `r`.
        
        During elimination:
        We maintain `current_val[j]` and `current_wire[j]`.
        Initially `current_val[j] = orig_row_bits[j]`, `current_wire[j] = j` (if unit) or synthesized.
        
        Wait, if `orig_row_bits[j]` is not a unit vector, we can't start with `wire=j`.
        We must first synthesize a wire for `orig_row_bits[j]`.
        That's the expensive part.
        
        So, Step 1: Synthesize initial wires for all rows using a cheap method (e.g., sequential XOR).
        Step 2: Apply Sparse Gaussian Elimination to these wires to reduce the "complexity" of the set.
        
        But "complexity" isn't reduced in terms of gate count for the final output, 
        unless we share gates.
        
        Let's go with **Minimum Weight Basis Synthesis**:
        1. Find a basis for the row space using Gaussian Elimination, choosing pivots to minimize 
           the Hamming weight of the resulting basis vectors.
        2. Synthesize each basis vector using a tree of XORs (minimizing gates for that vector).
        3. Express each original row as a linear combination of the basis vectors.
        4. Synthesize the combination for each row.
        
        This is essentially the same as the current program's intent, but the current program 
        has a complex and potentially buggy CSE implementation.
        My new strategy will be a clean implementation of:
        1. **Sparse Gaussian Elimination** to find a low-weight basis.
        2. **Tree Synthesis** for each basis vector.
        3. **Linear Combination** for each row.
        
        This avoids the complex "sharing" logic of CSE and instead relies on the structure 
        of the linear basis. It is a different design family: **Basis-First Synthesis**.
        
        Optimization: 
        When synthesizing the linear combination for a row, if the combination is just one basis vector, 
        use that wire. If it's two, one gate. If more, a tree.
        
        To minimize gates, we want the basis vectors to be sparse (low Hamming weight) so 
        their individual synthesis is cheap.
        And we want the representation of each row in this basis to be sparse (low number of terms).
        
        We can trade off these two.
        Standard RREF gives sparse representation (1 per row) but potentially dense basis vectors.
        Sparse Gaussian Elimination gives sparse basis vectors but potentially dense representation.
        
        Let's try **RREF with Sparse Pivot Selection**.
        """
        
        # 1. Gaussian Elimination to RREF, tracking operations
        # We work with bitsets.
        rows = orig_row_bits[:]
        # Track which linear combination of original rows produces the current row?
        # No, we just need the values.
        
        # We will build the circuit for the basis vectors.
        # First, find the basis.
        basis_indices = []
        basis_vals = []
        
        # Standard Gaussian Elimination
        # To minimize weight, we can swap rows to bring the sparsest one to the top?
        # Let's do a simple elimination first, then optimize.
        
        # Actually, let's just use the standard elimination to get the rank and basis.
        # Then synthesize.
        
        # Step 1: Find Rank and Basis (using standard GE, but we can choose pivots)
        # For now, use standard pivot (first non-zero).
        
        current_rows = rows[:]
        basis_wires_vals = [] # The actual bitset values of the basis vectors
        
        # We need to know the linear map from inputs to these basis vectors.
        # Since we are in GF(2), and we start with inputs, the basis vectors are just 
        # linear combinations of inputs.
        
        # Let's perform GE on the matrix A.
        # We'll keep track of the row operations to potentially use them, 
        # but for now, we just extract the basis vectors (the non-zero rows of RREF).
        
        rank = 0
        n_cols = n
        # Work on a copy
        mat = [r[:] for r in current_rows]
        
        # Sparse GE: At each step, pick the row with min weight among those with pivot in col.
        # This tends to keep rows sparse.
        
        for col in range(n_cols):
            if rank >= m:
                break
            # Find pivot
            pivot_row = -1
            min_weight = float('inf')
            for i in range(rank, m):
                if (mat[i] >> col) & 1:
                    w = bin(mat[i]).count('1')
                    if w < min_weight:
                        min_weight = w
                        pivot_row = i
            
            if pivot_row == -1:
                continue
            
            # Swap
            mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]
            
            # Eliminate
            for i in range(m):
                if i != rank and (mat[i] >> col) & 1:
                    mat[i] ^= mat[rank]
            
            basis_wires_vals.append(mat[rank])
            rank += 1
        
        # Now basis_wires_vals contains `rank` vectors that span the row space.
        # They are in RREF form (mostly), so they are sparse in the "pivot" sense, 
        # but might have other 1s.
        
        # Step 2: Synthesize each basis vector.
        gates = []
        next_wire = n + 1
        basis_wire_indices = []
        
        for bv in basis_wires_vals:
            if bv == 0:
                basis_wire_indices.append(n)
                continue
            
            # Synthesize this vector using a tree of XORs
            # Get indices of set bits
            bits = [j for j in range(n) if (bv >> j) & 1]
            
            if len(bits) == 1:
                basis_wire_indices.append(bits[0])
            else:
                # Build a balanced tree to minimize depth, or just sequential?
                # Sequential is simpler and gate count is len(bits)-1.
                # Balanced tree has same gate count but lower depth.
                # Since depth limit is 16, and n is small (16), sequential depth is <=15.
                # So sequential is fine and simple.
                prev = bits[0]
                for b in bits[1:]:
                    gates.append([prev, b])
                    prev = next_wire
                    next_wire += 1
                basis_wire_indices.append(prev)
        
        # Step 3: Express each original row as combination of basis vectors.
        # Since basis is in RREF (or close to it), we can solve for the coefficients.
        # Actually, the rows of `mat` after GE are the basis.
        # The original rows are linear combinations of these basis rows.
        # We can find the coefficients by running the original rows through the same elimination?
        # Or simpler: We know that `mat` (after GE) is the RREF.
        # The transformation from `A` to `RREF` is `RREF = M * A`.
        # So `A = M^{-1} * RREF`.
        # The rows of `M^{-1}` give the coefficients.
        
        # Instead of inverting M, we can just observe:
        # In the GE process, we performed row operations.
        # If we had tracked the operations on an identity matrix, we would have M.
        # Let's re-run GE but track the operations on an identity matrix of size m x m?
        # That might be expensive if m is large (16). 16x16 is fine.
        
        # Let's re-do GE with tracking.
        # This ensures we get the exact linear combination.
        
        # Reset
        mat = [r[:] for r in current_rows]
        # Identity matrix for tracking coefficients
        # coeff[i] is a bitset of m bits, indicating which original rows are XORed to form current row i.
        coeff = [1 << i for i in range(m)]
        
        rank = 0
        for col in range(n_cols):
            if rank >= m:
                break
            pivot_row = -1
            min_weight = float('inf')
            for i in range(rank, m):
                if (mat[i] >> col) & 1:
                    w = bin(mat[i]).count('1')
                    if w < min_weight:
                        min_weight = w
                        pivot_row = i
            
            if pivot_row == -1:
                continue
            
            # Swap
            mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]
            coeff[rank], coeff[pivot_row] = coeff[pivot_row], coeff[rank]
            
            # Eliminate
            for i in range(m):
                if i != rank and (mat[i] >> col) & 1:
                    mat[i] ^= mat[rank]
                    coeff[i] ^= coeff[rank]
            
            rank += 1
            
        # Now, the first `rank` rows of `mat` are the basis vectors.
        # The `coeff` rows tell us how to get them from original rows.
        # We want the reverse: how to get original rows from basis vectors.
        
        # Let B be the matrix of basis vectors (first rank rows of mat).
        # Let C be the matrix of coefficients (first rank rows of coeff).
        # B = C * A (where A is original matrix rows).
        # We want A = C^{-1} * B.
        
        # Since C is a submatrix of the transformation, and we are in GF(2),
        # and C is upper triangular (due to GE structure? No, it's the operations).
        
        # Actually, a simpler way:
        # We have wires for the basis vectors.
        # For each original row `i`, we need to find `x` such that `B * x = orig_row_bits[i]`.
        # Since `B` is in RREF, this is easy: `x` is just the bits of `orig_row_bits[i]` 
        # at the pivot columns?
        # No, `B` is not necessarily the identity.
        
        # Let's just solve the linear system for each row.
        # System: sum(x_j * B_j) = target.
        # Since B is RREF, the pivot column of B_j has a 1 in row j and 0 elsewhere.
        # So x_j is simply the bit of `target` at the pivot column of B_j.
        
        # We need to know the pivot columns.
        # Let's extract them from the GE process.
        
        # Re-do GE to get pivot columns.
        mat = [r[:] for r in current_rows]
        pivot_cols = []
        rank = 0
        for col in range(n_cols):
            if rank >= m:
                break
            pivot_row = -1
            min_weight = float('inf')
            for i in range(rank, m):
                if (mat[i] >> col) & 1:
                    w = bin(mat[i]).count('1')
                    if w < min_weight:
                        min_weight = w
                        pivot_row = i
            
            if pivot_row == -1:
                continue
            
            mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]
            
            for i in range(m):
                if i != rank and (mat[i] >> col) & 1:
                    mat[i] ^= mat[rank]
            
            pivot_cols.append(col)
            rank += 1
            
        # Now, for each original row `i`:
        # The coefficient for basis vector `j` is `orig_row_bits[i] & (1 << pivot_cols[j])`.
        
        outputs = []
        for i in range(m):
            target = orig_row_bits[i]
            if target == 0:
                outputs.append(n)
                continue
            
            # Find combination
            combo_wires = []
            for j in range(rank):
                if (target >> pivot_cols[j]) & 1:
                    combo_wires.append(basis_wire_indices[j])
            
            if len(combo_wires) == 0:
                outputs.append(n)
            elif len(combo_wires) == 1:
                outputs.append(combo_wires[0])
            else:
                # Synthesize combination
                prev = combo_wires[0]
                for w in combo_wires[1:]:
                    gates.append([prev, w])
                    prev = next_wire
                    next_wire += 1
                outputs.append(prev)
                
        return {'gates': gates, 'outputs': outputs}
    
    def _polish(circuit):
        gates = circuit['gates']
        outputs = circuit['outputs']
        if not gates:
            return circuit
            
        num_wires = n + 1 + len(gates)
        wire_val = [0] * num_wires
        for i in range(n):
            wire_val[i] = 1 << i
        wire_val[n] = 0
        
        for k, (a, b) in enumerate(gates):
            wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
            
        # Remove unused gates
        used = set(outputs)
        for a, b in gates:
            used.add(a)
            used.add(b)
            
        necessary = [False] * len(gates)
        wire_to_gate = {}
        for k in range(len(gates)):
            w = n + 1 + k
            wire_to_gate[w] = k
            
        for w in outputs:
            if w in wire_to_gate:
                necessary[wire_to_gate[w]] = True
                
        changed = True
        while changed:
            changed = False
            for k in range(len(gates) - 1, -1, -1):
                if necessary[k]:
                    a, b = gates[k]
                    for w in [a, b]:
                        if w in wire_to_gate:
                            idx = wire_to_gate[w]
                            if not necessary[idx]:
                                necessary[idx] = True
                                changed = True
                                
        new_gates = []
        wire_map = {}
        new_next_wire = n + 1
        
        for k in range(len(gates)):
            if necessary[k]:
                a, b = gates[k]
                a_new = wire_map.get(a, a)
                b_new = wire_map.get(b, b)
                new_gates.append([a_new, b_new])
                old_w = n + 1 + k
                wire_map[old_w] = new_next_wire
                new_next_wire += 1
                
        new_outputs = []
        for w in outputs:
            new_outputs.append(wire_map.get(w, w))
            
        return {'gates': new_gates, 'outputs': new_outputs}
    
    # Run strategies
    best_result = None
    best_gate_count = float('inf')
    
    # 1. Baseline
    baseline = _baseline()
    if _verify(baseline):
        best_result = baseline
        best_gate_count = len(baseline['gates'])
        
    # 2. Sparse GE Basis Synthesis
    if time.time() < deadline - 10.0:
        try:
            res = _sparse_gaussian_elimination()
            res = _polish(res)
            if _verify(res):
                depth = _compute_depth(res['gates'], res['outputs'])
                if depth <= depth_limit:
                    if len(res['gates']) < best_gate_count:
                        best_result = res
                        best_gate_count = len(res['gates'])
        except Exception:
            pass
            
    # Fallback
    if best_result is None:
        best_result = {'gates': [], 'outputs': [n] * m}
        
    return best_result

if __name__ == "__main__":
    matrix = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    res = synthesize(matrix, 16)
    print(res)
