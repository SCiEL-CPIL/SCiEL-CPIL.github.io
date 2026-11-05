"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix.
    Returns {'gates': [[wire, wire], ...], 'outputs': [wire, ...]}
    
    Strategy:
    1. Convert rows to bitmasks (integers) for fast XOR operations.
    2. Perform Gaussian elimination over GF(2) to find a basis.
    3. Express each output row as a linear combination of basis rows.
    4. Synthesize gates for the basis rows (sharing common sub-expressions where possible,
       though for now we just compute basis vectors directly).
    5. For each output, if it's a basis vector, point to that gate.
       If it's a combination, we need to XOR the corresponding basis gates.
       
    Wait, the interface requires gates to be pairwise XORs.
    If an output is a combination of k basis gates, we need k-1 additional gates to combine them,
    unless we can re-use existing gates.
    
    However, the "baseline" mentioned in the prompt is "independent-row baseline".
    The current program creates gates for each unique input pattern.
    
    Let's refine the current approach:
    The current program groups by exact input pattern.
    A better approach is to use the basis to reduce the number of *unique* linear forms we need to compute.
    
    Let R be the rank.
    We compute R basis vectors B_0, ..., B_{R-1}.
    Each B_i is a linear combination of inputs.
    We create gates to compute each B_i.
    
    Then each output row A_j is a linear combination of B_i's.
    Specifically, A_j = XOR_{i in S_j} B_i.
    
    If |S_j| == 0, output is 0.
    If |S_j| == 1, output is B_i.
    If |S_j| > 1, we need to compute XOR of these B_i's.
    
    Can we share these combination gates?
    Yes, if two outputs have the same set S_j, they can share the final combination gate.
    
    But can we share intermediate combination steps?
    This is the "circuit minimization" problem, which is hard.
    However, for small dimensions (8x8, 12x16, 16x16), we can be smarter.
    
    Let's look at the current program's weakness:
    It creates gates for every unique *input* pattern.
    If the matrix has low rank, many rows are dependent.
    Example: A = [[1,0], [0,1], [1,1]].
    Current:
    Row 0: input 0 -> output 0
    Row 1: input 1 -> output 1
    Row 2: inputs 0,1 -> gate XOR(0,1)
    Total gates: 1.
    
    Basis approach:
    Basis: [1,0], [0,1].
    Gates to compute basis:
    B_0 = input 0 (no gate needed, wire 0)
    B_1 = input 1 (no gate needed, wire 1)
    Output 2: XOR(B_0, B_1) -> gate XOR(0,1).
    Total gates: 1.
    
    Another example: A = [[1,1,0], [1,0,1], [0,1,1]].
    Current:
    Row 0: inputs 0,1 -> gate 0 (XOR 0,1)
    Row 1: inputs 0,2 -> gate 1 (XOR 0,2)
    Row 2: inputs 1,2 -> gate 2 (XOR 1,2)
    Total gates: 3.
    
    Basis approach:
    Rank is 3. Basis is the rows themselves (if independent).
    B_0 = 110, B_1 = 101, B_2 = 011.
    To compute B_0: XOR(0,1) -> Gate 0.
    To compute B_1: XOR(0,2) -> Gate 1.
    To compute B_2: XOR(1,2) -> Gate 2.
    Outputs map to these gates.
    Total gates: 3.
    
    Is there a case where basis approach is better?
    Consider A = [[1,1,0,0], [1,1,0,0], [0,0,1,1], [0,0,1,1]].
    Current:
    Row 0: inputs 0,1 -> Gate 0
    Row 1: same -> Gate 0
    Row 2: inputs 2,3 -> Gate 1
    Row 3: same -> Gate 1
    Total gates: 2.
    
    Basis:
    B_0 = 1100, B_1 = 0011.
    Compute B_0: XOR(0,1) -> Gate 0.
    Compute B_1: XOR(2,3) -> Gate 1.
    Outputs map to Gate 0, Gate 0, Gate 1, Gate 1.
    Total gates: 2.
    
    It seems the current "unique pattern" approach is already effectively doing this if the rows are identical.
    
    Where does the current approach fail?
    When rows are linearly dependent but not identical.
    Example: A = [[1,1,0], [0,1,1], [1,0,1]].
    Row 2 = Row 0 XOR Row 1.
    Current:
    Row 0: inputs 0,1 -> Gate 0 (XOR 0,1)
    Row 1: inputs 1,2 -> Gate 1 (XOR 1,2)
    Row 2: inputs 0,2 -> Gate 2 (XOR 0,2)
    Total gates: 3.
    
    Basis approach:
    Basis: B_0 = 110, B_1 = 011. (Row 0 and Row 1 are independent).
    Row 2 = B_0 XOR B_1.
    Compute B_0: Gate 0 (XOR 0,1).
    Compute B_1: Gate 1 (XOR 1,2).
    Compute Output 2: XOR(Gate 0, Gate 1) -> Gate 2.
    Total gates: 3.
    
    Still 3.
    
    What if we can share the computation of the basis vectors more efficiently?
    Or what if we choose a different basis?
    
    Actually, the minimal number of gates to compute a set of linear forms is related to the "linear complexity" or similar metrics.
    For small N, we can try to find a circuit that computes all required linear forms with minimal gates.
    
    Let's try a different strategy:
    1. Identify all unique non-trivial linear forms required (rows of A).
    2. Try to build a circuit that computes these forms.
    
    A simple improvement over the current code:
    The current code builds a tree for each unique pattern.
    If pattern P1 is a subset of P2, can we reuse?
    No, XOR is not monotonic in that sense.
    
    However, we can use the fact that if we have computed a wire W that represents a subset of inputs, and we need XOR of that subset plus more inputs, we can XOR W with the remaining inputs.
    
    Let's implement a more sophisticated sharing mechanism:
    We maintain a dictionary `wire_to_mask` mapping wire ID to the bitmask of inputs it represents.
    We start with inputs.
    We want to generate all required masks (rows of A).
    We can use a BFS or greedy approach to generate gates that produce new masks which are useful.
    
    Since N is small (<=16 inputs, <=16 outputs), the number of possible masks is 2^16 = 65536.
    We can do a shortest path or similar search in the space of masks?
    No, that's too slow if we do it naively.
    
    But we can use a heuristic:
    1. Start with available wires: inputs (masks 1<<i) and 0.
    2. Target wires: masks of rows of A.
    3. Repeatedly pick two available wires (masks m1, m2) such that m1 ^ m2 is a target mask or helps reach a target mask.
    
    Actually, a simpler and very effective method for small N:
    Gaussian elimination gives a basis.
    The number of gates to compute the basis is at most R * (avg weight - 1).
    Then the number of gates to combine basis elements for outputs is sum(|S_j| - 1).
    
    Total gates = Gates(Basis) + Gates(Combinations).
    
    Can we reduce Gates(Basis)?
    If we choose the basis to have minimum weight, we reduce the first term.
    Gaussian elimination with partial pivoting (choosing pivot with minimum weight) can help.
    
    Let's modify the Gaussian elimination to prefer pivots with fewer 1s.
    This minimizes the cost of computing the basis vectors.
    
    Also, for the combination step:
    If an output is a combination of k basis vectors, it costs k-1 gates.
    Can we share these combination gates?
    Yes, if the same combination of basis indices appears multiple times.
    
    Let's refine the algorithm:
    1. Perform Gaussian elimination to find a basis, preferring rows with minimum Hamming weight as pivots.
    2. Record the basis vectors and the representation of each original row in terms of the basis.
    3. Synthesize the basis vectors:
       - For each basis vector, if it's a single input, use that input wire.
       - If it's a combination of inputs, create gates to compute it.
       - To minimize gates for the basis itself, we can just use the standard tree method, but since the basis is reduced, the weights are often small.
    4. Synthesize the outputs:
       - For each output, look at its basis representation (a bitmask of basis indices).
       - If the representation is 0, output 0.
       - If it's a single bit, output the corresponding basis wire.
       - If it's multiple bits, we need to XOR the corresponding basis wires.
       - We can share these "combination" gates if the same basis-index pattern appears in multiple outputs.
    
    This should be at least as good as the current approach, and often better because:
    - Minimum weight basis reduces the cost of computing basis vectors.
    - Sharing combination gates based on basis representation is more likely to find duplicates than sharing based on input representation (since input representation is unique per row in many cases, but basis representation might have duplicates if rows are linearly dependent in specific ways? No, if rows are linearly dependent, their basis representation is unique. Wait.
    
    Let's re-evaluate.
    If Row A = Row B, they have the same input mask, so current code shares the gate.
    In basis representation, Row A and Row B have the same basis vector mask, so the new code also shares the gate.
    
    If Row C = Row A XOR Row B.
    Input masks: A, B, A^B. All distinct. Current code: 3 sets of gates.
    Basis masks: Suppose A=b0, B=b1. Then C=b0^b1.
    Basis gates: Compute b0, b1. (2 gates if weight 2).
    Combination gates: b0 (direct), b1 (direct), b0^b1 (1 gate).
    Total: 3 gates.
    
    It seems comparable.
    
    However, consider the case where the basis vectors themselves can be computed more cheaply than the original rows.
    Example:
    A = [[1,1,1,0], [1,1,0,1], [0,0,1,1], [1,1,1,1]]
    Weights: 3, 3, 2, 4.
    
    Current:
    Row 0: 3 inputs -> 2 gates.
    Row 1: 3 inputs -> 2 gates.
    Row 2: 2 inputs -> 1 gate.
    Row 3: 4 inputs -> 3 gates.
    Total: 8 gates. (Assuming no sharing, which there isn't).
    
    Basis approach:
    Let's find a min-weight basis.
    Row 2 (0011) is good. Weight 2.
    Row 0 (1110) ^ Row 2 (0011) = 1101. Weight 3.
    Row 1 (1101) is already in span?
    Let's do elimination.
    Pivot 1: Col 2 (Row 2: 0011).
    Eliminate Col 2 from others.
    Row 0: 1110 ^ 0011 = 1101.
    Row 1: 1101 ^ 0011 = 1110.
    Row 3: 1111 ^ 0011 = 1100.
    
    New rows:
    R0: 1101
    R1: 1110
    R2: 0011 (Pivot)
    R3: 1100
    
    Pivot 2: Col 0.
    R0 (1101) has 1. Weight 3.
    R1 (1110) has 1. Weight 3.
    R3 (1100) has 1. Weight 2.
    Choose R3 (1100) as pivot.
    Swap R3 to pos 1.
    Eliminate Col 0 from R0, R1.
    R0: 1101 ^ 1100 = 0001.
    R1: 1110 ^ 1100 = 0010.
    
    New rows:
    R0: 0001
    R1: 0010
    R2: 0011
    R3: 1100
    
    Pivot 3: Col 1.
    R1 (0010) has 1. Weight 1.
    Choose R1.
    
    New rows:
    R0: 0001
    R1: 0010 (Pivot)
    R2: 0011
    R3: 1100
    
    Eliminate Col 1 from others?
    R0: 0001 (no 1 in col 1)
    R2: 0011 ^ 0010 = 0001.
    R3: 1100 (no 1 in col 1)
    
    New rows:
    R0: 0001
    R1: 0010
    R2: 0001
    R3: 1100
    
    Pivot 4: Col 0.
    R0 (0001) no.
    R2 (0001) no.
    R3 (1100) yes. Weight 2.
    Choose R3.
    
    Basis:
    B0: 0001 (from R0)
    B1: 0010 (from R1)
    B2: 0011 (from R2? No, R2 became 0001 which is B0. So R2 is dependent).
    Wait, my manual elimination is getting messy.
    
    Let's just implement the code.
    
    Key improvements:
    1. Min-weight pivot selection in Gaussian elimination.
    2. Represent basis vectors as masks.
    3. Represent outputs as masks of basis indices.
    4. Build gates for basis vectors.
    5. Build gates for output combinations, sharing identical basis-index masks.
    
    This should strictly improve or match the current score.
    """
    start_time = time.time()
    deadline = start_time + 50.0
    
    A = np.array(matrix, dtype=np.uint8)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    n_rows, n_cols = A.shape
    
    # Convert rows to integer masks
    # Mask bit j is 1 if A[i, j] is 1
    row_masks = []
    for i in range(n_rows):
        mask = 0
        for j in range(n_cols):
            if A[i, j] == 1:
                mask |= (1 << j)
        row_masks.append(mask)
    
    # Gaussian Elimination to find basis and representations
    # We work with a list of (mask, original_index)
    # Actually, we need to track the transformation.
    
    # Let's use a standard GF(2) Gaussian elimination that tracks the basis.
    # basis[i] is the mask of the i-th basis vector.
    # repr[i] is the mask of basis indices that sum to row i.
    
    # To minimize gate count, we want basis vectors with low weight.
    # We'll perform elimination, but at each step, choose the pivot row with minimum weight
    # among those that have a 1 in the current pivot column.
    
    # Initialize
    # We have n_rows vectors.
    # We'll create a list of [mask, repr_mask]
    # repr_mask starts as 1 << i for row i.
    
    vectors = []
    for i in range(n_rows):
        vectors.append([row_masks[i], 1 << i])
    
    basis = []  # List of masks
    reprs = []  # List of repr masks corresponding to basis vectors
    # Actually, we want to express original rows in terms of the final basis.
    # Standard approach:
    # Perform elimination on the matrix of row_masks.
    # Keep track of how each row is formed from original rows.
    
    # Let's do it properly.
    # A_work is a list of [mask, coeff_mask]
    # coeff_mask represents which original rows were XORed to get this mask.
    
    A_work = [[row_masks[i], 1 << i] for i in range(n_rows)]
    
    r = 0
    pivot_cols = []
    
    for col in range(n_cols):
        # Find pivot
        pivot_idx = -1
        min_weight = n_cols + 1
        for i in range(r, n_rows):
            if (A_work[i][0] >> col) & 1:
                w = bin(A_work[i][0]).count('1')
                if w < min_weight:
                    min_weight = w
                    pivot_idx = i
        
        if pivot_idx == -1:
            continue
        
        # Swap
        A_work[r], A_work[pivot_idx] = A_work[pivot_idx], A_work[r]
        
        # Eliminate
        for i in range(n_rows):
            if i != r and ((A_work[i][0] >> col) & 1):
                A_work[i][0] ^= A_work[r][0]
                A_work[i][1] ^= A_work[r][1]
        
        pivot_cols.append(col)
        r += 1
    
    # r is the rank
    # The first r rows of A_work are the basis.
    # A_work[i][0] is the basis vector mask.
    # A_work[i][1] is the combination of original rows that forms this basis vector.
    # This is the reverse of what we want. We want to express original rows in terms of basis.
    
    # To get the representation of original rows in terms of basis:
    # The matrix T such that Basis = T * Original is formed by the coeff_masks of the basis rows.
    # We need Original = T^{-1} * Basis.
    # Since we did elimination, T is lower triangular (if we order correctly) or can be inverted.
    
    # Alternative: Just re-express each original row mask using the basis vectors.
    # For each original row mask M, find basis vectors B_i such that M = XOR B_i.
    # This is easy because basis vectors have unique pivot columns.
    
    basis_vectors = [A_work[i][0] for i in range(r)]
    pivot_cols_basis = pivot_cols
    
    # Map pivot col to basis index
    pivot_to_basis_idx = {col: i for i, col in enumerate(pivot_cols_basis)}
    
    # For each original row, find its basis representation
    row_basis_reprs = []
    for i in range(n_rows):
        M = row_masks[i]
        repr_mask = 0
        temp_M = M
        for col in pivot_cols_basis:
            if (temp_M >> col) & 1:
                b_idx = pivot_to_basis_idx[col]
                repr_mask |= (1 << b_idx)
                temp_M ^= basis_vectors[b_idx]
        # temp_M should be 0
        row_basis_reprs.append(repr_mask)
    
    # Now we have:
    # basis_vectors: list of r masks
    # row_basis_reprs: list of n_rows masks (each mask has r bits)
    
    # Synthesize gates
    # 1. Compute gates for basis vectors
    # Wire n_cols is 0.
    # Wires 0..n_cols-1 are inputs.
    
    gates = []
    gate_to_mask = {} # wire_id -> mask of inputs represented
    
    # Initialize wire_to_mask for inputs and zero
    for i in range(n_cols):
        gate_to_mask[i] = 1 << i
    gate_to_mask[n_cols] = 0
    
    basis_wire_ids = []
    
    # Compute basis vectors
    for b_idx in range(r):
        bv_mask = basis_vectors[b_idx]
        if bv_mask == 0:
            basis_wire_ids.append(n_cols)
            continue
        
        # Check if this mask is already available
        # (It shouldn't be, unless it's an input or zero, but basis vectors are independent)
        # If bv_mask is a single bit, it's an input.
        if bv_mask & (bv_mask - 1) == 0:
            # Single bit
            wire_id = int(np.log2(bv_mask))
            basis_wire_ids.append(wire_id)
            continue
        
        # Need to compute it
        # Simple strategy: XOR all set bits
        # We can do this more efficiently by reusing existing gates if possible,
        # but for now, let's just build a tree.
        # Optimization: If we have a gate that represents a subset of bv_mask, use it.
        # This is complex. Let's stick to simple tree for basis vectors first.
        # Actually, we can check if any existing wire matches a subset.
        
        # For simplicity and correctness, build tree from inputs.
        # But we can be smart: if bv_mask is small, just XOR inputs.
        
        # Let's try to find if any existing gate (other than inputs) helps.
        # Given the small size, a simple greedy is fine.
        
        # Get list of input indices
        inputs_needed = []
        for j in range(n_cols):
            if (bv_mask >> j) & 1:
                inputs_needed.append(j)
        
        if len(inputs_needed) == 0:
            basis_wire_ids.append(n_cols)
            continue
        elif len(inputs_needed) == 1:
            basis_wire_ids.append(inputs_needed[0])
            continue
        else:
            # Build tree
            current_wire = inputs_needed[0]
            for w in inputs_needed[1:]:
                # Check if we already have a gate for current_wire ^ w
                target_mask = gate_to_mask[current_wire] ^ gate_to_mask[w]
                # Search for existing wire with this mask
                found = False
                for wid, m in gate_to_mask.items():
                    if m == target_mask:
                        current_wire = wid
                        found = True
                        break
                if not found:
                    gate_id = n_cols + 1 + len(gates)
                    gates.append([current_wire, w])
                    gate_to_mask[gate_id] = target_mask
                    current_wire = gate_id
            
            basis_wire_ids.append(current_wire)
    
    # 2. Compute gates for outputs
    outputs = []
    
    # Map basis-repr mask to wire id
    # We need to compute XOR of basis_wire_ids for each output's repr.
    # We can share these if the repr mask is the same.
    
    repr_to_wire = {0: n_cols} # 0 repr means output 0
    
    for i in range(n_rows):
        repr_mask = row_basis_reprs[i]
        
        if repr_mask in repr_to_wire:
            outputs.append(repr_to_wire[repr_mask])
            continue
        
        # Need to compute this combination
        # Find which basis indices are set
        active_basis = []
        for b_idx in range(r):
            if (repr_mask >> b_idx) & 1:
                active_basis.append(b_idx)
        
        if len(active_basis) == 0:
            outputs.append(n_cols)
            repr_to_wire[repr_mask] = n_cols
            continue
        
        if len(active_basis) == 1:
            w = basis_wire_ids[active_basis[0]]
            outputs.append(w)
            repr_to_wire[repr_mask] = w
            continue
        
        # Multiple basis vectors
        current_wire = basis_wire_ids[active_basis[0]]
        for b_idx in active_basis[1:]:
            w = basis_wire_ids[b_idx]
            target_mask = gate_to_mask[current_wire] ^ gate_to_mask[w]
            
            # Check if this mask (in terms of inputs) is already available
            # Note: target_mask here is the input mask, not the basis repr mask.
            # We should check if we have a gate that produces this input mask.
            found = False
            for wid, m in gate_to_mask.items():
                if m == target_mask:
                    current_wire = wid
                    found = True
                    break
            
            if not found:
                gate_id = n_cols + 1 + len(gates)
                gates.append([current_wire, w])
                gate_to_mask[gate_id] = target_mask
                current_wire = gate_id
        
        outputs.append(current_wire)
        repr_to_wire[repr_mask] = current_wire
    
    result = {'gates': gates, 'outputs': outputs}
    
    # Verify
    if not _verify(A, result):
        # Fallback to simple direct synthesis
        gates = []
        outputs = []
        for i in range(n_rows):
            input_indices = [j for j in range(n_cols) if A[i, j] == 1]
            if len(input_indices) == 0:
                outputs.append(n_cols)
            elif len(input_indices) == 1:
                outputs.append(input_indices[0])
            else:
                wires = input_indices
                current_wire = wires[0]
                for w in wires[1:]:
                    gate_id = n_cols + 1 + len(gates)
                    gates.append([current_wire, w])
                    current_wire = gate_id
                outputs.append(current_wire)
        result = {'gates': gates, 'outputs': outputs}
    
    return result


def _verify(A, result):
    """Verify that the circuit computes the correct outputs."""
    n_cols = A.shape[1]
    gates = result['gates']
    outputs = result['outputs']
    
    # Use bitmasks for verification
    if n_cols > 64:
        return True
    
    wire_values = [1 << i for i in range(n_cols)]
    wire_values.append(0)  # zero wire
    
    for gate in gates:
        w1, w2 = gate
        if w1 >= len(wire_values) or w2 >= len(wire_values):
            return False
        wire_values.append(wire_values[w1] ^ wire_values[w2])
    
    for i, out_wire in enumerate(outputs):
        if out_wire >= len(wire_values):
            return False
        expected = 0
        for j in range(n_cols):
            if A[i, j] == 1:
                expected ^= (1 << j)
        if wire_values[out_wire] != expected:
            return False
    
    return True
# EVOLVE-BLOCK-END
