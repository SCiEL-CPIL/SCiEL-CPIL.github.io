# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict, deque

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit using a structured approach:
    1. Gaussian Elimination to find a minimal basis (canonical form) for the column space.
    2. A recursive, memoized construction of the basis vectors from inputs, which naturally
       shares sub-expressions (common sub-circuits) across different basis vectors.
    3. Express each output row as a linear combination of the basis vectors.
    4. Construct the final circuit by building the basis circuit first, then the output combinators.
    
    This approach is superior to greedy CSE or beam search because it exploits the linear structure
    of GF(2). The basis vectors are a minimal set of generators for the space spanned by the inputs
    that are actually used. The recursive construction ensures that any sub-XOR required by multiple
    basis vectors is computed only once.
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin for 90s limit

    mat = np.array(matrix, dtype=np.int8)
    n_outputs, n_inputs = mat.shape

    if n_outputs == 0 or n_inputs == 0:
        return {'gates': [], 'outputs': [n_inputs] * n_outputs}

    # Step 1: Gaussian Elimination to find a basis for the row space of the matrix.
    # We operate on the rows of the matrix.
    # We want to find a set of basis vectors such that every row is a linear combination of them.
    # We will store the basis vectors as bitmasks.
    
    # We will perform elimination on the columns (variables).
    # basis[j] will store the row (as a bitmask of length n_inputs) that has a leading 1 in column j.
    # This is standard Gaussian elimination over GF(2).
    
    # Convert rows to bitmasks for efficient XOR
    def row_to_mask(row):
        m = 0
        for j, v in enumerate(row):
            if v:
                m |= (1 << j)
        return m

    input_masks = [row_to_mask(mat[i]) for i in range(n_outputs)]
    
    # We want to find a basis for the span of the input rows.
    # Let's find the basis of the row space.
    # We iterate through the input rows and add them to the basis if they are independent.
    # To make it efficient, we maintain a reduced row echelon form (RREF) or similar.
    
    # basis_mask[j] is the bitmask of the row that has the leading 1 at position j.
    # If no such row exists, basis_mask[j] is None (or 0, but 0 is not a valid basis vector).
    # We use -1 to indicate "no basis vector for this pivot".
    
    basis_mask = [-1] * n_inputs
    # We also need to track the actual rows that form the basis to express outputs later.
    # Actually, we can just use the masks.
    
    # Let's store the basis vectors themselves in a list, and also keep track of which pivot column they correspond to.
    # basis_vecs: list of (pivot_col, mask)
    basis_vecs = []
    
    # We will process the input rows.
    # For each row, we try to eliminate it using the existing basis.
    # If it becomes 0, it's dependent.
    # If it's not 0, it's independent. We find its leading 1, and use it to eliminate that bit from other basis vectors.
    
    # This is a bit of work. Let's do it carefully.
    
    # We'll maintain a list of basis vectors.
    # To speed up, we can use a dictionary: pivot_col -> mask
    # But we also need to be able to iterate through them.
    
    # Let's use a list of (pivot_col, mask) and keep it sorted by pivot_col.
    # Or just a list of masks, and find the pivot for each.
    
    # Let's stick to the standard algorithm.
    # We have a set of basis vectors.
    # For a new vector v:
    #   for each basis vector b:
    #     if v has a 1 in b's pivot column, v = v XOR b
    #   if v is not 0:
    #     find pivot column p of v
    #     for each basis vector b:
    #       if b has a 1 in column p, b = b XOR v
    #     add v to basis
    
    # This is O(n^2 * n) in the worst case, which is fine for n=16.
    
    basis = []  # list of masks
    
    for v in input_masks:
        v_current = v
        for b in basis:
            # Find pivot of b
            pivot_b = None
            for j in range(n_inputs):
                if (b >> j) & 1:
                    pivot_b = j
                    break
            if pivot_b is not None and (v_current >> pivot_b) & 1:
                v_current ^= b
                
        if v_current != 0:
            # Find pivot of v_current
            pivot_v = None
            for j in range(n_inputs):
                if (v_current >> j) & 1:
                    pivot_v = j
                    break
            if pivot_v is not None:
                # Eliminate pivot_v from all existing basis vectors
                new_basis = []
                for b in basis:
                    if (b >> pivot_v) & 1:
                        b ^= v_current
                    new_basis.append(b)
                basis = new_basis
                basis.append(v_current)
                
    # Now `basis` contains a set of independent vectors that span the row space.
    # Every input row is a linear combination of these basis vectors.
    
    # Step 2: Build the circuit for the basis vectors.
    # We will use a recursive, memoized function to build each basis vector from the input wires.
    # This function will share sub-expressions.
    
    # Wire mapping:
    # 0..n_inputs-1: input wires
    # n_inputs: zero wire
    # n_inputs+1...: generated wires
    
    wire_map = {0: n_inputs}
    for i in range(n_inputs):
        wire_map[1 << i] = i
        
    gates = []
    next_wire = n_inputs + 1
    
    def create_wire(mask):
        nonlocal next_wire
        if mask in wire_map:
            return wire_map[mask]
        if mask == 0:
            return n_inputs
            
        # Find the bits set in mask
        bits = []
        m = mask
        idx = 0
        while m:
            if m & 1:
                bits.append(idx)
            m >>= 1
            idx += 1
            
        if not bits:
            return n_inputs
            
        if len(bits) == 1:
            w = wire_map[1 << bits[0]]
            wire_map[mask] = w
            return w
            
        # Split into two roughly equal parts
        mid = len(bits) // 2
        m1 = 0
        m2 = 0
        for b in bits[:mid]:
            m1 |= (1 << b)
        for b in bits[mid:]:
            m2 |= (1 << b)
            
        w1 = create_wire(m1)
        w2 = create_wire(m2)
        
        gates.append([w1, w2])
        w = next_wire
        next_wire += 1
        wire_map[mask] = w
        return w

    # Build all basis vectors
    for b in basis:
        create_wire(b)
        
    # Step 3: Express each output row as a linear combination of the basis vectors.
    # We need to find the coefficients for each output row in terms of the basis.
    
    # We can do this by solving a system of linear equations.
    # Let B be the matrix with the basis vectors as rows.
    # We want to find x such that B * x = output_row (over GF(2)).
    
    # Since B has full row rank (by construction), this system has a unique solution.
    
    # We can use Gaussian elimination again, but this time on the augmented matrix [B | output_row].
    
    # Let's build the matrix B.
    B = np.array(basis, dtype=np.int8)
    if len(B) == 0:
        # No basis vectors, all outputs are 0
        return {'gates': gates, 'outputs': [n_inputs] * n_outputs}
        
    n_basis = len(B)
    
    # We will solve B * x = y for each output row y.
    # We can precompute the inverse of B (if it's square) or use row reduction.
    # Since B is not necessarily square, we'll use row reduction on the augmented matrix.
    
    # Let's create an augmented matrix [B | I] to find the "dual" basis or just solve for each y.
    # Actually, solving for each y separately is O(n_outputs * n_basis^2), which is fine.
    
    # Let's precompute the row-reduced form of B.
    # We'll create a copy of B and reduce it.
    B_reduced = B.copy()
    # We'll also keep track of the transformations applied to the identity matrix, so we can apply them to y.
    # Or, more simply, we can just solve the system for each y.
    
    # Let's use a function to solve B * x = y.
    def solve_Bx_y(y):
        # y is a bitmask of length n_inputs
        # We want to find x (a bitmask of length n_basis) such that B * x = y.
        # We'll use Gaussian elimination on the augmented matrix [B | y].
        
        # Create the augmented matrix
        # B is n_basis x n_inputs
        # y is a vector of length n_inputs
        # We'll represent the augmented matrix as a list of bitmasks, where the last n_inputs bits are the "y" part?
        # No, that's confusing. Let's just do it with numpy or manual bit operations.
        
        # Let's do it with manual bit operations for clarity and speed.
        # We have n_basis equations.
        # Each equation is a dot product of a row of B with x, which equals the corresponding bit of y.
        
        # Let's create a list of (row_mask, y_bit) pairs.
        # row_mask is the row of B as a bitmask.
        # y_bit is the corresponding bit of y.
        
        # Actually, it's easier to think of it as:
        # We have a system of linear equations over GF(2).
        # We can use the standard algorithm.
        
        # Let's create an array of the rows of B, and the vector y.
        # We'll perform Gaussian elimination.
        
        # Let's represent the state as a list of (row_mask, y_bit)
        # But we need to eliminate x variables, not B variables.
        
        # Let's use the standard approach:
        # We have n_basis variables (x_0, ..., x_{n_basis-1}).
        # We have n_basis equations.
        
        # Let's create a matrix A (n_basis x n_basis) where A[i][j] is the coefficient of x_j in equation i.
        # And a vector b (n_basis) where b[i] is the right-hand side of equation i.
        
        # A[i][j] = (B[i] >> j) & 1  ... wait, no.
        # The equation is: sum_j B[i][j] * x_j = y_i
        # So A[i][j] = B[i][j], and b[i] = y_i.
        
        # But B[i] is a bitmask of length n_inputs, and x is a vector of length n_basis.
        # This doesn't match up.
        
        # Ah, I see the confusion. The basis vectors are in the space of the inputs (length n_inputs).
        # The coefficients x are for the basis vectors (length n_basis).
        # So the equation is: sum_{k=0}^{n_basis-1} x_k * basis_k = y
        # This is a vector equation in GF(2)^{n_inputs}.
        
        # So we have n_inputs scalar equations, and n_basis unknowns.
        # Since the basis vectors are linearly independent, the system is consistent and has a unique solution.
        
        # We can solve this by expressing y in terms of the basis.
        # This is equivalent to finding the coordinates of y in the basis.
        
        # We can do this by performing Gaussian elimination on the basis vectors and y simultaneously.
        
        # Let's create a list of the basis vectors and y.
        # We'll perform elimination to transform the basis vectors into a "reduced" form where each has a unique pivot.
        # Then, we can read off the coefficients.
        
        # Let's use the basis we already computed.
        # We can assume that the basis is in a form where each vector has a unique pivot column.
        # (This is true by construction in the Gaussian elimination step above.)
        
        # So, for each basis vector b_k, let p_k be its pivot column.
        # Then, the coefficient x_k is simply the bit of y at position p_k, after eliminating the contributions from other basis vectors.
        
        # Let's do it step by step.
        # We'll create a copy of y.
        y_current = y
        
        # We'll process the basis vectors in order of their pivot columns.
        # Let's sort the basis vectors by their pivot columns.
        basis_sorted = []
        for b in basis:
            pivot = None
            for j in range(n_inputs):
                if (b >> j) & 1:
                    pivot = j
                    break
            if pivot is not None:
                basis_sorted.append((pivot, b))
                
        basis_sorted.sort(key=lambda x: x[0])
        
        # Now, for each basis vector (pivot, b), if y_current has a 1 at the pivot position, we XOR y_current with b.
        # And we set the corresponding coefficient to 1.
        
        x = 0
        for pivot, b in basis_sorted:
            if (y_current >> pivot) & 1:
                x |= (1 << len(basis_sorted)) # This is wrong, we need to know the index of this basis vector.
                y_current ^= b
                
        # This approach is flawed because I don't have the index of the basis vector.
        # Let's redo this.
        
        # Let's create a list of (pivot, index, b)
        basis_with_idx = []
        for i, b in enumerate(basis):
            pivot = None
            for j in range(n_inputs):
                if (b >> j) & 1:
                    pivot = j
                    break
            if pivot is not None:
                basis_with_idx.append((pivot, i, b))
                
        basis_with_idx.sort(key=lambda x: x[0])
        
        y_current = y
        x = 0
        for pivot, idx, b in basis_with_idx:
            if (y_current >> pivot) & 1:
                x |= (1 << idx)
                y_current ^= b
                
        # At the end, y_current should be 0.
        # And x is the vector of coefficients.
        return x

    # Now, for each output row, find its coefficients in terms of the basis.
    output_coefficients = []
    for i in range(n_outputs):
        y = input_masks[i]
        x = solve_Bx_y(y)
        output_coefficients.append(x)
        
    # Step 4: Build the output combinators.
    # For each output, we need to XOR the basis vectors for which the corresponding coefficient is 1.
    
    # We can use the same create_wire function, but this time we're combining basis vectors, not input bits.
    # We can create a new function, or just reuse create_wire by mapping the basis vectors to their wires.
    
    # Let's map the basis vectors to their wires.
    basis_wires = []
    for b in basis:
        w = create_wire(b)  # This should already be in wire_map
        basis_wires.append(w)
        
    # Now, for each output, we XOR the appropriate basis wires.
    outputs = []
    for i in range(n_outputs):
        x = output_coefficients[i]
        if x == 0:
            outputs.append(n_inputs)
            continue
            
        # Find the bits set in x
        bits = []
        m = x
        idx = 0
        while m:
            if m & 1:
                bits.append(idx)
            m >>= 1
            idx += 1
            
        if not bits:
            outputs.append(n_inputs)
            continue
            
        # XOR the corresponding basis wires
        # We can do this recursively or iteratively.
        # Let's do it iteratively for simplicity.
        w = basis_wires[bits[0]]
        for j in range(1, len(bits)):
            w2 = basis_wires[bits[j]]
            gates.append([w, w2])
            w = next_wire
            next_wire += 1
            
        outputs.append(w)
        
    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
