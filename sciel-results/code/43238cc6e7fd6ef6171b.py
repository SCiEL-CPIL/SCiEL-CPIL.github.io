"""XOR-circuit synthesis: greedy decomposition with cost-based reuse."""
import time
import numpy as np

def synthesize(matrix, depth_limit):
    """
    Synthesize an exact XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D array-like of 0/1 values (m x n).
        depth_limit: int, max allowed circuit depth (<=16).
    
    Returns:
        dict with keys:
          'gates': list of [wire_a, wire_b] pairs (XOR gates)
          'outputs': list of wire indices (one per output row)
    """
    t_start = time.time()
    deadline = t_start + 55.0  # safety margin before 60s limit

    m = len(matrix)
    if m == 0:
        return {'gates': [], 'outputs': []}
    n = len(matrix[0])
    # Constant-zero wire is index n
    zero_wire = n

    # Work with numpy arrays for speed
    A = np.array(matrix, dtype=np.uint8)  # m x n

    # Each output row is a vector in GF(2)^n
    # We want to express each row as XOR of a subset of input wires (0..n-1)
    # and possibly intermediate gates.
    #
    # Strategy:
    # 1. Build a basis of intermediate gates that are reused across rows.
    # 2. For each row, decompose it using existing wires (inputs + gates).
    # 3. Greedily select decomposition that minimizes total gate count.
    #
    # Key insight: each input wire i corresponds to the unit vector e_i in GF(2)^n.
    # A gate produces XOR of two existing wires. We want to cover all row vectors
    # using as few gates as possible, where each gate is XOR of two previously
    # available vectors.

    # Represent vectors as Python ints (bitmasks) for fast XOR
    # Wire i (0-indexed) has vector v_i
    # Input wires: v_i = 1 << i for i in 0..n-1
    # Zero wire: v_n = 0

    # Available wires: list of (vector_int, wire_index)
    # Start with input wires and zero
    # We'll track which vectors we have as a dict: vector -> wire_index
    # But multiple wires can have the same vector (we only need one per vector)

    # For decomposition: given a target vector t, find a subset of available
    # vectors whose XOR equals t. This is a linear algebra problem over GF(2).
    # We use Gaussian elimination on the available vectors to find a representation.

    # To minimize gate count, we want to:
    # - Reuse gates across rows
    # - Create gates that are useful for multiple rows

    # Approach:
    # 1. Collect all row vectors.
    # 2. Find a common basis / useful intermediate gates.
    # 3. For each row, express it as XOR of basis vectors + input wires.
    # 4. Count total unique gates needed.

    # Let's use a different strategy: 
    # - Maintain a set of "available" vectors (as ints) with their wire indices.
    # - Initially: all input unit vectors + zero.
    # - For each row (processed in order of complexity), find a decomposition
    #   using available vectors. If not all vectors in the decomposition are
    #   available, create new gates.
    # - To minimize gates, prefer decompositions that use existing vectors.

    # More efficient: use Gaussian elimination to find a basis of the row space,
    # then express each row in terms of that basis. The basis vectors that are
    # not input wires require gates.

    # Let me think about this differently.
    # Each output is XOR of some subset of inputs. The "cost" is the number of
    # additional XOR gates needed beyond the inputs.
    # 
    # If output j = XOR of inputs in set S_j, and |S_j| = k, then we need
    # k-1 gates to compute it (chain of XORs). But if multiple outputs share
    # intermediate XORs, we can save gates.
    #
    # Optimal approach: find a tree/forest of XOR operations that computes all
    # outputs. This is related to the "minimum XOR circuit" problem.
    #
    # Greedy approach that works well:
    # 1. For each row, find its support (which inputs are 1).
    # 2. Process rows in order of increasing support size (simpler first).
    # 3. For each row, express it as XOR of available vectors.
    # 4. When creating a new gate, prefer creating gates that combine vectors
    #    that appear in many remaining rows.

    # Let's implement a clean version:
    # - available: dict mapping vector_int -> wire_index
    # - gates: list of [a, b] pairs
    # - wire_to_vector: list where wire_to_vector[i] = vector_int for wire i
    # - Initially wires 0..n-1 are inputs, wire n is zero.

    wire_to_vec = [1 << i for i in range(n)] + [0]
    vec_to_wire = {v: i for i, v in enumerate(wire_to_vec)}
    gates = []
    next_wire = n + 1

    # For each row, we need to express it as XOR of available vectors.
    # Use Gaussian elimination: find a basis of available vectors, then
    # express the target in terms of that basis.

    def find_decomposition(target, available_vecs):
        """
        Find a subset of available_vecs whose XOR equals target.
        Returns list of wire indices, or None if not possible.
        available_vecs: dict vector_int -> wire_index
        """
        if target == 0:
            return []
        if target in available_vecs:
            return [available_vecs[target]]
        
        # Gaussian elimination
        # Sort vectors by bit-length for efficiency
        vecs = sorted(available_vecs.keys(), key=lambda v: v.bit_length())
        # Build basis
        basis = []
        for v in vecs:
            if v == 0:
                continue
            # Reduce v by existing basis
            for b in basis:
                if v.bit_length() == b.bit_length() or (v ^ b).bit_length() < v.bit_length():
                    v ^= b
            if v != 0:
                basis.append(v)
        
        # Check if target is in span
        t = target
        for b in basis:
            if t.bit_length() >= b.bit_length() and ((t >> (b.bit_length()-1)) & 1):
                t ^= b
        if t != 0:
            return None  # Not representable
        
        # Now express target as XOR of basis vectors
        # Redo with tracking
        basis_with_info = []  # (vector, [wires])
        for v in vecs:
            if v == 0:
                continue
            wires = [available_vecs[v]]
            reduced = v
            for bv, bw in basis_with_info:
                if reduced.bit_length() >= bv.bit_length() and ((reduced >> (bv.bit_length()-1)) & 1):
                    reduced ^= bv
                    wires = wires + bw
            if reduced != 0:
                basis_with_info.append((reduced, wires))
        
        # Express target
        t = target
        result_wires = []
        for bv, bw in basis_with_info:
            if t.bit_length() >= bv.bit_length() and ((t >> (bv.bit_length()-1)) & 1):
                t ^= bv
                result_wires = result_wires + bw
        if t != 0:
            return None
        
        # result_wires may have duplicates (XOR of same wire twice cancels)
        # Simplify: XOR all vectors, but we need wire indices
        # Actually, we just need the XOR of the wire vectors to equal target.
        # The list may have duplicates; XOR of a wire with itself is 0.
        # Let's just return the list; the caller will handle it.
        return result_wires

    # This approach is getting complex. Let me use a simpler but effective strategy:
    # 
    # 1. Compute all row vectors as ints.
    # 2. Find the row space basis using Gaussian elimination on the rows.
    # 3. Express each row as XOR of basis vectors.
    # 4. The basis vectors that aren't input wires need to be computed via gates.
    # 5. To compute a basis vector, we can use a chain of XORs of input wires.
    # 6. Minimize total gates by sharing intermediate XORs.

    # Actually, the simplest correct approach:
    # Each output row j has support S_j (set of input indices where A[j,i]=1).
    # To compute output j, we XOR all inputs in S_j. This takes |S_j|-1 gates.
    # Total gates = sum(|S_j|-1 for j) if no sharing.
    # With sharing, we can do better.
    #
    # Baseline (xor_baseline) likely does something simple. Let me aim to beat it.
    #
    # Key optimization: if two rows share a common subset of inputs, we can
    # compute the XOR of that subset once and reuse it.
    #
    # Greedy algorithm:
    # 1. Start with all input wires available.
    # 2. For each row (in order of decreasing support size, to maximize reuse),
    #    find a decomposition using available wires.
    # 3. If the row vector is not directly available, create new gates to produce it.
    #    To create a gate, XOR two available wires to get a new wire.
    #    Choose pairs that are "useful" (appear in many remaining rows).
    # 4. Continue until all rows are expressed.

    # Let me implement a practical version:

    # Convert rows to int vectors
    row_vecs = []
    for j in range(m):
        v = 0
        for i in range(n):
            if A[j, i]:
                v |= (1 << i)
        row_vecs.append(v)

    # Available wires: vector -> wire_index
    # Start with inputs and zero
    wire_to_vec = [1 << i for i in range(n)] + [0]
    vec_to_wire = {v: i for i, v in enumerate(wire_to_vec)}
    gates = []
    next_wire = n + 1

    # For each row, we need to express it as XOR of available vectors.
    # We'll maintain a list of "needed" vectors (the row vectors not yet available).
    # Process rows in order of complexity (fewest new gates needed first).

    # Helper: compute cost of expressing a vector using current available set
    def expr_cost(target, vec_to_wire, wire_to_vec):
        """Estimate number of new gates needed to express target."""
        if target == 0:
            return 0
        if target in vec_to_wire:
            return 0
        # Use Gaussian elimination to find minimum number of available vectors
        # whose XOR gives target, then cost = (num_vectors - 1) if we chain them
        # But we might create intermediate gates.
        # Simple estimate: popcount of target / 2 (each gate combines 2 wires)
        # Better: use the actual decomposition
        return target.bit_count() // 2  # rough estimate

    # Process rows
    # Order: sort by bit_count descending (more complex first, to create useful gates)
    row_order = sorted(range(m), key=lambda j: row_vecs[j].bit_count(), reverse=True)

    outputs = [0] * m

    for j in row_order:
        target = row_vecs[j]
        if target == 0:
            outputs[j] = zero_wire
            continue
        if target in vec_to_wire:
            outputs[j] = vec_to_wire[target]
            continue

        # Need to create gates to produce this vector
        # Strategy: find a chain of XORs from available wires to produce target
        # Use Gaussian elimination to find a representation
        available = {v: w for v, w in vec_to_wire.items() if v != 0}
        
        # Find basis of available vectors
        basis = []
        for v in sorted(available.keys(), key=lambda x: x.bit_length()):
            reduced = v
            for b in basis:
                if reduced.bit_length() >= b.bit_length() and ((reduced >> (b.bit_length()-1)) & 1):
                    reduced ^= b
            if reduced != 0:
                basis.append(reduced)
        
        # Check if target is in span
        t = target
        for b in basis:
            if t.bit_length() >= b.bit_length() and ((t >> (b.bit_length()-1)) & 1):
                t ^= b
        if t != 0:
            # Should not happen if all inputs are available
            # Fallback: use raw inputs
            wires = [i for i in range(n) if (target >> i) & 1]
            if not wires:
                outputs[j] = zero_wire
                continue
            # Chain XOR
            acc = wires[0]
            for w in wires[1:]:
                gates.append([acc, w])
                wire_to_vec.append(wire_to_vec[acc] ^ wire_to_vec[w])
                vec_to_wire[wire_to_vec[-1]] = next_wire
                acc = next_wire
                next_wire += 1
                if time.time() > deadline:
                    break
            outputs[j] = acc
            continue

        # Express target as XOR of basis vectors
        # Redo with wire tracking
        basis_wires = []  # (vector, wire_index)
        for v in sorted(available.keys(), key=lambda x: x.bit_length()):
            reduced = v
            rw = [available[v]]
            for bv, bw in basis_wires:
                if reduced.bit_length() >= bv.bit_length() and ((reduced >> (bv.bit_length()-1)) & 1):
                    reduced ^= bv
                    rw.append(bw)
            if reduced != 0:
                basis_wires.append((reduced, rw))
        
        # Express target
        t = target
        result_wires = []
        for bv, bw in basis_wires:
            if t.bit_length() >= bv.bit_length() and ((t >> (bv.bit_length()-1)) & 1):
                t ^= bv
                result_wires.extend(bw)
        
        if not result_wires:
            outputs[j] = zero_wire
            continue

        # result_wires is a list of wire indices whose XOR gives target
        # But there may be duplicates. XOR of same wire twice = 0.
        # Simplify: XOR all vectors together
        # Actually, we just need to compute the XOR of all these wires.
        # If there are k wires, we need k-1 gates (chained).
        # But some intermediate XORs might already be available.
        
        # Simplify: remove duplicate wires (they cancel)
        from collections import Counter
        cnt = Counter(result_wires)
        simplified = [w for w, c in cnt.items() if c % 2 == 1]
        
        if not simplified:
            outputs[j] = zero_wire
            continue
        if len(simplified) == 1:
            outputs[j] = simplified[0]
            continue

        # Check if the XOR of all simplified wires is already available
        combined = 0
        for w in simplified:
            combined ^= wire_to_vec[w]
        if combined in vec_to_wire:
            outputs[j] = vec_to_wire[combined]
            continue

        # Chain XOR to create the result
        acc = simplified[0]
        for w in simplified[1:]:
            # Check if wire_to_vec[acc] ^ wire_to_vec[w] is already available
            new_vec = wire_to_vec[acc] ^ wire_to_vec[w]
            if new_vec in vec_to_wire:
                acc = vec_to_wire[new_vec]
            else:
                gates.append([acc, w])
                wire_to_vec.append(new_vec)
                vec_to_wire[new_vec] = next_wire
                acc = next_wire
                next_wire += 1
                if time.time() > deadline:
                    break
        outputs[j] = acc

    return {'gates': gates, 'outputs': outputs}
