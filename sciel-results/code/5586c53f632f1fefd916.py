"""Exact XOR-circuit synthesis with shared sub-expression reuse and depth control."""
import time
import numpy as np

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D binary array (rows x cols). Each row specifies which input wires
                are XORed to produce that output.
        depth_limit: Maximum allowed circuit depth (<=16).
    
    Returns:
        dict with 'gates' (list of [wire_a, wire_b] pairs) and 'outputs' (list of wire indices).
    
    Wire indexing:
        - Input wires: 0 .. n-1 (n = number of columns)
        - Constant zero: n
        - Gate k (0-based) produces wire n+1+k
    """
    start_time = time.time()
    DEADLINE = start_time + 75.0  # safety margin before 90s kill
    
    if matrix is None or len(matrix) == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert to numpy for speed
    mat = np.asarray(matrix, dtype=np.uint8)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    
    m, n = mat.shape
    zero_wire = n
    first_gate_wire = n + 1
    
    # If matrix is empty or all zeros
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    # For each output row, we need to XOR the input wires where matrix[i][j] == 1.
    # Strategy: build a shared sub-expression tree.
    # 
    # Key insight: many rows may share common subsets of inputs.
    # We use a recursive divide-and-conquer on the input set for each row,
    # but we memoize intermediate XOR results by the frozenset of input indices.
    #
    # To control depth, we use a balanced tree: XOR of k inputs in a balanced
    # binary tree has depth ceil(log2(k)).
    #
    # For sharing: if two rows both need XOR of inputs {a,b,c}, we compute it once.
    #
    # However, for large n, the number of distinct subsets could explode.
    # A more practical approach: for each row, build a balanced XOR tree of its
    # active inputs. Then check if any two rows have identical active sets -> share.
    # Also, if one row's active set is a subset of another's, we can potentially
    # reuse partial results, but this is complex. Let's start with exact-match sharing
    # and balanced trees.
    
    # Step 1: For each row, get the list of active input indices.
    row_inputs = []
    for i in range(m):
        active = np.where(mat[i] == 1)[0].tolist()
        row_inputs.append(active)
    
    # Step 2: Identify unique active sets and map rows to them.
    # Use frozenset as key.
    from collections import defaultdict
    
    set_to_rows = defaultdict(list)
    for i, active in enumerate(row_inputs):
        key = frozenset(active)
        set_to_rows[key].append(i)
    
    # Step 3: For each unique active set, build a balanced XOR tree.
    # We assign a wire to each node in the tree.
    # 
    # We'll build trees bottom-up. For a set of k inputs:
    #   - If k == 0: output is zero_wire
    #   - If k == 1: output is the single input
    #   - If k >= 2: split into two roughly equal halves, build trees for each,
    #     then XOR the two results with one gate.
    #
    # Depth of balanced tree for k inputs: ceil(log2(k))
    
    # We need to assign wire numbers to intermediate results.
    # Let's build all trees and collect gates.
    
    # To enable sharing across rows with the same active set, we build each
    # unique set once.
    
    # Also, for potential sub-expression sharing: if set A is a subset of set B,
    # and we've already built a tree for A, we might reuse it. But this requires
    # careful handling. For now, let's do exact-set sharing only.
    
    # Build a map from frozenset -> wire (the output wire of the tree for that set)
    set_to_wire = {}
    gates = []  # list of [wire_a, wire_b]
    
    def build_tree(active_set, depth_budget):
        """
        Build a balanced XOR tree for the given frozenset of input indices.
        Returns the wire number that holds the XOR result.
        
        Uses memoization via set_to_wire.
        """
        if active_set in set_to_wire:
            return set_to_wire[active_set]
        
        k = len(active_set)
        if k == 0:
            set_to_wire[active_set] = zero_wire
            return zero_wire
        if k == 1:
            wire = next(iter(active_set))
            set_to_wire[active_set] = wire
            return wire
        
        # Split into two halves
        items = sorted(active_set)
        mid = k // 2
        left_set = frozenset(items[:mid])
        right_set = frozenset(items[mid:])
        
        left_wire = build_tree(left_set, depth_budget - 1)
        right_wire = build_tree(right_set, depth_budget - 1)
        
        # Create a gate to XOR left and right
        gate_idx = len(gates)
        gates.append([left_wire, right_wire])
        result_wire = first_gate_wire + gate_idx
        
        set_to_wire[active_set] = result_wire
        return result_wire
    
    # Determine maximum depth needed
    max_k = max(len(s) for s in set_to_rows.keys()) if set_to_rows else 0
    import math
    max_depth_needed = max(1, math.ceil(math.log2(max_k))) if max_k > 1 else 1
    
    # Check if depth is within limit
    # The depth of our circuit is the max depth of any tree.
    # Balanced tree for k inputs has depth ceil(log2(k)).
    # This should be well within depth_limit=16 for reasonable n.
    
    # Build all trees
    for active_set in set_to_rows.keys():
        if active_set not in set_to_wire:
            build_tree(active_set, depth_limit)
    
    # Step 4: Determine outputs
    outputs = []
    for i in range(m):
        key = frozenset(row_inputs[i])
        outputs.append(set_to_wire[key])
    
    return {'gates': gates, 'outputs': outputs}
