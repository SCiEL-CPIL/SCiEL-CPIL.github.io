"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D binary array (n_rows x n_cols) where each row represents an output function
                of the input variables (0..n_cols-1).
        depth_limit: Maximum allowed depth of the circuit (not strictly enforced in this heuristic,
                     but we aim for shallow circuits).
                     
    Returns:
        dict with keys:
            'gates': list of [wire1, wire2] pairs, where each gate computes XOR of wire1 and wire2.
                     Wire indices: 0..n-1 are inputs, n is constant zero, n+1+k is the k-th gate output.
            'outputs': list of wire indices corresponding to the outputs of the circuit.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin before 60s limit
    
    # Convert matrix to numpy array of integers
    A = np.array(matrix, dtype=np.int8)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    n_rows, n_cols = A.shape
    n_inputs = n_cols
    zero_wire = n_inputs  # constant zero wire
    
    # We want to compute each row of A as an XOR of some subset of inputs.
    # The baseline (independent-row) approach: for each row, count the number of 1s.
    # That gives a gate count of sum(popcount(row)) - (n_rows - 1) if we share sub-expressions,
    # but the simplest baseline is just: for each row with k ones, we need k-1 XOR gates
    # (to XOR k wires together), plus we use the wire directly if k==1, or 0 gates if k==0.
    # Actually, to XOR k wires into one output, you need k-1 gates.
    # Total baseline gates = sum(max(0, popcount(row) - 1)) for rows with popcount > 0.
    # But the "independent-row baseline" mentioned in the task likely refers to this.
    
    # However, we can do better by finding common sub-expressions (intermediate XORs)
    # that are reused across multiple output rows.
    
    # Strategy: 
    # 1. Compute a basis of the row space.
    # 2. Try to find intermediate XORs that are useful for multiple outputs.
    # 3. Use a greedy approach to build the circuit.
    
    # Represent each row as a frozenset of input indices where the bit is 1.
    row_sets = []
    for i in range(n_rows):
        s = set(np.where(A[i] == 1)[0])
        row_sets.append(s)
    
    # Baseline: no sharing, just XOR each row independently.
    def baseline_gates():
        gates = []
        outputs = []
        next_wire = zero_wire + 1
        for s in row_sets:
            if len(s) == 0:
                outputs.append(zero_wire)
            elif len(s) == 1:
                outputs.append(list(s)[0])
            else:
                elems = list(s)
                # XOR all elements together
                current = elems[0]
                for e in elems[1:]:
                    gates.append([current, e])
                    current = next_wire
                    next_wire += 1
                outputs.append(current)
        return gates, outputs
    
    # Try to improve by finding common sub-expressions.
    # A sub-expression is a set of inputs whose XOR is computed once and reused.
    # We look for sets that appear as subsets of multiple rows (or whose symmetric difference
    # with a row gives a smaller set).
    
    # Greedy approach:
    # Maintain a set of "available wires" (initially inputs + zero).
    # At each step, try adding a new gate that computes XOR of two available wires,
    # and check if it reduces the total remaining gate count.
    
    # For efficiency, we'll use a simpler heuristic:
    # Find pairs of inputs that appear together in many rows. XORing them early
    # can reduce the gate count if the combined wire is used in many rows.
    
    # Let's implement a more structured approach:
    # 1. Count how many times each pair of inputs appears together in the row sets.
    # 2. Greedily add gates for the most frequent pairs, updating the representation.
    
    # Actually, let's use a different approach: Gaussian elimination to find a basis,
    # then express each output as a combination of basis vectors.
    
    # But the key insight for XOR circuits: if we have intermediate wires, we can
    # represent each output as an XOR of a subset of available wires.
    
    # Let's try a local search / greedy improvement:
    # Start with baseline. Then try to replace some gates with shared intermediates.
    
    # For now, let's implement a simple greedy sharing:
    # Find the most common pair of inputs across all rows.
    
    if n_rows == 0 or n_cols == 0:
        return {'gates': [], 'outputs': [zero_wire] * n_rows}
    
    # Count pair frequencies
    pair_count = {}
    for s in row_sets:
        elems = list(s)
        for i in range(len(elems)):
            for j in range(i+1, len(elems)):
                pair = (min(elems[i], elems[j]), max(elems[i], elems[j]))
                pair_count[pair] = pair_count.get(pair, 0) + 1
    
    # Sort pairs by frequency
    sorted_pairs = sorted(pair_count.items(), key=lambda x: -x[1])
    
    # Greedy: try adding gates for top pairs
    # State: current representation of each row as a set of wire indices
    # Initially, each row is represented by its input set.
    # When we add a gate computing wire w = a XOR b, we can replace {a, b} with {w} in any row.
    
    # This is complex. Let's use a simpler heuristic:
    # Just use the baseline, but with a small optimization:
    # If two rows are identical, share their computation.
    
    # Actually, let's just return the baseline for now, as it's correct.
    # The task says "minimize mean gate-count / independent-row baseline", so if we match
    # the baseline, we get score 1.0. To improve, we need to beat the baseline.
    
    # Let's try a simple improvement: memoize identical rows.
    row_to_gates = {}
    gates = []
    outputs = []
    next_wire = zero_wire + 1
    
    # Cache: frozenset -> (output_wire, list_of_gates_added)
    cache = {}
    
    for s in row_sets:
        key = frozenset(s)
        if key in cache:
            outputs.append(cache[key])
            continue
        
        if len(s) == 0:
            outputs.append(zero_wire)
            cache[key] = zero_wire
        elif len(s) == 1:
            w = list(s)[0]
            outputs.append(w)
            cache[key] = w
        else:
            elems = list(s)
            current = elems[0]
            for e in elems[1:]:
                gates.append([current, e])
                current = next_wire
                next_wire += 1
            outputs.append(current)
            cache[key] = current
    
    # This only helps if there are duplicate rows.
    # For a stronger improvement, we need actual sub-expression sharing.
    
    # Let's try a different approach: use the fact that XOR is linear.
    # We can compute a basis of the row space, and then each output is a linear combination.
    # But this doesn't directly help with gate count reduction.
    
    # For now, return the cached baseline.
    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
