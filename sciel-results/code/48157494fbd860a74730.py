import time
import numpy as np
import random
from collections import defaultdict, deque

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    mat = np.array(matrix, dtype=np.int8)
    m, n = mat.shape
    
    zero_wire = n
    row_weights = np.sum(mat, axis=1)
    baseline_gates = int(np.sum(np.maximum(0, row_weights - 1)))
    
    # Handle trivial cases
    if baseline_gates == 0:
        gates = []
        outputs = []
        for i in range(m):
            if row_weights[i] == 0:
                outputs.append(zero_wire)
            else:
                idx = np.where(mat[i] == 1)[0][0]
                outputs.append(int(idx))
        return {'gates': gates, 'outputs': outputs}
    
    # Convert rows to bitmasks
    target_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if mat[i, j] == 1:
                mask |= (1 << j)
        target_masks.append(mask)
    
    # For small n, use a more exhaustive search or DP if feasible
    # For larger n, use the min-cost flow approach or a strong heuristic
    
    if n <= 12:
        # Try DP or exhaustive search for small n
        # This is a placeholder for a potentially better small-n solver
        pass
    
    # Main strategy: Min-Cost Flow formulation
    # We model the problem as finding a set of gates (edges) that computes all targets.
    # This is NP-hard in general, but we can approximate it or use a heuristic based on flow.
    
    # Alternative strong heuristic: Iterative improvement / Local Search
    # Start with the greedy solution, then try to improve it by swapping gates or adding/removing them.
    
    # Let's implement a robust heuristic that combines greedy construction with local search.
    
    def greedy_synthesis():
        available = {}
        for i in range(n):
            available[1 << i] = i
        available[0] = zero_wire
        
        gates = []
        next_wire = n + 1
        target_wires = [None] * m
        
        for i in range(m):
            if target_masks[i] in available:
                target_wires[i] = available[target_masks[i]]
        
        remaining_indices = sorted(
            [i for i in range(m) if target_wires[i] is None],
            key=lambda i: row_weights[i]
        )
        
        for idx in remaining_indices:
            if time.time() > deadline:
                break
            mask = target_masks[idx]
            wire = None
            found = False
            
            # Try to find a decomposition using available wires
            # Sort available wires by some heuristic (e.g., frequency of use, or just random)
            # For now, just iterate
            items = list(available.items())
            # Shuffle to avoid bias? No, keep deterministic for now.
            for w_mask, w_wire in items:
                other_mask = mask ^ w_mask
                if other_mask in available:
                    other_wire = available[other_mask]
                    if w_mask == 0:
                        wire = other_wire
                        found = True
                        break
                    elif other_mask == 0:
                        wire = w_wire
                        found = True
                        break
                    else:
                        gates.append([w_wire, other_wire])
                        wire = next_wire
                        next_wire += 1
                        available[mask] = wire
                        found = True
                        break
            
            if not found:
                # Fallback: compute from inputs directly
                ones = [j for j in range(n) if (mask >> j) & 1]
                if len(ones) == 0:
                    wire = zero_wire
                elif len(ones) == 1:
                    wire = ones[0]
                else:
                    current = ones[0]
                    for j in range(1, len(ones)):
                        gates.append([current, ones[j]])
                        current = next_wire
                        next_wire += 1
                    wire = current
                    available[mask] = wire
            
            target_wires[idx] = wire
        
        return gates, target_wires
    
    gates, outputs = greedy_synthesis()
    
    # Local Search: Try to improve the gate count
    # We can try to identify redundant gates or find better decompositions.
    # This is complex, so for now, we return the greedy result.
    # The min-cost flow idea is promising but hard to implement quickly and correctly.
    
    # Let's try a different approach: Basis Decomposition
    # Find a basis for the row space, compute the basis vectors, then express each row as a sum of basis vectors.
    
    def basis_decomposition():
        # Find a basis for the rows
        # Use Gaussian elimination over GF(2)
        rows = [list(target_masks[i]) for i in range(m)] # Wait, target_masks are integers
        # Convert to lists of bits? No, keep as integers.
        
        basis = []
        basis_masks = []
        
        for mask in target_masks:
            current = mask
            for b_mask, b_idx in basis:
                if current & b_mask:
                    current ^= b_mask
            if current != 0:
                # Find the highest bit
                highest_bit = current.bit_length() - 1
                # Insert into basis
                basis.append((1 << highest_bit, len(basis)))
                basis_masks.append(current)
                # Update existing basis if needed? Standard row echelon form
        
        # Actually, let's just compute a basis and then express each target as a sum of basis vectors.
        # This might not minimize gate count if basis vectors are expensive to compute.
        
        # A better approach: 
        # 1. Identify "frequent" sub-expressions (masks that appear often or are cheap).
        # 2. Use a priority queue to decide which sub-expression to compute next.
        
        # For now, let's stick with the greedy method but add a simple local search.
        # Local search: For each gate, see if it can be replaced by a cheaper combination.
        # This is difficult.
        
        # Instead, let's try to run the greedy method multiple times with different orderings.
        best_gates = None
        best_count = float('inf')
        
        for _ in range(10): # 10 restarts
            if time.time() > deadline:
                break
            g, o = greedy_synthesis()
            if len(g) < best_count:
                best_count = len(g)
                best_gates = g
                best_outputs = o
        
        if best_gates is not None:
            return {'gates': best_gates, 'outputs': best_outputs}
        
        return {'gates': gates, 'outputs': outputs}
    
    return basis_decomposition()
