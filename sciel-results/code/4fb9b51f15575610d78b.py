import time
import numpy as np
from collections import deque

def synthesize(matrix, depth_limit):
    start_time = time.time()
    # 90s wall clock limit. 80s is a safe internal deadline.
    DEADLINE = start_time + 80.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Convert matrix to numpy array for faster operations
    mat = np.array(matrix, dtype=np.uint8) % 2

    # Parse rows as frozensets of column indices
    rows = []
    for i in range(m):
        row_set = frozenset(np.where(mat[i] == 1)[0].tolist())
        rows.append(row_set)

    # Wire assignment
    # 0..n-1 are inputs
    # n is constant zero
    next_wire = n + 1
    gates = []

    # Memo: frozenset -> (wire_id, depth)
    # We use a dictionary for O(1) lookup.
    # For efficiency in subset checking, we rely on the fact that we only
    # check subsets that are likely to be useful (small size or high frequency).
    # However, the current program iterates ALL memo keys.
    # To improve speed, we can limit the iteration or use a smarter data structure.
    # But for now, let's stick to the robust approach and add a deadline check.
    
    memo = {}
    for j in range(n):
        memo[frozenset([j])] = (j, 0)
    memo[frozenset()] = (n, 0)

    def elapsed():
        return time.time() - start_time

    def add_gate(a, b, d_a, d_b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        depth = max(d_a, d_b) + 1
        return w, depth

    def build_balanced_tree(s):
        """Fallback: build a balanced tree for set s, memoizing intermediate results."""
        if s in memo:
            return memo[s]
        if len(s) == 0:
            memo[s] = (n, 0)
            return memo[s]
        if len(s) == 1:
            j = next(iter(s))
            memo[s] = (j, 0)
            return memo[s]
        
        elems = list(s)
        wires_depths = [(j, 0) for j in elems]
        while len(wires_depths) > 1:
            new_wires = []
            for i in range(0, len(wires_depths), 2):
                if i + 1 < len(wires_depths):
                    w1, d1 = wires_depths[i]
                    w2, d2 = wires_depths[i+1]
                    w, d = add_gate(w1, w2, d1, d2)
                    new_wires.append((w, d))
                else:
                    new_wires.append(wires_depths[i])
            wires_depths = new_wires
        result = wires_depths[0]
        memo[s] = result
        return result

    def compute_set(s):
        if s in memo:
            return memo[s]
        
        # If deadline exceeded, fallback to balanced tree to ensure termination
        if elapsed() > DEADLINE:
            return build_balanced_tree(s)

        # Strategy: Find a split s = A U B (disjoint) where A, B are in memo or cheap
        # We prefer A, B in memo (cost 1 gate)
        # Then we prefer A in memo, B small (cost 1 + cost(B))
        
        best_cost = float('inf')
        best_A = None
        best_B = None
        best_wire_A = None
        best_depth_A = None
        best_wire_B = None
        best_depth_B = None
        best_is_both_memo = False

        # Optimization: Iterate through memo keys that are subsets of s.
        # To speed this up, we can filter by size. 
        # Ideally, we want A to be "large" enough to reduce the problem significantly,
        # but not so large that B is huge.
        # A common heuristic is to look for A of size roughly half of s.
        # However, iterating all subsets is O(2^n) in the worst case if memo is full.
        # Since memo is sparse, iterating keys is O(|memo|).
        
        # We can prune: if len(A) > len(s)//2, then B is smaller.
        # We should check both A and B to avoid duplicates?
        # Actually, if we find A, B, it's the same as finding B, A.
        # So we can restrict len(A) <= len(s)//2 to avoid duplicate checks?
        # No, because the cost function might be asymmetric if B is not in memo.
        # But XOR is commutative.
        # Let's just iterate all and pick the best.
        
        # To avoid O(|memo|) scan if |memo| is huge, we can stop early if we find a 1-gate solution.
        # A 1-gate solution is optimal for this step.
        
        for A, (wire_A, depth_A) in memo.items():
            if not A.issubset(s):
                continue
            
            B = s - A # Disjoint difference
            
            if B in memo:
                wire_B, depth_B = memo[B]
                # Cost is 1 gate
                # We found a 1-gate solution. This is the best possible for this step.
                # We can break immediately.
                return (wire_A, depth_A, wire_B, depth_B, True) # Signal to caller
            else:
                # B is not in memo. Estimate cost of B.
                # Cost of B is roughly popcount(B) - 1 if computed from scratch.
                # But it might share sub-expressions.
                # For now, use popcount(B) as a heuristic cost.
                cost = 1 + len(B)
                if cost < best_cost:
                    best_cost = cost
                    best_A = A
                    best_B = B
                    best_wire_A = wire_A
                    best_depth_A = depth_A
                    best_wire_B = None
                    best_depth_B = None
                    best_is_both_memo = False

        # If we found a 1-gate solution, it would have returned.
        # So we proceed with the best heuristic found.
        if best_A is not None:
            if best_is_both_memo:
                w, d = add_gate(best_wire_A, best_wire_B, best_depth_A, best_depth_B)
                memo[s] = (w, d)
                return (w, d)
            else:
                # Compute B
                w_B, d_B = compute_set(best_B)
                w, d = add_gate(best_wire_A, w_B, best_depth_A, d_B)
                memo[s] = (w, d)
                return (w, d)
        
        # Fallback: Balanced tree
        return build_balanced_tree(s)
    
    outputs = []
    # Sort rows by popcount to compute simpler ones first?
    # This helps populate memo with small sets first, which are more likely to be reused.
    sorted_indices = sorted(range(m), key=lambda i: len(rows[i]))
    
    for i in sorted_indices:
        if elapsed() > DEADLINE:
            # If we hit the deadline, we must ensure we still produce a valid result.
            # compute_set has a fallback, so it will terminate.
            pass
        
        w, d = compute_set(rows[i])
        outputs.append(w)
        
    # Re-map outputs to original order
    final_outputs = [0] * m
    for idx, orig_idx in enumerate(sorted_indices):
        final_outputs[orig_idx] = outputs[idx]
    
    return {'gates': gates, 'outputs': final_outputs}
