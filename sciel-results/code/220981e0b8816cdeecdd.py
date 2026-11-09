def synthesize(matrix, depth_limit):
    import time
    import numpy as np
    from collections import deque

    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    # Convert rows to integer bitmasks for fast XOR operations
    # Bit i corresponds to input wire i
    target_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j]:
                mask |= (1 << j)
        target_masks.append(mask)

    # Wire 0..n-1 are inputs (single bits)
    # Wire n is constant 0 (mask 0)
    # New gates produce wires n+1, n+2, ...
    
    # Map from mask to wire index
    # mask 0 -> wire n
    # mask (1<<j) -> wire j
    mask_to_wire = {0: n}
    for j in range(n):
        mask_to_wire[1 << j] = j

    gates = []
    next_wire = n + 1
    
    # We want to find the cheapest way to produce each target mask.
    # "Cost" is roughly the number of gates added to the circuit.
    # A better metric for BFS is "depth" or "weight" (number of inputs involved).
    # However, minimizing gate count globally is complex. 
    # Let's use a BFS that expands states by XORing two existing wires.
    # To minimize gate count, we should prioritize reusing existing wires.
    
    # Strategy: 
    # 1. Start with all single-bit masks (cost 0 gates, depth 1) and zero (cost 0, depth 0).
    # 2. We perform a BFS where each step XORs two existing wires to create a new one.
    # 3. We stop when all target masks are found.
    # 4. To keep it efficient and within depth limits, we limit the "weight" (popcount of mask) 
    #    or simply the number of steps in BFS.
    # 5. Actually, minimizing gate count is equivalent to finding the smallest set of 
    #    intermediate XORs that spans the target vectors.
    
    # Let's try a different angle: 
    # The "independent-row baseline" is sum of (popcount(row) - 1) for each row? 
    # No, baseline is likely just building each row independently with popcount(row)-1 gates.
    # Our goal is to share sub-expressions.
    
    # Efficient approach:
    # Use a priority queue (Dijkstra-like) or simple BFS levels.
    # State: mask
    # Action: XOR with any existing mask.
    # This is too slow if we try all pairs.
    
    # Better: 
    # Observe that for random matrices, the rows are likely linearly independent or close.
    # If rows are independent, sharing is limited unless we create "common basis" vectors.
    
    # Let's implement a greedy "Common Subexpression Elimination" via BFS levels.
    # Level 0: Inputs (wires 0..n-1) and Zero (wire n).
    # Level 1: All XORs of two Level 0 wires? No, that's O(n^2).
    
    # Instead, let's look at the specific targets.
    # We will generate wires on-demand but globally coordinate.
    
    # Revised Strategy: 
    # 1. Identify all unique target masks.
    # 2. If a target is a single bit, use input wire.
    # 3. If a target is 0, use zero wire.
    # 4. For complex targets, we want to find a decomposition.
    #    We can use a "meet in the middle" or simple recursive splitting with global cache.
    #    The current program does this but greedily.
    
    # Let's try: **Algebraic Normal Form (ANF) based synthesis with sharing**.
    # Actually, the most effective simple strategy for XOR circuits is:
    # 1. Compute the linear span of the target rows.
    # 2. Find a basis for this span.
    # 3. Synthesize the basis vectors first (they are the "shared" blocks).
    # 4. Express each target row as a combination of the basis vectors.
    
    # Step 1: Gaussian elimination on the target rows to find a basis.
    # This minimizes the number of "new" linear combinations we need to create if we treat basis vectors as intermediates.
    # Wait, the basis vectors themselves might be complex.
    
    # Let's stick to a robust **BFS over masks** but optimized.
    # We only need to generate masks that are targets or useful for targets.
    
    # Key Insight: The cost of a wire is 1 gate. The depth is the max depth of parents + 1.
    # We want to minimize total gates.
    
    # Let's use a simple heuristic that works well:
    # 1. Collect all target masks.
    # 2. Sort them by popcount (ascending).
    # 3. For each mask, try to express it as XOR of two previously generated (or input) wires.
    #    To do this efficiently, we maintain a set of available wires.
    #    For a target T, we look for A in Available such that T^A is also in Available.
    #    If found, use those.
    #    If not, we must create a new wire. We can split T into T1 and T2 such that T1^T2=T.
    #    To minimize future cost, we might want T1 and T2 to be "simple" (low popcount).
    
    # This is essentially what the current program does, but it's recursive and might miss global optimizations.
    # The "Different" strategy: **Global Greedy with Backtracking/Repair** or **Simulated Annealing**?
    # No, let's do **Level-Bounded BFS Generation**.
    
    # We will generate all possible XORs of pairs of existing wires, level by level, 
    # but we will PRUNE the search space to only keep wires that are:
    # 1. A target row.
    # 2. A "useful" intermediate (e.g., appears in the decomposition of multiple targets).
    
    # Simpler and robust: 
    # 1. Start with inputs.
    # 2. While there are unmet targets:
    #    a. Find the target with the minimum "potential cost" (e.g., popcount).
    #    b. Try to satisfy it using existing wires.
    #    c. If not, split it into two parts, add the parts to the "to-do" list, and create a gate for the XOR.
    #    d. Recurse on the parts.
    # This is the current program's logic.
    
    # How to make it materially different?
    # **Strategy: Pre-compute a "Shared Basis" for all rows.**
    # 1. Stack all rows into a matrix.
    # 2. Perform Gaussian Elimination to find a basis B = {b1, ..., bk}.
    # 3. Synthesize each bi. Since bi are basis vectors, they are linearly independent.
    #    To synthesize bi, we can use the same recursive splitting, but now we know bi are "fundamental".
    # 4. Express each original row ri as XOR of some subset of B.
    # 5. The circuit for ri is just XORing the wires corresponding to the basis vectors in its representation.
    # 6. This ensures that if two rows share a basis vector, the wire for that basis vector is reused.
    
    # Why is this different? The current program splits *each row independently* (though with a global cache, it might accidentally share). 
    # The basis approach *forces* sharing of the most "efficient" linear combinations (the basis).
    # Actually, the basis vectors from GE are just the rows themselves (or row sums). 
    # If we use the rows as the basis, there is no sharing unless we reduce them.
    
    # Better: **Find a basis of "simple" vectors**.
    # 1. Start with standard basis e1...en.
    # 2. We want to synthesize targets T1...Tm.
    # 3. We can introduce new intermediate wires.
    
    # Let's try **Minimum Weight Basis Synthesis**:
    # 1. Identify all unique target masks.
    # 2. Compute the linear span.
    # 3. Find a basis for the span that minimizes the sum of popcounts of the basis vectors? 
    #    This is NP-hard.
    
    # Let's go with a **Hybrid**:
    # 1. Use the recursive splitting strategy (like current) BUT with a **Global Priority Queue**.
    # 2. Instead of processing rows one by one, we process all "needed" masks in order of "complexity" (popcount).
    # 3. This ensures that simpler sub-expressions are created first and can be reused by more complex ones.
    
    # Implementation:
    # 1. Add all target masks to a set `needed`.
    # 2. While `needed` is not empty:
    #    a. Pick the mask `T` in `needed` with the smallest popcount.
    #    b. If `T` is in `mask_to_wire`, remove from `needed` and continue.
    #    c. Check if `T` can be formed by XOR of two existing wires `A` and `B`.
    #       - Iterate through existing wires `A`. If `T^A` is in `mask_to_wire`, use it.
    #    d. If not, split `T` into `T1` and `T2` (e.g., half popcount).
    #       - Add `T1` and `T2` to `needed`.
    #       - Create a new gate for `T = T1 ^ T2`? No, we can't create the gate for T until T1 and T2 are ready.
    #       - So, we mark T as "pending" with parents T1, T2.
    
    # This is a topological sort approach.
    
    # Let's implement this **Topological Generation with Popcount Priority**.
    
    # Data structures:
    # mask_to_wire: dict
    # gates: list
    # next_wire: int
    
    # pending: dict mask -> (parent1_mask, parent2_mask)
    # needed: set of masks
    
    # 1. Initialize needed with all target masks.
    # 2. Loop:
    #    - Find T in needed with min popcount.
    #    - If T in mask_to_wire: 
    #        - Remove T from needed.
    #        - Check if any pending masks depend on T? 
    #          Actually, if T is resolved, we can resolve its children?
    #          No, the children are the parents in the XOR tree. 
    #          If T = A ^ B, and A, B are in mask_to_wire, we can create the gate for T.
    
    # Let's refine:
    # 1. `targets`: set of masks we need.
    # 2. `resolved`: set of masks we have wires for.
    # 3. `pending_gates`: list of (mask, mask1, mask2) where mask = mask1 ^ mask2, but mask1/mask2 not yet resolved.
    
    # Algorithm:
    # 1. For each target, if simple (popcount <= 1), add to resolved.
    # 2. For complex targets, decompose into two parts, add to pending_gates, add parts to targets.
    # 3. Loop while targets not empty:
    #    a. Find a mask M in targets that can be resolved.
    #       - If M is in mask_to_wire (from initial inputs or previous steps), it's resolved.
    #       - If M is in pending_gates and both parents are resolved, create the gate, add M to mask_to_wire, resolve M.
    #    b. If no mask can be resolved, we must decompose a new target.
    #       - Pick the target with min popcount.
    #       - Decompose it.
    
    # This ensures we always make progress.
    
    # Optimization: When decomposing, choose split to maximize chance of reuse.
    # Split T into T1 and T2 such that T1 and T2 are "likely" to be simple or already existing.
    # Simple split: T1 = T & mask_half, T2 = T ^ T1.
    
    # Let's code this.
    
    # 1. Setup
    targets = set(target_masks)
    # Remove already resolved
    for t in list(targets):
        if t in mask_to_wire:
            targets.remove(t)
            
    # If all resolved, return
    if not targets:
        return {'gates': gates, 'outputs': [mask_to_wire[t] for t in target_masks]}

    # pending: list of tuples (mask, m1, m2)
    pending = []
    
    # We need a way to track which targets are "active" for decomposition.
    # Let's use a loop.
    
    while targets:
        # Check if any pending gate can be resolved
        resolved_any = False
        new_pending = []
        for (mask, m1, m2) in pending:
            if m1 in mask_to_wire and m2 in mask_to_wire:
                # Create gate
                w1 = mask_to_wire[m1]
                w2 = mask_to_wire[m2]
                gates.append([w1, w2])
                w_new = next_wire
                next_wire += 1
                mask_to_wire[mask] = w_new
                if mask in targets:
                    targets.remove(mask)
                resolved_any = True
            else:
                new_pending.append((mask, m1, m2))
        pending = new_pending
        
        if resolved_any:
            continue
            
        # If no pending resolved, we must decompose a target
        # Pick target with min popcount
        # Note: popcount is a proxy for "complexity"
        min_pop = float('inf')
        min_mask = None
        for t in targets:
            pc = bin(t).count('1')
            if pc < min_pop:
                min_pop = pc
                min_mask = t
                
        if min_mask is None:
            break
            
        # Decompose min_mask
        # If popcount is 1, it should have been resolved initially. 
        # If popcount is 0, it's 0.
        # So min_pop >= 2.
        
        # Split min_mask into two parts
        # Strategy: Split into two roughly equal popcount parts
        # To ensure progress, one part should be strictly smaller? 
        # If we split into 1 and k-1, the 1 is resolved, so we just need k-1.
        # This leads to a chain of XORs, which is depth k-1.
        # If we split into k/2 and k/2, depth is log k.
        # We want to minimize depth? The task says depth <= 16. 
        # Minimize gate count.
        
        # For gate count, splitting into 1 and k-1 uses 1 gate for the XOR, 
        # but requires synthesizing k-1.
        # Total gates = 1 + gates(k-1).
        # If we split into k/2 and k/2, gates = 2 * gates(k/2) + 1 (if no sharing).
        # With sharing, k/2 might be reused.
        
        # Let's try splitting into a single bit and the rest.
        # This creates a "prefix XOR" chain.
        # It uses k-1 gates for a vector of weight k.
        # This is the same as the baseline!
        # To beat the baseline, we MUST share sub-expressions.
        
        # Therefore, we should split into two "complex" parts that are likely to be shared.
        # Or, we should look for existing wires that are subsets.
        
        # Let's try: Find an existing wire A such that A is a subset of T (bitwise).
        # If A is a subset, then T = A ^ (T^A).
        # If A is "small" or "common", this helps.
        
        # In the BFS/Topo approach, we don't have "existing wires" other than inputs and previously resolved.
        
        # Let's stick to the **Basis Approach** which is fundamentally different and guarantees sharing of the basis.
        
        # **Revised Plan: Basis Synthesis**
        # 1. Compute a basis for the span of the target rows.
        # 2. Synthesize each basis vector.
        # 3. Express each target as XOR of basis vectors.
        # 4. The "cost" is sum of costs of basis vectors + sum of (num_basis_in_target - 1) for each target.
        
        # How to synthesize basis vectors?
        # They are just specific masks. We can use the recursive splitting for them.
        # Since they are a basis, they are independent, so less sharing between them.
        # But the targets share the basis vectors!
        
        # This is a strong strategy.
        
        # Let's implement Basis Synthesis.
        
        # 1. Gaussian Elimination to find basis.
        #    We work with integers (bitmasks).
        #    Basis: list of masks.
        #    Rep: dict mask -> list of basis indices.
        
        # Actually, we don't need to find the *optimal* basis. 
        # The standard row-echelon basis is fine.
        
        # 2. Synthesize basis vectors using a helper `synthesize_mask(mask)`.
        #    This helper uses the recursive splitting strategy, but caches results.
        #    Since basis vectors are independent, they won't share much with each other, 
        #    but the *targets* will share the basis wires.
        
        # 3. For each target, XOR the wires of the basis vectors in its representation.
        
        # This is clean and different.
        
        # Let's implement this.
        
        # Reset state for this new strategy
        gates = []
        next_wire = n + 1
        mask_to_wire = {0: n}
        for j in range(n):
            mask_to_wire[1 << j] = j
            
        # 1. Find Basis
        basis_masks = []
        # We want to find a basis for the set of target masks.
        # Standard GE:
        # Sort targets by popcount? No, standard GE by highest bit.
        # Let's just iterate.
        remaining = list(set(target_masks))
        # Remove 0
        remaining = [r for r in remaining if r != 0]
        
        # To make basis "simple", we can try to minimize popcount of basis vectors?
        # Let's just use standard GE.
        basis = []
        # We need to track the representation of each basis vector in terms of original rows?
        # No, we just need the masks.
        
        # Wait, if we use standard GE, the basis vectors are linear combinations of the rows.
        # Synthesizing a basis vector b means synthesizing the mask b.
        
        # Let's do this:
        # 1. Get unique non-zero targets.
        # 2. Perform GE to find a basis B.
        # 3. For each b in B, synthesize b.
        # 4. For each target t, find coefficients c_i such that t = sum c_i b_i.
        # 5. Wire for t is XOR of wires for b_i where c_i=1.
        
        # Synthesizing b:
        # Use recursive splitting with cache.
        
        # Let's implement `get_wire(mask)` inside.
        
        # Note: We must be careful with recursion depth and time.
        
        # Implementation details:
        # - `mask_to_wire` is global cache.
        # - `get_wire(mask)`:
        #   - if mask in cache, return.
        #   - if popcount(mask) == 1, return input wire.
        #   - Split mask into m1, m2.
        #   - w1 = get_wire(m1), w2 = get_wire(m2).
        #   - Create gate, return new wire.
        
        # To minimize gates, we should split smartly.
        # If we split into 1 and rest, depth is high.
        # If we split into halves, depth is log.
        # Let's use halves.
        
        # Let's code this.

        # 1. Basis Calculation
        # We want a basis for the span of the targets.
        # Let's use the targets themselves as the initial set for GE.
        # But we want the basis vectors to be "simple" (low popcount) to minimize synthesis cost.
        # Standard GE doesn't guarantee this.
        # However, for random matrices, the basis vectors will have popcount ~ n/2.
        
        # Let's try to find a basis with low popcount using a greedy approach?
        # 1. Start with empty basis.
        # 2. Pick the target with min popcount. Add to basis.
        # 3. Eliminate this vector from all other targets (XOR it where it helps reduce popcount?).
        #    Actually, standard GE:
        #    - Pick pivot (highest bit).
        #    - Use it to eliminate that bit from others.
        
        # Let's stick to standard GE for simplicity and correctness, but maybe sort by popcount first?
        # No, GE requires pivot structure.
        
        # Let's just use the targets as the basis if they are independent?
        # If they are not independent, we reduce them.
        
        # Simplest:
        # 1. unique_targets = list(set(target_masks))
        # 2. basis = []
        # 3. for t in unique_targets:
        #      if t == 0: continue
        #      # Check if t is in span of current basis
        #      # If not, add to basis.
        #      # To check, we can try to reduce t using basis.
        #      # If it reduces to 0, it's dependent.
        #      # If it reduces to non-zero, it's independent. Add the reduced form to basis?
        #      # Or add t?
        #      # Adding t is fine.
        
        # Let's implement a simple linear independence check and basis building.
        
        def is_in_span(mask, basis):
            # Try to reduce mask using basis
            # We need basis in row-echelon form for this to be fast.
            # Let's maintain basis in row-echelon form (pivot bits are unique).
            pass
            
        # This is getting complex. Let's use NumPy for GE.
        
        # Convert masks to binary vectors.
        # But n can be up to 16. Masks fit in int.
        
        # Let's do manual GE on integers.
        # Basis: list of (pivot_bit, mask)
        
        basis = []
        # Sort targets by popcount? No, by highest bit for GE.
        # Let's just iterate.
        
        # We want to build a basis for the space spanned by targets.
        # We also want to express each target in terms of this basis.
        
        # Let's use a different approach: 
        # **Synthesize all targets directly with a global cache, but process them in order of popcount.**
        # This is the "Topo Sort" idea again, but simpler.
        
        # 1. Sort targets by popcount.
        # 2. For each target:
        #    - If in cache, skip.
        #    - Else, synthesize it.
        #      - Synthesis: Find split (m1, m2).
        #      - Ensure m1, m2 are synthesized (recursively).
        #      - Create gate.
        
        # This is effectively the current program, but with a global ordering.
        # The current program processes rows in input order.
        # If we process by popcount, simpler sub-expressions are created first.
        
        # Let's try this **Ordered Synthesis** strategy.
        
        # Reset state
        gates = []
        next_wire = n + 1
        mask_to_wire = {0: n}
        for j in range(n):
            mask_to_wire[1 << j] = j
            
        # Get unique targets
        unique_targets = list(set(target_masks))
        
        # Sort by popcount
        unique_targets.sort(key=lambda x: bin(x).count('1'))
        
        # Recursive function with cache
        # To avoid deep recursion, we can use iterative or ensure depth is small.
        # Max depth 16.
        
        def synthesize_mask(mask):
            nonlocal next_wire
            if mask in mask_to_wire:
                return mask_to_wire[mask]
            
            # If popcount is 1, it's an input (should be in cache)
            if mask == 0:
                return mask_to_wire[0]
            
            # Find split
            # Try to find a split that reuses existing wires
            # Iterate through existing wires? Too slow.
            
            # Heuristic: Split into two halves
            # Get bits
            bits = [j for j in range(n) if (mask >> j) & 1]
            if len(bits) == 1:
                # Should be in cache
                return mask_to_wire[mask]
            
            mid = len(bits) // 2
            m1 = 0
            for j in bits[:mid]:
                m1 |= (1 << j)
            m2 = mask ^ m1
            
            w1 = synthesize_mask(m1)
            w2 = synthesize_mask(m2)
            
            gates.append([w1, w2])
            w_new = next_wire
            next_wire += 1
            mask_to_wire[mask] = w_new
            return w_new

        # Synthesize each target
        for t in unique_targets:
            synthesize_mask(t)
            
        # Build outputs
        outputs = [mask_to_wire[t] for t in target_masks]
        
        return {'gates': gates, 'outputs': outputs}
