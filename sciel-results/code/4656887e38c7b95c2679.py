# EVOLVE-BLOCK-START
import time
import random

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 50.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Parse matrix to masks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    # Wire assignment
    next_wire = n + 1
    gates = []
    memo = {}
    for j in range(n):
        memo[1 << j] = j
    memo[0] = n

    def elapsed():
        return time.time() - start_time

    def add_gate(a, b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        return w

    # Collect unique non-zero targets
    target_masks = set(row_masks)
    target_masks.discard(0)

    if not target_masks:
        return {'gates': [], 'outputs': [n] * m}

    # Strategy: Find a set of "useful" intermediate terms
    # An intermediate term is useful if it appears as a subexpression in multiple target decompositions
    
    # Approach: 
    # 1. For each target, generate multiple possible decompositions
    # 2. Choose decompositions that share common subterms
    # 3. Synthesize shared subterms once
    
    # Simpler effective approach:
    # Use a "common subexpression" detection
    # 1. Start with all single bits
    # 2. Repeatedly add the XOR of two existing terms that appears in the most target decompositions
    
    # Let's use a greedy set cover approach:
    # 1. For each target mask, the cost to synthesize it from scratch is popcount-1
    # 2. We want to find intermediate terms that reduce total cost
    
    # Better: Use the fact that XOR is associative and commutative
    # Optimal synthesis of a single mask with k bits requires k-1 gates
    # But if we share subterms across multiple masks, we can do better
    
    # Key insight: If two masks share a common submask S, we can compute S once and reuse it
    
    # Algorithm:
    # 1. Find all "frequent" submasks across target masks
    # 2. Synthesize them in order of increasing popcount
    # 3. Use them to synthesize targets
    
    # Find submasks that appear in multiple targets
    # A submask S of target T means S is a subset of bits in T
    
    # For efficiency, let's limit to submasks of size 2-4
    
    # Actually, let's use a different proven approach:
    # 1. Compute RREF to find independent rows
    # 2. Synthesize independent rows with maximum sharing
    # 3. Express dependent rows as XOR of independent rows
    
    # Phase 1: Find a basis via Gaussian elimination
    basis = []
    basis_masks = []
    
    # Sort rows by popcount (descending) to get "dense" basis first
    indexed_rows = [(row_masks[i], i) for i in range(m)]
    indexed_rows.sort(key=lambda x: x[0].bit_count(), reverse=True)
    
    for mask, idx in indexed_rows:
        if mask == 0:
            continue
        # Check if mask is independent of current basis
        reduced = mask
        for bmask in basis_masks:
            # XOR with basis if leading bit matches
            # Find highest set bit
            if bmask:
                hbit = bmask.bit_length() - 1
                if (reduced >> hbit) & 1:
                    reduced ^= bmask
        if reduced != 0:
            basis.append(idx)
            basis_masks.append(reduced)
    
    # If we have a basis, synthesize basis vectors with sharing
    # Then express each row as combination of basis vectors
    
    # But this might not be optimal. Let's try a hybrid:
    # Use the balanced tree approach but with global sharing
    
    # Collect all submasks from balanced trees of all targets
    needed = set()
    
    def balanced_split(mask):
        if mask == 0:
            return (0, 0)
        bits = [j for j in range(n) if (mask >> j) & 1]
        if len(bits) <= 1:
            return (mask, 0)
        mid = len(bits) // 2
        sub1 = 0
        sub2 = 0
        for i, b in enumerate(bits):
            if i < mid:
                sub1 |= (1 << b)
            else:
                sub2 |= (1 << b)
        return (sub1, sub2)
    
    def collect_submasks(mask):
        if mask == 0 or mask in memo:
            return
        if mask in needed:
            return
        needed.add(mask)
        sub1, sub2 = balanced_split(mask)
        if sub1 != 0:
            collect_submasks(sub1)
        if sub2 != 0:
            collect_submasks(sub2)
    
    for mask in target_masks:
        collect_submasks(mask)
    
    # Sort by popcount for synthesis order
    sorted_needed = sorted(needed, key=lambda x: (x.bit_count(), x))
    
    # Synthesize
    for mask in sorted_needed:
        if mask in memo:
            continue
        if elapsed() > DEADLINE:
            break
        
        sub1, sub2 = balanced_split(mask)
        
        if sub1 == 0 and sub2 == 0:
            memo[mask] = n
        elif sub1 == 0:
            memo[mask] = memo[sub2]
        elif sub2 == 0:
            memo[mask] = memo[sub1]
        else:
            if sub1 in memo and sub2 in memo:
                w = add_gate(memo[sub1], memo[sub2])
                memo[mask] = w
            else:
                # Fallback: direct synthesis
                bits = [j for j in range(n) if (mask >> j) & 1]
                if len(bits) == 1:
                    memo[mask] = bits[0]
                elif len(bits) > 1:
                    w = bits[0]
                    for b in bits[1:]:
                        w = add_gate(w, b)
                    memo[mask] = w
                else:
                    memo[mask] = n
    
    # Ensure all targets are in memo
    for mask in target_masks:
        if mask not in memo:
            bits = [j for j in range(n) if (mask >> j) & 1]
            if len(bits) == 1:
                memo[mask] = bits[0]
            elif len(bits) > 1:
                w = bits[0]
                for b in bits[1:]:
                    w = add_gate(w, b)
                memo[mask] = w
            else:
                memo[mask] = n
    
    outputs = [memo.get(mask, n) for mask in row_masks]
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
