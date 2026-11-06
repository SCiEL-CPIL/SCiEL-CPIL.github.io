# EVOLVE-BLOCK-START
import time
import heapq

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 55.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Wire assignment: inputs 0..n-1, const-zero n, gates n+1...
    next_wire = n + 1
    gates = []

    # memo: bitmask -> wire_id
    memo = {}
    # Initialize with inputs
    for j in range(n):
        memo[1 << j] = j
    # Const zero
    memo[0] = n

    def elapsed():
        return time.time() - start_time

    def add_gate(a, b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        return w

    # Parse matrix to masks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    # Collect all unique non-zero masks that need to be synthesized
    all_masks = set(row_masks)
    all_masks.discard(0)

    # Estimate cost of synthesizing a mask: popcount
    def mask_cost(mask):
        return mask.bit_count()

    # We'll use a priority queue to process masks in order of increasing cost.
    # For each mask, we need to decide how to split it into two sub-masks.
    # Strategy: split into two parts with roughly equal popcount.
    # This is a heuristic for building a balanced XOR tree.

    # pending: dict mask -> set of submasks needed
    # We'll do a worklist algorithm.
    
    # Start by adding all target masks
    targets = list(all_masks)
    
    # We need a function to get the wire for a mask, possibly creating gates.
    # But we want to be smart about ordering.
    
    # Let's use a different approach: 
    # 1. Identify all "minimal" masks (those that cannot be decomposed further into smaller masks in the set).
    #    Actually, every mask can be decomposed into single-bit masks.
    # 2. We want to maximize sharing.
    
    # Alternative structural approach:
    # Build a DAG of XOR operations where nodes are masks.
    # We want to find a set of intermediate masks to synthesize such that all target masks are covered.
    # This is a set cover-like problem, but we can use a greedy heuristic.
    
    # Greedy: 
    # 1. Start with all target masks as "unsatisfied".
    # 2. Repeatedly pick the mask that appears in the most unsatisfied masks (or some other metric).
    # 3. Synthesize it, mark it satisfied, and update.
    
    # Actually, a simpler and often effective approach for XOR circuits is to use a 
    # "balanced tree" decomposition for each mask, but with global memoization.
    # The key is to choose the decomposition that maximizes sub-mask reuse.
    
    # Let's try this:
    # For each mask, we can decompose it as mask = sub1 ^ sub2, where sub1 and sub2 are non-empty and disjoint.
    # We want to choose (sub1, sub2) such that sub1 and sub2 are "common" or "cheap".
    
    # Heuristic: Split the mask into two parts by taking the first half of the bits and the second half.
    # This is deterministic and simple.
    
    def split_mask(mask):
        if mask == 0:
            return (0, 0)
        bits = [j for j in range(n) if (mask >> j) & 1]
        if len(bits) == 1:
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
    
    # Now, we'll use a recursive memoized function to synthesize a mask.
    # But we want to process masks in a good order.
    # Let's collect all sub-masks that will be needed.
    
    needed = set()
    def collect_needed(mask):
        if mask in memo:
            return
        if mask in needed:
            return
        needed.add(mask)
        sub1, sub2 = split_mask(mask)
        if sub1 != 0:
            collect_needed(sub1)
        if sub2 != 0:
            collect_needed(sub2)
    
    for mask in all_masks:
        collect_needed(mask)
    
    # Now we have all masks we need to synthesize.
    # Sort them by popcount (ascending) to synthesize simpler ones first.
    sorted_needed = sorted(needed, key=lambda m: m.bit_count())
    
    # Synthesize each mask in order
    for mask in sorted_needed:
        if mask in memo:
            continue
        
        sub1, sub2 = split_mask(mask)
        
        if sub1 == 0 and sub2 == 0:
            memo[mask] = n
            continue
            
        if sub1 == 0:
            memo[mask] = memo[sub2]
            continue
            
        if sub2 == 0:
            memo[mask] = memo[sub1]
            continue
        
        # Both sub1 and sub2 should be in memo now (since we're processing in order)
        if sub1 not in memo:
            # This shouldn't happen if collect_needed was correct
            continue
        if sub2 not in memo:
            continue
            
        w = add_gate(memo[sub1], memo[sub2])
        memo[mask] = w
    
    # Now all target masks should be in memo
    outputs = []
    for mask in row_masks:
        outputs.append(memo.get(mask, n))
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
