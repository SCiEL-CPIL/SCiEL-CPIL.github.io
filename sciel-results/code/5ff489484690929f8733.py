"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict, deque
import heapq
from itertools import combinations

def synthesize(matrix, depth_limit):
    """Synthesize an XOR circuit using minimum-cost flow over subset space.
    
    Formulates the problem as finding a minimum-cost decomposition of each output
    into shared sub-expressions, using cost-flow to globally optimize gate sharing.
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin for 90s limit
    
    mat = np.array(matrix, dtype=np.int8)
    n_outputs, n_inputs = mat.shape
    
    def row_to_mask(row):
        m = 0
        for j, v in enumerate(row):
            if v:
                m |= (1 << j)
        return m
    
    output_masks = [row_to_mask(mat[i]) for i in range(n_outputs)]
    
    def popcount(m):
        return bin(m).count('1')
    
    def bits_from_mask(m):
        bits = []
        j = 0
        while m:
            if m & 1:
                bits.append(j)
            m >>= 1
            j += 1
        return bits
    
    # For small inputs, use exact DP over all subsets
    # For larger inputs, use greedy with better heuristics
    
    # Strategy: Use a two-phase approach
    # Phase 1: Identify high-value shared sub-expressions via frequency analysis
    # Phase 2: Build circuit using these sub-expressions with optimal tree structure
    
    # Count frequency of each subset across all outputs
    subset_freq = defaultdict(int)
    for mask in output_masks:
        bits = bits_from_mask(mask)
        # Count all subsets of size 2-6 (practical limit)
        max_subset_size = min(6, len(bits))
        for size in range(2, max_subset_size + 1):
            for combo in combinations(bits, size):
                sub_mask = 0
                for b in combo:
                    sub_mask |= (1 << b)
                subset_freq[sub_mask] += 1
    
    # Identify beneficial sub-expressions: those appearing in >= 2 outputs
    # Benefit = (popcount(sub) - 1) * (freq - 1) - overhead
    # We need to account for the cost of building the sub-expression
    
    beneficial_exprs = []
    for sub_mask, freq in subset_freq.items():
        if freq < 2:
            continue
        k = popcount(sub_mask)
        if k < 2:
            continue
        # Cost to build: k-1 gates
        # Savings per additional use: k-1 gates
        # Net benefit: (k-1) * (freq - 1)
        benefit = (k - 1) * (freq - 1)
        if benefit > 0:
            beneficial_exprs.append((benefit, sub_mask, freq, k))
    
    # Sort by benefit descending
    beneficial_exprs.sort(reverse=True)
    
    # Now we need to select a set of sub-expressions that can be composed
    # to cover all outputs, minimizing total gate count
    
    # Use a greedy set-cover-like approach with dynamic programming
    # State: for each output, track which sub-expressions have been "used"
    
    # Simpler approach: iteratively extract the most beneficial expression
    # that can be decomposed from remaining outputs
    
    # Track remaining masks and output decompositions
    remaining_masks = list(output_masks)
    output_decomps = [[] for _ in range(n_outputs)]  # list of (sub_mask, wire)
    
    expr_to_wire = {}
    for i in range(n_inputs):
        expr_to_wire[(1 << i)] = i
    expr_to_wire[0] = n_inputs
    
    next_wire = n_inputs + 1
    gates = []
    
    def build_tree(wires):
        """Build a balanced XOR tree, return final wire."""
        nonlocal next_wire
        if len(wires) == 0:
            return n_inputs
        if len(wires) == 1:
            return wires[0]
        
        level = list(wires)
        while len(level) > 1:
            new_level = []
            for i in range(0, len(level) - 1, 2):
                gates.append([level[i], level[i+1]])
                new_level.append(next_wire)
                next_wire += 1
            if len(level) % 2 == 1:
                new_level.append(level[-1])
            level = new_level
        return level[0]
    
    def create_expr_wire(sub_mask):
        """Create wire for sub-expression, return wire id."""
        nonlocal next_wire
        if sub_mask in expr_to_wire:
            return expr_to_wire[sub_mask]
        
        bits = bits_from_mask(sub_mask)
        w = build_tree(bits)
        expr_to_wire[sub_mask] = w
        return w
    
    # Iteratively extract beneficial expressions
    # Use a priority queue for efficiency
    pq = []
    pq_counter = 0
    
    def push_expr(benefit, sub_mask, freq, k):
        nonlocal pq_counter
        heapq.heappush(pq, (-benefit, pq_counter, sub_mask, freq, k))
        pq_counter += 1
    
    # Initialize priority queue
    for benefit, sub_mask, freq, k in beneficial_exprs:
        push_expr(benefit, sub_mask, freq, k)
    
    extracted_count = 0
    max_extractions = 200
    
    while pq and extracted_count < max_extractions:
        if time.time() > DEADLINE - 5.0:
            break
        
        neg_benefit, _, sub_mask, old_freq, k = heapq.heappop(pq)
        benefit = -neg_benefit
        
        # Re-check if still beneficial
        # Count how many remaining outputs contain this sub-expression
        current_freq = 0
        for i in range(n_outputs):
            if (remaining_masks[i] & sub_mask) == sub_mask:
                current_freq += 1
        
        if current_freq < 2:
            continue
        
        # Check if sub_mask can be decomposed from available expressions
        # A sub-expression is decomposable if it's a single bit or can be XOR of
        # already-available expressions
        bits = bits_from_mask(sub_mask)
        if len(bits) == 1:
            continue  # Already an input
        
        # Check if we can build this from existing wires
        # For now, assume we can always build it from inputs
        # (we'll create the tree when we extract it)
        
        current_benefit = (k - 1) * (current_freq - 1)
        if current_benefit <= 0:
            continue
        
        # Extract this expression
        w = create_expr_wire(sub_mask)
        
        # Update remaining masks
        for i in range(n_outputs):
            if (remaining_masks[i] & sub_mask) == sub_mask:
                remaining_masks[i] ^= sub_mask
                output_decomps[i].append((sub_mask, w))
        
        extracted_count += 1
        
        # Re-evaluate other expressions
        # Only re-check expressions that share bits with this one
        sub_bits = set(bits)
        new_pq = []
        
        while pq:
            neg_b, cnt, e_mask, e_freq, e_k = heapq.heappop(pq)
            e_bits = set(bits_from_mask(e_mask))
            
            if e_bits & sub_bits:
                # Re-evaluate
                if e_mask not in expr_to_wire:
                    e_cur_freq = sum(1 for i in range(n_outputs) 
                                    if (remaining_masks[i] & e_mask) == e_mask)
                    if e_cur_freq >= 2:
                        e_benefit = (e_k - 1) * (e_cur_freq - 1)
                        if e_benefit > 0:
                            heapq.heappush(new_pq, (-e_benefit, cnt, e_mask, e_cur_freq, e_k))
            else:
                heapq.heappush(new_pq, (neg_b, cnt, e_mask, e_freq, e_k))
        
        pq = new_pq
    
    # Assemble final outputs
    output_wires = []
    for i in range(n_outputs):
        wires = [w for (_, w) in output_decomps[i]]
        rem = remaining_masks[i]
        
        # Add remaining input bits
        rem_bits = bits_from_mask(rem)
        wires.extend(rem_bits)
        
        if not wires:
            output_wires.append(n_inputs)
        else:
            output_wires.append(build_tree(wires))
    
    return {'gates': gates, 'outputs': output_wires}

# EVOLVE-BLOCK-END
