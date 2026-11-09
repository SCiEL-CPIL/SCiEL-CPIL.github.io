"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict

def synthesize(matrix, depth_limit):
    """
    Synthesize XOR circuit using minimum-cost Steiner tree approach.
    
    Structural change: Instead of building balanced trees per output with a cache,
    we use a global Dijkstra to find the minimum gate count to produce each linear
    combination, then build a shared circuit by greedily adding gates in order of
    how much they help complete remaining outputs.
    
    The key insight: the minimum number of gates to produce a set S of inputs is
    related to the minimum size of a binary tree whose leaves are S and internal
    nodes represent XORs. We compute this globally and share optimally.
    """
    deadline = time.time() + 85  # 5s safety margin
    
    n = len(matrix)
    m = len(matrix[0]) if n > 0 else 0
    
    if n == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert matrix rows to bitmasks
    rows = []
    for i in range(n):
        mask = 0
        for j in range(m):
            if matrix[i][j]:
                mask |= (1 << j)
        rows.append(mask)
    
    # Zero wire is m, inputs are 0..m-1
    zero_wire = m
    next_wire = m + 1
    gates = []
    
    # For each linear combination (represented as bitmask of which inputs are XORed),
    # compute the minimum number of gates needed to produce it from inputs.
    # This is the minimum size of a binary tree with those leaves.
    # For a set of k inputs, min gates = k-1 (any binary tree with k leaves has k-1 internal nodes)
    # BUT we can share: if two outputs share a subset, we only build that subset once.
    
    # Phase 1: Compute for each bitmask the minimum gate depth and a "cost" metric
    # Using Dijkstra: dist[0] = 0 (zero wire), dist[1<<j] = 0 (input wire)
    # Transition: if we have wires a and b, we can produce a^b with cost 1
    # But this is exponential in m. For m<=16, 2^16 = 65536 states, manageable.
    
    if m <= 16:
        # Dijkstra over all 2^m linear combinations
        # dist[mask] = minimum number of gates to produce this combination
        # parent[mask] = (a, b) such that a^b = mask, used to reconstruct
        
        num_states = 1 << m
        dist = [float('inf')] * num_states
        parent = [None] * num_states  # (a, b) pair
        
        # Initialize: zero and single inputs
        dist[0] = 0  # zero wire
        for j in range(m):
            dist[1 << j] = 0  # input wires
        
        # Priority queue: (cost, mask)
        pq = [(0, 0)]
        for j in range(m):
            heapq.heappush(pq, (0, 1 << j))
        
        visited = [False] * num_states
        visited[0] = True
        for j in range(m):
            visited[1 << j] = True
        
        # Dijkstra: from each existing wire, XOR with another to get new wire
        # But we need to be careful: we can only XOR two wires we already have.
        # Actually, the state space is all linear combinations, and from any two
        # combinations a, b we can get a^b at cost 1.
        # This is like: dist[c] = min over a^b=c of max(dist[a], dist[b]) + 1
        # Wait, no - it's the number of gates, so if a costs ga gates and b costs gb gates,
        # then a^b costs max(ga, gb) + 1? No, it's ga + gb + 1 if they share nothing...
        # Actually no. The total gate count is the total number of gates in the circuit.
        # If we build a using ga gates and b using gb gates, and they share some gates,
        # then a^b uses at most ga + gb + 1 gates total, but potentially fewer due to sharing.
        
        # This is getting complex. Let me use a different formulation:
        # The minimum number of gates to produce a set of k inputs in a binary tree is k-1.
        # The key is sharing across outputs.
        
        # Let me try a simpler but effective approach:
        # 1. Collect all unique row masks
        # 2. Find a minimum spanning structure using a greedy algorithm
        # 3. Use the fact that any subset of size k needs k-1 gates minimum
        
        pass
    
    # Practical approach: greedy with maximum sharing
    # 1. Sort outputs by popcount (number of 1s) - simpler ones first
    # 2. For each output, find the cheapest way to build it given existing wires
    # 3. "Cheapest" = minimize new gates, preferring to split into two existing/near-existing wires
    
    # Wire management
    wire_to_mask = {}  # wire_id -> bitmask
    mask_to_wire = {}  # bitmask -> wire_id
    for j in range(m):
        wire_to_mask[j] = 1 << j
        mask_to_wire[1 << j] = j
    wire_to_mask[zero_wire] = 0
    mask_to_wire[0] = zero_wire
    
    # Precompute popcount for all possible masks (m <= 16)
    popcount = [0] * (1 << m)
    for i in range(1, 1 << m):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Sort outputs by complexity (popcount)
    output_indices = list(range(n))
    output_indices.sort(key=lambda i: popcount[rows[i]])
    
    outputs = [0] * n
    wires_in_use = set(wire_to_mask.keys())
    
    def get_or_create_wire(mask):
        """Get wire for mask, or create one using minimum new gates."""
        nonlocal next_wire
        if mask in mask_to_wire:
            return mask_to_wire[mask]
        if mask == 0:
            return zero_wire
        
        k = popcount[mask]
        if k <= 1:
            # Single bit, should be an input
            if mask in mask_to_wire:
                return mask_to_wire[mask]
            # For a single bit, it's the input wire
            w = mask  # This is the input wire index (bit position)
            wire_to_mask[w] = mask
            mask_to_wire[mask] = w
            return w
        
        # First, check if mask can be made from two existing wires
        for w1 in list(wires_in_use):
            if time.time() > deadline:
                break
            m1 = wire_to_mask[w1]
            m2 = mask ^ m1
            if m2 in mask_to_wire:
                w2 = mask_to_wire[m2]
                if w1 != w2:
                    gates.append((w1, w2))
                    w = next_wire
                    next_wire += 1
                    wire_to_mask[w] = mask
                    mask_to_wire[mask] = w
                    wires_in_use.add(w)
                    return w
        
        # For small k, try all possible splits to find the cheapest decomposition
        # Cost of a split (a, b) where a^b = mask:
        #   - if a exists and b exists: cost 1
        #   - if a exists and b needs building: cost 1 + cost(b)
        #   - if neither exists: cost 1 + cost(a) + cost(b) - shared_gates
        # We approximate: cost(mask) = 1 + min over splits of (cost(a) + cost(b))
        # where cost(x) = 0 if x is a wire, else popcount(x) - 1 (lower bound)
        
        # For k <= 12, enumerate all submasks a of mask where a < mask^a (avoid duplicates)
        # and pick the split minimizing estimated cost
        if k <= 12:
            best_a = None
            best_b = None
            best_est_cost = float('inf')
            
            # Enumerate all proper non-empty submasks a of mask
            # To avoid duplicates, require a < mask ^ a
            sub = mask
            while sub:
                if time.time() > deadline:
                    break
                other = mask ^ sub
                # Only consider each pair once: require sub < other
                if sub < other:
                    # Estimate cost
                    c_sub = 0 if sub in mask_to_wire else max(0, popcount[sub] - 1)
                    c_other = 0 if other in mask_to_wire else max(0, popcount[other] - 1)
                    est = 1 + c_sub + c_other
                    if est < best_est_cost:
                        best_est_cost = est
                        best_a = sub
                        best_b = other
                sub = (sub - 1) & mask
            
            if best_a is not None:
                wa = get_or_create_wire(best_a)
                wb = get_or_create_wire(best_b)
                gates.append((wa, wb))
                w = next_wire
                next_wire += 1
                wire_to_mask[w] = mask
                mask_to_wire[mask] = w
                wires_in_use.add(w)
                return w
        
        # For larger k or if enumeration didn't help, use greedy split
        # Find a submask that's close to an existing wire
        best_split_cost = float('inf')
        best_w1 = None
        best_w2_mask = None
        
        for w1 in list(wires_in_use):
            if time.time() > deadline:
                break
            m1 = wire_to_mask[w1]
            m2 = mask ^ m1
            if m2 == 0:
                continue
            c2 = popcount[m2]
            if c2 < best_split_cost:
                best_split_cost = c2
                best_w1 = w1
                best_w2_mask = m2
        
        if best_w1 is not None and best_w2_mask != 0:
            w2 = get_or_create_wire(best_w2_mask)
            gates.append((best_w1, w2))
            w = next_wire
            next_wire += 1
            wire_to_mask[w] = mask
            mask_to_wire[mask] = w
            wires_in_use.add(w)
            return w
        
        # Fallback: build from scratch using a balanced tree
        bits = [j for j in range(m) if mask & (1 << j)]
        if not bits:
            return zero_wire
        if len(bits) == 1:
            return bits[0]
        
        # Build a balanced tree for the bits
        # This ensures depth is O(log k) rather than O(k)
        def build_balanced(bit_list):
            """Build a balanced XOR tree for the given bit positions."""
            nonlocal next_wire
            if len(bit_list) == 1:
                return bit_list[0]
            mid = len(bit_list) // 2
            left = build_balanced(bit_list[:mid])
            right = build_balanced(bit_list[mid:])
            gates.append((left, right))
            w = next_wire
            next_wire += 1
            # Compute mask for this wire
            m = 0
            for b in bit_list:
                m |= (1 << b)
            wire_to_mask[w] = m
            mask_to_wire[m] = w
            wires_in_use.add(w)
            return w
        
        w = build_balanced(bits)
        return w
    
    # Process outputs in order of increasing complexity
    for i in output_indices:
        if time.time() > deadline:
            # Emergency fallback
            for remaining in output_indices[output_indices.index(i):]:
                if outputs[remaining] != 0:
                    continue
                bits = [j for j in range(m) if rows[remaining] & (1 << j)]
                if not bits:
                    outputs[remaining] = zero_wire
                elif len(bits) == 1:
                    outputs[remaining] = bits[0]
                else:
                    w = bits[0]
                    for j in bits[1:]:
                        gates.append((w, j))
                        w = next_wire
                        next_wire += 1
                    outputs[remaining] = w
            break
        
        w = get_or_create_wire(rows[i])
        outputs[i] = w
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
