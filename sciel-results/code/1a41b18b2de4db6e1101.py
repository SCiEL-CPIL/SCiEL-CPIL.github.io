"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit for search, 5s buffer
    
    N = 256
    # Squares up to 2*N
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Valid nodes: 2*v not in squares (diagonal constraint a=a means 2a not square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
    
    if not valid_nodes:
        return [1]  # Fallback
    
    # Map valid nodes to indices 0..M-1
    M = len(valid_nodes)
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    
    # Build adjacency: conflict if a+b is a perfect square
    # adj[i] = set of j such that valid_nodes[i] + valid_nodes[j] is a square
    adj = [set() for _ in range(M)]
    for i in range(M):
        for j in range(i + 1, M):
            if valid_nodes[i] + valid_nodes[j] in squares:
                adj[i].add(j)
                adj[j].add(i)
    
    # Greedy initial solution: pick nodes in order of degree (ascending), add if no conflict
    degrees = [len(adj[i]) for i in range(M)]
    order = sorted(range(M), key=lambda x: (degrees[x], -x))
    
    greedy_set = set()
    for idx in order:
        if all(j not in greedy_set for j in adj[idx]):
            greedy_set.add(idx)
    
    best_set = set(greedy_set)
    best_size = len(best_set)
    
    # Local search: try adding one node at a time, removing conflicting nodes
    def try_add_one(s, idx):
        """Try to add idx to s, removing all neighbors of idx that are in s."""
        if idx in s:
            return s, False
        conflicts = [j for j in s if j in adj[idx]]
        if len(conflicts) >= 1:
            # Only add if we gain net size (add 1, remove len(conflicts))
            # For cardinality, we only add if len(conflicts) == 0
            # But we might want to swap: remove k, add 1, net = 1-k
            # For now, only add if no conflicts
            pass
        new_s = set(s)
        new_s.add(idx)
        return new_s, True
    
    # Branch and Bound for Maximum Independent Set
    # Use bitmasks for efficiency if M <= 64, otherwise use sets
    # M is around 256 - (number of v where 2v is square)
    # 2v = square => v = square/2, so v must be even and square/2 integer
    # Squares: 4, 16, 36, 64, 100, 144, 196, 256 -> v = 2, 8, 18, 32, 50, 72, 98, 128
    # So 8 nodes are invalid, M = 248
    
    # For M=248, bitmasks are too large. Use a different approach.
    # Use a simple B&B with pruning.
    
    # Sort nodes by degree for better branching
    node_order = sorted(range(M), key=lambda x: -degrees[x])
    
    # Build conflict masks using Python integers (bitmasks) for speed
    # M=248 fits in Python big integers
    conflict_masks = [0] * M
    for i in range(M):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        conflict_masks[i] = mask
    
    # All nodes mask
    all_mask = (1 << M) - 1
    
    # B&B state: (available_mask, current_set_size, current_selection_mask)
    # We'll use a recursive B&B with memoization-like pruning
    
    best_size_global = [best_size]
    best_set_global = [set(best_set)]
    
    def popcount(x):
        return bin(x).count('1')
    
    def bbb(available_mask, current_size, current_mask):
        """Branch and bound for MIS."""
        # Check deadline
        if time.time() > deadline:
            return
        
        # Prune: if current_size + popcount(available_mask) <= best_size_global[0], prune
        if current_size + popcount(available_mask) <= best_size_global[0]:
            return
        
        if available_mask == 0:
            if current_size > best_size_global[0]:
                best_size_global[0] = current_size
                # Reconstruct set from current_mask
                s = set()
                mask = current_mask
                while mask:
                    lsb = mask & (-mask)
                    idx = lsb.bit_length() - 1
                    s.add(idx)
                    mask ^= lsb
                best_set_global[0] = s
            return
        
        # Choose a node to branch on: pick the node with highest degree in available
        # For simplicity, pick the lowest set bit
        lsb = available_mask & (-available_mask)
        node = lsb.bit_length() - 1
        
        # Branch 1: include node
        # Remove node and all its neighbors from available
        new_available = available_mask & ~(1 << node) & ~conflict_masks[node]
        bbb(new_available, current_size + 1, current_mask | (1 << node))
        
        # Branch 2: exclude node
        bbb(available_mask & ~(1 << node), current_size, current_mask)
    
    # Start B&B from the full set
    bbb(all_mask, 0, 0)
    
    # Check if we found a better solution
    if len(best_set_global[0]) > len(best_set):
        best_set = best_set_global[0]
    
    # Local search polish: try to add nodes that don't conflict
    improved = True
    while improved:
        improved = False
        for idx in range(M):
            if idx not in best_set:
                if all(j not in best_set for j in adj[idx]):
                    best_set.add(idx)
                    improved = True
    
    # Convert back to original values
    result = [valid_nodes[i] for i in sorted(best_set)]
    
    if not result:
        result = [1]
    
    return result
# EVOLVE-BLOCK-END
