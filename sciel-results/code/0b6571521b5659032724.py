import time
import sys
import math
from collections import defaultdict

def construct(n):
    if n <= 4:
        # Known optimal for small n
        if n == 1: return "1"
        if n == 2: return "121"
        if n == 3: return "123121321"
        if n == 4: return "12341231423124312134213241214321"
        
    # Generate all permutations
    import itertools
    perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    N = len(perms)
    
    # Build the Permutation Graph
    # Edge u -> v if suffix(u, n-1) == prefix(v, n-1)
    # For each u, there is exactly one v.
    next_in_cycle = [0] * N
    
    # Map suffix (length n-1) to permutation index
    # Suffix is a tuple of length n-1
    suffix_to_idx = {}
    for i, p in enumerate(perms):
        suffix = p[1:]
        # There should be only one permutation with this suffix?
        # No, multiple permutations can have the same suffix?
        # If suffix is (2,3,4), then perm must be (x, 2, 3, 4).
        # x must be the missing symbol.
        # So yes, unique.
        suffix_to_idx[suffix] = i
        
    for i, p in enumerate(perms):
        prefix = p[:-1]
        # Find v such that prefix(v) == suffix(p)
        # v is the permutation starting with suffix(p)
        # We can construct v: v = (missing_symbol) + suffix(p)
        # But we need the index.
        # Let's just use a map from prefix to index?
        # Prefix of v is suffix of u.
        # So we need a map: prefix_tuple -> index_of_perm_with_that_prefix
        pass

    # Re-do mapping: prefix to index
    prefix_to_idx = {}
    for i, p in enumerate(perms):
        prefix = p[:-1]
        prefix_to_idx[prefix] = i
        
    for i, p in enumerate(perms):
        suffix = p[1:]
        next_in_cycle[i] = prefix_to_idx[suffix]
        
    # Find all cycles
    visited = [False] * N
    cycles = []
    for i in range(N):
        if not visited[i]:
            cycle = []
            curr = i
            while not visited[curr]:
                visited[curr] = True
                cycle.append(curr)
                curr = next_in_cycle[curr]
            cycles.append(cycle)
            
    # Greedy Stitching
    # State: current_perm_idx, unvisited_cycles (list of cycle lists)
    current_idx = cycles[0][0]
    word = list(perms[current_idx])
    
    unvisited = cycles[1:]
    
    # Precompute for each cycle the list of (node_idx, prev_node_idx)
    # prev_node_idx is the node that points to node_idx in the cycle
    
    # To speed up search, we can just scan all nodes in unvisited cycles.
    
    while unvisited:
        best_cycle_idx = -1
        best_node_idx = -1
        best_cost = float('inf')
        best_overlap = -1
        
        current_perm = perms[current_idx]
        
        for ci, cycle in enumerate(unvisited):
            for node in cycle:
                perm = perms[node]
                # Calculate overlap between current_perm suffix and perm prefix
                # Max overlap k < n
                # We want to append perm[k:]
                # Cost is n - k
                # We want to maximize k.
                # k is the length of the longest suffix of current_perm that is a prefix of perm.
                # Note: k must be < n because if k==n, perm==current_perm, which is not in unvisited.
                
                # Check overlaps from n-1 down to 0
                # Optimization: just check n-1, n-2...
                # For n=7, max 6 checks.
                found_overlap = False
                for k in range(n - 1, -1, -1):
                    if current_perm[n-k:] == perm[:k]:
                        # Overlap k
                        if k > best_overlap:
                            best_overlap = k
                            best_node_idx = node
                            best_cycle_idx = ci
                        break
                # If no overlap found, k=0 is always valid (append whole perm)
                # But we break at first match. If no match down to 0?
                # k=0: current_perm[0:] == perm[:0] (empty) is always True.
                # So we always find at least k=0.
                
        if best_cycle_idx == -1:
            break
            
        # Get the best cycle and node
        target_cycle = unvisited.pop(best_cycle_idx)
        entry_node = best_node_idx
        
        # Append the bridge
        entry_perm = perms[entry_node]
        k = best_overlap
        # Append entry_perm[k:]
        word.extend(entry_perm[k:])
        
        # Update current_idx to entry_node
        current_idx = entry_node
        
        # Traverse the rest of the cycle
        # The cycle is a list of nodes.
        # We are at entry_node.
        # We need to visit all nodes in the cycle.
        # The cycle is directed.
        # Find the position of entry_node in the cycle.
        # Then iterate forward.
        
        # Note: The cycle list `target_cycle` is in order of traversal?
        # My cycle finding:
        # curr = i
        # while not visited:
        #   visited[curr] = True
        #   cycle.append(curr)
        #   curr = next_in_cycle[curr]
        # Yes, `target_cycle` is in the order of the edges.
        
        # Find index of entry_node in target_cycle
        entry_pos = target_cycle.index(entry_node)
        
        # We need to traverse the cycle starting from entry_node.
        # We have already appended the entry_node's "tail" (from bridge).
        # Now we need to append the next nodes.
        # Next node is target_cycle[(entry_pos + 1) % len(target_cycle)]
        
        num_nodes = len(target_cycle)
        for j in range(1, num_nodes):
            next_node = target_cycle[(entry_pos + j) % num_nodes]
            next_perm = perms[next_node]
            # Append the last character of next_perm
            # Because overlap with current is n-1
            word.append(next_perm[-1])
            current_idx = next_node
            
        # After loop, current_idx is the node before entry_node in the cycle.
        # This is the exit node.
        
    return "".join(map(str, word))
