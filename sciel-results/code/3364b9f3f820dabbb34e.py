"""Optimized program for finding max subset of [1,128] avoiding equal-lcm triples."""
# EVOLVE-BLOCK-START
import time
import itertools
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_bad_triple(a, b, c):
    """Check if three distinct numbers have all pairwise LCMs equal."""
    lab = _lcm(a, b)
    if lab != _lcm(a, c):
        return False
    return lab == _lcm(b, c)

def construct():
    start_time = time.time()
    deadline = start_time + 88  # 88s safety margin for 90s limit
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute all bad triples
    bad_triples = []
    for a, b, c in itertools.combinations(elements, 3):
        if time.time() > deadline - 20:
            break
        if _is_bad_triple(a, b, c):
            bad_triples.append((a - 1, b - 1, c - 1))  # 0-indexed
    
    if not bad_triples:
        return elements
    
    # Build conflict structures using bitmasks for speed
    # For each node, store bitmask of nodes that form bad triples with it
    # conflict_mask[i] has bit j set if i and j are in some bad triple together
    conflict_mask = [0] * n
    # For each node, store the set of (other1, other2) pairs that form a bad triple
    pairs_by_node = [set() for _ in range(n)]
    for a, b, c in bad_triples:
        pairs_by_node[a].add((b, c))
        pairs_by_node[b].add((a, c))
        pairs_by_node[c].add((a, b))
        conflict_mask[a] |= (1 << b) | (1 << c)
        conflict_mask[b] |= (1 << a) | (1 << c)
        conflict_mask[c] |= (1 << a) | (1 << b)
    
    # Count conflicts per node
    conflict_count = [len(pairs_by_node[i]) for i in range(n)]
    
    # Precompute pair masks for each node: pair_mask[node] is a list of (mask1, mask2) 
    # where mask1 and mask2 are bitmasks of the two other nodes
    # This allows faster can_add checks
    pair_masks = [[] for _ in range(n)]
    for node in range(n):
        for other1, other2 in pairs_by_node[node]:
            pair_masks[node].append(((1 << other1) | (1 << other2), other1, other2))
    
    def can_add(selected_mask, node):
        """Check if adding node to selected doesn't complete any bad triple.
        selected_mask is a bitmask where bit i is 1 if node i is selected."""
        for mask, o1, o2 in pair_masks[node]:
            if (selected_mask & mask) == mask:
                return False
        return True
    
    def greedy_init(order):
        """Greedy initialization with given order of nodes."""
        selected_mask = 0
        for node in order:
            if time.time() > deadline - 10:
                break
            if can_add(selected_mask, node):
                selected_mask |= (1 << node)
        return selected_mask
    
    def mask_to_set(mask):
        """Convert bitmask to set of nodes."""
        return {i for i in range(n) if (mask >> i) & 1}
    
    def local_search_v2(selected_mask, max_iter=5000):
        """Local search: try to add unselected nodes, then try swaps."""
        selected_mask = selected_mask
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > deadline - 15:
                break
            
            # Phase 1: Try to add any unselected node
            for node in range(n):
                if time.time() > deadline - 15:
                    break
                if not (selected_mask & (1 << node)):
                    if can_add(selected_mask, node):
                        selected_mask |= (1 << node)
                        improved = True
            
            # Phase 2: Try swaps - remove one, add two
            if not improved:
                swap_found = False
                for s in range(n):
                    if time.time() > deadline - 15:
                        break
                    if swap_found:
                        break
                    if not (selected_mask & (1 << s)):
                        continue
                    # Try removing s
                    new_mask = selected_mask & ~(1 << s)
                    original_mask = selected_mask
                    # Try adding pairs from unselected
                    for u1 in range(n):
                        if time.time() > deadline - 15:
                            break
                        if swap_found:
                            break
                        if (new_mask & (1 << u1)):
                            continue
                        if can_add(new_mask, u1):
                            mask_with_u1 = new_mask | (1 << u1)
                            for u2 in range(u1 + 1, n):
                                if time.time() > deadline - 15:
                                    break
                                if (mask_with_u1 & (1 << u2)):
                                    continue
                                if can_add(mask_with_u1, u2):
                                    selected_mask = mask_with_u1 | (1 << u2)
                                    swap_found = True
                                    improved = True
                                    break
                    if not swap_found:
                        # No swap found, restore original
                        selected_mask = original_mask
        
        return selected_mask
    
    # Multiple restarts with different orderings
    best_mask = 0
    best_size = 0
    
    # Ordering 1: by conflict count (ascending)
    order1 = sorted(range(n), key=lambda i: conflict_count[i])
    sol1 = greedy_init(order1)
    sol1 = local_search_v2(sol1, max_iter=3000)
    if bin(sol1).count('1') > best_size:
        best_size = bin(sol1).count('1')
        best_mask = sol1
    
    # Ordering 2: reverse
    order2 = list(reversed(order1))
    sol2 = greedy_init(order2)
    sol2 = local_search_v2(sol2, max_iter=3000)
    if bin(sol2).count('1') > best_size:
        best_size = bin(sol2).count('1')
        best_mask = sol2
    
    # Ordering 3: natural order
    order3 = list(range(n))
    sol3 = greedy_init(order3)
    sol3 = local_search_v2(sol3, max_iter=3000)
    if bin(sol3).count('1') > best_size:
        best_size = bin(sol3).count('1')
        best_mask = sol3
    
    # Ordering 4: reverse natural
    order4 = list(range(n - 1, -1, -1))
    sol4 = greedy_init(order4)
    sol4 = local_search_v2(sol4, max_iter=3000)
    if bin(sol4).count('1') > best_size:
        best_size = bin(sol4).count('1')
        best_mask = sol4
    
    # Additional local search passes on best solution
    if time.time() < deadline - 20:
        for _ in range(5):
            if time.time() > deadline - 15:
                break
            improved_mask = local_search_v2(best_mask, max_iter=2000)
            if bin(improved_mask).count('1') > best_size:
                best_size = bin(improved_mask).count('1')
                best_mask = improved_mask
    
    # Final attempt: try to add any remaining nodes
    if time.time() < deadline - 5:
        for node in range(n):
            if not (best_mask & (1 << node)):
                if can_add(best_mask, node):
                    best_mask |= (1 << node)
    
    return [i + 1 for i in range(n) if (best_mask >> i) & 1]

# EVOLVE-BLOCK-END
