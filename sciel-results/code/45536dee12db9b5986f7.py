import time
import itertools
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_bad_triple(a, b, c):
    lab = _lcm(a, b)
    if lab != _lcm(a, c):
        return False
    return lab == _lcm(b, c)

def construct():
    start_time = time.time()
    # The evaluator kills after 90s. We use a safety margin of 5s.
    deadline = start_time + 85
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute all bad triples and conflict masks
    # conflict_mask[i] is a bitmask of all nodes j such that there exists k 
    # where (i, j, k) is a bad triple.
    # Wait, that's not quite right for the "can_add" check.
    # To check if adding node `v` to `selected` creates a bad triple:
    # We need to check if there exist u, w in `selected` such that (u, v, w) is a bad triple.
    # This is equivalent to checking if `selected` intersects with `partners[v]` where 
    # `partners[v]` is the set of all nodes that form a bad triple with `v` and some other node?
    # No.
    # Let's define `conflict_pairs[v]` as a set of (u, w) pairs? No, too complex.
    # Let's stick to the definition:
    # A triple (a, b, c) is bad.
    # If we add `v` to `S`, we violate if there exist `u, w in S` such that `{u, v, w}` is a bad triple.
    # This means for a fixed `v`, the set of pairs `{u, w}` that form a bad triple with `v` must not have both `u` and `w` in `S`.
    
    # Efficient check:
    # For each node `v`, precompute a list of masks `M_v` such that if `S` contains any mask from `M_v` fully, it's bad?
    # No, that's not right either.
    
    # Let's use the structure from the current program but optimize it.
    # conflict[v][u] = bitmask of w such that (v, u, w) is bad.
    # Check: for all u in S, if (S & conflict[v][u]) != 0, then adding v is bad.
    
    # Precomputation
    conflict = [dict() for _ in range(n)]
    for a, b, c in itertools.combinations(elements, 3):
        if _is_bad_triple(a, b, c):
            ia, ib, ic = a - 1, b - 1, c - 1
            conflict[ia].setdefault(ib, 0)
            conflict[ia][ib] |= (1 << ic)
            conflict[ib].setdefault(ia, 0)
            conflict[ib][ia] |= (1 << ic)
            conflict[ic].setdefault(ia, 0)
            conflict[ic][ia] |= (1 << ib)
            conflict[ic].setdefault(ib, 0)
            conflict[ic][ib] |= (1 << ia)
            
    # Convert to static lists for faster iteration
    # conflict_lists[v] = list of (u, mask_w)
    conflict_lists = []
    for i in range(n):
        conflict_lists.append(list(conflict[i].items()))
        
    # Precompute popcount for 16-bit chunks to speed up bin(x).count('1')
    # Or just use int.bit_count() which is very fast in Python 3.8+
    def popcount(x):
        return x.bit_count()
        
    def can_add(selected_mask, node):
        """Check if adding node to selected_mask doesn't complete any bad triple."""
        for other, partners in conflict_lists[node]:
            if (selected_mask >> other) & 1:
                if selected_mask & partners:
                    return False
        return True
        
    # Greedy initialization with multiple orderings
    def greedy_init(order):
        selected = 0
        for node in order:
            if time.time() > deadline - 5:
                break
            if can_add(selected, node):
                selected |= (1 << node)
        return selected
        
    # Orderings
    conflict_counts = [len(conflict_lists[i]) for i in range(n)]
    order1 = sorted(range(n), key=lambda i: conflict_counts[i])
    order2 = list(reversed(order1))
    order3 = list(range(n))
    order4 = list(reversed(range(n)))
    
    best_mask = 0
    best_size = 0
    for order in [order1, order2, order3, order4]:
        sol = greedy_init(order)
        sz = popcount(sol)
        if sz > best_size:
            best_size = sz
            best_mask = sol
            
    # Branch and Bound
    # We will search for a better solution.
    # State: (idx, selected_mask, remaining_mask)
    # We process nodes in a fixed order (e.g., order1) to ensure consistency and allow memoization if we wanted, 
    # but memoization is hard with masks. We rely on pruning.
    
    # To make B&B effective, we should order nodes such that we decide on "conflicting" nodes early?
    # Or "independent" nodes early?
    # Usually, branching on nodes with high degree (conflicts) first helps prune.
    # Let's sort by conflict count descending for the B&B order.
    bb_order = sorted(range(n), key=lambda i: conflict_counts[i], reverse=True)
    bb_order_idx = {node: i for i, node in enumerate(bb_order)}
    
    # We need to map the B&B decision path to the actual nodes.
    # Let's just iterate through bb_order.
    
    def branch_and_bound(idx, selected_mask, remaining_mask, current_size):
        nonlocal best_size, best_mask
        
        if time.time() > deadline - 2:
            return
            
        # Prune 1: Upper bound
        # Max possible size = current_size + popcount(remaining_mask)
        if current_size + popcount(remaining_mask) <= best_size:
            return
            
        # Prune 2: If no remaining nodes, update best
        if idx >= n:
            if current_size > best_size:
                best_size = current_size
                best_mask = selected_mask
            return
            
        node = bb_order[idx]
        node_bit = 1 << node
        
        # Option 1: Exclude node
        # This is usually the safer branch to try first to reduce remaining set quickly?
        # Or include first to get a good lower bound?
        # Let's try Include first if possible, as it increases current_size, which might tighten the bound for the Exclude branch?
        # Actually, standard MIS heuristics suggest branching on the vertex with highest degree.
        
        # Try Include
        if can_add(selected_mask, node):
            new_sel = selected_mask | node_bit
            new_rem = remaining_mask & ~node_bit
            branch_and_bound(idx + 1, new_sel, new_rem, current_size + 1)
            
        # Try Exclude
        new_rem = remaining_mask & ~node_bit
        branch_and_bound(idx + 1, selected_mask, new_rem, current_size)
        
    # Initialize B&B with the greedy solution?
    # It's hard to "resume" B&B from a partial solution easily with a fixed order.
    # Better to run B&B from scratch, but the greedy solution provides a strong lower bound (best_size).
    # The B&B will prune heavily because best_size is already high.
    
    all_mask = (1 << n) - 1
    branch_and_bound(0, 0, all_mask, 0)
    
    # Final verification and conversion
    result = [i + 1 for i in range(n) if (best_mask >> i) & 1]
    
    if not result:
        return [1]
        
    return result

# EVOLVE-BLOCK-END
