import time
import random

_N = 256
_DEADLINE = 85.0

def _build_conflict_bitsets():
    """Build bitsets where bit i is set if i conflicts with j (i.e., i*j divisible by i+j)."""
    # For each node i, a bitset of all j that conflict with i
    # We'll use Python integers as bitsets (bit i represents node i+1)
    conflict = [0] * (_N + 1)
    for i in range(1, _N + 1):
        mask = 0
        for j in range(i + 1, _N + 1):
            if (i * j) % (i + j) == 0:
                mask |= (1 << j)
        conflict[i] = mask
        # Also set the reverse direction for j
        conflict[j] |= (1 << i)
    return conflict

def _greedy_init(conflict, order):
    """Greedy construction: add nodes in given order if no conflict with current set."""
    S = []
    S_set = set()
    S_mask = 0
    for x in order:
        if not (conflict[x] & S_mask):
            S.append(x)
            S_set.add(x)
            S_mask |= (1 << x)
    return S, S_set, S_mask

def _branch_and_bound(conflict, nodes, current_mask, current_size, best_size, time_limit, t0):
    """
    Branch and bound for maximum independent set.
    nodes: list of candidate nodes still available
    current_mask: bitset of nodes in current solution
    current_size: number of nodes in current solution
    best_size: best size found so far
    Returns (best_size, best_mask)
    """
    if time.time() > time_limit:
        return best_size, (1 << 1) - 1  # dummy mask
    
    if not nodes:
        return best_size, current_mask
    
    # Upper bound: current_size + len(nodes)
    if current_size + len(nodes) <= best_size:
        return best_size, current_mask
    
    # Try to find a good ordering: pick nodes with fewest conflicts among remaining
    # Use a simple heuristic: process nodes in order of increasing degree in remaining set
    remaining_mask = 0
    for x in nodes:
        remaining_mask |= (1 << x)
    
    # Sort nodes by number of conflicts within remaining set (ascending)
    def degree_in_remaining(x):
        return bin(conflict[x] & remaining_mask).count('1')
    
    nodes_sorted = sorted(nodes, key=degree_in_remaining)
    
    best_mask = current_mask
    
    # Branch: for each node, try including it
    i = 0
    while i < len(nodes_sorted):
        if time.time() > time_limit:
            break
        
        x = nodes_sorted[i]
        
        # Try including x
        new_mask = current_mask | (1 << x)
        new_nodes = [y for y in nodes_sorted[i+1:] if not (conflict[y] & (1 << x))]
        
        # Quick upper bound check
        if current_size + 1 + len(new_nodes) > best_size:
            # Recurse
            best_size, best_mask = _branch_and_bound(
                conflict, new_nodes, new_mask, current_size + 1, 
                best_size, time_limit, t0
            )
        
        # Update remaining mask for exclusion
        # Remove x from consideration
        i += 1
    
    return best_size, best_mask

def _local_search(conflict, S_set, S_mask, time_limit):
    """Local search: try to add nodes, then try swaps."""
    S = list(S_set)
    S_set = set(S)
    S_mask = S_mask
    complement = [x for x in range(1, _N + 1) if x not in S_set]
    start = time.time()
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        
        # Phase 1: Add any node that fits
        for x in list(complement):
            if not (conflict[x] & S_mask):
                S.append(x)
                S_set.add(x)
                S_mask |= (1 << x)
                complement.remove(x)
                improved = True
        
        if time.time() >= time_limit:
            break
        
        # Phase 2: Try 1-for-k swaps
        if not improved:
            for u in list(S):
                if time.time() >= time_limit:
                    break
                
                # Remove u
                S_set.discard(u)
                S_mask &= ~(1 << u)
                
                # Try to add multiple nodes
                added_any = False
                for x in list(complement):
                    if not (conflict[x] & S_mask):
                        S_set.add(x)
                        S_mask |= (1 << x)
                        complement.remove(x)
                        added_any = True
                
                if added_any:
                    improved = True
                    break
                else:
                    # Restore u
                    S_set.add(u)
                    S_mask |= (1 << u)
                    complement.append(u)
    
    return S, S_set, S_mask

def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    # Build conflict bitsets
    conflict = _build_conflict_bitsets()
    
    best_S = []
    best_size = 0
    best_mask = 0
    
    # Try multiple greedy initializations with different orderings
    for i in range(200):
        if time.time() - t0 > _DEADLINE - 15:
            break
        
        # Different ordering strategies
        if i % 4 == 0:
            # By degree (fewest conflicts first)
            degrees = [bin(conflict[x]).count('1') for x in range(1, _N + 1)]
            order = list(range(1, _N + 1))
            order.sort(key=lambda x: (degrees[x - 1], x))
        elif i % 4 == 1:
            # Random order
            order = list(range(1, _N + 1))
            rng.shuffle(order)
        elif i % 4 == 2:
            # Reverse degree (most conflicts first - sometimes helps)
            degrees = [bin(conflict[x]).count('1') for x in range(1, _N + 1)]
            order = list(range(1, _N + 1))
            order.sort(key=lambda x: (-degrees[x - 1], x))
        else:
            # Least conflicting dynamic greedy
            S, S_set, S_mask = [], set(), 0
            remaining = set(range(1, _N + 1))
            while remaining:
                best_x = None
                best_c = None
                for x in remaining:
                    c = bin(conflict[x] & S_mask).count('1')
                    if best_c is None or c < best_c:
                        best_x = x
                        best_c = c
                    if c == 0:
                        break
                if best_c == 0:
                    S.append(best_x)
                    S_set.add(best_x)
                    S_mask |= (1 << best_x)
                    remaining.discard(best_x)
                else:
                    break
            if len(S) > best_size:
                best_S = S
                best_size = len(S)
                best_mask = S_mask
            continue
        
        S, S_set, S_mask = _greedy_init(conflict, order)
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
            best_mask = S_mask
    
    # Local search on best found
    if time.time() - t0 < _DEADLINE - 5:
        S_final, S_set_final, S_mask_final = _local_search(conflict, set(best_S), best_mask, _DEADLINE - 5)
        if len(S_final) > best_size:
            best_S = S_final
            best_size = len(S_final)
            best_mask = S_mask_final
    
    # Try branch and bound from best solution
    if time.time() - t0 < _DEADLINE - 2:
        # Get remaining nodes
        remaining = [x for x in range(1, _N + 1) if x not in set(best_S)]
        bb_size, bb_mask = _branch_and_bound(
            conflict, remaining, best_mask, best_size,
            best_size, _DEADLINE - 2, t0
        )
        if bb_size > best_size:
            # Extract solution from mask
            new_S = [x for x in range(1, _N + 1) if (bb_mask >> x) & 1]
            best_S = new_S
            best_size = bb_size
            best_mask = bb_mask
    
    # Final local search polish
    if time.time() - t0 < _DEADLINE - 1:
        S_final, S_set_final, S_mask_final = _local_search(conflict, set(best_S), best_mask, _DEADLINE - 1)
        if len(S_final) > best_size:
            best_S = S_final
            best_size = len(S_final)
            best_mask = S_mask_final
    
    return best_S
