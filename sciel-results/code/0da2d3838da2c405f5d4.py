"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

N = 512
DEADLINE = 85.0  # Safety margin before 90s kill

def _lcm(a, b):
    return a // gcd(a, b) * b

def _build_conflict_bitsets():
    """Build conflict graph as bitsets (Python ints) for fast operations."""
    # conflict_bits[i] is a bitmask where bit j is set if lcm(i,j) > N
    conflict_bits = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if _lcm(a, b) > N:
                conflict_bits[a] |= (1 << b)
                conflict_bits[b] |= (1 << a)
    return conflict_bits

def _greedy_color_upper_bound_fast(conflict_bits, candidates_mask):
    """
    Fast greedy coloring upper bound using bit operations.
    Returns upper bound on max independent set size.
    """
    if candidates_mask == 0:
        return 0
    
    remaining = candidates_mask
    num_colors = 0
    
    while remaining:
        num_colors += 1
        # Greedily pick a maximal independent set from remaining
        # Use a simple heuristic: pick nodes with lowest degree first
        current_color_set = 0
        uncolored = remaining
        
        # Try to build a large independent set for this color
        # Process nodes in order of increasing degree
        temp = uncolored
        nodes_by_degree = []
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            temp ^= lsb
            deg = bin(conflict_bits[v] & uncolored).count('1')
            nodes_by_degree.append((deg, v))
        nodes_by_degree.sort()
        
        for deg, v in nodes_by_degree:
            if not (uncolored & (1 << v)):
                continue
            # Check if v is compatible with all nodes in current_color_set
            if not (conflict_bits[v] & current_color_set):
                current_color_set |= (1 << v)
                uncolored &= ~(1 << v)
        
        remaining = uncolored
    
    return num_colors

def _max_independent_set_bnb(conflict_bits, initial_set, candidates_mask, start_time, deadline):
    """
    Branch and bound for maximum independent set with improved heuristics.
    initial_set: set of already-selected nodes
    candidates_mask: bitmask of nodes that can still be added (no conflicts with initial_set)
    Returns the best independent set found.
    """
    best_set = set(initial_set)
    best_size = len(initial_set)
    node_count = 0
    
    def bnb(cands_mask, current_set):
        nonlocal best_set, best_size, node_count
        
        if (time.time() - start_time) > deadline:
            return
        
        if cands_mask == 0:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
        
        # Quick upper bound: if even taking all candidates doesn't beat best, prune
        cands_count = bin(cands_mask).count('1')
        if len(current_set) + cands_count <= best_size:
            return
        
        # Upper bound: current size + greedy coloring of remaining candidates
        ub = len(current_set) + _greedy_color_upper_bound_fast(conflict_bits, cands_mask)
        if ub <= best_size:
            return
        
        # Select a node to branch on: pick the one with highest degree in cands
        # but prefer nodes that are in many potential solutions
        node = None
        max_score = -1
        temp_mask = cands_mask
        while temp_mask:
            lsb = temp_mask & (-temp_mask)
            v = lsb.bit_length() - 1
            temp_mask ^= lsb
            deg = bin(conflict_bits[v] & cands_mask).count('1')
            # Score: high degree means more conflicts, so including it removes more options
            # But we want to branch on nodes that are "important" - high degree
            score = deg
            if score > max_score:
                max_score = score
                node = v
        
        if node is None:
            return
        
        # Branch 1: include node (try first since it might lead to larger sets)
        new_cands = cands_mask & ~(1 << node) & ~conflict_bits[node]
        new_current = current_set | {node}
        bnb(new_cands, new_current)
        
        # Branch 2: exclude node
        new_cands2 = cands_mask & ~(1 << node)
        bnb(new_cands2, current_set)
    
    bnb(candidates_mask, initial_set)
    return best_set

def _get_divisors(m):
    """Get all divisors of m that are <= N."""
    divs = set()
    for i in range(1, int(m**0.5) + 1):
        if m % i == 0:
            if i <= N:
                divs.add(i)
            if m // i <= N:
                divs.add(m // i)
    return divs

def _greedy_fill(conflict_bits, S, start_time, deadline):
    """Greedily add compatible elements to set S using bitset operations."""
    S = set(S)
    
    # Build a bitmask of current S
    S_mask = 0
    for x in S:
        S_mask |= (1 << x)
    
    # Find all candidates that don't conflict with S
    # A candidate x is compatible if conflict_bits[x] & S_mask == 0
    # But we also need to ensure pairwise compatibility among added elements
    
    # Use a greedy approach: repeatedly add the element with fewest conflicts
    remaining = 0
    for x in range(1, N + 1):
        if not (S_mask & (1 << x)):
            if not (conflict_bits[x] & S_mask):
                remaining |= (1 << x)
    
    # Now greedily add from remaining, preferring elements with fewer conflicts among remaining
    while remaining:
        if (time.time() - start_time) > deadline:
            break
        
        # Find element in remaining with fewest conflicts within remaining
        best_x = None
        min_conf = float('inf')
        temp = remaining
        while temp:
            lsb = temp & (-temp)
            x = lsb.bit_length() - 1
            temp ^= lsb
            conf = bin(conflict_bits[x] & remaining).count('1')
            if conf < min_conf:
                min_conf = conf
                best_x = x
        
        if best_x is None:
            break
        
        # Add best_x to S
        S.add(best_x)
        S_mask |= (1 << best_x)
        # Remove best_x and all its conflicts from remaining
        remaining &= ~(1 << best_x)
        remaining &= ~conflict_bits[best_x]
    
    return S

def _local_search_fast(conflict_bits, S, start_time, deadline):
    """Fast local search: try to add elements that become compatible after removing one element."""
    S = set(S)
    S_mask = 0
    for x in S:
        S_mask |= (1 << x)
    
    improved = True
    iterations = 0
    
    while improved and (time.time() - start_time) < deadline and iterations < 50:
        improved = False
        iterations += 1
        
        # Try 1-swap: remove one element, try to add one or more new elements
        for elem in list(S):
            if (time.time() - start_time) > deadline:
                return S
            
            S_without_mask = S_mask & ~(1 << elem)
            
            # Find all elements not in S that are compatible with S_without
            addable = []
            for x in range(1, N + 1):
                if x == elem:
                    continue
                if S_mask & (1 << x):
                    continue
                if not (conflict_bits[x] & S_without_mask):
                    addable.append(x)
            
            if not addable:
                continue
            
            # Try to greedily fill from addable
            new_S_mask = S_without_mask
            new_S = S - {elem}
            
            # Sort addable by number of conflicts within addable
            addable_with_conf = []
            for x in addable:
                conf = bin(conflict_bits[x] & 0).count('1')  # Will recalculate below
                addable_with_conf.append((x, conf))
            
            # Greedily add from addable
            remaining_add = set(addable)
            while remaining_add:
                # Pick element with fewest conflicts in remaining_add
                best_x = None
                min_conf = float('inf')
                for x in remaining_add:
                    conf = 0
                    for y in remaining_add:
                        if y != x and (conflict_bits[x] >> y) & 1:
                            conf += 1
                    if conf < min_conf:
                        min_conf = conf
                        best_x = x
                
                if best_x is None:
                    break
                
                new_S.add(best_x)
                new_S_mask |= (1 << best_x)
                remaining_add.discard(best_x)
                # Remove conflicts of best_x from remaining_add
                to_remove = []
                for y in remaining_add:
                    if (conflict_bits[best_x] >> y) & 1:
                        to_remove.append(y)
                for y in to_remove:
                    remaining_add.discard(y)
            
            if len(new_S) > len(S):
                S = new_S
                S_mask = new_S_mask
                improved = True
                break
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Build conflict graph as bitsets
    conflict_bits = _build_conflict_bitsets()
    
    best_S = set()
    best_size = 0
    
    # Strategy 1: Try divisor sets of highly composite numbers
    # These give good initial lower bounds
    high_composite_numbers = [360, 240, 180, 120, 480, 336, 420, 504, 288, 192, 168, 144, 96, 72, 48, 36, 24, 12, 6, 4, 2, 1]
    
    seen_divisor_sets = set()
    for m in high_composite_numbers:
        if (time.time() - start_time) > deadline:
            break
        divs = _get_divisors(m)
        # Use a hash of the divisor set to avoid duplicates
        divs_frozen = frozenset(divs)
        if divs_frozen in seen_divisor_sets:
            continue
        seen_divisor_sets.add(divs_frozen)
        
        # Greedy fill
        S = _greedy_fill(conflict_bits, divs, start_time, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # Ensure we have at least {1}
    if not best_S:
        best_S = {1}
        best_size = 1
    
    # Strategy 2: Branch and bound to improve
    if (time.time() - start_time) < deadline - 5:
        # Compute which nodes can be added to best_S (no conflicts)
        best_S_mask = 0
        for x in best_S:
            best_S_mask |= (1 << x)
        
        candidates_mask = 0
        for x in range(1, N + 1):
            if not (best_S_mask & (1 << x)):
                if not (conflict_bits[x] & best_S_mask):
                    candidates_mask |= (1 << x)
        
        if candidates_mask:
            # Run B&B
            result = _max_independent_set_bnb(conflict_bits, best_S, candidates_mask, start_time, deadline - 5)
            if len(result) > best_size:
                best_size = len(result)
                best_S = result
    
    # Strategy 3: Local search
    if (time.time() - start_time) < deadline - 2:
        S_final = _local_search_fast(conflict_bits, best_S, start_time, deadline - 2)
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_S = S_final
    
    # Final greedy fill
    if (time.time() - start_time) < deadline - 1:
        S_final = _greedy_fill(conflict_bits, best_S, start_time, deadline - 1)
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_S = S_final
    
    if not best_S:
        best_S = {1}
    
    return list(best_S)
# EVOLVE-BLOCK-END
