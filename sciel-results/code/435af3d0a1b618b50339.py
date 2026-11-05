"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

N = 512
DEADLINE = 85.0  # Safety margin before 90s kill

def _lcm(a, b):
    return a // gcd(a, b) * b

def _is_compatible_with_set(x, S):
    """Check if x can be added to set S (all pairs have lcm <= N)."""
    for y in S:
        if _lcm(x, y) > N:
            return False
    return True

def _build_conflict_graph():
    """Build conflict graph: edge between a,b if lcm(a,b) > N."""
    # Use adjacency as sets for fast lookup
    adj = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if _lcm(a, b) > N:
                adj[a].add(b)
                adj[b].add(a)
    return adj

def _greedy_color_upper_bound(adj, candidates):
    """
    Compute an upper bound on the maximum independent set size in candidates
    using greedy coloring. The chromatic number is an upper bound on the
    independence number... wait, no. The independence number is at most the
    number of colors in any valid coloring. So we need a proper coloring
    of the induced subgraph on candidates, and the number of colors used
    is an upper bound on alpha (independence number).
    """
    if not candidates:
        return 0
    
    # Greedy coloring: assign colors to nodes in candidates
    color = {}
    nodes = list(candidates)
    for node in nodes:
        # Find the smallest color not used by any neighbor in candidates
        used_colors = set()
        for neighbor in adj[node]:
            if neighbor in color:
                used_colors.add(color[neighbor])
        c = 1
        while c in used_colors:
            c += 1
        color[node] = c
    
    return max(color.values()) if color else 0

def _branch_and_bound(adj, candidates, current_set, lower_bound, start_time, deadline):
    """
    Branch and bound for maximum independent set.
    candidates: set of nodes still available to add
    current_set: set of nodes already selected
    Returns the best independent set found.
    """
    best_set = set(current_set)
    best_size = len(current_set)
    
    if not candidates or (time.time() - start_time) > deadline:
        return best_set
    
    # Upper bound: current size + greedy coloring of remaining candidates
    ub = len(current_set) + _greedy_color_upper_bound(adj, candidates)
    if ub <= best_size:
        return best_set
    
    # Select a node to branch on: pick the one with highest degree in candidates
    # (or just pick the first one for simplicity, but degree-based is better)
    node = None
    max_deg = -1
    for v in candidates:
        deg = len(adj[v] & candidates)
        if deg > max_deg:
            max_deg = deg
            node = v
    
    if node is None:
        return best_set
    
    # Branch 1: include node
    new_candidates = candidates - {node} - adj[node]
    new_current = current_set | {node}
    
    result1 = _branch_and_bound(adj, new_candidates, new_current, lower_bound, start_time, deadline)
    if len(result1) > best_size:
        best_size = len(result1)
        best_set = result1
    
    # Branch 2: exclude node
    new_candidates2 = candidates - {node}
    result2 = _branch_and_bound(adj, new_candidates2, current_set, lower_bound, start_time, deadline)
    if len(result2) > best_size:
        best_size = len(result2)
        best_set = result2
    
    return best_set

def _try_divisors_of(m, start_time, deadline):
    """Get divisors of m, then greedily add compatible elements."""
    divs = []
    for i in range(1, int(m**0.5) + 1):
        if m % i == 0:
            divs.append(i)
            if i != m // i:
                divs.append(m // i)
    divs = [d for d in divs if d <= N]
    S = set(divs)
    # Greedy add
    for x in range(1, N + 1):
        if (time.time() - start_time) > deadline:
            break
        if x not in S and _is_compatible_with_set(x, S):
            S.add(x)
    return S

def _local_search(S, start_time, deadline):
    """Local search: try to swap out elements for better ones to grow the set."""
    S = set(S)
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        current_size = len(S)
        S_list = sorted(S)
        
        for i, elem in enumerate(S_list):
            if (time.time() - start_time) > deadline:
                break
            S_without = S - {elem}
            for x in range(1, N + 1):
                if x in S:
                    continue
                if _is_compatible_with_set(x, S_without):
                    new_S = S_without | {x}
                    if len(new_S) > current_size:
                        S = new_S
                        improved = True
                        break
            if improved:
                break
    return S

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_S = set()
    best_size = 0
    
    # Build the conflict graph once
    adj = _build_conflict_graph()
    
    # Strategy 1: Divisors of highly composite numbers
    for m in [360, 240, 180, 120, 480, 336, 420, 504]:
        if (time.time() - start_time) > deadline:
            break
        S = _try_divisors_of(m, start_time, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # Strategy 2: Branch and bound on the conflict graph
    if (time.time() - start_time) < deadline:
        # Start with the best set found so far as a lower bound
        initial_lower = len(best_S)
        
        # Candidates: all numbers from 1 to N
        candidates = set(range(1, N + 1))
        
        # Use branch and bound to find a larger independent set
        # We'll do a limited search: try to add nodes to the current best set
        # by exploring which non-conflicting nodes can be added
        
        # First, compute which nodes conflict with the current best set
        conflicting_with_best = set()
        for node in best_S:
            conflicting_with_best |= adj[node]
        
        # Nodes that can potentially be added (don't conflict with best_S)
        addable = candidates - best_S - conflicting_with_best
        
        # Now do branch and bound on addable nodes, starting from best_S
        # But we need to be careful: adding a node might conflict with other addable nodes
        # So we need to find an independent set in the induced subgraph on addable
        
        # For efficiency, let's do a greedy approach first, then branch and bound
        # Greedy: sort addable by degree (ascending) and add if compatible
        addable_list = sorted(addable, key=lambda x: len(adj[x] & addable))
        greedy_add = set(best_S)
        for x in addable_list:
            if (time.time() - start_time) > deadline:
                break
            # Check if x conflicts with any node in greedy_add that was added from addable
            # Actually, we need to check against all of greedy_add
            compatible = True
            for y in greedy_add:
                if x != y and _lcm(x, y) > N:
                    compatible = False
                    break
            if compatible:
                greedy_add.add(x)
        
        if len(greedy_add) > best_size:
            best_size = len(greedy_add)
            best_S = greedy_add
        
        # Now try branch and bound to improve further
        # Rebuild candidates: nodes not in best_S that don't conflict with best_S
        conflicting_with_best = set()
        for node in best_S:
            conflicting_with_best |= adj[node]
        addable = set(range(1, N + 1)) - best_S - conflicting_with_best
        
        if addable and (time.time() - start_time) < deadline - 10:
            # Branch and bound on the induced subgraph
            # To make this tractable, we'll limit the search
            # Use a recursive approach with pruning
            best_bnb = set(best_S)
            
            def bnb_search(cands, current, best_ref, start_t, dl):
                nonlocal best_bnb, best_size
                if not cands or (time.time() - start_t) > dl:
                    return
                if len(current) > len(best_bnb):
                    best_bnb = set(current)
                    best_size = len(current)
                
                # Upper bound
                ub = len(current) + _greedy_color_upper_bound(adj, cands)
                if ub <= len(best_bnb):
                    return
                
                # Pick a node to branch on
                node = None
                max_deg = -1
                for v in cands:
                    deg = len(adj[v] & cands)
                    if deg > max_deg:
                        max_deg = deg
                        node = v
                
                if node is None:
                    return
                
                # Branch 1: include node
                new_cands = cands - {node} - adj[node]
                new_current = current | {node}
                bnb_search(new_cands, new_current, best_ref, start_t, dl)
                
                # Branch 2: exclude node
                new_cands2 = cands - {node}
                bnb_search(new_cands2, current, best_ref, start_t, dl)
            
            bnb_search(addable, set(best_S), best_S, start_time, deadline - 5)
            
            if len(best_bnb) > best_size:
                best_size = len(best_bnb)
                best_S = best_bnb
    
    # Strategy 3: Local search from best
    if (time.time() - start_time) < deadline:
        S_final = _local_search(best_S, start_time, deadline)
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_S = S_final
    
    if not best_S:
        best_S = {1}
    
    return list(best_S)
# EVOLVE-BLOCK-END
