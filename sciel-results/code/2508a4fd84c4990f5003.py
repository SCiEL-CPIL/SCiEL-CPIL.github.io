"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

N = 256

def _build_conflict_graph():
    """Build conflict graph: edge (a,b) exists if there exists c in [1,N]
    such that {a,b,c} forms a 3-AP (non-constant)."""
    # A 3-AP is (x, x+d, x+2d) with d>0.
    # For a pair (a,b) with a<b, they conflict if:
    #   - There exists c such that (a,b,c) is a 3-AP: b-a = c-b => c = 2b-a, need 1<=c<=N, c!=a,b
    #   - There exists c such that (a,c,b) is a 3-AP: c-a = b-c => 2c=a+b, need a+b even, 1<=c<=N, c!=a,b
    #   - There exists c such that (c,a,b) is a 3-AP: a-c = b-a => c = 2a-b, need 1<=c<=N, c!=a,b
    conflicts = [set() for _ in range(N+1)]
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            conflict = False
            # Case 1: a, b, c with c = 2b - a
            c = 2*b - a
            if 1 <= c <= N and c != a and c != b:
                conflict = True
            # Case 2: a, c, b with 2c = a+b
            if (a+b) % 2 == 0:
                c = (a+b)//2
                if 1 <= c <= N and c != a and c != b:
                    conflict = True
            # Case 3: c, a, b with c = 2a - b
            c = 2*a - b
            if 1 <= c <= N and c != a and c != b:
                conflict = True
            if conflict:
                conflicts[a].add(b)
                conflicts[b].add(a)
    return conflicts

def _greedy_max_weight_independent(conflicts, deadline):
    """Greedy: pick nodes in descending weight order if no conflict with already picked."""
    selected = set()
    for x in range(N, 0, -1):  # descending weight = ascending x
        if all(c not in selected for c in conflicts[x]):
            selected.add(x)
    return selected

def _branch_and_bound(conflicts, deadline):
    """Branch and bound for maximum weight independent set.
    Weight of node x is 1/x. Process nodes in order 1..N (highest weight first).
    At each step, decide include or exclude node i."""
    best_weight = 0.0
    best_set = set()
    
    # Precompute weights
    weights = [0.0] + [1.0/x for x in range(1, N+1)]
    
    # For upper bound: sum of weights of all remaining feasible nodes
    # We process nodes 1, 2, ..., N in order.
    # State: (idx, current_set, current_weight)
    # Upper bound at idx: current_weight + sum(weights[i] for i in idx..N if i compatible with current_set)
    
    # To speed up, precompute for each node the set of nodes it conflicts with
    # and use bitmasks or sorted lists.
    
    # Let's use a recursive approach with memoization-like pruning.
    # Order: process 1, 2, ..., N
    
    # For efficiency, we'll track which nodes are still "available" (not conflicting with selected)
    # Use a bitmask or a set of available nodes.
    
    # Since N=256, we can use Python sets or bitmasks.
    # Let's use a list of booleans or a frozenset.
    
    # Actually, let's use a different approach: 
    # At each node i, we can either include it (if compatible) or skip it.
    # The upper bound is current_weight + sum of weights of all j >= i that are compatible with current selection.
    
    # To compute upper bound efficiently, precompute suffix sums of weights.
    suffix_sum = [0.0] * (N+2)
    for i in range(N, 0, -1):
        suffix_sum[i] = suffix_sum[i+1] + weights[i]
    
    # For the bound, we need: among nodes i..N, which are compatible with current_set?
    # A node j is compatible with current_set if no element in current_set conflicts with j.
    # i.e., for all s in current_set, j not in conflicts[s].
    
    # Let's precompute for each subset... no, that's too large.
    # Instead, maintain a "forbidden" set: nodes that conflict with at least one selected node.
    
    sys.setrecursionlimit(10000)
    
    def solve(idx, current_weight, current_set, forbidden):
        nonlocal best_weight, best_set
        
        if time.time() > deadline:
            return
        
        # Upper bound: current_weight + sum of weights[j] for j in [idx..N] if j not in forbidden
        # Quick bound: current_weight + suffix_sum[idx] (ignoring forbidden)
        if current_weight + suffix_sum[idx] <= best_weight:
            return
        
        if idx > N:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set(current_set)
            return
        
        # Try including idx
        if idx not in forbidden:
            new_forbidden = set(forbidden)
            new_forbidden.update(conflicts[idx])
            new_forbidden.add(idx)
            new_set = set(current_set)
            new_set.add(idx)
            solve(idx+1, current_weight + weights[idx], new_set, new_forbidden)
        
        # Try excluding idx
        solve(idx+1, current_weight, current_set, forbidden)
    
    solve(1, 0.0, set(), set())
    return best_set

def _local_search_improve(selected, conflicts, deadline):
    """Improve via single swaps and additions."""
    selected = set(selected)
    best_weight = sum(1.0/x for x in selected)
    best_set = set(selected)
    
    # Try adding each missing element
    for x in range(1, N+1):
        if time.time() > deadline:
            break
        if x not in selected:
            if all(c not in selected for c in conflicts[x]):
                new_set = selected | {x}
                new_weight = best_weight + 1.0/x
                if new_weight > best_weight:
                    best_weight = new_weight
                    best_set = new_set
                    selected = new_set
    
    # Try swaps: remove y, add x
    for x in range(1, N+1):
        if time.time() > deadline:
            break
        if x not in selected:
            for y in list(selected):
                if time.time() > deadline:
                    break
                # Check if x is compatible with selected - {y}
                compatible = True
                for c in conflicts[x]:
                    if c in selected and c != y:
                        compatible = False
                        break
                if compatible:
                    new_weight = best_weight - 1.0/y + 1.0/x
                    if new_weight > best_weight:
                        best_weight = new_weight
                        best_set = (selected - {y}) | {x}
                        selected = set(best_set)
    
    return best_set

def construct():
    deadline = time.time() + 80.0
    
    conflicts = _build_conflict_graph()
    
    # Get a good initial solution via greedy
    initial = _greedy_max_weight_independent(conflicts, deadline)
    
    # Improve with local search
    result = _local_search_improve(initial, conflicts, deadline)
    
    # Try branch and bound if time permits
    if time.time() < deadline - 10:
        bb_result = _branch_and_bound(conflicts, deadline)
        if bb_result:
            bb_weight = sum(1.0/x for x in bb_result)
            cur_weight = sum(1.0/x for x in result)
            if bb_weight > cur_weight:
                result = bb_result
    
    return list(result)
# EVOLVE-BLOCK-END
