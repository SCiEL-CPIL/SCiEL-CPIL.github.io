"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
from collections import defaultdict

def construct():
    """
    Return a nonempty subset of [1, 180] with no THREE DISTINCT elements a,b,c
    satisfying 1/a = 1/b + 1/c. Maximize cardinality.
    """
    N = 180
    
    # Precompute all forbidden triples {a, b, c} with a, b, c distinct and 1/a = 1/b + 1/c
    # Note: a is always the smallest in the triple.
    triples_by_vertex = defaultdict(list)
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    triple = (a, b, c)
                    triples_by_vertex[a].append((b, c))
                    triples_by_vertex[b].append((a, c))
                    triples_by_vertex[c].append((a, b))
    
    # Heuristic Initial Solution: All Odds
    # We proved that no 3 distinct odds form a triple.
    initial_set = set(i for i in range(1, N + 1) if i % 2 == 1)
    best_set = set(initial_set)
    best_size = len(best_set)
    
    # Branch and Bound
    # State:
    # - idx: current vertex index in the processing order
    # - chosen: set of vertices chosen so far
    # - constraints: list of pairs (u, v) that cannot both be chosen
    # - excluded: set of vertices that are forced to be excluded
    
    # We process vertices in decreasing order (N down to 1).
    # This is because if we choose a large vertex v, it only constrains pairs of smaller vertices.
    # If we choose a small vertex a, it constrains pairs of larger vertices.
    # Processing large to small might allow earlier pruning?
    # Let's stick to decreasing order.
    
    order = list(range(N, 0, -1))
    
    # To speed up B&B, we use a time limit.
    start_time = time.time()
    time_limit = 45.0
    
    # We'll use a recursive B&B with a global "best" tracker.
    # To avoid deep recursion issues, we can use an explicit stack or rely on Python's recursion limit (set high).
    sys.setrecursionlimit(10000)
    
    # Precompute for each vertex v, the list of pairs (u, w) it conflicts with.
    # This is triples_by_vertex.
    
    def upper_bound(chosen, constraints, excluded, remaining_indices):
        # Simple bound: current size + number of remaining vertices not excluded
        # We can do better by ignoring constraints for a moment?
        # No, constraints are pairwise.
        # A tighter bound: Greedy Independent Set on the constraint graph of remaining vertices?
        # The constraint graph is dynamic.
        # For now, use |chosen| + |remaining \ excluded|.
        rem_count = 0
        for i in remaining_indices:
            if i not in excluded:
                rem_count += 1
        return len(chosen) + rem_count

    def solve(idx, chosen, constraints, excluded):
        nonlocal best_set, best_size, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
        
        # Pruning
        if len(chosen) + (N - idx) <= best_size:
            return
            
        if idx == N:
            if len(chosen) > best_size:
                best_size = len(chosen)
                best_set = set(chosen)
            return
        
        v = order[idx]
        
        # Option 1: Exclude v
        # No new constraints
        solve(idx + 1, chosen, constraints, excluded)
        
        # Option 2: Include v
        if v not in excluded:
            # Check if v conflicts with any already chosen vertex?
            # No, conflicts are pairs.
            # But if v is in a constraint pair (u, w) and u is chosen, then w is excluded.
            # If w is chosen, then u is excluded.
            # If both u, w are chosen, it's a conflict (should have been pruned).
            
            new_chosen = set(chosen)
            new_chosen.add(v)
            new_constraints = list(constraints)
            new_excluded = set(excluded)
            
            conflict = False
            # Add constraints for v
            for (u, w) in triples_by_vertex[v]:
                # Constraint: not (u and w)
                
                # Check against existing chosen
                if u in new_chosen:
                    if w in new_chosen:
                        conflict = True
                        break
                    else:
                        # w must be excluded
                        new_excluded.add(w)
                elif w in new_chosen:
                    if u in new_chosen:
                        conflict = True
                        break
                    else:
                        # u must be excluded
                        new_excluded.add(u)
                else:
                    # Both undecided or one excluded
                    # If one is already excluded, the constraint is satisfied.
                    if u in new_excluded or w in new_excluded:
                        continue
                    else:
                        # Add constraint
                        # Normalize pair
                        pair = (min(u, w), max(u, w))
                        new_constraints.append(pair)
            
            if not conflict:
                # Check consistency of new_excluded with chosen
                # If any chosen vertex is in new_excluded, it's a conflict.
                # This can happen if we excluded w because u was chosen, but w was already chosen?
                # We checked "if w in new_chosen" above.
                # But what if w was in `chosen` (old) and we added it to `new_excluded`?
                # The logic above:
                # if u in new_chosen:
                #    if w in new_chosen: conflict
                #    else: new_excluded.add(w)
                # This ensures we don't exclude a chosen vertex.
                
                # Check if any new_excluded vertex is in new_chosen
                # This is redundant if the above logic is correct, but safe to check.
                if new_chosen.isdisjoint(new_excluded):
                     # Additional Pruning: Check if any constraint is violated by chosen?
                     # Constraints are only on undecided vertices mostly.
                     # If a constraint (u,w) has u in chosen, w must be excluded.
                     # We handled this when adding constraints.
                     
                     # Upper Bound Check
                     remaining_indices = list(range(idx + 1, N))
                     ub = len(new_chosen) + sum(1 for i in remaining_indices if i not in new_excluded)
                     if ub > best_size:
                         solve(idx + 1, new_chosen, new_constraints, new_excluded)

    # Run B&B
    solve(0, set(), [], set())
    
    # Fallback if B&B didn't find anything better (shouldn't happen as initial is valid)
    if not best_set:
         best_set = set(initial_set)
         
    return sorted(best_set)

# EVOLVE-BLOCK-END
