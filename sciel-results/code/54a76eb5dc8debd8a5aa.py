"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # 1. Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l

    # 2. Identify conflict graph
    # A conflict exists between pairs if they form a bad triple with some third element?
    # No, the constraint is: No three distinct elements {a, b, c} such that lcm(a,b)==lcm(a,c)==lcm(b,c).
    # This is not a simple pairwise constraint. It's a hypergraph constraint (triples).
    # However, we can model this as a graph where edges represent "potential bad triple partners"?
    # Actually, standard MIS requires pairwise conflicts. 
    # Let's re-evaluate. If we pick a set S, it is valid if for any a,b,c in S, they don't form a bad triple.
    # This is a 3-uniform hypergraph independent set problem.
    # Branch and bound on hypergraphs is harder. 
    # But we can convert this to a graph problem?
    # If we pick a and b, and they are part of a bad triple with c, then we cannot pick c.
    # So, if we have selected a set S, adding x is invalid if there exist a, b in S such that {a, b, x} is a bad triple.
    # This is exactly the logic used in the current greedy program.
    # To use Branch and Bound effectively, we need a good upper bound.
    # Upper bound: |S| + (number of remaining vertices that could potentially be added).
    # A simple bound is |S| + |R|, but that's weak.
    # A better bound: For each remaining vertex v, check if it is "compatible" with S.
    # If not, it can't be added. Let R' be the set of remaining vertices compatible with S.
    # Then max size <= |S| + |R'|.
    # Even better: Compute the maximum independent set of the conflict graph induced by R'?
    # Wait, the conflicts are not pairwise. 
    # Let's define a graph G_conflict where an edge (u, v) exists if there is some w such that {u, v, w} is a bad triple.
    # This is NOT sufficient. If u and v are not connected, they might still conflict with a third node w.
    # The condition is: {a, b, c} is bad.
    # If we have selected a and b, we must exclude all c such that {a, b, c} is bad.
    # Let's precompute for each pair (a, b), the set of c's that form a bad triple with them.
    # bad_with_pair[a][b] = set of c such that lcm(a,b)==lcm(a,c)==lcm(b,c).
    
    bad_with_pair = [[set() for _ in range(n + 1)] for _ in range(n + 1)]
    
    # Precompute bad triples
    # For efficiency, iterate over possible LCM values?
    # LCMs can be large. Max LCM for 1..128 is 128*127 = 16256.
    # Let's just iterate a, b, c.
    # N=128, N^3/6 ~ 350k. This is fast.
    
    triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    triples.append((a, b, c))
                    bad_with_pair[a][b].add(c)
                    bad_with_pair[b][a].add(c)
                    bad_with_pair[a][c].add(b)
                    bad_with_pair[c][a].add(b)
                    bad_with_pair[b][c].add(a)
                    bad_with_pair[c][b].add(a)

    if time.time() > deadline - 5:
        # Fallback to simple greedy
        selected = []
        for x in range(1, 129):
            valid = True
            for i in range(len(selected)):
                for j in range(i+1, len(selected)):
                    a = selected[i]
                    b = selected[j]
                    if x in bad_with_pair[a][b]:
                        valid = False
                        break
                if not valid: break
            if valid:
                selected.append(x)
        return selected

    # Branch and Bound for Maximum Independent Set in a 3-uniform hypergraph
    # State: (S, R) where S is selected, R is remaining candidates.
    # We want to maximize |S|.
    
    # To make B&B efficient, we need a good ordering of vertices and a tight bound.
    # Let's use a recursive backtracking search.
    
    best_set = []
    
    # Precompute degrees or some metric for ordering
    # Vertices involved in many bad triples are more "constrained".
    # Let's sort vertices by degree in the "conflict hypergraph" descending?
    # Or ascending? Usually, branching on constrained variables first is better.
    
    vertex_conflict_count = [0] * (n + 1)
    for a, b, c in triples:
        vertex_conflict_count[a] += 1
        vertex_conflict_count[b] += 1
        vertex_conflict_count[c] += 1
    
    # Order: Most constrained first
    ordered_vertices = sorted(numbers, key=lambda x: vertex_conflict_count[x], reverse=True)
    
    # We will search in this order.
    # State: index in ordered_vertices, current set S.
    # This is a bit tricky because "remaining" depends on S.
    # Standard MIS B&B works on a graph. Here, the "conflict" is dynamic based on pairs in S.
    
    # Alternative approach: Convert to a graph problem?
    # If we pick a, we remove all b such that... no, that's not right.
    # The constraint is only violated if we pick a TRIPLE.
    # So, if we have S, and we consider adding x, we check if any pair in S conflicts with x.
    
    # Let's implement a recursive solver.
    # To prune, we can use an upper bound.
    # UB = |S| + |R| is too weak.
    # Better UB: For each x in R, if x is incompatible with S, it can't be added.
    # Let R_valid = {x in R | x is compatible with S}.
    # Then UB = |S| + |R_valid|.
    # We can further bound |R_valid| by the size of the maximum independent set of the conflict graph induced by R_valid?
    # But the conflict is not pairwise.
    # However, if x and y are in R_valid, can they both be added?
    # They can be added if there is no z in S such that {x, y, z} is bad, AND no pair in S conflicts with x or y individually?
    # Actually, if we add x and y to S, we must ensure that no pair in S_new forms a bad triple with any future element.
    # This is getting complex.
    
    # Let's try a different heuristic: Local Search / Simulated Annealing on the set.
    # Start with a greedy set.
    # Try to swap elements to improve size.
    # Or, use the branch and bound but with a simpler bound:
    # If we are at index i in ordered_vertices, and we have selected S,
    # the maximum possible size is |S| + (len(ordered_vertices) - i).
    # If this is <= best_size, prune.
    # This is a weak bound, but with good ordering (most constrained first), it might work.
    
    # Let's try to improve the bound.
    # For each remaining vertex v, if it conflicts with S, it's out.
    # Let's precompute for each vertex v, the set of pairs (a,b) in S that conflict with v.
    # If any such pair exists, v is out.
    
    def is_compatible(x, S):
        # Check if x can be added to S
        # S is a set of ints
        # For each pair (a, b) in S, if x in bad_with_pair[a][b], return False
        # Optimization: Iterate through S elements? No, pairs are O(|S|^2).
        # Better: Precompute for each x, the list of pairs (a,b) that form a bad triple with x.
        # conflict_pairs[x] = list of (a,b) such that {a,b,x} is bad.
        # Then check if any (a,b) in conflict_pairs[x] has both a,b in S.
        
        for a, b in conflict_pairs[x]:
            if a in S and b in S:
                return False
        return True

    # Precompute conflict_pairs
    conflict_pairs = [set() for _ in range(n + 1)]
    for a, b, c in triples:
        conflict_pairs[c].add((a, b))
        conflict_pairs[b].add((a, c))
        conflict_pairs[a].add((b, c))
        # Store as frozenset for hashing? No, just tuples is fine.
        # To speed up "a in S and b in S", we can keep S as a set.
        
    # To make the check faster, we can maintain a set of "active pairs" in S?
    # No, let's just use the set S.
    
    # Let's try a simple Backtracking with Pruning.
    # Order vertices by degree descending.
    
    sys.setrecursionlimit(10000)
    
    def solve(idx, S, best_ref):
        nonlocal best_set
        
        # Prune 1: Upper bound
        # Max possible size = |S| + (n - idx)
        if len(S) + (n - idx) <= len(best_ref):
            return
        
        # Prune 2: Check if all remaining can be added?
        # If yes, update best and return.
        # This is expensive. Skip.
        
        if idx == n:
            if len(S) > len(best_ref):
                best_set = list(S)
            return
        
        v = ordered_vertices[idx]
        
        # Option 1: Skip v
        solve(idx + 1, S, best_ref)
        
        # Option 2: Include v
        # Check if v is compatible with S
        # We need to check if any pair in conflict_pairs[v] is in S.
        compatible = True
        S_list = list(S) # For faster iteration? No, set is O(1) lookup.
        for a, b in conflict_pairs[v]:
            if a in S and b in S:
                compatible = False
                break
        
        if compatible:
            S.add(v)
            solve(idx + 1, S, best_ref)
            S.remove(v)
            
    # Initialize with a good greedy solution to set a high lower bound
    # Greedy: Pick vertices in order of increasing conflict count?
    # Or just use the current best from a quick greedy pass.
    
    greedy_set = set()
    for v in numbers:
        if is_compatible(v, greedy_set):
            greedy_set.add(v)
    
    best_set = list(greedy_set)
    
    # Run B&B
    # To avoid recursion depth issues and speed up, we can use iterative deepening or just hope it's fast enough.
    # N=128 is small enough for B&B if the bound is tight enough.
    # The bound |S| + (n - idx) is very loose.
    # Let's try to tighten it.
    # For each remaining vertex, if it's incompatible with S, it can't be added.
    # Let's count how many remaining vertices are compatible with S.
    # This is O((n-idx) * |conflict_pairs[v]|).
    
    def tight_bound(idx, S):
        count = 0
        for j in range(idx, n):
            v = ordered_vertices[j]
            # Check if v is compatible with S
            comp = True
            for a, b in conflict_pairs[v]:
                if a in S and b in S:
                    comp = False
                    break
            if comp:
                count += 1
        return len(S) + count

    def solve_tight(idx, S, best_ref):
        nonlocal best_set
        
        # Tight bound
        if tight_bound(idx, S) <= len(best_ref):
            return
            
        if idx == n:
            if len(S) > len(best_ref):
                best_set = list(S)
            return
        
        v = ordered_vertices[idx]
        
        # Option 1: Skip v
        solve_tight(idx + 1, S, best_ref)
        
        # Option 2: Include v
        compatible = True
        for a, b in conflict_pairs[v]:
            if a in S and b in S:
                compatible = False
                break
        
        if compatible:
            S.add(v)
            solve_tight(idx + 1, S, best_ref)
            S.remove(v)
            
    # Run with time check
    # We need to handle the time limit inside the recursion?
    # Python doesn't easily allow breaking out of deep recursion with a time check without global flags.
    
    global_time_check = [False]
    
    def solve_with_time(idx, S, best_ref):
        nonlocal best_set
        if global_time_check[0]:
            return
        if time.time() > deadline - 2:
            global_time_check[0] = True
            return
            
        if tight_bound(idx, S) <= len(best_ref):
            return
            
        if idx == n:
            if len(S) > len(best_ref):
                best_set = list(S)
            return
        
        v = ordered_vertices[idx]
        
        # Try include first if it seems promising?
        # Usually, include first leads to larger sets faster, improving the bound.
        
        compatible = True
        for a, b in conflict_pairs[v]:
            if a in S and b in S:
                compatible = False
                break
        
        if compatible:
            S.add(v)
            solve_with_time(idx + 1, S, best_ref)
            S.remove(v)
            
        # Option 1: Skip v
        solve_with_time(idx + 1, S, best_ref)
        
    S_init = set()
    solve_with_time(0, S_init, best_set)
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
