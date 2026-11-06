"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import gcd, lcm
    import sys
    sys.setrecursionlimit(10000)
    
    N = 128
    # Precompute pairwise LCMs
    # lcm[i][j] for i, j in 1..128
    # We'll use 0-indexed internally, so index i corresponds to value i+1
    
    # Precompute LCM matrix
    lcm_mat = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            lcm_mat[i][j] = (i // g) * j
            lcm_mat[j][i] = lcm_mat[i][j]
    
    # A triple (a, b, c) with a < b < c is bad if:
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    # Let's precompute for each pair (a,b), which c > b would make it bad.
    # Actually, we need to check triples where all three pairwise LCMS are equal.
    
    # For a given pair (a, b), the condition is lcm(a,b) == lcm(a,c) == lcm(b,c).
    # This means c must be such that lcm(a,c) = lcm(a,b) and lcm(b,c) = lcm(a,b).
    # So c must divide lcm(a,b), and lcm(a,c) = lcm(a,b) implies that for every prime power in lcm(a,b),
    # max(v_p(a), v_p(c)) = v_p(lcm(a,b)).
    
    # Let's precompute bad triples as a set of frozensets or just track conflicts.
    # Since we want maximum independent set in 3-uniform hypergraph, we need to avoid any bad triple.
    
    # Alternative approach: For each element x, precompute which pairs (a,b) with a,b < x would form a bad triple with x.
    # When adding x to the selected set, we need to ensure no pair (a,b) in selected forms a bad triple with x.
    
    # Let's precompute for each x (from 1 to 128), the set of pairs (a,b) such that {a,b,x} is a bad triple.
    # We'll store this as a dictionary: conflict_pairs[x] = set of (a,b) tuples where a<b<x and the triple is bad.
    
    # Actually, for the branch-and-bound with bitsets, let's think differently.
    # We'll use a recursive backtracking approach where we decide for each number 1..128 whether to include it.
    # State: current selected set (as a bitset or list), and we process numbers in some order.
    
    # Key insight: The greedy algorithm in the seed processes numbers 1 to 128 in order and includes if no conflict.
    # This gives a valid set but not necessarily maximum.
    
    # For a stronger approach, let's first understand the structure better.
    # A triple (a,b,c) with lcm(a,b)=lcm(a,c)=lcm(b,c)=L means all three divide L and for each prime p|L,
    # at least two of a,b,c have the full p-adic valuation of L.
    
    # Let's precompute all bad triples.
    bad_triples = set()
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            L_ab = lcm_mat[a][b]
            for c in range(b+1, N+1):
                if lcm_mat[a][c] == L_ab and lcm_mat[b][c] == L_ab:
                    bad_triples.add((a,b,c))
    
    # Now we have all bad triples. Let's count them.
    # For the branch-and-bound, we'll use a recursive approach with memoization or just careful pruning.
    
    # Let's use a different strategy: since N=128 is small enough, we can use a maximum independent set approach
    # on the 3-uniform hypergraph using branch and bound.
    
    # Represent the hypergraph: vertices are 1..128, hyperedges are bad triples.
    # We want the maximum subset with no hyperedge fully contained.
    
    # Branch and bound:
    # - Process vertices in some order (e.g., 1 to 128)
    # - At each step, decide to include or exclude the current vertex
    # - Prune if the current size + remaining vertices < best found
    
    # For efficiency, we need to quickly check if adding a vertex creates a bad triple with two already-selected vertices.
    # Precompute for each vertex v, the list of pairs (u,w) with u<w<v such that (u,w,v) is a bad triple.
    
    # Let's precompute: for each v, conflicts[v] = list of (u,w) pairs
    conflicts = [[] for _ in range(N+1)]
    for (a,b,c) in bad_triples:
        # c is the largest
        conflicts[c].append((a,b))
        # Also, a and b could be the "new" one if we process in different orders,
        # but if we process in increasing order, only c is added last.
    
    # But we might process in a different order for better pruning. Let's stick with increasing order first.
    
    # Actually, for maximum independent set in hypergraphs, the standard branch and bound works like:
    # f(S, U) where S is selected, U is remaining candidates
    # Try adding each v in U to S, or skip v.
    
    # Let's implement a recursive solver with bitset representation.
    
    # We'll use Python integers as bitsets. Bit i (0-indexed) represents whether number i+1 is selected.
    
    # Precompute for each v (1-indexed), a bitset of pairs? That's complex.
    
    # Alternative: use a list-based approach with careful pruning.
    
    best = []
    best_size = 0
    
    selected = []
    
    # Precompute for quick checking: for each pair (a,b) in selected, we need to check if x conflicts.
    # Let's precompute a 2D array: conflict_with[a][b] = set of x > b such that (a,b,x) is bad.
    # Actually, let's precompute: for each x, which pairs (a,b) with a<b<x conflict with x.
    
    # Let's restructure: precompute bad_pairs_for[x] = set of (a,b) where a<b<x and (a,b,x) is bad.
    bad_pairs_for = [set() for _ in range(N+1)]
    for (a,b,c) in bad_triples:
        bad_pairs_for[c].add((a,b))
    
    def is_compatible(x, selected):
        """Check if x can be added to selected without forming a bad triple."""
        for (a,b) in bad_pairs_for[x]:
            if a in selected and b in selected:
                return False
        return True
    
    # Convert selected to a set for O(1) lookup
    def backtrack(idx, selected, selected_set):
        nonlocal best, best_size
        if idx > N:
            if len(selected) > best_size:
                best_size = len(selected)
                best = list(selected)
            return
        
        # Upper bound: current size + remaining elements
        remaining = N - idx + 1
        if len(selected) + remaining <= best_size:
            return
        
        # Option 1: skip idx
        backtrack(idx+1, selected, selected_set)
        
        # Option 2: include idx if compatible
        if is_compatible(idx, selected_set):
            selected.append(idx)
            selected_set.add(idx)
            backtrack(idx+1, selected, selected_set)
            selected.pop()
            selected_set.remove(idx)
    
    # This might be too slow. Let's add more pruning.
    # We can also try to order the vertices differently for better pruning.
    
    # Actually, let's first try the greedy approach to get a good initial bound, then use branch and bound.
    
    # Greedy: process in order, add if compatible.
    greedy_selected = []
    greedy_set = set()
    for x in range(1, N+1):
        if is_compatible(x, greedy_set):
            greedy_selected.append(x)
            greedy_set.add(x)
    
    best = list(greedy_selected)
    best_size = len(greedy_selected)
    
    # Now try to improve with local search or branch and bound.
    # Given time constraints, let's try a simple hill-climbing: try to swap elements.
    
    # Actually, let's try a different ordering for the branch and bound to improve pruning.
    # Sort by some heuristic: elements with more conflicts first.
    
    # Let's try the branch and bound with the current ordering but with the initial bound from greedy.
    
    # The recursion above should work. Let's hope it's fast enough for N=128.
    
    # Let's run the backtrack.
    backtrack(1, [], set())
    
    return best

# EVOLVE-BLOCK-END
