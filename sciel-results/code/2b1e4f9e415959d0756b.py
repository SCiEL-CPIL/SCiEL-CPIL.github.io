"""
MWIS Branch-and-Bound for 4-AP-free subset of [1, 256] maximizing 3-APs.
"""
import time
import sys
import numpy as np

N = 256
DEADLINE_FRACTION = 0.85  # Safety margin for 90s limit

# Precompute 3-APs: list of (a, b, c) such that a, a+d, a+2d are in [1, N]
# and d > 0.
# We will use a bitmask representation for the set of selected numbers.
# However, N=256 is too large for a single Python int bitmask in some contexts?
# No, Python ints are arbitrary precision. 2^256 is fine.
# But iterating over bits might be slow.
# Let's use a boolean array for state, but precomputed lists for structure.

# Precompute all 3-APs
_3aps = []
_3ap_indices = {} # Map element -> list of 3-AP indices it participates in
for d in range(1, (N + 1) // 2):
    max_a = N - 2 * d
    if max_a < 1:
        break
    for a in range(1, max_a + 1):
        b = a + d
        c = a + 2 * d
        idx = len(_3aps)
        _3aps.append((a, b, c))
        # Map each element to the APs it's in
        if a not in _3ap_indices: _3ap_indices[a] = []
        if b not in _3ap_indices: _3ap_indices[b] = []
        if c not in _3ap_indices: _3ap_indices[c] = []
        _3ap_indices[a].append(idx)
        _3ap_indices[b].append(idx)
        _3ap_indices[c].append(idx)

num_3aps = len(_3aps)

# Precompute 4-AP constraints
# A 4-AP is (a, b, c, d) with b-a = c-b = d-c.
# Constraint: Not all 4 can be selected.
# This is a hypergraph where hyperedges are 4-tuples.
# For MWIS, we can't select a set of vertices that contains any hyperedge.
# We will model this as a conflict graph for pruning?
# Or simply check conflicts during branching.
# Given the density, checking all 4-APs containing a new vertex is efficient if precomputed.

_4ap_conflicts = {i: [] for i in range(1, N + 1)}
# List of 4-APs to check
_4aps = []
for d in range(1, (N + 1) // 3):
    max_a = N - 3 * d
    if max_a < 1:
        break
    for a in range(1, max_a + 1):
        b = a + d
        c = a + 2 * d
        e = a + 3 * d
        _4aps.append((a, b, c, e))
        _4ap_conflicts[a].append((b, c, e))
        _4ap_conflicts[b].append((a, c, e))
        _4ap_conflicts[c].append((a, b, e))
        _4ap_conflicts[e].append((a, b, c))

# Weights for vertices:
# The objective is to maximize the number of 3-APs.
# This is a 3-way interaction, not pairwise.
# Standard MWIS uses pairwise edges. Here we have hyperedges for objective and constraints.
# We can approximate the weight of a vertex v as the number of 3-APs it participates in.
# However, adding v only completes a 3-AP if the other two are already present.
# This makes it a submodular-like objective (actually supermodular for maximization? No, diminishing returns).
# Let's define the state by the set of selected vertices S.
# Score(S) = count of 3-APs fully contained in S.

# Strategy:
# 1. Greedy construction to get a good initial lower bound (LB).
# 2. Branch and Bound:
#    - Order vertices by potential weight (degree in 3-APs).
#    - At each step, decide to include or exclude the current vertex.
#    - Upper Bound: Current Score + Sum of weights of remaining vertices that *could* contribute.
#      A simple bound: Current Score + (Number of remaining 3-APs that don't have a conflict yet).
#      Better: For each remaining 3-AP, if none of its vertices are excluded, it *might* be counted.
#      So UB = Current Score + Count of "active" 3-APs (where no vertex is excluded and not all are included? No, if all included, it's already counted).
#      Let's refine:
#      State: Included set I, Excluded set E, Remaining set R.
#      Current Score: 3-APs fully in I.
#      Potential Score: 3-APs with at least one vertex in R and no vertices in E.
#      UB = Current Score + Potential Score.
#      If UB <= LB, prune.

# To make this fast, we need efficient updates.
# N=256 is small enough for bitmasks?
# Let's use Python sets for I and E for clarity, and precomputed lists for speed.

def _count_3aps_in_set(S):
    """Count 3-APs fully contained in set S."""
    S_set = set(S)
    count = 0
    for a, b, c in _3aps:
        if a in S_set and b in S_set and c in S_set:
            count += 1
    return count

def _greedy_construct():
    """Greedy construction: add vertices that increase score and don't violate 4-AP."""
    S = set()
    flags = [False] * (N + 1)
    # Order vertices by degree in 3-APs (higher degree = higher potential)
    degrees = [len(_3ap_indices.get(i, [])) for i in range(1, N + 1)]
    order = sorted(range(1, N + 1), key=lambda x: -degrees[x-1])
    
    for x in order:
        # Check if adding x creates a 4-AP
        conflict = False
        for others in _4ap_conflicts[x]:
            if all(flags[o] for o in others):
                conflict = True
                break
        if not conflict:
            S.add(x)
            flags[x] = True
    return S, _count_3aps_in_set(S)

def _branch_and_bound(deadline):
    """
    Branch and bound search.
    """
    # Initial LB from greedy
    S_greedy, LB = _greedy_construct()
    best_set = S_greedy.copy()
    
    # Precompute vertex weights (degree in 3-APs) for ordering
    # Vertices with higher degree are "more valuable"
    weights = [0] + [len(_3ap_indices.get(i, [])) for i in range(1, N + 1)]
    
    # Order vertices for branching
    # High weight first? Or maybe low weight first to prune?
    # Usually, branching on high impact variables first is better for finding good solutions early.
    order = sorted(range(1, N + 1), key=lambda x: -weights[x])
    
    # State:
    # I: set of included vertices
    # E: set of excluded vertices
    # R: list of remaining vertices (subset of order)
    
    # We will use a recursive function with a stack to avoid recursion limits, 
    # or just careful recursion. Depth is 256, which might hit limits or be slow.
    # Let's use an iterative approach with a stack of states.
    # State: (I, E, R, current_score, active_3aps_count)
    # active_3aps: 3-APs that have no vertices in E and at least one in R?
    # Actually, to calculate UB efficiently:
    # UB = current_score + number of 3-APs that are "still possible".
    # A 3-AP is possible if none of its vertices are in E.
    # We can maintain a count of possible 3-APs.
    
    # Let's precompute for each vertex v, the list of 3-APs it belongs to.
    # When we exclude v, we must subtract the 3-APs that ONLY v could have saved?
    # No, if any vertex of a 3-AP is excluded, the 3-AP is impossible.
    # So, maintain `possible_3aps` set or count.
    # Initially, all 3-APs are possible.
    # When we exclude v, we remove all 3-APs containing v from `possible_3aps`.
    # When we include v, we don't change `possible_3aps` immediately, but we might complete some.
    # Completed 3-APs are added to `current_score` and removed from `possible_3aps`?
    # If a 3-AP is completed, it's no longer "possible" in the sense of being a future gain, it's a current gain.
    # So, `possible_3aps` should be: 3-APs with no vertices in E, and NOT fully in I.
    # If a 3-AP becomes fully in I, we add 1 to score and remove it from possible.
    
    # This requires tracking the status of each 3-AP.
    # Status: 
    # 0: Impossible (has vertex in E)
    # 1: Possible (no vertex in E, not fully in I)
    # 2: Completed (fully in I)
    
    # Initial state:
    # I = {}, E = {}
    # All 3-APs are status 1.
    # possible_count = num_3aps
    # current_score = 0
    # UB = 0 + num_3aps
    
    # We need to update these counts efficiently when adding to I or E.
    
    # Let's map 3-AP index to its vertices.
    # And vertex to 3-AP indices.
    
    # Data structures:
    # ap_status: array of int (0, 1, 2) length num_3aps
    # ap_vertex_count_in_I: array of int length num_3aps (how many vertices are in I)
    
    # When adding x to I:
    # For each 3-AP idx in _3ap_indices[x]:
    #   if ap_status[idx] == 0: continue
    #   ap_vertex_count_in_I[idx] += 1
    #   if ap_vertex_count_in_I[idx] == 3:
    #       ap_status[idx] = 2
    #       current_score += 1
    #       possible_count -= 1  # It was possible, now completed
    #   # If it was 1, it stays 1 until completed or excluded.
    #   # Wait, if it's 1 and we add a vertex, it's still 1 unless completed.
    
    # When adding x to E:
    # For each 3-AP idx in _3ap_indices[x]:
    #   if ap_status[idx] == 2: 
    #       # This shouldn't happen if we check consistency, but logically:
    #       # If a 3-AP is completed, it's already scored. Excluding a vertex doesn't un-score it?
    #       # No, if we exclude a vertex, we are saying "this vertex is NOT in the final set".
    #       # If the 3-AP was completed, it means all 3 vertices were in I.
    #       # But now we are excluding one of them? Contradiction.
    #       # So we must ensure we never exclude a vertex that is in I.
    #       # And we never include a vertex that is in E.
    #       # So if ap_status is 2, all vertices are in I. So x is in I.
    #       # We are branching on x. If we are in the "exclude" branch, x is not in I yet.
    #       # So ap_status cannot be 2 for any 3-AP containing x if x is not in I.
    #       # Correct.
    #   if ap_status[idx] == 1:
    #       ap_status[idx] = 0
    #       possible_count -= 1
    
    # This logic holds.
    
    # Stack element: (I_set, E_set, R_list, current_score, possible_count, ap_status_tuple, ap_count_in_I_tuple)
    # Tuples are immutable and hashable, good for stack. But copying them is O(N).
    # N=256, num_3aps ~ 25000. Copying 25000 ints per node is expensive.
    # We need a more efficient way to track state or use a different bound.
    
    # Alternative Bound:
    # UB = current_score + sum of weights of remaining vertices / 3?
    # Or simpler: current_score + (number of remaining vertices * max_degree / 3)
    # This is loose.
    
    # Let's try a simpler heuristic first:
    # Greedy + Local Search was 1.18.
    # MWIS B&B might be too slow if not optimized.
    
    # Let's implement a "Weighted Greedy with Backtracking" (Limited B&B).
    # We will only backtrack if the potential gain is high.
    
    # Actually, let's look at the "Inspiration 2" which used bitmask.
    # Bitmask 4-AP check is very fast.
    # Let's combine Bitmask state with a smarter search.
    
    # State: bitmask of selected vertices.
    # Precompute for each vertex x, a bitmask of "conflicts"?
    # No, 4-AP is a 4-way constraint.
    # If we have a bitmask S, adding x is valid if no 4-AP containing x is fully covered by S U {x}.
    # This means: for every 4-AP (x, y, z, w), if y, z, w are in S, then x cannot be added.
    # So, precompute for each x, a list of "blocking sets"?
    # Or, precompute for each x, a bitmask `conflict_mask[x]`?
    # No, a single bitmask can't represent "if y,z,w are in S".
    
    # Let's stick to the boolean array approach but optimize the search.
    # The current program uses `_check_add_4ap` which is O(N) per check.
    # With N=256, this is fast.
    
    # The bottleneck is the search space.
    # Let's try a "Simulated Annealing" or "Tabu Search" with a better move set.
    # The prompt suggests MWIS B&B.
    
    # Let's implement a simplified B&B that uses the "Active 3-APs" count for UB.
    # To avoid copying large arrays, we will use a "copy-on-write" strategy or just accept the cost if the tree is small.
    # Given N=256, the tree might explode.
    # We need a strong pruning.
    
    # Let's try a different structural change:
    # Use a **Genetic Algorithm** with a repair mechanism.
    # Population of sets.
    # Fitness: 3-AP count.
    # Constraint: No 4-AP.
    # Repair: If a crossover creates a 4-AP, remove one of the vertices involved.
    # This is a different design family (Evolutionary) from Local Search (Optimization) and B&B (Exact/Heuristic).
    
    # However, the prompt specifically asks for: "Model the problem as a Maximum Weight Independent Set ... and solve it using a branch-and-bound heuristic".
    # I should follow the directive.
    
    # Optimized B&B:
    # 1. Precompute `ap_vertices` list of tuples.
    # 2. Precompute `vertex_aps` list of lists.
    # 3. Use a global `ap_status` array and `ap_count_in_I` array.
    # 4. Recursive function with backtracking.
    # 5. Prune if `current_score + possible_count <= best_score`.
    
    # To make it fast, we will limit the depth or branching factor.
    # Or use a "Best-First" search with a priority queue (A*).
    # Node: (UB, I, E, R, score, possible_count, ap_status, ap_count)
    # This is memory heavy.
    
    # Let's try a **Greedy Randomized Adaptive Search Procedure (GRASP)** with a strong local search.
    # This is a metaheuristic, different from pure B&B.
    # But the prompt says "branch-and-bound heuristic".
    
    # Okay, I will implement a **Depth-First Branch and Bound** with:
    # - Vertex ordering by degree.
    # - UB = current_score + possible_count.
    # - Early termination if deadline reached.
    # - To handle state copying, I will use `copy` for the small lists, but `ap_status` and `ap_count` are arrays.
    #   I will use `np.array` copies. 25000 ints is ~100KB. 1000 nodes = 100MB. Might be okay.
    
    # Let's refine the UB.
    # `possible_count` counts 3-APs that are not impossible and not completed.
    # This is a valid upper bound on future gains.
    
    # One issue: `possible_count` decreases when we exclude a vertex.
    # It also decreases when we complete a 3-AP.
    # When we complete a 3-AP, we add to `current_score`.
    # So `current_score + possible_count` is indeed an upper bound on the final score.
    
    # Let's implement this.

    # Global variables for B&B
    global_best_score = LB
    global_best_set = S_greedy
    node_count = 0
    start_time = time.time()
    
    # Initialize state
    ap_status = np.ones(num_3aps, dtype=np.int8) # 1 = possible
    ap_count_in_I = np.zeros(num_3aps, dtype=np.int8) # 0 = none in I
    
    # Order vertices
    # We branch on vertices.
    # To make the tree smaller, we can use "Include" or "Exclude".
    # Usually, "Include" is tried first if it looks promising.
    
    def _branch(idx, I_mask, E_mask, score, possible_count):
        nonlocal global_best_score, global_best_set, node_count
        
        if time.time() > deadline:
            return
        
        # UB Check
        if score + possible_count <= global_best_score:
            return
            
        if idx == len(order):
            if score > global_best_score:
                global_best_score = score
                # Reconstruct set from I_mask
                # I_mask is a bitmask or set?
                # Using bitmask for I and E is faster for checking?
                # But we need to update ap_status.
                # Let's use sets for I and E for the B&B state, as bitmask manipulation for 4-AP check is complex.
                # Wait, I used I_mask in signature. Let's use sets.
                pass
            return
            
        x = order[idx]
        
        # Option 1: Include x
        # Check if x conflicts with I?
        # If x is in E, we can't include. But we branch on unvisited vertices.
        # So x is not in I or E.
        
        # Check 4-AP constraint for including x
        valid_include = True
        # We need to check if adding x creates a 4-AP with vertices in I.
        # For each 4-AP containing x, if the other 3 are in I, then invalid.
        # We can check this using a precomputed structure or by iterating.
        # Since we have I as a set, we can check.
        for others in _4ap_conflicts[x]:
            if all(o in I for o in others):
                valid_include = False
                break
                
        if valid_include:
            # Tentatively include x
            I.add(x)
            
            # Update ap_status and ap_count_in_I
            changes = []
            new_score = score
            new_possible = possible_count
            
            for ap_idx in _3ap_indices.get(x, []):
                if ap_status[ap_idx] == 0:
                    continue
                # It was 1 (possible)
                ap_count_in_I[ap_idx] += 1
                changes.append((ap_idx, 1, ap_count_in_I[ap_idx])) # (idx, old_count, new_count)
                
                if ap_count_in_I[ap_idx] == 3:
                    # Completed
                    ap_status[ap_idx] = 2
                    new_score += 1
                    new_possible -= 1
                    changes.append((ap_idx, 2, 2)) # Mark status change
                    
            _branch(idx + 1, I, E, new_score, new_possible)
            
            # Backtrack
            for ap_idx, old_status, new_count in reversed(changes):
                if ap_status[ap_idx] == 2:
                    ap_status[ap_idx] = 1
                    new_possible += 1 # Restore possible count? 
                    # Wait, the `new_possible` passed to recursion was local.
                    # We need to restore the global arrays.
                    # The `changes` list needs to store enough info to revert.
                    # Let's simplify: just re-copy the arrays? No, too slow.
                    # Let's store old values.
                    
            # My change tracking is messy. Let's use a simpler approach:
            # Since N is small, maybe just pass copies of the arrays?
            # 25000 ints * 2 arrays = 50KB per node.
            # If we visit 10,000 nodes, that's 500MB. Risky.
            # But pruning should be effective.
            
            # Let's restart with a cleaner state management.
            pass

    # The state management in B&B is tricky to implement correctly and fast in Python without C extensions.
    # Given the 60s limit, a complex B&B might time out.
    
    # Alternative: **Simulated Annealing** with a strong cooling schedule and a diverse move set.
    # This is a different design family (Stochastic Optimization) from the current Local Search (Deterministic/Greedy).
    # The prompt asks for MWIS B&B. I will try to implement a *simplified* B&B that is robust.
    
    # Actually, let's look at the "Inspiration" scores.
    # 1.18 is the current best.
    # A simple SA might easily beat this if tuned well.
    # But I must follow the directive: "Model ... as MWIS ... solve using B&B".
    
    # I will implement a **Greedy B&B** that only branches on the top K vertices, effectively limiting the search space.
    # Or, I will implement **Branch and Bound** but with a very tight time check and a simple state representation.
    
    # Let's use a **Bitmask** for the set I.
    # And precompute `conflict_bitmask[x]`?
    # No, 4-AP is not pairwise.
    
    # Let's go with **Simulated Annealing** as the "Structural Change" because implementing a correct, fast B&B for this specific hypergraph in Python within the time limit is high-risk.
    # Wait, the directive is explicit: "Strategy to try in this change: Model the problem as a Maximum Weight Independent Set ... and solve it using a branch-and-bound heuristic".
    # I MUST try B&B.
    
    # I will implement a **Best-First Search** (A* variant) with a priority queue.
    # Node: (UB, I, E, score, possible_count, ap_status, ap_count)
    # I will limit the number of nodes expanded.
    
    # To avoid copying large arrays, I will use a **persistent array** trick or just accept the copy cost if the tree is pruned heavily.
    # Given the strong UB (`score + possible_count`), the tree should be pruned well.
    
    # Let's implement it.

def construct():
    start_time = time.time()
    deadline = start_time + 90 * DEADLINE_FRACTION
    
    # 1. Greedy Initial Solution
    S_greedy, LB = _greedy_construct()
    global_best_score = LB
    global_best_set = S_greedy.copy()
    
    # 2. Branch and Bound
    # Precomputation
    ap_status = np.ones(num_3aps, dtype=np.int8)
    ap_count_in_I = np.zeros(num_3aps, dtype=np.int8)
    
    # Vertex order: High degree first
    weights = [0] + [len(_3ap_indices.get(i, [])) for i in range(1, N + 1)]
    order = sorted(range(1, N + 1), key=lambda x: -weights[x])
    
    # Use a stack for DFS
    # State: (idx, I, E, score, possible_count, ap_status, ap_count_in_I)
    # I and E are sets.
    
    initial_I = set()
    initial_E = set()
    
    stack = [(0, initial_I, initial_E, 0, num_3aps, ap_status.copy(), ap_count_in_I.copy())]
    
    nodes_expanded = 0
    
    while stack:
        if time.time() > deadline:
            break
            
        idx, I, E, score, possible_count, ap_status, ap_count_in_I = stack.pop()
        nodes_expanded += 1
        
        # UB Check
        if score + possible_count <= global_best_score:
            continue
            
        if idx == len(order):
            if score > global_best_score:
                global_best_score = score
                global_best_set = I.copy()
            continue
            
        x = order[idx]
        
        # Option 1: Exclude x
        # New State: I, E U {x}
        # Update ap_status: for each AP in x, if status==1, set to 0.
        new_E = E | {x}
        new_ap_status = ap_status.copy()
        new_ap_count = ap_count_in_I.copy() # No change in count for exclude
        new_possible = possible_count
        
        for ap_idx in _3ap_indices.get(x, []):
            if new_ap_status[ap_idx] == 1:
                new_ap_status[ap_idx] = 0
                new_possible -= 1
                
        # Push Exclude branch
        # We push Include first so it's popped first (LIFO)
        # Or we can push Exclude then Include.
        stack.append((idx + 1, I, new_E, score, new_possible, new_ap_status, new_ap_count))
        
        # Option 2: Include x
        # Check 4-AP conflict with I
        valid_include = True
        for others in _4ap_conflicts[x]:
            if all(o in I for o in others):
                valid_include = False
                break
                
        if valid_include:
            new_I = I | {x}
            new_ap_status_inc = ap_status.copy()
            new_ap_count_inc = ap_count_in_I.copy()
            new_score = score
            new_possible_inc = possible_count
            
            for ap_idx in _3ap_indices.get(x, []):
                if new_ap_status_inc[ap_idx] == 0:
                    continue
                # It was 1
                new_ap_count_inc[ap_idx] += 1
                if new_ap_count_inc[ap_idx] == 3:
                    new_ap_status_inc[ap_idx] = 2
                    new_score += 1
                    new_possible_inc -= 1
                    
            stack.append((idx + 1, new_I, E, new_score, new_possible_inc, new_ap_status_inc, new_ap_count_inc))
            
    return list(global_best_set)

# EVOLVE-BLOCK-END
