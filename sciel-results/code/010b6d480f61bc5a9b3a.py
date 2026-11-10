"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import time
import sys
from functools import lru_cache

num_points = N * N
coords = [(i // N, i % N) for i in range(num_points)]

# Precompute squared distances
dist = [[0] * num_points for _ in range(num_points)]
for i in range(num_points):
    x1, y1 = coords[i]
    for j in range(i + 1, num_points):
        x2, y2 = coords[j]
        d = (x1 - x2) ** 2 + (y1 - y2) ** 2
        dist[i][j] = d
        dist[j][i] = d

# Precompute conflict graph: conflict[i] is a set of points j such that 
# i and j form a "conflict" if they were both in the set? No.
# The constraint is on triples.
# Let's use the bitset approach for "can_add".
# For a point p, and a set S, p can be added iff for all pairs (a,b) in S, 
# {p,a,b} is not isosceles.
# This is equivalent to: p is not in the "forbidden set" for any pair in S.
# Let's precompute for each pair (a,b) the set of points p that form an isosceles triangle with a,b.
# forbidden[pair] -> set of p
# But pairs are 256*255/2 = 32640. Sets could be large.
# Total memory: 32640 * (avg set size).
# Let's try a different representation.
# For each point p, precompute a bitset conflict_mask[p].
# conflict_mask[p] has bit q set if p and q "conflict" in some sense?
# No, the conflict is ternary.
# 
# Let's stick to the set of isosceles triples but use a faster check.
# is_isosceles(i, j, k) is O(1) with dist table.
# can_add(p, S) checks all pairs in S.
# If S has size k, this is O(k^2).
# 
# The strategy suggests "Maximum Independent Set in 3-uniform hypergraph".
# This is NP-hard.
# The current program uses greedy + local search (removal).
# The inspiration uses greedy + local search (removal/refill?).
# 
# Let's try a structural change: **Simulated Annealing** or **Tabu Search**.
# Objective: Minimize |S| such that S is maximal no-isosceles.
# State: S (set of points).
# Moves:
# 1. Remove a point p from S. (Decreases size by 1).
#    - If S\{p} is still no-isosceles (always true) and MAXIMAL, accept.
#    - If S\{p} is not maximal, we MUST add points to make it maximal.
#      This is a "repair" step.
#      Let S' = maximal_extension(S \ {p}).
#      If |S'| < |S|, accept.
# 2. Swap: Remove p, add q?
#    - Remove p. Add q if valid. Then repair to maximal.
#
# Key Insight: "Maximal" is a hard constraint. We can't just minimize size freely.
# We are searching the space of *maximal* no-isosceles sets.
#
# Algorithm:
# 1. Generate a random maximal set S0 (Greedy).
# 2. Local Search:
#    - While time remains:
#      - Pick a point p in S randomly.
#      - Try to remove p.
#      - Let S_temp = S \ {p}.
#      - Check if S_temp is maximal.
#        - If yes: New S = S_temp. (Size decreased by 1).
#        - If no: Find a point q not in S_temp that can be added.
#          - If we can add q such that the resulting set S_new = S_temp U {q} is valid (no isosceles) 
#            and then we repair S_new to maximal, does it help?
#          - Actually, if S_temp is not maximal, it means there exists some r not in S_temp 
#            such that r can be added to S_temp.
#            If we add r, we get S_temp U {r}. This set might not be maximal either.
#            We need to find a *maximal* extension of S_temp.
#            Let S_ext = maximal_extension(S_temp).
#            If |S_ext| < |S|, accept S_ext.
#            Note: S_ext contains S_temp. So S_ext = S_temp U A where A is some set of added points.
#            Since S was maximal, S_temp (which is S\{p}) is "almost" maximal.
#            The points that can be added to S_temp are exactly the points that were "blocked" by p 
#            in S, plus maybe others?
#            Actually, if p was in S, it blocked some points. Removing p unblocks them.
#            So the set of addable points to S_temp is non-empty.
#            The maximal extension will add ALL possible points? No, greedy adds some.
#            To minimize size, we want to add as FEW points as possible to make it maximal.
#            This is the "Minimum Maximal Extension" problem.
#            Greedy might add too many.
#
# Let's refine the move:
# Move: Remove p.
# Repair: Find a *minimum* set of points A such that (S \ {p}) U A is maximal and no-isosceles.
# This is hard.
# Approximation:
# 1. Remove p.
# 2. Find all points that can be added to S\{p}. Let this be C (candidates).
# 3. We must add at least one point from C? No, we must add enough points so that NO point in C' (new candidates) can be added.
#    Actually, maximality means: For every point q not in S_new, q cannot be added to S_new.
#    This is equivalent to: S_new is a maximal independent set in the hypergraph.
#
# Simpler Local Search Strategy (2-Swap):
# 1. Start with maximal S.
# 2. Pick two points p, q in S.
# 3. Remove p and q.
# 4. Repair to maximal.
# 5. If new size < old size, accept.
#
# Even simpler: "Remove and Re-add optimally"
# 1. Remove p.
# 2. Identify the set of points P_blocked_by_p in S that were ONLY blocked by p? No.
# 3. Identify points that can be added to S\{p}.
# 4. We want to add a set A such that (S\{p}) U A is maximal.
# 5. To minimize |A|, we should pick points that block the most other candidates.
#    This is a Set Cover problem:
#    Universe: Points not in S\{p} that can be added to S\{p}. (Let's call this C).
#    Wait, if we add a point a in C, it might block other points in C.
#    We want to cover the "maximality condition".
#    Condition: No point outside S_new can be added.
#    This is tricky.
#
# Let's try a different structural approach: **Genetic Algorithm**.
# Representation: Bitmask of size 256.
# Fitness: -|S| if S is maximal no-isosceles, else -infinity.
# But checking maximality is expensive.
#
# Let's go back to the prompt's suggestion: "branch-and-bound ... bitset-based conflict propagation".
# This suggests a solver for the *Minimum Maximal* problem directly.
# However, with 256 points, exact B&B might be too slow.
#
# Hybrid Approach:
# 1. Fast Greedy to get a good initial S.
# 2. Local Search with a "Smart Repair".
#    - Remove p.
#    - Calculate C = {q not in S\{p} | q can be added to S\{p}}.
#    - If C is empty, S\{p} is maximal. Accept.
#    - If C is not empty, we must add some points from C.
#    - To minimize added points, we solve a small Set Cover/Maximal Independent Set problem on the induced subgraph of C?
#      No, the points in C interact with each other.
#      If we add a in C, it might make b in C invalid (if a,b and some c in S form isosceles).
#      Actually, a and b are both valid w.r.t S\{p}.
#      If we add a, does it block b?
#      b can be added to S\{p} U {a} iff b does not form isosceles with any pair in (S\{p}) U {a}.
#      Since b was valid w.r.t S\{p}, the only new conflict is with pairs involving a.
#      So b is blocked by a if there exists c in S\{p} such that {a,b,c} is isosceles.
#      This defines a conflict graph on C.
#      We need to find a *maximal independent set* in this conflict graph?
#      No. We need to add a set A subset of C such that:
#      1. A is an independent set in the conflict graph of C (i.e., no two in A conflict with each other w.r.t S\{p}).
#      2. A is "maximal" in the sense that it blocks all other points in C.
#         i.e., for every c in C \ A, c conflicts with some a in A.
#      This is exactly a **Maximal Independent Set** in the conflict graph G_C.
#      And we want to MINIMIZE |A|.
#      So we want a **Minimum Maximal Independent Set** (MMIS) in G_C.
#      G_C is much smaller than the original graph (size |C|).
#      We can solve MMIS on G_C using a simple greedy or local search, as |C| is likely small.
#
# This is a solid structural improvement.

def build_conflict_graph_for_removal(S_set, p):
    """
    Given a maximal set S_set and a point p in it,
    compute the conflict graph on the set of candidates C
    that can be added to S_set \ {p}.
    Returns the conflict graph as a list of sets (adjacency lists).
    """
    S_temp = S_set - {p}
    
    # Find candidates C
    C = []
    for q in range(num_points):
        if q in S_temp:
            continue
        if _can_add(q, S_temp):
            C.append(q)
            
    if not C:
        return None, None  # S_temp is already maximal
        
    # Map C indices to actual points for easier handling
    # Build conflict graph on C
    # c_i conflicts with c_j if there exists c in S_temp 
    # such that {c_i, c_j, c} is isosceles.
    
    n_c = len(C)
    adj = [set() for _ in range(n_c)]
    
    for i in range(n_c):
        pi = C[i]
        for j in range(i + 1, n_c):
            pj = C[j]
            # Check if pi and pj conflict w.r.t S_temp
            # They conflict if there is some c in S_temp 
            # such that dist[pi][pj] == dist[pi][c] or dist[pi][pj] == dist[pj][c] or dist[pi][c] == dist[pj][c]
            # Wait, the definition of conflict for the repair step:
            # We add A. A must be an independent set in the original hypergraph restricted to C?
            # No, A must be a valid set (no isosceles triples within A U S_temp).
            # Since C points are valid w.r.t S_temp individually,
            # two points pi, pj in C are incompatible if they form an isosceles triangle with some c in S_temp.
            # OR if pi, pj and some other point in C form an isosceles?
            # The set S_new = S_temp U A must be no-isosceles.
            # This requires:
            # 1. No isosceles triple entirely in A.
            # 2. No isosceles triple with 2 in A, 1 in S_temp.
            # 3. No isosceles triple with 1 in A, 2 in S_temp (already satisfied by C definition).
            
            # So, pi and pj are "conflicting" for the purpose of choosing A if:
            # - They form an isosceles triangle with some c in S_temp.
            # - OR they are part of an isosceles triangle with some other point in A? (This is handled by the independent set constraint on A itself).
            
            # Let's define the graph G_C where edges represent "cannot be in A together".
            # Edge (i,j) exists if:
            # a) There exists c in S_temp such that {C[i], C[j], c} is isosceles.
            # b) {C[i], C[j], k} is isosceles for some k in C? No, that's a triangle within C.
            #    If {C[i], C[j], C[k]} is isosceles, then we can't have all three in A.
            #    This is a 3-uniform hyperedge in C.
            
            # To simplify, let's just check condition (a) for pairwise conflicts.
            # If we ignore condition (b), A might contain an isosceles triple entirely in C.
            # We can check this later or include it in the graph as a clique?
            # Given |C| is small, we can do a more robust check during the MMIS search on C.
            
            conflict = False
            for c in S_temp:
                if is_isosceles(C[i], C[j], c):
                    conflict = True
                    break
            if conflict:
                adj[i].add(j)
                adj[j].add(i)
                
    return C, adj

def find_min_maximal_independent_set_small(vertices, adj):
    """
    Find a minimum maximal independent set in a small graph.
    vertices: list of vertex indices (0..n-1)
    adj: list of sets of neighbors
    Returns a set of vertex indices.
    """
    n = len(vertices)
    if n == 0:
        return set()
    
    best_set = set(range(n))  # Initial: all vertices (maximal, but not minimal)
    best_size = n
    
    # Simple Local Search / Greedy
    # Start with a random maximal independent set
    random.shuffle(vertices)
    current_set = set()
    for v in vertices:
        # Check if v can be added to current_set
        valid = True
        for u in current_set:
            if u in adj[v]:
                valid = False
                break
        if valid:
            current_set.add(v)
            
    # Ensure it's maximal (greedy construction usually yields maximal if we iterate all)
    # Actually, the greedy loop above adds v if it doesn't conflict with ANY in current_set.
    # This produces a maximal independent set?
    # Yes, because if a vertex v was not added, it conflicted with some u in current_set.
    # So no vertex outside current_set can be added.
    
    current_size = len(current_set)
    
    # Local search: try to remove vertices
    improved = True
    while improved:
        improved = False
        for v in list(current_set):
            trial = current_set - {v}
            # Check if trial is maximal
            is_maximal = True
            for u in vertices:
                if u in trial:
                    continue
                # Check if u can be added to trial
                can_add = True
                for w in trial:
                    if w in adj[u]:
                        can_add = False
                        break
                if can_add:
                    is_maximal = False
                    break
            
            if is_maximal:
                current_set = trial
                current_size -= 1
                improved = True
                if current_size < best_size:
                    best_size = current_size
                    best_set = current_set.copy()
                break # Restart loop with new set
    
    return best_set

def _can_add(p, s_set):
    """Check if point p can be added to set s_set without creating an isosceles triangle."""
    if len(s_set) < 2:
        return True
    s_list = list(s_set)
    for idx_a in range(len(s_list)):
        a = s_list[idx_a]
        for idx_b in range(idx_a + 1, len(s_list)):
            b = s_list[idx_b]
            if is_isosceles(p, a, b):
                return False
    return True

def is_isosceles(i, j, k):
    d1 = dist[i][j]
    d2 = dist[i][k]
    d3 = dist[j][k]
    return d1 == d2 or d1 == d3 or d2 == d3

def construct():
    start_time = time.time()
    time_limit = 295
    
    best_size = num_points
    best_set = None
    random.seed(42)
    
    while time.time() - start_time < time_limit:
        # 1. Randomized greedy construction
        order = list(range(num_points))
        random.shuffle(order)
        
        current_set = set()
        for p in order:
            if _can_add(p, current_set):
                current_set.add(p)
        
        # 2. Repair to maximal
        changed = True
        while changed and time.time() - start_time < time_limit:
            changed = False
            for p in range(num_points):
                if p in current_set:
                    continue
                if _can_add(p, current_set):
                    current_set.add(p)
                    changed = True
        
        # 3. Structural Local Search: Remove and Smart Repair
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            s_list = list(current_set)
            random.shuffle(s_list)
            
            for p in s_list:
                if time.time() - start_time > time_limit:
                    break
                
                # Try removing p
                C, adj = build_conflict_graph_for_removal(current_set, p)
                
                if C is None:
                    # S \ {p} is already maximal
                    current_set.discard(p)
                    improved = True
                    if len(current_set) < best_size:
                        best_size = len(current_set)
                        best_set = current_set.copy()
                    break
                else:
                    # Solve MMIS on C
                    # Map C indices back to points?
                    # The MMIS solver works on indices 0..len(C)-1.
                    # The result is a set of indices.
                    # We need to map them back to points in C.
                    min_idx_set = find_min_maximal_independent_set_small(range(len(C)), adj)
                    
                    # Convert indices to points
                    points_to_add = [C[i] for i in min_idx_set]
                    
                    # Construct new set
                    new_set = (current_set - {p}) | set(points_to_add)
                    
                    # Verify validity (safety check)
                    valid = True
                    new_list = list(new_set)
                    for i in range(len(new_list)):
                        a = new_list[i]
                        for j in range(i+1, len(new_list)):
                            b = new_list[j]
                            for k in range(j+1, len(new_list)):
                                c = new_list[k]
                                if is_isosceles(a,b,c):
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    
                    if valid and len(new_set) < len(current_set):
                        # Verify maximality of new_set?
                        # The MMIS on C should ensure maximality w.r.t C.
                        # But what about points not in C?
                        # Points not in C were already blocked by S\{p}.
                        # Adding points from C only adds more blocks.
                        # So new_set should be maximal.
                        current_set = new_set
                        improved = True
                        if len(current_set) < best_size:
                            best_size = len(current_set)
                            best_set = current_set.copy()
                        break
        
        if best_set is None:
             best_set = current_set.copy()
             best_size = len(current_set)
    
    if best_set is None:
        # Fallback
        chosen = set()
        for p in range(num_points):
            if _can_add(p, chosen):
                chosen.add(p)
        changed = True
        while changed:
            changed = False
            for p in range(num_points):
                if p not in chosen and _can_add(p, chosen):
                    chosen.add(p)
                    changed = True
        best_set = chosen
        
    return [coords[i] for i in sorted(best_set)]

# EVOLVE-BLOCK-END
