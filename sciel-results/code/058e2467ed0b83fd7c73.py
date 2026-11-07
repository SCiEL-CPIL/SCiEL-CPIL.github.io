import time
import math
from functools import lru_cache

# Precompute conflict graph for the pool 1..256
# A conflict exists between a and b if adding them to a set creates a divisibility conflict
# in their subset sums. Since we are building the set incrementally, we need to know if
# {a, b} is a valid subset. If {a, b} is valid, it might still conflict with other elements
# later, but for the graph, we define an edge if {a, b} is INVALID.
# Wait, the condition is: "no two distinct resulting values may divide one another".
# For a set S, let Sums(S) be the set of all non-empty subset sums.
# S is valid if for all x, y in Sums(S), x != y implies x does not divide y.
#
# The conflict graph approach:
# Nodes are integers 1..256.
# Edge (u, v) exists if {u, v} is NOT a valid set.
# If we find a Maximum Independent Set in this graph, is it guaranteed to be a valid set?
# NO. Validity is not a pairwise property. A set {a, b, c} might be valid even if
# {a, b} is invalid? No, if {a, b} is invalid, it means there are subset sums of {a, b}
# that divide each other. These sums are also subset sums of {a, b, c}. So {a, b, c}
# would also be invalid.
# Conversely, if all pairs {a, b} are valid, is {a, b, c} valid?
# Not necessarily. A conflict might arise from sums involving 3 elements.
# Example: a=1, b=2, c=3.
# Sums({1,2}) = {1, 2, 3}. 1 divides 2, 1 divides 3, 2 doesn't divide 3. Wait, 1 divides everything.
# So any set containing 1 is invalid because 1 divides all other sums.
# Actually, if 1 is in the set, sum=1 is in Sums. Any other sum s>1 is divisible by 1.
# So 1 cannot be in any valid set of size > 1.
#
# Let's check if validity is pairwise.
# If S is valid, every subset of S is valid.
# If every pair {a, b} is valid, is S valid?
# Counter-example search:
# We need x, y in Sums(S) such that x|y.
# x and y are sums of subsets.
# If x and y are sums of disjoint subsets, their union has sum x+y.
# The condition "no two distinct subset sums divide one another" is quite strong.
#
# Let's verify if the conflict graph (pairwise invalidity) is sufficient.
# If {a, b} is invalid, there exist sums s1, s2 from {a, b} such that s1 | s2.
# These sums are also sums from S. So S is invalid.
# So, an independent set in the conflict graph (where edges are pairwise conflicts)
# is a set where every pair is valid.
# Is it possible that every pair is valid, but the whole set is invalid?
# Yes, if a conflict arises from a sum involving 3 or more elements.
# Example: a, b, c.
# Sums: a, b, c, a+b, a+c, b+c, a+b+c.
# Suppose a+b = 2, a+c = 4. Then (a+b) | (a+c).
# But {a, b} has sums {a, b, a+b}. If a=1, b=1 (not distinct), or a=1, b=1? Distinct integers.
# Let a=1, b=1 is not allowed.
# Let a=1, b=1... wait distinct integers.
# Let a=1, b=1 is impossible.
# Let a=1, b=2. Sums {1, 2, 3}. 1|2, 1|3. Invalid.
# So any set with 1 is invalid.
#
# Let's try to find a set where all pairs are valid but triple is not.
# We need x, y in Sums({a,b,c}) such that x|y, but x, y are not in Sums({a,b}), Sums({a,c}), or Sums({b,c}).
# This implies x and y must involve at least 3 elements? Or one involves 2 and the other 3?
# If x involves 2 elements, say a+b, and y involves 3 elements a+b+c.
# Then (a+b) | (a+b+c) => (a+b) | c.
# So if c is a multiple of a+b, and {a,b} is valid, {a,b,c} might be invalid.
# Is {a,b} valid?
# Sums({a,b}) = {a, b, a+b}.
# Validity: a !| b, a !| a+b (always true if b>0), b !| a+b (always true if a>0), a !| b, b !| a.
# So {a,b} is valid iff a !| b and b !| a.
# Now add c such that c is a multiple of a+b.
# Let a=2, b=3. a+b=5.
# {2,3} is valid (2 !| 3, 3 !| 2). Sums {2,3,5}. No divisibility.
# Let c=5. c is a multiple of 5 (1*5).
# Sums({2,3,5}) = {2, 3, 5, 5, 7, 8, 10}. Set: {2, 3, 5, 7, 8, 10}.
# Check divisibility:
# 2 | 8, 2 | 10. Conflict!
# So {2,3,5} is invalid.
# But {2,3} is valid, {2,5} is valid (2 !| 5), {3,5} is valid (3 !| 5).
# So the conflict graph (pairwise) would say {2,3,5} is an independent set (no edges),
# but it is actually invalid.
#
# Therefore, Maximum Independent Set on the pairwise conflict graph is NOT sufficient.
# We need a different approach.
#
# However, we can use the conflict graph for a *relaxation* or to guide the search,
# or we can define the conflict graph differently.
# But the directive says: "Model the problem as a maximum independent set in a conflict graph... then solve it".
# If the model is flawed, the result will be invalid.
#
# Alternative interpretation:
# Maybe the "conflict" is defined not just on pairs of elements, but on the *subset sums*?
# No, the nodes are integers 1-256.
#
# Let's re-read the directive carefully.
# "Model the problem as a maximum independent set in a conflict graph where nodes are integers 1-256 and edges connect pairs whose subset sums create a divisibility conflict"
# This implies the pairwise model.
# If the pairwise model is insufficient for validity, then any set found will likely be invalid.
# UNLESS... the task constraints or the nature of the numbers 1-256 make pairwise conflicts sufficient?
# No, I just found a counter-example with 2,3,5 which are in 1-256.
#
# So, if I implement a pure MIS solver on the pairwise graph, I will return an invalid set.
# The evaluator will score 0.
#
# How to fix this while keeping the "MIS" structural change?
# 1. Use MIS to find a candidate set.
# 2. Verify the candidate set for full validity.
# 3. If invalid, prune the set (remove elements causing conflicts) or use the MIS result as a seed for local search.
#
# The directive asks to "solve it using a branch-and-bound algorithm... to efficiently explore the search space".
# If I add a post-processing verification and repair step, is it still "solving it as MIS"?
# It's using MIS as a heuristic to find a *near* solution.
#
# Let's try a hybrid:
# 1. Build the pairwise conflict graph.
# 2. Use BnB to find a large Independent Set (IS).
# 3. Check if the IS is fully valid.
# 4. If not, iteratively remove elements that cause the most conflicts in the full subset-sum sense.
# 5. Use the remaining set as a seed for a fast local search that checks full validity.
#
# This is a structural change: the search space is explored via graph MIS BnB, not greedy addition.
# The verification is rigorous.
#
# Implementation details:
# - Conflict Graph: N=256.
# - Edge (u,v) if {u,v} is invalid.
# - BnB for MIS:
#   - Use bitsets for adjacency.
#   - Use greedy coloring for upper bound.
#   - Python doesn't have native bitsets, but we can use integers or lists of bools.
#   - 256 nodes is small enough for list-of-lists or integer bitmasks (if we pack 256 bits into 4 64-bit ints or just use Python's arbitrary precision ints).
#   - Python ints are efficient for bit operations.
#
# Let's implement the MIS BnB.
#
# Algorithm for MIS BnB (Tomita's algorithm variant):
# 1. Order vertices by degree (descending).
# 2. Branch on a vertex: include or exclude.
# 3. Upper bound: size of current solution + greedy coloring bound on remaining candidates.
#
# Greedy Coloring Bound:
# - Assign colors to remaining vertices such that no two adjacent vertices have the same color.
# - The number of colors is an upper bound on the size of the independent set in that subgraph.
# - We want to maximize the number of colors? No, the bound is the number of colors used.
# - To get a tight bound, we should color in a way that minimizes colors? No, the bound is valid for ANY coloring.
# - Actually, the standard trick is: sort vertices, assign smallest possible color. The max color index is the bound.
# - To make the bound tight (small), we want to color efficiently.
# - But for BnB, we want the bound to be as small as possible to prune early.
# - So we should use a greedy coloring that minimizes the number of colors?
# - Yes.
#
# Wait, Tomita's algorithm uses a specific ordering and coloring.
#
# Let's simplify.
# 1. Precompute conflict graph G.
# 2. Run BnB to find a large Independent Set IS.
# 3. Validate IS.
# 4. If invalid, repair.
#
# Repair strategy:
# - While IS is invalid:
#   - Find a conflicting pair of subset sums.
#   - Identify elements involved.
#   - Remove one of them.
#   - (This is slow).
#
# Better:
# - Use the MIS result as a *starting point* for a local search that checks full validity.
# - The MIS result ensures no *pairwise* conflicts, so it's a good candidate.
# - Local search: try to swap elements to fix full-validity conflicts while maintaining size.
#
# Actually, if the pairwise conflicts are the main bottleneck, the MIS set is likely close to valid.
# The 3-element conflicts (like 2,3,5) are rare.
#
# Let's implement:
# 1. Build pairwise conflict graph.
# 2. Find MIS using BnB.
# 3. Check full validity.
# 4. If valid, return.
# 5. If not, enter local search to fix.
#
# To ensure speed, the BnB must be fast.
#
# Let's code the BnB.

def _build_conflict_graph():
    """Build adjacency list/bitmask for pairwise conflicts."""
    n = 256
    # adj[i] is a bitmask of nodes that conflict with i
    adj = [0] * n
    for i in range(n):
        u = i + 1
        for j in range(i + 1, n):
            v = j + 1
            # Check if {u, v} is valid
            # Sums: u, v, u+v
            # Valid if:
            # u !| v
            # v !| u
            # u !| u+v (always true for v>0)
            # v !| u+v (always true for u>0)
            # (u+v) !| u (impossible)
            # (u+v) !| v (impossible)
            # So just check u !| v and v !| u.
            if u % v != 0 and v % u != 0:
                continue # No conflict
            else:
                # Conflict
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _greedy_color_bound(candidates, adj):
    """
    candidates: list of indices in 0..255
    adj: list of bitmasks
    Returns: upper bound on MIS size in the subgraph induced by candidates.
    """
    if not candidates:
        return 0
    
    # Simple greedy coloring
    # Sort candidates by degree in the subgraph (descending) to get a tighter bound?
    # Or just iterate.
    # Tomita's algorithm sorts by degree.
    
    # To keep it simple and fast, let's just assign colors greedily.
    # The bound is the number of colors.
    # We want a SMALL number of colors to prune.
    # So we should try to reuse colors.
    
    colors = {} # node -> color
    next_color = 0
    for node in candidates:
        # Find smallest color not used by neighbors
        used_colors = set()
        neighbors = adj[node] & sum(1 << c for c in candidates) # This is slow
        # Better: iterate neighbors
        for nb in candidates:
            if nb == node: continue
            if (adj[node] >> nb) & 1:
                if nb in colors:
                    used_colors.add(colors[nb])
        
        # Find smallest available color
        c = 0
        while c in used_colors:
            c += 1
        colors[node] = c
        if c + 1 > next_color:
            next_color = c + 1
            
    return next_color

def _mis_branch_and_bound(adj, deadline):
    """
    Find a Maximum Independent Set using Branch and Bound.
    """
    n = 256
    best_size = 0
    best_set = []
    
    # Order vertices by degree descending
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    # Remap indices to 0..n-1 based on order?
    # No, let's just use the original indices but pick the next vertex from the ordered list.
    # This is tricky for bitmasks.
    
    # Let's stick to the original 0..255 indices.
    # We maintain a set of available candidates.
    
    def bnb(candidates_mask, current_set, current_size):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
        
        if not candidates_mask:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
            
        # Upper bound
        # Candidates are indices where bit is set in candidates_mask
        cand_list = [i for i in range(n) if (candidates_mask >> i) & 1]
        
        if current_size + len(cand_list) <= best_size:
            return
            
        # Greedy coloring bound
        # We need to color the subgraph induced by cand_list
        # For speed, if len(cand_list) is small, we can just use len(cand_list) as bound
        # But coloring is better.
        
        # Simplified bound: just use len(cand_list)
        # This is weak. Let's try a quick coloring.
        
        # To speed up, we can skip coloring if the bound is already too high?
        # No, we need the bound to prune.
        
        # Let's implement a fast coloring.
        # Since n=256, cand_list can be large.
        # Greedy coloring is O(V*E).
        
        # Heuristic: Pick a vertex to branch on.
        # Tomita: pick the vertex with highest degree in the colored subgraph?
        # Or just the first one in the order.
        
        # Let's pick the vertex with the highest degree among candidates.
        max_deg = -1
        pivot = -1
        for v in cand_list:
            deg = bin(adj[v] & candidates_mask).count('1')
            if deg > max_deg:
                max_deg = deg
                pivot = v
                
        if pivot == -1:
            return
            
        # Branch 1: Include pivot
        # New candidates: candidates that do not conflict with pivot
        new_mask = candidates_mask & ~(1 << pivot) & ~adj[pivot]
        current_set.append(pivot)
        bnb(new_mask, current_set, current_size + 1)
        current_set.pop()
        
        # Branch 2: Exclude pivot
        new_mask = candidates_mask & ~(1 << pivot)
        bnb(new_mask, current_set, current_size)
        
    initial_mask = (1 << n) - 1
    bnb(initial_mask, [], 0)
    
    return best_set

def _is_valid_full(S):
    """Full validity check."""
    if not S:
        return True
    n = len(S)
    if n > 20:
        # Fallback to partial check if too large for full enumeration?
        # But we need to be sure.
        # If n is large, 2^n is huge.
        # However, the MIS result should be small enough?
        # Max size is likely < 15.
        pass
        
    sums = {0}
    for x in S:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        sums |= new_sums
    sums.discard(0)
    
    sums_list = sorted(sums)
    sums_set = sums
    total = sum(S)
    for s in sums_list:
        if s == 0:
            continue
        m = 2 * s
        while m <= total:
            if m in sums_set:
                return False
            m += s
    return True

def _local_search_fix(S, deadline):
    """Try to fix invalidity by removing elements or swapping."""
    current = list(S)
    if _is_valid_full(current):
        return current
        
    # If invalid, we need to remove elements.
    # Find a conflict.
    # This is hard to do efficiently.
    # Strategy: Randomly remove elements until valid?
    # Or: Iterate through elements, try removing each, check if it reduces conflict count?
    
    # Simple approach:
    # While invalid:
    #   Pick a random element, remove it.
    #   If valid, stop.
    #   If size becomes 0, stop.
    
    # This is dumb but works as a fallback.
    # Better: Try to swap elements to maintain size.
    
    # Let's try a simple hill climb:
    # 1. Try to remove one element to make it valid.
    # 2. If none works, remove two?
    
    # Actually, if the MIS set is pairwise valid, it's very likely close to fully valid.
    # The 2,3,5 example is rare.
    
    # Let's just try removing the "most conflicting" element.
    # How to measure conflict?
    # Number of subset sums that divide another.
    
    # For now, let's just try removing each element one by one and checking validity.
    # O(n * 2^n * n). Might be slow.
    
    # Alternative: Just return the largest subset of the MIS set that is valid.
    # Since n is small (likely < 20), we can brute force subsets of the MIS set?
    # No, 2^20 is 1M.
    
    # Let's try a greedy removal:
    # While invalid:
    #   For each element x in current:
    #     If removing x makes it valid (or reduces conflicts):
    #       Remove x.
    #       Break.
    #   If no element removal helps:
    #     Remove the largest element?
    
    # This is getting complicated.
    # Let's assume the MIS set is valid or very close.
    # If it's invalid, we can just do a random restart of the MIS solver with a different pivot selection?
    
    # For this proposal, let's just return the MIS set if it's valid,
    # otherwise, fall back to a simple greedy construction that checks full validity.
    
    return current

def construct():
    start = time.time()
    deadline = start + 55
    
    # 1. Build Conflict Graph
    adj = _build_conflict_graph()
    
    # 2. Find MIS
    mis_set = _mis_branch_and_bound(adj, deadline)
    
    # 3. Validate
    if _is_valid_full(mis_set):
        return [int(x + 1) for x in mis_set]
        
    # 4. Fallback: Greedy with full validity check
    # This ensures we return a valid set even if MIS fails.
    current = []
    pool = list(range(1, 257))
    # Sort by some heuristic?
    # Randomize to avoid local optima?
    import random
    random.seed(42)
    
    # Try multiple random orders
    best = []
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        random.shuffle(pool)
        curr = []
        for x in pool:
            if time.time() > deadline - 2:
                break
            # Check if adding x is valid
            test = curr + [x]
            if _is_valid_full(test):
                curr.append(x)
        if len(curr) > len(best):
            best = curr
            
    if _is_valid_full(best):
        return [int(x) for x in best]
        
    # Ultimate fallback: empty set or single element
    return [256]

# EVOLVE-BLOCK-END
