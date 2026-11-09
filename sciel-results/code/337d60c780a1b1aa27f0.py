"""
Strategy: Branch-and-Bound for Minimum Maximal Independent Set (MMIS)
on a 3-uniform hypergraph derived from isosceles triangles.

The problem is to find a subset S of points such that:
1. S is an independent set (no three points in S form an isosceles triangle).
2. S is maximal (no point outside S can be added to S without violating condition 1).
3. |S| is minimized.

We use bitsets (Python integers) to represent sets of points for fast intersection and propagation.
"""

N = 16
TOTAL_POINTS = N * N
DEADLINE = 330.0
MARGIN = 5.0

import time
import random

# Precompute the hypergraph
# vertices: 0 .. TOTAL_POINTS - 1
# hyperedges: sets of 3 vertices that form an isosceles triangle

def _point_to_idx(x, y):
    return x * N + y

def _idx_to_point(idx):
    return (idx // N, idx % N)

# Precompute for each vertex v, the set of vertices u such that {v, u} is part of an isosceles triangle with some w?
# Actually, for independence, we need to ensure no hyperedge is fully contained in S.
# A hyperedge is a triple {a, b, c} such that dist(a,b)==dist(a,c) or dist(a,b)==dist(b,c) or dist(a,c)==dist(b,c).

# To speed up propagation:
# If we pick vertex v, we must exclude any hyperedge containing v.
# This means for any other vertex u, if there exists w such that {v, u, w} is a hyperedge,
# then we cannot pick both u and w.
# This creates a "conflict" between u and w conditional on v being picked.
# This is complex for simple bitset propagation.

# Simpler Propagation:
# Standard Independent Set propagation:
# If we pick v, we must remove all vertices u such that there is an edge (v, u) in the underlying graph?
# No, the constraint is on triples.
# If we pick v, and we pick u, then we must NOT pick any w such that {v, u, w} is a hyperedge.
# This is a "forbidden pair" (u, w) given v.

# Let's precompute:
# conflict[v][u] = set of w such that {v, u, w} is an isosceles triangle.
# Note: If we pick v, and we pick u, then all w in conflict[v][u] are forbidden.

# Precomputation
points = [(x, y) for x in range(N) for y in range(N)]
# Map index to point
idx_to_pt = {i: _idx_to_point(i) for i in range(TOTAL_POINTS)}

# Precompute distances squared
def _d2(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

# Build conflict sets
# conflict[v][u] is a bitset of w
# This is O(N^3) which is 16^3 = 4096 triples. Very fast.
# But we need to store it efficiently.
# conflict[v][u] is a list of w.
# We can store it as a bitset integer for each pair (v, u).

# Let's store conflict as a 2D array of bitsets?
# conflict[v][u] = bitset of w.
# Since N=16, TOTAL_POINTS=256.
# A bitset of 256 bits fits in a Python int.

conflict = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]

# Iterate over all triples
for i in range(TOTAL_POINTS):
    pi = idx_to_pt[i]
    for j in range(i+1, TOTAL_POINTS):
        pj = idx_to_pt[j]
        dij = _d2(pi, pj)
        for k in range(j+1, TOTAL_POINTS):
            pk = idx_to_pt[k]
            dik = _d2(pi, pk)
            djk = _d2(pj, pk)
            if dij == dik or dij == djk or dik == djk:
                # {i, j, k} is an isosceles triangle
                # If we pick i and j, k is forbidden.
                # If we pick i and k, j is forbidden.
                # If we pick j and k, i is forbidden.
                conflict[i][j] |= (1 << k)
                conflict[i][k] |= (1 << j)
                conflict[j][i] |= (1 << k)
                conflict[j][k] |= (1 << i)
                conflict[k][i] |= (1 << j)
                conflict[k][j] |= (1 << i)

# Also, for maximality check, we need to know if a point p can be added to S.
# p can be added to S if there is no hyperedge {p, u, v} with u, v in S.
# This is equivalent to: for all u in S, (conflict[p][u] & S) == 0.
# Or: for all u in S, (conflict[u][p] & S) == 0. (Symmetric)

def _can_add(p, S_bitset):
    # Check if p can be added to S
    # For each u in S, check if any v in S is in conflict[p][u]
    # Actually, it's enough to check if there exists u, v in S such that {p, u, v} is a hyperedge.
    # This means u is in S and v is in S and v is in conflict[p][u].
    # So, for each u in S, if (conflict[p][u] & S) != 0, then p cannot be added.
    # We can iterate through bits in S.
    # But S is a bitset.
    # Let's iterate through u in S.
    temp = S_bitset
    while temp:
        lsb = temp & -temp
        u = lsb.bit_length() - 1
        temp ^= lsb
        if (conflict[p][u] & S_bitset):
            return False
    return True

def _is_maximal(S_bitset):
    # Check if S is maximal
    # For every p not in S, _can_add(p, S) must be False.
    # If there exists p not in S such that _can_add(p, S) is True, S is not maximal.
    not_S = ((1 << TOTAL_POINTS) - 1) ^ S_bitset
    temp = not_S
    while temp:
        lsb = temp & -temp
        p = lsb.bit_length() - 1
        temp ^= lsb
        if _can_add(p, S_bitset):
            return False
    return True

def _branch_and_bound(current_S, available, best_size, best_S, start_time):
    """
    current_S: bitset of points already chosen.
    available: bitset of points that can still be chosen (independent of current_S).
    best_size: current best size.
    best_S: current best bitset.
    """
    if time.time() - start_time > DEADLINE - MARGIN:
        return best_size, best_S

    size = current_S.bit_count()
    if size >= best_size:
        return best_size, best_S

    # If no available points, check maximality
    if available == 0:
        if _is_maximal(current_S):
            if size < best_size:
                best_size = size
                best_S = current_S
        return best_size, best_S

    # Heuristic: Pick a point to branch on.
    # Choose a point with high degree of conflict?
    # Or just the first available point?
    # Let's pick the point with the most conflicts with other available points?
    # Actually, we want to prune branches that cannot lead to a smaller set.
    
    # Lower bound: size + (number of points needed to cover all available points' conflicts?)
    # Simple lower bound: size.
    
    # Select a pivot point
    # Let's pick the point with the highest "degree" in the conflict graph restricted to available?
    # Degree of p in available: number of u in available such that conflict[p][u] is not empty?
    # Or number of pairs (u, v) in available such that {p, u, v} is a hyperedge?
    
    # Let's just pick the first available point for simplicity, or one that blocks the most.
    # Point p blocks u and v if {p, u, v} is a hyperedge.
    # If we pick p, we remove from available any u, v such that there exists w in available with {p, u, w} or {p, v, w}?
    # No, if we pick p, we must remove from available any pair (u, v) that forms a hyperedge with p.
    # This means if we pick p, u and v cannot both be in the final set.
    # This is a constraint on the remaining set.
    
    # Propagation:
    # If we pick p, the new available set is:
    # available' = { u in available | for all v in available', {p, u, v} is not a hyperedge }
    # This is hard to compute directly as a bitset without iterating.
    
    # Let's try a different propagation:
    # If we pick p, we cannot pick any u such that there exists v in current_S with {p, u, v} hyperedge.
    # But current_S is fixed.
    # The constraint is on the NEW points we pick.
    
    # Let's use a simpler branching:
    # Branch 1: Include p.
    # Branch 2: Exclude p.
    
    # For Branch 1 (Include p):
    # New current_S = current_S | (1 << p)
    # New available:
    # Any u in available such that there exists v in available with {p, u, v} hyperedge must be handled carefully.
    # Actually, if we pick p, we must ensure that no two points u, v in the REMAINING set form a hyperedge with p.
    # This means the remaining set must be an independent set in the graph G_p, where edges are (u, v) if {p, u, v} is a hyperedge.
    # This is a standard Independent Set problem on G_p.
    
    # This suggests that simple bitset propagation for hypergraphs is tricky.
    
    # Alternative: Use a greedy lower bound.
    # If we pick p, we can estimate the minimum additional points needed.
    
    # Let's stick to a simpler B&B:
    # 1. Pick a point p from available.
    # 2. Branch Include p:
    #    - Add p to S.
    #    - Remove from available any point u such that {p, u, v} is a hyperedge for some v in S? No, v is in new S.
    #    - Actually, if we add p to S, we must remove from available any u such that there exists v in S (including p) with {p, u, v} hyperedge?
    #    - No, the constraint is that NO three points in S form a hyperedge.
    #    - If p is in S, and u is in S, then no v in S can be such that {p, u, v} is a hyperedge.
    #    - So, if we add p to S, we must remove from available any u such that there exists v in S (old S) with {p, u, v} hyperedge.
    #    - AND, for any u, v in new available, if {p, u, v} is a hyperedge, then we cannot pick both u and v.
    #    - This second part is a constraint on the subproblem.
    
    # Given the complexity, let's try a different approach:
    # Randomized Local Search with Maximality Check.
    # Start with a random independent set.
    # Try to add points.
    # Try to remove points to see if it remains maximal.
    # This is what the current program does.
    
    # The directive asks for a DIFFERENT strategy.
    # Let's try "Simulated Annealing" or "Genetic Algorithm" on the bitset representation.
    # Or "Constraint Programming" with backtracking.
    
    # Let's implement a backtracking search with a better heuristic.
    # Heuristic: Order points by "blocking power".
    # Point p has high blocking power if it forms many isosceles triangles with other points.
    
    # Precompute blocking power
    block_power = [0] * TOTAL_POINTS
    for i in range(TOTAL_POINTS):
        for j in range(TOTAL_POINTS):
            if i != j and conflict[i][j]:
                block_power[i] += conflict[i][j].bit_count()
        
    # Sort points by block_power descending
    sorted_points = sorted(range(TOTAL_POINTS), key=lambda x: block_power[x], reverse=True)
    
    # Backtracking
    # State: current_S, available
    # We process points in sorted_points order.
    
    def _search(idx, current_S, available, best_size, best_S):
        nonlocal best_size, best_S
        if time.time() - start_time > DEADLINE - MARGIN:
            return
        
        size = current_S.bit_count()
        if size >= best_size:
            return
            
        if idx == TOTAL_POINTS:
            if _is_maximal(current_S):
                if size < best_size:
                    best_size = size
                    best_S = current_S
            return
            
        p = sorted_points[idx]
        
        # Option 1: Exclude p
        _search(idx + 1, current_S, available, best_size, best_S)
        
        # Option 2: Include p
        if (available & (1 << p)) != 0:
            # Check if p can be added to current_S
            # i.e., no u, v in current_S such that {p, u, v} is hyperedge
            # This is _can_add(p, current_S)
            if _can_add(p, current_S):
                new_S = current_S | (1 << p)
                # Update available
                # Remove p from available
                new_available = available & ~(1 << p)
                # Remove any u from new_available such that {p, u, v} is hyperedge for some v in new_S?
                # No, the constraint is on the FINAL set.
                # If we pick p, we must ensure that no two points u, v in the REMAINING picks form a hyperedge with p.
                # This means for any u in new_available, if there exists v in new_available such that {p, u, v} is hyperedge,
                # then we cannot pick both u and v.
                # This is a constraint on the subproblem, not a simple removal.
                
                # However, we CAN remove u from new_available if there exists v in new_S (which includes p) such that {p, u, v} is hyperedge.
                # Since p is in new_S, we check if there exists v in new_S (old S) such that {p, u, v} is hyperedge.
                # If so, u cannot be picked.
                
                # Let's compute the forbidden points due to p and old S.
                forbidden = 0
                temp = new_S
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    temp ^= lsb
                    if v != p:
                        forbidden |= conflict[p][v]
                
                new_available &= ~forbidden
                
                _search(idx + 1, new_S, new_available, best_size, best_S)

    # Start search
    all_points_bitset = (1 << TOTAL_POINTS) - 1
    _search(0, 0, all_points_bitset, TOTAL_POINTS, 0)
    
    return best_size, best_S

def construct():
    start_time = time.time()
    
    # Initial greedy to get a good upper bound
    random.seed(42)
    best_size = TOTAL_POINTS
    best_S = 0
    
    # Greedy construction
    available = (1 << TOTAL_POINTS) - 1
    current_S = 0
    points_list = list(range(TOTAL_POINTS))
    random.shuffle(points_list)
    
    for p in points_list:
        if time.time() - start_time > DEADLINE - MARGIN:
            break
        if _can_add(p, current_S):
            current_S |= (1 << p)
            
    best_size = current_S.bit_count()
    best_S = current_S
    
    # Try to improve with B&B
    # B&B might be too slow for N=16.
    # Let's try a few random restarts with greedy + local search.
    
    for trial in range(50):
        if time.time() - start_time > DEADLINE - MARGIN:
            break
        
        random.seed(1000 + trial)
        available = (1 << TOTAL_POINTS) - 1
        current_S = 0
        points_list = list(range(TOTAL_POINTS))
        random.shuffle(points_list)
        
        for p in points_list:
            if time.time() - start_time > DEADLINE - MARGIN:
                break
            if _can_add(p, current_S):
                current_S |= (1 << p)
        
        size = current_S.bit_count()
        if size < best_size:
            best_size = size
            best_S = current_S
            
        # Local search: try to remove points
        # If removing a point keeps the set maximal, we have a smaller maximal set.
        improved = True
        while improved:
            improved = False
            temp_S = best_S
            for p in range(TOTAL_POINTS):
                if time.time() - start_time > DEADLINE - MARGIN:
                    break
                if (temp_S & (1 << p)) != 0:
                    # Try removing p
                    new_S = temp_S & ~(1 << p)
                    if _is_maximal(new_S):
                        temp_S = new_S
                        improved = True
                        break # Restart loop with new S
            
            if temp_S.bit_count() < best_size:
                best_size = temp_S.bit_count()
                best_S = temp_S

    # Convert best_S to list of (x, y)
    result = []
    temp = best_S
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        temp ^= lsb
        result.append(_idx_to_point(idx))
        
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
