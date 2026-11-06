import time
import sys
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    N = 256
    
    # Precompute conflicts: conflict[x] is a set of y such that x and y cannot both be in S
    # x and y conflict if there exists z in [1,N] such that {x,y,z} is a 3-AP
    # This happens if:
    # 1. x+y is even, mid=(x+y)/2 is in [1,N] -> x,y are endpoints
    # 2. 2y-x is in [1,N] and != x,y -> x,y are first/second (or second/first)
    # 3. 2x-y is in [1,N] and != x,y -> y,x are first/second (or second/first)
    # Actually, simpler: x and y are in conflict if they appear together in any 3-AP.
    # For any pair (x,y) with x!=y, they are in a 3-AP if:
    # - (x+y) is even and 1 <= (x+y)/2 <= N
    # - 2y-x is in [1,N] and 2y-x != x and 2y-x != y
    # - 2x-y is in [1,N] and 2x-y != x and 2x-y != y
    
    conflict = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            if x == y:
                continue
            # Check if x, y are in a 3-AP
            # Case 1: x, y are endpoints
            if (x + y) % 2 == 0:
                mid = (x + y) // 2
                if 1 <= mid <= N:
                    conflict[x].add(y)
                    conflict[y].add(x)
            # Case 2: x, y are consecutive terms (x, y, z) or (z, x, y)
            # If x, y are consecutive, the third term is 2y-x or 2x-y
            third = 2 * y - x
            if 1 <= third <= N and third != x and third != y:
                conflict[x].add(y)
                conflict[y].add(x)
            third = 2 * x - y
            if 1 <= third <= N and third != x and third != y:
                conflict[x].add(y)
                conflict[y].add(x)
    
    # Precompute reciprocals
    recips = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recips[i] = 1.0 / i

    # Greedy heuristic to get a good initial solution
    # Sort by weight descending, add if no conflict with current set
    def greedy_initial():
        S = set()
        # Try different orderings
        best_S = set()
        best_score = 0.0
        
        # Order 1: descending weight (small numbers first)
        order = sorted(range(1, N + 1), key=lambda x: -recips[x])
        S = set()
        for x in order:
            if not any(y in S for y in conflict[x]):
                S.add(x)
        score = sum(recips[x] for x in S)
        if score > best_score:
            best_score = score
            best_S = S.copy()
        
        # Order 2: random shuffles
        for _ in range(20):
            order = list(range(1, N + 1))
            random.shuffle(order)
            S = set()
            for x in order:
                if not any(y in S for y in conflict[x]):
                    S.add(x)
            score = sum(recips[x] for x in S)
            if score > best_score:
                best_score = score
                best_S = S.copy()
        
        return best_S, best_score

    best_S, best_score = greedy_initial()
    
    # Branch and Bound for Maximum Weight Independent Set
    # We solve for the complement: find a minimum weight vertex cover? No, that's not right.
    # We want max weight independent set.
    # B&B:
    # State: (candidates, current_set, current_weight)
    # candidates: set of vertices not yet decided (neither in current_set nor excluded)
    # We branch on a vertex v in candidates:
    #   1. Include v: add v to current_set, remove v and its neighbors from candidates
    #   2. Exclude v: remove v from candidates
    # Bound: current_weight + sum of weights of all vertices in candidates (optimistic, assuming all can be added)
    # Better bound: greedy coloring of the subgraph induced by candidates, then sum of color class weights?
    # Or simply: current_weight + sum(w(u) for u in candidates) is a valid upper bound (since we can't add more than all candidates)
    # But this is loose. A tighter bound: for each u in candidates, we can only add u if it doesn't conflict with current_set.
    # Actually, the standard MWIS B&B uses:
    # Upper bound = current_weight + sum of weights of a maximal independent set in the candidate subgraph (greedy)
    # Or even simpler: current_weight + sum(w(u) for u in candidates if u not in conflict with current_set)
    # But that's still loose.
    
    # Let's use a simple bound: current_weight + sum(w(u) for u in candidates)
    # And branch on the vertex with highest weight in candidates.
    
    # To speed up, we can use bitmasks if N were smaller, but N=256 is too large for 256-bit integers in Python easily (though possible with int).
    # Let's use sets for simplicity, but optimize the conflict check.
    
    # We'll use a recursive B&B with pruning.
    # To avoid recursion depth issues, we might need to increase recursion limit or use iterative.
    sys.setrecursionlimit(10000)
    
    # Precompute neighbor lists as sorted lists for faster iteration
    conflict_list = [sorted(conflict[i]) for i in range(N + 1)]
    
    # For faster "can add" check, we maintain a set of excluded vertices (neighbors of current_set)
    # State: (candidates, current_weight, excluded)
    # candidates: set of vertices that can still be added (not in current_set, not excluded)
    # excluded: set of vertices that cannot be added (neighbors of current_set)
    
    # Actually, candidates should be: all vertices not in current_set and not excluded.
    # When we add v to current_set, we remove v and all its neighbors from candidates.
    
    # Let's implement the B&B.
    # We'll use a global best_S and best_score.
    
    # To make it faster, we can order the vertices by weight and always branch on the highest weight vertex in candidates.
    
    # Initialize candidates as all vertices
    initial_candidates = set(range(1, N + 1))
    
    # We'll use a stack for iterative B&B to avoid recursion limits
    # Stack items: (candidates, current_weight, current_set)
    # But storing current_set is expensive. We can just store the list of added vertices.
    
    # Alternatively, we can use the "include/exclude" branching on a fixed order of vertices.
    # Let's sort vertices by weight descending: v1, v2, ..., vN
    # At step i, we decide if vi is in the set or not.
    # If vi is in the set, we must exclude all its neighbors from future consideration.
    # If vi is not in the set, we just move to vi+1.
    
    # This is a standard approach.
    
    # Let's use this approach.
    # Order: sorted by weight descending
    order = sorted(range(1, N + 1), key=lambda x: -recips[x])
    
    # We'll use a recursive function with memoization? No, the state is too large.
    # We'll use simple B&B with the bound: current_weight + sum of weights of remaining candidates that are not excluded.
    
    # To speed up the bound calculation, we can precompute prefix sums of weights in the order.
    # But the excluded set changes, so we need to check each remaining vertex.
    
    # Let's try a different approach: use the greedy coloring bound.
    # For the subgraph induced by candidates, compute a greedy coloring.
    # The weight of each color class is the sum of weights of vertices in that class.
    # The upper bound is current_weight + sum of the maximum weight color class? No, that's not right.
    # The correct bound is: current_weight + sum of weights of all color classes? No.
    # Actually, the standard bound for MWIS using coloring is:
    # Upper bound = current_weight + sum of weights of the color classes in the greedy coloring of the candidate subgraph.
    # Wait, no. The independent set can contain at most one vertex from each color class.
    # So the upper bound is current_weight + sum of (max weight in each color class)? No, that's not right either.
    # Actually, the bound is: current_weight + sum of weights of all vertices in the candidate set, but that's loose.
    # The coloring bound is: current_weight + sum of weights of the color classes, where each color class is an independent set.
    # But we can pick at most one vertex from each color class? No, we can pick multiple vertices from a color class if they are not adjacent.
    # Wait, in a proper coloring, no two vertices in the same color class are adjacent.
    # So each color class is an independent set.
    # Therefore, the maximum weight independent set in the candidate subgraph is at most the sum of the weights of all vertices in the candidate subgraph.
    # But that's the same as the loose bound.
    
    # Actually, the standard bound is:
    # Upper bound = current_weight + sum of weights of a maximal independent set in the candidate subgraph (computed greedily).
    # This is a valid upper bound because any independent set is contained in some maximal independent set? No, that's not true.
    # Actually, the greedy maximal independent set is not necessarily an upper bound.
    
    # Let's use the simple bound: current_weight + sum(w(u) for u in candidates if u not in excluded)
    # And hope it's fast enough.
    
    # To make it faster, we can use bitmasks.
    # Let's represent the candidate set as a bitmask (Python int).
    # N=256, so we need 256 bits. Python ints can handle this.
    
    # Let's re-implement with bitmasks.
    
    # Map each vertex to a bit position
    # vertex i -> bit (i-1)
    
    # Precompute conflict masks
    conflict_mask = [0] * (N + 1)
    for x in range(1, N + 1):
        mask = 0
        for y in conflict[x]:
            mask |= (1 << (y - 1))
        conflict_mask[x] = mask
    
    # Precompute weight array
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = recips[i]
    
    # Order vertices by weight descending
    order = sorted(range(1, N + 1), key=lambda x: -weights[x])
    
    # We'll use the branch-and-bound with the simple bound.
    # To speed up, we'll use a recursive function with the following state:
    # (index, candidates_mask, current_weight, current_set_list)
    # index: current position in the order
    # candidates_mask: bitmask of vertices that can still be added
    # current_weight: sum of weights of vertices in current_set
    # current_set_list: list of vertices in current_set (for reconstruction)
    
    # But this might be slow due to the large number of states.
    
    # Let's try a different approach: use the "include/exclude" branching on the highest weight vertex in candidates.
    
    # We'll use a stack for iterative B&B.
    # Stack items: (candidates_mask, current_weight, current_set_list)
    
    # Initial state: all vertices are candidates
    initial_mask = (1 << N) - 1
    # But we only have vertices 1..N, so bits 0..N-1
    # Let's use bits 0..N-1 for vertices 1..N
    
    # Actually, let's use a simpler approach: just use the greedy solution and then try to improve it with local search.
    # The B&B might be too slow for N=256.
    
    # Let's stick with the local search but improve it.
    
    # We'll use the best_S from greedy_initial as the starting point.
    
    # Convert best_S to a set for fast lookup
    current_set = set(best_S)
    current_score = best_score
    
    # Local search: try to add a vertex, remove a vertex, or swap
    # We'll use simulated annealing with a better cooling schedule.
    
    T = 1.0
    last_improvement = time.time()
    steps = 0
    
    # Precompute a list of vertices for random selection
    all_vertices = list(range(1, N + 1))
    
    while time.time() < deadline - 2:
        if time.time() - last_improvement > 5.0:
            T *= 0.9
            if T < 0.01:
                break
        
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add a random vertex not in current_set
            candidates = [x for x in all_vertices if x not in current_set]
            if not candidates:
                continue
            y = random.choice(candidates)
            # Check if y can be added
            can_add = True
            for z in current_set:
                if y in conflict[z]:
                    can_add = False
                    break
            if can_add:
                current_set.add(y)
                current_score += recips[y]
                if current_score > best_score:
                    best_score = current_score
                    best_S = current_set.copy()
                    last_improvement = time.time()
                else:
                    # Undo
                    current_set.remove(y)
                    current_score -= recips[y]
            else:
                # Rejected by constraint
                pass
                
        elif move_type < 0.7:
            # Try to remove a random vertex from current_set
            if not current_set:
                continue
            x = random.choice(list(current_set))
            current_set.remove(x)
            current_score -= recips[x]
            # Always accept removal? No, we should only accept if it leads to improvement later.
            # For now, let's just do swaps.
            # Undo
            current_set.add(x)
            current_score += recips[x]
            
        else:
            # Swap: remove x, add y
            if not current_set:
                continue
            x = random.choice(list(current_set))
            candidates = [y for y in all_vertices if y not in current_set]
            if not candidates:
                continue
            y = random.choice(candidates)
            # Check if y can be added after removing x
            # y must not conflict with any vertex in current_set \ {x}
            can_add = True
            for z in current_set:
                if z == x:
                    continue
                if y in conflict[z]:
                    can_add = False
                    break
            if can_add:
                delta = recips[y] - recips[x]
                if delta > 0 or random.random() < (2.718281828 ** (delta / T)):
                    # Accept swap
                    current_set.remove(x)
                    current_set.add(y)
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_S = current_set.copy()
                        last_improvement = time.time()
                else:
                    # Reject swap
                    pass
            else:
                # y not valid, do nothing
                pass
        
        T *= 0.999
        steps += 1
    
    return list(best_S)
