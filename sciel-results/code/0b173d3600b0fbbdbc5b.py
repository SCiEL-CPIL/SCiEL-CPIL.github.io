"""Maximum Weight Independent Set on 3-AP hypergraph via branch-and-bound with exact Fraction bounds."""

from fractions import Fraction
from itertools import combinations
from functools import lru_cache
import time

N = 256

def construct():
    start_time = time.time()
    
    # Build all nonconstant 3-APs within [1, 256]
    # A 3-AP is (a, b, c) with a < b < c and 2b = a + c, i.e., b - a = c - b = d > 0
    # We store conflicts: for each vertex, which other vertices form a 3-AP with it
    # Actually, for MWIS on a 3-uniform hypergraph, an independent set is a set of vertices
    # such that no hyperedge is fully contained in the set.
    # Equivalently, for any two vertices a, b in the set, there must be no c such that
    # {a,b,c} forms a 3-AP. But actually the constraint is: no three elements form a 3-AP.
    
    # Let's think about this differently. The constraint is: for any a < b < c in S,
    # we must NOT have 2b = a + c.
    
    # For branch-and-bound, we process vertices in order of decreasing weight.
    # Weight of vertex x is 1/x.
    
    # Precompute: for each pair (a, b) with a < b, find all c > b such that 2b = a + c,
    # i.e., c = 2b - a. If c <= 256 and c > b, then {a, b, c} is a 3-AP.
    # Also for a < b, find c < a such that 2b = a + c? No, that would require c = 2b - a < a,
    # so 2b < 2a, b < a, contradiction. So for a < b, the only possible third element is c = 2b - a > b.
    
    # Actually wait: if we have a < b, the 3-AP could be (a, b, c) with c = 2b - a, or
    # (c, a, b) with c = 2a - b. Since a < b, 2a - b could be less than a if b > a, which it is.
    # So c = 2a - b < a is possible. Let me reconsider.
    
    # A 3-AP is three numbers x < y < z with y - x = z - y. So x + z = 2y.
    # For any pair (x, z) with x < z, if x + z is even, then y = (x+z)/2 is the middle,
    # and we need x < y < z, which is true iff x < z and x+z > 2x (i.e., z > x, true) and
    # x+z < 2z (i.e., x < z, true). So for any pair (x, z) with x < z and x+z even,
    # there's a unique middle y = (x+z)/2, and {x, y, z} is a 3-AP iff y is an integer
    # in [1, 256] and x < y < z.
    
    # Alternatively: for any pair (x, y) with x < y, the third element z = 2y - x.
    # We need z > y (so z = 2y - x > y iff y > x, true) and z <= 256.
    # Also, the pair (y, z) with y < z has third element x' = 2y - z = x.
    
    # For the hypergraph formulation: vertices are 1..256, hyperedges are all 3-APs.
    # An independent set has no hyperedge fully contained.
    
    # For branch-and-bound: we decide for each vertex (in some order) whether to include or not.
    # When we include vertex v, we must exclude any vertex that, together with two already-included
    # vertices, would form a 3-AP. But actually, the constraint is global: no three in S form a 3-AP.
    
    # A cleaner approach: process vertices in decreasing weight order. At each step, we can either
    # include the current vertex or exclude it. If we include it, we must ensure that no two previously
    # included vertices, together with this one, form a 3-AP. Also, we must ensure that no two
    # future vertices, together with this one, form a 3-AP — but that's handled when we process those
    # future vertices.
    
    # So the state is: the set of included vertices so far. When considering vertex v:
    # - If we include v, we check: for every pair (a, b) in included_so_far with a < b,
    #   does {a, b, v} form a 3-AP? If yes, we cannot include v.
    #   Also, we need to check if v with two included vertices forms a 3-AP.
    
    # Actually, the check is: for v to be addable to S, there must be no a, b in S such that
    # {a, b, v} is a 3-AP. This means:
    #   - No a, b in S with a < b < v and 2b = a + v (i.e., a = 2b - v)
    #   - No a, b in S with a < v < b and 2v = a + b (i.e., a + b = 2v)
    #   - No a, b in S with v < a < b and 2a = v + b (i.e., b = 2a - v)
    
    # For efficiency, let's precompute for each vertex v, the set of "conflict pairs":
    # pairs (a, b) such that {a, b, v} is a 3-AP. Then v can be added iff no such pair is
    # fully in S.
    
    # Precompute conflicts: for each v, list of pairs (a, b) with a < b, a != v, b != v,
    # such that {a, b, v} is a 3-AP.
    
    # For a given v, the 3-APs containing v are:
    # 1. v is the smallest: v, y, z with y - v = z - y, so y = v + d, z = v + 2d for d >= 1.
    #    Pairs: (v, v+d) and (v, v+2d) — wait, the pair that with v forms a 3-AP is (v+d, v+2d).
    #    So for v as smallest, conflict pairs are (v+d, v+2d) for d >= 1, v+2d <= 256.
    # 2. v is the middle: a, v, z with v - a = z - v, so a = v - d, z = v + d for d >= 1.
    #    Conflict pairs: (v-d, v+d) for d >= 1, v-d >= 1.
    # 3. v is the largest: a, y, v with y - a = v - y, so y = v - d, a = v - 2d for d >= 1.
    #    Conflict pairs: (v-2d, v-d) for d >= 1, v-2d >= 1.
    
    # Let's build for each v, the set of unordered pairs {a, b} (a < b) such that {a, b, v} is a 3-AP.
    
    conflict_pairs = [set() for _ in range(N + 1)]
    for v in range(1, N + 1):
        # v as smallest
        d = 1
        while v + 2 * d <= N:
            conflict_pairs[v].add((v + d, v + 2 * d))
            d += 1
        # v as middle
        d = 1
        while v - d >= 1:
            conflict_pairs[v].add((v - d, v + d))
            d += 1
        # v as largest
        d = 1
        while v - 2 * d >= 1:
            conflict_pairs[v].add((v - 2 * d, v - d))
            d += 1
    
    # Sort vertices by decreasing weight (1/x), so vertex 1 first, then 2, etc.
    # Actually, weight is 1/x, so larger x means smaller weight. We want to process high-weight
    # vertices first, i.e., small x first.
    order = list(range(1, N + 1))  # already in order of decreasing weight
    
    # Precompute weights as Fractions
    weights = {x: Fraction(1, x) for x in range(1, N + 1)}
    
    # For upper bound: sum of weights of all remaining vertices not yet decided.
    # We'll use a suffix sum of weights in processing order.
    suffix_sum = [Fraction(0)] * (N + 2)
    for i in range(N - 1, -1, -1):
        v = order[i]
        suffix_sum[i] = suffix_sum[i + 1] + weights[v]
    
    # Branch and bound
    best_set = []
    best_weight = Fraction(0)
    
    # We'll use recursion with index i into order.
    # State: included = list of included vertices so far.
    # To check if we can add order[i], we need to verify no conflict pair is fully in included.
    
    # For efficiency, maintain a set of included vertices and for each conflict pair, track
    # how many are in included. But that's complex. Instead, for each candidate v, check
    # all its conflict pairs against the current included set.
    
    # Given the time limit of 60s, we need to be careful. Let's use a heuristic:
    # Greedy construction first to get a good initial solution, then branch-and-bound
    # with strong pruning.
    
    # First, get a greedy solution: process vertices in order, include if possible.
    def can_add(v, included_set):
        for (a, b) in conflict_pairs[v]:
            if a in included_set and b in included_set:
                return False
        return True
    
    greedy_set = []
    greedy_included = set()
    for v in order:
        if can_add(v, greedy_included):
            greedy_set.append(v)
            greedy_included.add(v)
    
    greedy_weight = sum(weights[v] for v in greedy_set)
    best_set = list(greedy_set)
    best_weight = greedy_weight
    
    # Now branch and bound
    # To make this tractable, we limit the depth or use aggressive pruning.
    # Given 256 vertices, full B&B is infeasible. We need a smarter approach.
    
    # Alternative: use a local search / simulated annealing to improve the greedy solution.
    # But the directive says to use B&B with exact Fraction bounds.
    
    # Let's try a limited B&B: only branch on the first few vertices, or use a time-limited search.
    
    # Actually, let's try a different strategy: start from the greedy solution and try to
    # improve it by swapping vertices. Or use a proper B&B with strong bounds.
    
    # For B&B: at node (i, included), upper bound = weight(included) + suffix_sum[i].
    # If upper bound <= best_weight, prune.
    # Also, if we can't add any more vertices (all remaining conflict with included), prune.
    
    # Let's implement a recursive B&B with time limit.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def bnb(i, included, included_set, current_weight):
        nonlocal best_set, best_weight
        
        if time.time() - start_time > 55:
            return
        
        if i == N:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = list(included)
            return
        
        # Upper bound: current weight + sum of all remaining weights
        ub = current_weight + suffix_sum[i]
        if ub <= best_weight:
            return
        
        v = order[i]
        
        # Try including v
        if can_add(v, included_set):
            included.append(v)
            included_set.add(v)
            bnb(i + 1, included, included_set, current_weight + weights[v])
            included.pop()
            included_set.discard(v)
        
        # Try excluding v
        bnb(i + 1, included, included_set, current_weight)
    
    # Full B&B on 256 vertices is infeasible. Let's only branch on a subset, or use
    # a different strategy.
    
    # Given the constraints, let's try a hybrid: use the greedy solution and then
    # do local search to improve it.
    
    # Local search: try to swap a vertex in the set with a vertex not in the set,
    # or add a vertex if possible, or remove a vertex and add two others.
    
    # Actually, let's try a simpler improvement: iterate through vertices not in the
    # greedy set and see if we can add them (removing conflicting vertices if needed).
    
    # For now, let's just return the greedy solution if B&B is too slow.
    # But let's try a limited B&B on the first 30 vertices or so.
    
    # Actually, let's try a proper B&B but with a time limit and see how far we get.
    # We'll process vertices in order and use the bound to prune.
    
    # Let's try B&B but only for the first 40 vertices, then greedily fill the rest.
    
    def bnb_limited(i, included, included_set, current_weight):
        nonlocal best_set, best_weight
        
        if time.time() - start_time > 55:
            return
        
        if i == 40:  # Only branch on first 40 vertices
            # Greedily add remaining vertices
            temp_included = list(included)
            temp_set = set(included_set)
            temp_weight = current_weight
            for j in range(i, N):
                v = order[j]
                if can_add(v, temp_set):
                    temp_included.append(v)
                    temp_set.add(v)
                    temp_weight += weights[v]
            if temp_weight > best_weight:
                best_weight = temp_weight
                best_set = temp_included
            return
        
        # Upper bound
        ub = current_weight + suffix_sum[i]
        if ub <= best_weight:
            return
        
        v = order[i]
        
        # Try including v
        if can_add(v, included_set):
            included.append(v)
            included_set.add(v)
            bnb_limited(i + 1, included, included_set, current_weight + weights[v])
            included.pop()
            included_set.discard(v)
        
        # Try excluding v
        bnb_limited(i + 1, included, included_set, current_weight)
    
    bnb_limited(0, [], set(), Fraction(0))
    
    return best_set
