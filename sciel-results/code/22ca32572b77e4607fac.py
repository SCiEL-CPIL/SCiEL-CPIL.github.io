"""Reciprocal weight of three-progression-free sets."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START

def _build_conflict_bitsets(n=256):
    """Precompute conflict bitsets.
    conflicts[i] is a bitset where bit j is set if i and j are in a 3-AP with some third element.
    Actually, we need: for a 3-AP-free set, no three elements form a 3-AP.
    The constraint is: for any a, b in S, 2b-a not in S (and a != b, and 2b-a != a, 2b-a != b).
    
    Better formulation: vertices 1..n. Edge between i and j if there exists k such that
    {i,j,k} forms a 3-AP. But that's not quite right for MWIS.
    
    Actually, the constraint is that S contains no 3-term AP. This is a 3-uniform hypergraph
    independent set problem. We can convert to a graph by noting that if we fix two elements,
    the third is determined.
    
    For branch-and-bound, we'll use the following approach:
    - Process vertices in some order
    - At each step, decide to include or exclude
    - Use bitsets to track which vertices are still available
    
    Let's precompute: for each pair (a, b) with a < b, the third element c = 2b - a.
    If c is in [1, n] and c != a and c != b, then {a, b, c} is a 3-AP.
    
    For the B&B, we can use a different approach:
    - Sort vertices by weight (1/x) in descending order (smallest x first)
    - Use a recursive search with pruning
    
    Actually, let's use a cleaner formulation. Define for each vertex v, the set of 
    "forbidden pairs". When we include v, it forbids certain other vertices from being 
    included together.
    
    Simpler: precompute for each vertex v, the list of vertices u such that v and u 
    conflict (i.e., there exists w such that {v, u, w} is a 3-AP). But this isn't quite 
    right because the conflict depends on all three.
    
    Let me think differently. The 3-AP-free condition means: for all distinct a, b, c in S,
    2b != a + c.
    
    Equivalently: for all a, b in S with a != b, 2b - a not in S (when 2b-a is in [1,n] 
    and distinct from a and b).
    
    So if we have a set S, it's valid iff for all a, b in S, a != b: 
    (2b - a) not in S or (2b - a) == a or (2b - a) == b or 2b-a not in [1,n].
    
    For B&B, I'll use the following:
    - Order vertices by index (1 to n)
    - At each vertex, decide include/exclude
    - Maintain the current set S
    - When considering adding vertex v to S, check if it creates a 3-AP with any pair in S
    - Prune if the upper bound on additional weight is less than current best
    
    For efficiency, precompute for each v, the set of pairs (a, b) in [1, v-1] such that
    {a, b, v} forms a 3-AP. When adding v, we need no such pair to be entirely in S.
    
    Let's precompute: for each v, conflicts_with_lower[v] = set of pairs (a, b) with a < b < v
    such that {a, b, v} is a 3-AP.
    """
    # For each v from 1 to n, find all pairs (a, b) with a < b < v such that 
    # {a, b, v} is a 3-AP
    # A 3-AP is (a, b, c) with 2b = a + c
    # So for fixed v, we need pairs (a, b) with a < b < v and 2b = a + v, i.e., a = 2b - v
    # Also v could be the middle: (a, v, c) with 2v = a + c, c > v
    # Or v could be the first: (v, b, c) with 2b = v + c, c > b > v
    
    # Let's precompute for each v, the set of "conflict pairs" that would be violated
    # if v is added to a set S.
    
    # Actually, for the B&B algorithm, let's use a simpler but effective approach:
    # Process vertices in order 1, 2, ..., n
    # State: current set S (as a bitset), current weight
    # When considering vertex v:
    #   - If we include v: check that no 3-AP is formed with existing elements
    #   - If we exclude v: just skip
    # Upper bound: current_weight + sum of 1/x for all x > v (optimistic)
    
    # Precompute suffix sums of 1/x for upper bound
    suffix_sum = [0.0] * (n + 2)
    for i in range(n, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + 1.0 / i
    
    # Precompute for each v, the set of pairs (a, b) with a < b < v that form a 3-AP with v
    # i.e., a + v = 2b, so a = 2b - v, and we need 1 <= a < b < v
    conflict_pairs = [set() for _ in range(n + 1)]
    for v in range(1, n + 1):
        # v is the third element: a + v = 2b, a = 2b - v, need 1 <= a < b < v
        for b in range(1, v):
            a = 2 * b - v
            if 1 <= a < b:
                conflict_pairs[v].add((a, b))
        # v is the middle: a + c = 2v, c = 2v - a, need a < v < c <= n
        # This creates a constraint: if a is in S, then c = 2v-a cannot be in S
        # But c > v, so this will be handled when we get to c
        # v is the first: v + c = 2b, c = 2b - v, need v < b < c <= n
        # Similarly handled later
    
    # For the B&B, when we include v, we need to check that for no pair (a, b) in 
    # conflict_pairs[v] are both a and b in S.
    
    return suffix_sum, conflict_pairs

def _greedy_initial(n=256):
    """Greedy: add numbers in increasing order if they don't create a 3-AP."""
    selected = []
    selected_set = set()
    for x in range(1, n + 1):
        # Check if adding x creates a 3-AP
        valid = True
        for a in selected_set:
            if a == x:
                continue
            # x as middle: (a-x, x, a) or (a, x, 2x-a)
            if a - x >= 1 and (a - x) in selected_set:
                valid = False
                break
            if 2 * x - a >= 1 and (2 * x - a) in selected_set and 2 * x - a != x:
                valid = False
                break
            # x as first: (x, a, 2a-x)
            if 2 * a - x >= 1 and (2 * a - x) in selected_set and 2 * a - x != x:
                valid = False
                break
        if valid:
            selected.append(x)
            selected_set.add(x)
    return selected

def _is_valid_3ap_free(selected):
    """Verify the entire set has no 3-term AP."""
    s = set(selected)
    for a in s:
        for b in s:
            if a == b:
                continue
            c = 2 * b - a
            if c in s and c != a and c != b:
                return False
    return True

def _reciprocal_sum(selected):
    """Compute exact reciprocal sum as Fraction."""
    total = Fraction(0, 1)
    for x in selected:
        total += Fraction(1, x)
    return total

def _branch_and_bound(n=256, deadline=None):
    """Branch-and-bound to find maximum weight 3-AP-free subset.
    
    Uses the following approach:
    - Process vertices in order 1 to n
    - At each vertex, decide to include or exclude
    - Use upper bound based on suffix sum of reciprocals for pruning
    - Track the best solution found
    """
    if deadline is None:
        deadline = time.time() + 80
    
    # Precompute suffix sums for upper bound
    suffix_sum = [0.0] * (n + 2)
    for i in range(n, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + 1.0 / i
    
    # Precompute conflict information
    # For each v, precompute which pairs (a, b) with a < b < v form a 3-AP with v
    # When v is included, no such pair can both be in S
    # Also need to handle v as middle or first
    
    # Let's precompute for each v, the set of "forbidden" elements given S
    # Actually, let's use a different approach:
    # For each v, precompute:
    #   - middle_conflicts[v]: set of (a, b) with a < v < b such that a + b = 2v
    #     If v is in S, then not both a and b can be in S
    #   - first_conflicts[v]: set of (b, c) with v < b < c such that v + c = 2b
    #     If v is in S, then not both b and c can be in S
    #   - third_conflicts[v]: set of (a, b) with a < b < v such that a + v = 2b
    #     If v is in S, then not both a and b can be in S
    
    # For efficiency, let's precompute a different structure:
    # For each pair (i, j) with i < j, compute k = 2j - i
    # If k is in [1, n] and k != i and k != j, then (i, j, k) is a 3-AP
    # We can store: ap_triples = list of (i, j, k) with i < j < k and i + k = 2j
    
    ap_triples = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            k = 2 * j - i
            if k <= n and k > j:
                ap_triples.append((i, j, k))
    
    # For each element v, precompute the list of triples it participates in
    # And for each triple, which elements are "lower" than v
    triples_by_element = [[] for _ in range(n + 1)]
    for (i, j, k) in ap_triples:
        triples_by_element[i].append((j, k))  # i is first
        triples_by_element[j].append((i, k))  # j is middle
        triples_by_element[k].append((i, j))  # k is third
    
    best_weight = 0.0
    best_set = set()
    
    # Use a recursive B&B with bitset representation
    # selected_mask: bit i set means element i+1 is selected
    # available_mask: bit i set means element i+1 is still available to consider
    
    # Actually, let's use a simpler recursive approach with pruning
    
    selected = set()
    
    def _upper_bound(current_weight, next_v):
        """Upper bound: current weight + sum of 1/x for all x >= next_v"""
        return current_weight + suffix_sum[next_v]
    
    def _can_add(v, current_selected):
        """Check if v can be added to current_selected without creating a 3-AP."""
        for a in current_selected:
            if a == v:
                continue
            # Check if (a, v, c) is a 3-AP for some c in S
            # v as middle: a + c = 2v => c = 2v - a
            c = 2 * v - a
            if c != v and c in current_selected:
                return False
            # v as first: v + c = 2b => need to check all b
            # Actually, let's just check all pairs
        # More efficient: for each a in S, check if adding v creates a 3-AP
        for a in current_selected:
            # v as middle: (a, v, 2v-a) - check if 2v-a in S
            c = 2 * v - a
            if c != v and c in current_selected:
                return False
            # v as third: (a, b, v) with a + v = 2b => b = (a+v)/2
            if (a + v) % 2 == 0:
                b = (a + v) // 2
                if b != a and b != v and b in current_selected:
                    return False
            # v as first: (v, b, c) with v + c = 2b => c = 2b - v
            # For each b in S, check if 2b-v in S and 2b-v != v
            for b in current_selected:
                if b == v or b == a:
                    continue
                c = 2 * b - v
                if c != v and c in current_selected:
                    return False
        return True
    
    # This is getting complex. Let's use a cleaner B&B approach.
    # 
    # Key insight: process elements in order 1, 2, ..., n
    # State: (v, selected_set, current_weight)
    # At element v:
    #   Option 1: Include v if it doesn't conflict with selected_set
    #   Option 2: Exclude v
    # Prune if current_weight + suffix_sum[v+1] <= best_weight
    #
    # For conflict checking, when adding v to S:
    #   For each a in S:
    #     - If 2v - a in S and 2v-a != v: conflict (v is middle)
    #     - If (a+v)/2 is integer and in S and != v: conflict (v is third)  
    #     - For each b in S: if 2b - v in S and != v: conflict (v is first or b is middle)
    #
    # Let's precompute for each v, the "conflict set" more efficiently.
    
    # Precompute: for each v, conflict_mask[v] = set of elements that cannot be 
    # in S along with v (i.e., there exists w such that {v, u, w} is a 3-AP)
    # But this isn't quite right because the conflict is pairwise conditional.
    
    # Let's just use the direct check with optimization.
    
    # Use iterative deepening or a different strategy.
    # Actually, for n=256, a well-implemented B&B should work.
    
    # Let's use numpy bitsets for speed
    import numpy as np
    
    # Precompute conflict information using bitsets
    # conflict_bitset[v] = bitset where bit u is set if v and u are in some 3-AP 
    # with a third element. But this overcounts.
    
    # Better: for the B&B, when we have a partial solution S and consider adding v:
    # The conflict check is: does S ∪ {v} contain a 3-AP?
    # This is equivalent to: for the new element v, does it form a 3-AP with 
    # any two elements in S?
    # 
    # v forms a 3-AP with a, b in S iff:
    #   2v = a + b (v is middle), or
    #   2a = v + b (a is middle, v is third), or  
    #   2b = v + a (b is middle, v is third) -- same as above
    #   2v = a + b is the same as a + b = 2v
    #   v + b = 2a means b = 2a - v
    #   v + a = 2b means a = 2b - v
    #
    # So when adding v to S, check:
    #   1. For each a in S: is 2v - a in S? (v is middle, a is one end)
    #   2. For each a in S: is 2a - v in S? (a is middle, v is one end)
    #
    # Let's precompute for each v, the set of "partner" elements.
    # partners[v] = set of u such that there exists w with {v, u, w} a 3-AP
    # When adding v, for each a in S:
    #   if 2v-a in S: conflict
    #   if 2a-v in S: conflict
    
    # For efficiency, let's precompute:
    # mid_partners[v] = set of a such that 2v - a is a valid element
    #   i.e., 1 <= 2v - a <= n, 2v-a != v, 2v-a != a
    #   => a != v, a != v (since 2v-a != v means a != v), a != v (since 2v-a != a means v != a)
    #   Actually: 2v - a != v => a != v. 2v - a != a => v != a. So just a != v.
    #   And 1 <= 2v - a <= n => 2v - n <= a <= 2v - 1
    #
    # first_partners[v] = set of a such that 2a - v is a valid element
    #   i.e., 1 <= 2a - v <= n, 2a-v != v, 2a-v != a
    #   => a != v (from 2a-v != v), a != v (from 2a-v != a => v != a)
    #   And 1 <= 2a - v <= n => (v+1)/2 <= a <= (v+n)/2
    
    # Let's just implement the B&B with the direct check and see if it's fast enough.
    
    # Use a different approach: since n=256 is small, use a smarter B&B
    # with memoization or a different ordering.
    
    # Actually, let's try a simpler approach: use the greedy solution as a starting 
    # point, then do local search with 2-swap and 3-swap moves.
    
    # But the directive says to try B&B. Let me implement a proper B&B.
    
    # Key optimization: order elements by 1/x (descending) = x (ascending)
    # Use bitset for the selected set
    # Precompute for each v, the conflict check can be done in O(|S|)
    
    # For n=256, |S| can be up to ~100, so O(100) per check is fine.
    # The B&B tree can be large, but pruning should help.
    
    # Let's implement it.
    
    selected = [False] * (n + 1)
    current_weight = 0.0
    best_weight = 0.0
    best_selected = set()
    
    # Precompute 1/x as float for fast comparison
    recip = [0.0] * (n + 1)
    for i in range(1, n + 1):
        recip[i] = 1.0 / i
    
    # Suffix sum for upper bound
    suffix = [0.0] * (n + 2)
    for i in range(n, 0, -1):
        suffix[i] = suffix[i + 1] + recip[i]
    
    def _check_conflict(v, sel):
        """Check if adding v to sel creates a 3-AP. sel is a list of selected elements."""
        for a in sel:
            # v as middle: 2v = a + b => b = 2v - a
            b = 2 * v - a
            if b != v and b in sel_set:
                return True
            # a as middle: 2a = v + b => b = 2a - v
            b = 2 * a - v
            if b != v and b in sel_set:
                return True
        return False
    
    sel_set = set()
    
    def _bbs(v, sel, weight):
        """B&B: process elements from v to n."""
        nonlocal best_weight, best_selected, sel_set
        
        if time.time() > deadline:
            return
        
        # Upper bound
        if weight + suffix[v] <= best_weight:
            return
        
        if v > n:
            if weight > best_weight:
                best_weight = weight
                best_selected = set(sel)
            return
        
        # Try excluding v
        _bbs(v + 1, sel, weight)
        
        # Try including v if it doesn't conflict
        if not _check_conflict(v, sel):
            sel.append(v)
            sel_set.add(v)
            _bbs(v + 1, sel, weight + recip[v])
            sel.pop()
            sel_set.discard(v)
    
    # Start B&B
    _bbs(1, [], 0.0)
    
    if best_selected:
        return list(best_selected)
    else:
        return _greedy_initial(n)

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s limit
    
    # First get a good initial solution via greedy
    result = _greedy_initial(256)
    
    # Try branch-and-bound
    try:
        bb_result = _branch_and_bound(256, deadline)
        if bb_result and len(bb_result) > len(result):
            result = bb_result
        elif bb_result and len(bb_result) == len(result):
            # Compare weights
            if _reciprocal_sum(bb_result) > _reciprocal_sum(result):
                result = bb_result
    except Exception:
        pass
    
    # Final verification
    if not _is_valid_3ap_free(result):
        # Fix by removing conflicting elements
        result_set = set(result)
        changed = True
        while changed:
            changed = False
            for a in list(result_set):
                for b in list(result_set):
                    if a == b:
                        continue
                    c = 2 * b - a
                    if c in result_set and c != a and c != b:
                        to_remove = max(a, b, c)
                        result_set.discard(to_remove)
                        changed = True
        result = list(result_set)
    
    return result
# EVOLVE-BLOCK-END
