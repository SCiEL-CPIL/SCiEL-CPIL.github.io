"""Rewritten program: branch-and-bound maximum independent set with bitset pruning."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _precompute_conflicts(n=256):
    """Precompute conflict bitmasks. conflict_mask[x] is a Python int whose bit j is set
    iff x and j+1 conflict (i.e., (x*(j+1)) % (x+(j+1)) == 0)."""
    conflict_mask = [0] * (n + 1)
    for a in range(1, n + 1):
        mask = 0
        for b in range(1, n + 1):
            if a != b and (a * b) % (a + b) == 0:
                mask |= (1 << (b - 1))
        conflict_mask[a] = mask
    return conflict_mask

def _branch_and_bound(conflict_mask, n, deadline):
    """Branch-and-bound for maximum independent set.
    
    Uses bitset representation: a Python int where bit i (0-indexed) means
    vertex (i+1) is available.
    
    Pruning:
    - Upper bound = popcount(candidates)
    - If best >= upper bound, prune
    - Choose vertex with max degree in candidate subgraph for branching
    """
    full_mask = (1 << n) - 1
    
    best_size = [0]
    best_set = [set()]
    
    # Precompute popcount for all masks? No, use int.bit_count()
    
    def popcount(x):
        return x.bit_count()
    
    def _branch(candidates, current_set, current_size, depth=0):
        """Recursive branch-and-bound."""
        if time.time() > deadline:
            return
        
        # Upper bound: can't do better than |candidates|
        upper = popcount(candidates)
        if current_size + upper <= best_size[0]:
            return
        
        if candidates == 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_set[0] = set(current_set)
            return
        
        # Choose branching vertex: one with max conflicts within candidates
        # Try a few heuristics and pick the one that gives best pruning
        # For speed, pick the vertex with max degree in the induced subgraph
        
        # Extract vertex indices from candidates
        # Find the vertex with max conflicts within candidates
        best_vertex = -1
        best_degree = -1
        temp = candidates
        while temp:
            # Get lowest set bit
            lb = temp & (-temp)
            idx = lb.bit_length() - 1  # 0-indexed
            v = idx + 1  # 1-indexed
            # Degree of v within candidates
            deg = popcount(candidates & conflict_mask[v])
            if deg > best_degree:
                best_degree = deg
                best_vertex = v
            temp ^= lb
        
        if best_vertex == -1:
            return
        
        # Branch 1: include best_vertex
        new_candidates = candidates & ~conflict_mask[best_vertex]
        new_candidates &= ~(1 << (best_vertex - 1))  # remove best_vertex itself
        current_set.add(best_vertex)
        _branch(new_candidates, current_set, current_size + 1, depth + 1)
        current_set.discard(best_vertex)
        
        # Branch 2: exclude best_vertex
        new_candidates = candidates & ~(1 << (best_vertex - 1))
        _branch(new_candidates, current_set, current_size, depth + 1)
    
    _branch(full_mask, set(), 0)
    return best_set[0]

def _greedy_max_independent(conflict_mask, n, order=None):
    """Greedy: add vertices in given order if no conflict with already selected."""
    if order is None:
        order = list(range(1, n + 1))
    selected = set()
    for x in order:
        ok = True
        for y in selected:
            if conflict_mask[x] & (1 << (y - 1)):
                ok = False
                break
        if ok:
            selected.add(x)
    return selected

def _local_search(conflict_mask, n, selected, deadline, max_iters=10000):
    """Local search: try to add vertices, or do 1-for-1 swaps."""
    selected = set(selected)
    best = set(selected)
    best_size = len(selected)
    
    universe = list(range(1, n + 1))
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try to add any unselected vertex with no conflicts
        rng = random.Random(it)
        unselected = [x for x in universe if x not in selected]
        rng.shuffle(unselected)
        for x in unselected:
            # Check if x conflicts with any selected
            ok = True
            for y in selected:
                if conflict_mask[x] & (1 << (y - 1)):
                    ok = False
                    break
            if ok:
                selected.add(x)
                improved = True
                break
        
        if not improved:
            # Try 1-for-1 swap
            sel_list = list(selected)
            rng.shuffle(sel_list)
            for rem in sel_list:
                selected.discard(rem)
                # Try to add any unselected
                unselected = [x for x in universe if x not in selected]
                rng.shuffle(unselected)
                added = False
                for x in unselected:
                    ok = True
                    for y in selected:
                        if conflict_mask[x] & (1 << (y - 1)):
                            ok = False
                            break
                    if ok:
                        selected.add(x)
                        added = True
                        break
                if added:
                    improved = True
                    break
                else:
                    selected.add(rem)  # undo
        
        if not improved:
            # Try remove-2-add-many
            if len(selected) > 4:
                sel_list = list(selected)
                rng.shuffle(sel_list)
                for i in range(min(len(sel_list), 15)):
                    rem1 = sel_list[i]
                    for j in range(i + 1, min(len(sel_list), i + 8)):
                        rem2 = sel_list[j]
                        saved = set(selected)
                        selected.discard(rem1)
                        selected.discard(rem2)
                        # Greedily add
                        unselected = [x for x in universe if x not in selected]
                        rng.shuffle(unselected)
                        for x in unselected:
                            ok = True
                            for y in selected:
                                if conflict_mask[x] & (1 << (y - 1)):
                                    ok = False
                                    break
                            if ok:
                                selected.add(x)
                        if len(selected) > len(saved):
                            improved = True
                            break
                        else:
                            selected = saved
                    if improved:
                        break
        
        if not improved:
            break
        
        if len(selected) > best_size:
            best_size = len(selected)
            best = set(selected)
    
    return best

def construct():
    start_time = time.time()
    # Safety margin: return well before 90s kill
    deadline = start_time + 82
    
    n = 256
    conflict_mask = _precompute_conflicts(n)
    
    # Phase 1: Multiple greedy initializations with different orderings
    best = set()
    best_size = 0
    
    # Natural order
    sel = _greedy_max_independent(conflict_mask, n, list(range(1, n + 1)))
    if len(sel) > best_size:
        best_size = len(sel)
        best = sel
    
    # Reverse natural
    sel = _greedy_max_independent(conflict_mask, n, list(range(n, 0, -1)))
    if len(sel) > best_size:
        best_size = len(sel)
        best = sel
    
    # Degree-sorted (fewest conflicts first)
    degrees = [conflict_mask[i].bit_count() for i in range(n + 1)]
    degree_order = sorted(range(1, n + 1), key=lambda x: (degrees[x], x))
    sel = _greedy_max_independent(conflict_mask, n, degree_order)
    if len(sel) > best_size:
        best_size = len(sel)
        best = sel
    
    # Reverse degree-sorted
    rev_degree_order = sorted(range(1, n + 1), key=lambda x: (-degrees[x], x))
    sel = _greedy_max_independent(conflict_mask, n, rev_degree_order)
    if len(sel) > best_size:
        best_size = len(sel)
        best = sel
    
    # Random orderings
    rng = random.Random(12345)
    for trial in range(200):
        if time.time() > deadline - 15:
            break
        order = list(range(1, n + 1))
        rng.shuffle(order)
        sel = _greedy_max_independent(conflict_mask, n, order)
        if len(sel) > best_size:
            best_size = len(sel)
            best = sel
    
    # Phase 2: Local search from best
    if time.time() < deadline - 10:
        improved = _local_search(conflict_mask, n, best, deadline - 5, max_iters=20000)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Phase 3: Branch-and-bound on a reduced problem
    # Use the best solution found so far as a lower bound
    # Try B&B on the full problem with the lower bound
    if time.time() < deadline - 5:
        bb_deadline = deadline - 3
        bb_result = _branch_and_bound(conflict_mask, n, bb_deadline)
        if len(bb_result) > best_size:
            best = bb_result
            best_size = len(best)
    
    # Phase 4: More local search if time permits
    while time.time() < deadline - 5:
        improved = _local_search(conflict_mask, n, best, deadline - 3, max_iters=10000)
        if len(improved) <= best_size:
            break
        best = improved
        best_size = len(best)
    
    return sorted(best)

# EVOLVE-BLOCK-END
