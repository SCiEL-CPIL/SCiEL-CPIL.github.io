"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256
# EVOLVE-BLOCK-START

def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal_sidon(selected, N):
    """Check if no missing element can be added without breaking Sidon property."""
    sel_set = set(selected)
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            s = a + selected[j]
            sums.add(s)
    
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if 2 * x in sums:
            continue
        blocked = False
        for a in selected:
            if x + a in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _precompute_hyperedges(N):
    """
    Precompute the sum-collision hypergraph.
    For each sum s in [2, 2N], the hyperedge is the set of pairs (a,b) with a<=b, a+b=s.
    Two elements x,y "conflict on sum s" if there exist a,b in selected such that a+b=s and one of x,y would create a duplicate.
    
    Actually, let's think differently:
    A Sidon set S is invalid if there exist distinct unordered pairs {a,b} and {c,d} (with a<=b, c<=d, and {a,b} != {c,d}) such that a+b = c+d.
    
    For a given sum s, the "collision hyperedge" consists of all pairs (a,b) with a<=b and a+b=s.
    If we select two or more pairs from the same hyperedge, we get a collision.
    
    For the vertex cover formulation:
    Vertices = elements [1..N]
    For each sum s, let P_s = {(a,b) : a<=b, a+b=s, 1<=a,b<=N}
    We need to select at most one pair from each P_s.
    
    This is equivalent to: for each s, the selected elements must form a matching in the "pair graph" for sum s, and since each element can appear in at most one pair per sum, it's actually:
    For each s, at most one pair from P_s is fully selected.
    
    A simpler view: x is "blocked" by selected set S if adding x creates a duplicate sum.
    x is blocked if:
    - 2x = a+b for some a,b in S (diagonal collision)
    - x+a = b+c for some a,b,c in S with {x,a} != {b,c}
    
    Let's use the hyperedge approach for the greedy:
    For each candidate x, compute how many "potential collisions" it would create if added.
    But we want to pick x that BLOCKS the most others.
    
    Alternative: think of it as each element x has a set of "sums it participates in":
    - 2x (diagonal)
    - x+a for each a in selected
    
    When we add x, it "covers" (blocks) any y such that:
    - 2y = 2x (i.e., y=x, trivial)
    - 2y = x+a for some a in selected (y = (x+a)/2)
    - y+b = 2x for some b in selected (y = 2x-b)
    - y+b = x+a for some a,b in selected (y = x+a-b)
    
    So the blocking set of x given selected S is:
    B(x, S) = {y : 2y = 2x or 2y = x+a or y+b = 2x or y+b = x+a for a,b in S}
    
    Simplified:
    B(x, S) = {x} ∪ {(x+a)/2 : a in S, (x+a) even} ∪ {2x-b : b in S} ∪ {x+a-b : a,b in S}
    
    We want to pick x that maximizes |B(x, S) ∩ Unblocked|.
    """
    pass

def _greedy_blocking_maximal(rng, N):
    """
    Generate a maximal Sidon set using hypergraph-inspired greedy.
    At each step, pick the unblocked element that blocks the most other unblocked elements.
    """
    selected = []
    sel_set = set()
    
    while True:
        # Compute blocked set
        sums = set()
        for a in selected:
            for b in selected:
                if a <= b:
                    sums.add(a + b)
        
        blocked = set()
        for s in sums:
            # 2y = s => y = s/2
            if s % 2 == 0:
                y = s // 2
                if 1 <= y <= N:
                    blocked.add(y)
            # y + a = s => y = s - a for each a in selected
            for a in selected:
                y = s - a
                if 1 <= y <= N:
                    blocked.add(y)
        
        unblocked = [x for x in range(1, N + 1) if x not in sel_set and x not in blocked]
        
        if not unblocked:
            break
        
        # For each unblocked x, compute how many unblocked elements it would block
        best_x = None
        best_count = -1
        
        for x in unblocked:
            # New sums if we add x
            new_sums = set(sums)
            new_sums.add(2 * x)
            for a in selected:
                new_sums.add(x + a)
            
            # Count how many unblocked y (y != x) become blocked
            count = 0
            for y in unblocked:
                if y == x:
                    continue
                # Check if y is blocked by new_sums
                is_blocked = False
                if 2 * y in new_sums:
                    is_blocked = True
                else:
                    for a in selected + [x]:
                        if y + a in new_sums:
                            is_blocked = True
                            break
                if is_blocked:
                    count += 1
            
            if count > best_count:
                best_count = count
                best_x = x
        
        if best_x is None:
            break
        
        selected.append(best_x)
        sel_set.add(best_x)
    
    return selected

def _shrink_maximal(selected, N, deadline):
    """Try to remove elements while maintaining maximality."""
    selected = list(selected)
    
    while time.time() < deadline:
        improved = False
        indices = list(range(len(selected)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            x = selected[idx]
            candidate = selected[:idx] + selected[idx+1:]
            
            if _is_maximal_sidon(candidate, N):
                selected = candidate
                improved = True
                break
        
        if not improved:
            break
            
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best = None
    best_size = float('inf')
    
    num_restarts = 1000
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(i * 12345 + 67890)
        
        # Generate initial maximal set using blocking heuristic
        sel = _greedy_blocking_maximal(rng, N)
        
        if not sel:
            continue
            
        # Shrink it
        sel = _shrink_maximal(sel, N, deadline)
        
        if len(sel) < best_size:
            best = sel
            best_size = len(sel)
            
            if best_size <= 16:
                break

    if best is None:
        # Fallback
        sel = _greedy_blocking_maximal(random.Random(0), N)
        return sel
        
    return best

# EVOLVE-BLOCK-END
