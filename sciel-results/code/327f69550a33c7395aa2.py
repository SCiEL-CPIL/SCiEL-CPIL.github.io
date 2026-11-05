import time
from math import gcd, isqrt
from itertools import combinations

N = 512
DEADLINE = 85.0  # Safety margin before 90s kill

def _lcm(a, b):
    return (a // gcd(a, b)) * b

def _build_conflict_bitsets():
    """Build bitsets where bit i is set if number i conflicts with current number."""
    # Numbers are 1..N, use 0-indexed bits (0 for number 1, 1 for number 2, etc.)
    # conflict[i] is a bitset of all j where lcm(i+1, j+1) > N
    conflict = [0] * (N + 1)  # 1-indexed: conflict[i] for number i
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i != j and _lcm(i, j) > N:
                mask |= (1 << j)
        conflict[i] = mask
    return conflict

def _greedy_color_upper_bound(vertices, conflict):
    """Greedy coloring to get upper bound on independent set size."""
    # vertices: set of node indices (1-indexed)
    # Returns upper bound on max independent set in induced subgraph
    if not vertices:
        return 0
    
    # Order vertices by degree in the induced subgraph (descending)
    v_list = list(vertices)
    degrees = []
    for v in v_list:
        # Count neighbors in vertices
        c = 0
        mask = conflict[v]
        for u in v_list:
            if u != v and (mask >> u) & 1:
                c += 1
        degrees.append((c, v))
    degrees.sort(reverse=True)
    
    # Greedy coloring
    color = {}
    num_colors = 0
    for _, v in degrees:
        # Find smallest available color
        used_colors = set()
        mask = conflict[v]
        for u in v_list:
            if u != v and (mask >> u) & 1 and u in color:
                used_colors.add(color[u])
        c = 1
        while c in used_colors:
            c += 1
        color[v] = c
        num_colors = max(num_colors, c)
    
    return num_colors

def _max_independent_set_bnb(vertices, conflict, deadline, current_best):
    """Branch and bound for maximum independent set.
    vertices: set of 1-indexed node indices
    Returns (best_set, best_size)
    """
    if not vertices:
        return set(), 0
    
    v_list = list(vertices)
    if len(v_list) <= 1:
        return set(v_list), len(v_list)
    
    best_set = set()
    best_size = 0
    
    # Use recursion with pruning
    def bnb(candidates, current):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if len(current) > best_size:
                best_size = len(current)
                best_set = set(current)
            return
        
        # Upper bound: |current| + greedy color bound on candidates
        ub = len(current) + _greedy_color_upper_bound(candidates, conflict)
        if ub <= best_size:
            return
        
        # Pick vertex with highest degree in candidates
        v_list_c = list(candidates)
        best_v = None
        best_deg = -1
        for v in v_list_c:
            mask = conflict[v]
            deg = 0
            for u in v_list_c:
                if u != v and (mask >> u) & 1:
                    deg += 1
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        if best_v is None:
            if len(current) > best_size:
                best_size = len(current)
                best_set = set(current)
            return
        
        # Branch 1: Include best_v
        new_current = current | {best_v}
        # Remove best_v and its neighbors from candidates
        mask = conflict[best_v]
        new_candidates = set()
        for u in candidates:
            if u != best_v and not (mask >> u) & 1:
                new_candidates.add(u)
        bnb(new_candidates, new_current)
        
        # Branch 2: Exclude best_v
        new_candidates2 = candidates - {best_v}
        bnb(new_candidates2, current)
    
    bnb(vertices, set())
    return best_set, best_size

def _get_divisors(n):
    divs = []
    for i in range(1, isqrt(n) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return divs

def _greedy_fill(S, conflict):
    """Greedily add compatible elements to S."""
    S = set(S)
    changed = True
    while changed:
        changed = False
        for x in range(1, N + 1):
            if x not in S:
                # Check if x is compatible with all in S
                mask = conflict[x]
                compatible = True
                for y in S:
                    if (mask >> y) & 1:
                        compatible = False
                        break
                if compatible:
                    S.add(x)
                    changed = True
    return S

def _local_search_fast(S, conflict, deadline):
    """Fast local search using bitset operations."""
    S = set(S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try remove 1, add 2
        for y in list(S):
            if time.time() > deadline:
                break
            S_temp = S - {y}
            
            # Find candidates compatible with S_temp
            candidates = []
            for x in range(1, N + 1):
                if x not in S_temp:
                    mask = conflict[x]
                    compatible = True
                    for z in S_temp:
                        if (mask >> z) & 1:
                            compatible = False
                            break
                    if compatible:
                        candidates.append(x)
            
            # Find pair x1, x2 in candidates such that they are compatible
            found_pair = False
            for i in range(len(candidates)):
                if time.time() > deadline:
                    break
                x1 = candidates[i]
                S_with_x1 = S_temp | {x1}
                for j in range(i + 1, len(candidates)):
                    x2 = candidates[j]
                    mask2 = conflict[x2]
                    compatible = True
                    for z in S_with_x1:
                        if (mask2 >> z) & 1:
                            compatible = False
                            break
                    if compatible:
                        new_S = S_with_x1 | {x2}
                        if len(new_S) > len(S):
                            S = new_S
                            improved = True
                            found_pair = True
                            break
                if found_pair:
                    break
            if found_pair:
                break  # Restart loop with new S

        if improved:
            continue

        # Try remove 2, add 3
        if time.time() < deadline - 10:
            found_triple = False
            for pair in combinations(list(S), 2):
                if time.time() > deadline:
                    break
                y1, y2 = pair
                S_temp = S - {y1, y2}
                
                # Find candidates compatible with S_temp
                candidates = []
                for x in range(1, N + 1):
                    if x not in S_temp:
                        mask = conflict[x]
                        compatible = True
                        for z in S_temp:
                            if (mask >> z) & 1:
                                compatible = False
                                break
                        if compatible:
                            candidates.append(x)
                
                # Try to add 3 greedily from candidates
                S_new = S_temp.copy()
                added = 0
                for x in candidates:
                    mask = conflict[x]
                    compatible = True
                    for z in S_new:
                        if (mask >> z) & 1:
                            compatible = False
                            break
                    if compatible:
                        S_new.add(x)
                        added += 1
                        if added == 3:
                            break
                
                if len(S_new) > len(S):
                    S = S_new
                    improved = True
                    found_triple = True
                    break
            
            if found_triple:
                continue

    return S

def construct():
    start_time = time.time()
    
    # Build conflict bitsets
    conflict = _build_conflict_bitsets()
    
    # Find highly composite numbers <= 512
    best_M = 1
    best_tau = 0
    for M in range(1, N + 1):
        tau = len(_get_divisors(M))
        if tau > best_tau:
            best_tau = tau
            best_M = M
    
    # Try multiple seeds: divisors of top highly composite numbers
    # Sort by number of divisors
    M_list = []
    for M in range(1, N + 1):
        tau = len(_get_divisors(M))
        M_list.append((tau, M))
    M_list.sort(reverse=True)
    
    best_S = set()
    best_size = 0
    
    # Try top 5 highly composite numbers as seeds
    for tau, M in M_list[:5]:
        if time.time() > DEADLINE:
            break
        seed_S = set(_get_divisors(M))
        S = _greedy_fill(seed_S, conflict)
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
    
    # Local search
    best_S = _local_search_fast(best_S, conflict, DEADLINE - 5)
    best_size = len(best_S)
    
    # If time permits, try branch-and-bound on the complement
    # (find max independent set in conflict graph)
    if time.time() < DEADLINE - 20:
        all_vertices = set(range(1, N + 1))
        # Use B&B with current best as lower bound
        bnb_deadline = min(time.time() + 15, DEADLINE - 5)
        bnb_set, bnb_size = _max_independent_set_bnb(all_vertices, conflict, bnb_deadline, best_size)
        if bnb_size > best_size:
            best_S = bnb_set
            best_size = bnb_size
    
    # Final local search polish
    if time.time() < DEADLINE - 2:
        best_S = _local_search_fast(best_S, conflict, DEADLINE - 1)
        best_size = len(best_S)
    
    return list(best_S)
