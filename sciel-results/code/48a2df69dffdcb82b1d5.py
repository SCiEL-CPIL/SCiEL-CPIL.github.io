import time
import numpy as np
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    N = 180
    
    # Precompute conflicts and triples
    # triples contains (a, b, c) such that 1/a = 1/b + 1/c with a < b < c
    triples = []
    
    for a in range(1, N + 1):
        a2 = a * a
        # Find factor pairs d, e of a^2 such that d < e
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d != 0:
                continue
            e = a2 // d
            if d >= e:
                continue
            b = a + d
            c = a + e
            if b <= N and c <= N:
                triples.append((a, b, c))
                
    # Precompute for each vertex, the list of triples it participates in
    vertex_triples = defaultdict(list)
    for idx, (a, b, c) in enumerate(triples):
        vertex_triples[a].append(idx)
        vertex_triples[b].append(idx)
        vertex_triples[c].append(idx)
        
    # Build adjacency structure for spectral clustering
    # Two vertices are adjacent if they appear together in some triple
    adj = defaultdict(set)
    for a, b, c in triples:
        adj[a].add(b)
        adj[a].add(c)
        adj[b].add(a)
        adj[b].add(c)
        adj[c].add(a)
        adj[c].add(b)
    
    # Find connected components in the conflict graph
    visited = set()
    components = []
    
    for v in range(1, N + 1):
        if v in visited:
            continue
        # BFS to find component
        comp = []
        queue = deque([v])
        visited.add(v)
        while queue:
            u = queue.popleft()
            comp.append(u)
            for w in adj[u]:
                if w not in visited:
                    visited.add(w)
                    queue.append(w)
        components.append(comp)
    
    # Sort components by size (larger first)
    components.sort(key=len, reverse=True)
    
    # For each component, try to find maximum independent set
    # Use different strategies based on component size
    S = set()
    
    for comp in components:
        if time.time() > deadline - 10:
            break
            
        comp_set = set(comp)
        comp_size = len(comp)
        
        # Get all triples that are entirely within this component
        comp_triples = []
        for idx, (a, b, c) in enumerate(triples):
            if a in comp_set and b in comp_set and c in comp_set:
                comp_triples.append((a, b, c))
        
        if not comp_triples:
            # No conflicts within this component, include all
            S.update(comp)
            continue
        
        # For small components (<= 15), use exhaustive search
        if comp_size <= 15:
            best_subset = _max_independent_set_small(comp, comp_triples)
            S.update(best_subset)
            continue
        
        # For medium components (16-30), use greedy with local search
        if comp_size <= 30:
            best_subset = _greedy_with_polish(comp, comp_triples, vertex_triples, triples)
            S.update(best_subset)
            continue
        
        # For large components, use MILP or greedy fallback
        if time.time() > deadline - 5:
            # Not enough time, use quick greedy
            subset = _quick_greedy(comp, vertex_triples, triples)
            S.update(subset)
            continue
        
        # Try MILP for this component
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            
            comp_indices = sorted(comp)
            index_map = {v: i for i, v in enumerate(comp_indices)}
            m = len(comp_indices)
            
            c_obj = -np.ones(m)
            A_ub = np.zeros((len(comp_triples), m))
            b_ub = np.full(len(comp_triples), 2)
            
            for idx, (a, b, c) in enumerate(comp_triples):
                A_ub[idx, index_map[a]] = 1
                A_ub[idx, index_map[b]] = 1
                A_ub[idx, index_map[c]] = 1
            
            bounds = Bounds(lb=np.zeros(m), ub=np.ones(m))
            integrality = np.ones(m)
            
            res = milp(
                c=c_obj,
                constraints=LinearConstraint(A_ub, ub=b_ub),
                bounds=bounds,
                integrality=integrality
            )
            
            if res.status == 0:
                subset = set()
                for i in range(m):
                    if res.x[i] > 0.5:
                        subset.add(comp_indices[i])
                
                # Verify
                valid = True
                for a, b, c in comp_triples:
                    if a in subset and b in subset and c in subset:
                        valid = False
                        break
                
                if valid:
                    # Polish
                    subset = _polish_subset(subset, comp_set, vertex_triples, triples, deadline)
                    S.update(subset)
                    continue
            
        except Exception:
            pass
        
        # Fallback: greedy
        subset = _quick_greedy(comp, vertex_triples, triples)
        subset = _polish_subset(subset, comp_set, vertex_triples, triples, deadline)
        S.update(subset)
    
    # Final global polish
    S = _polish_subset(S, set(range(1, N + 1)), vertex_triples, triples, deadline)
    
    return sorted(S)

def _max_independent_set_small(vertices, triples):
    """Exhaustive search for maximum independent set in small component."""
    n = len(vertices)
    vlist = list(vertices)
    
    # Map to indices
    vmap = {v: i for i, v in enumerate(vlist)}
    
    # Precompute which triples conflict with which vertices
    # For each vertex, store set of triple indices it's in
    v_in_triples = {v: set() for v in vertices}
    for idx, (a, b, c) in enumerate(triples):
        v_in_triples[a].add(idx)
        v_in_triples[b].add(idx)
        v_in_triples[c].add(idx)
    
    best_size = 0
    best_mask = 0
    
    # Use bitmasks for efficiency
    full_mask = (1 << n) - 1
    
    def dfs(mask, current_size):
        nonlocal best_size, best_mask
        
        if current_size > best_size:
            best_size = current_size
            best_mask = mask
        
        # Find first available vertex
        avail = mask & full_mask
        if avail == 0:
            return
        
        # Try adding each available vertex
        while avail:
            # Pick lowest set bit
            lsb = avail & (-avail)
            v_idx = lsb.bit_length() - 1
            v = vlist[v_idx]
            
            # Check if we can add v
            can_add = True
            for tidx in v_in_triples[v]:
                a, b, c = triples[tidx]
                # Check if other two are already in selected set
                others = [x for x in (a, b, c) if x != v]
                if others[0] in selected_set and others[1] in selected_set:
                    can_add = False
                    break
            
            if can_add:
                selected_set.add(v)
                # Remove v and all vertices that conflict with v
                new_mask = mask & ~(1 << v_idx)
                for tidx in v_in_triples[v]:
                    a, b, c = triples[tidx]
                    for other in (a, b, c):
                        if other != v:
                            other_idx = vmap[other]
                            new_mask &= ~(1 << other_idx)
                
                dfs(new_mask, current_size + 1)
                selected_set.remove(v)
            
            avail &= avail - 1
    
    selected_set = set()
    dfs(full_mask, 0)
    
    # Reconstruct best subset
    result = set()
    for i in range(n):
        if best_mask & (1 << i):
            result.add(vlist[i])
    
    return result

def _greedy_with_polish(vertices, triples, vertex_triples, all_triples):
    """Greedy with local search for medium components."""
    vlist = sorted(vertices, reverse=True)
    S = set()
    
    for v in vlist:
        can_add = True
        for tidx in vertex_triples[v]:
            a, b, c = all_triples[tidx]
            others = [x for x in (a, b, c) if x != v]
            if others[0] in S and others[1] in S:
                can_add = False
                break
        
        if can_add:
            S.add(v)
    
    return _polish_subset(S, set(vertices), vertex_triples, all_triples, time.time() + 10)

def _quick_greedy(vertices, vertex_triples, triples):
    """Quick greedy construction."""
    S = set()
    for v in sorted(vertices, reverse=True):
        can_add = True
        for tidx in vertex_triples[v]:
            a, b, c = triples[tidx]
            others = [x for x in (a, b, c) if x != v]
            if others[0] in S and others[1] in S:
                can_add = False
                break
        
        if can_add:
            S.add(v)
    
    return S

def _polish_subset(S, available, vertex_triples, triples, deadline):
    """Try to add more elements to S."""
    S = set(S)
    
    # Try adding each element not in S
    for v in sorted(available - S, reverse=True):
        if time.time() > deadline - 2:
            break
        
        can_add = True
        for tidx in vertex_triples[v]:
            a, b, c = triples[tidx]
            others = [x for x in (a, b, c) if x != v]
            if others[0] in S and others[1] in S:
                can_add = False
                break
        
        if can_add:
            S.add(v)
    
    # Try removing one and adding two (swap improvement)
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for remove_v in list(S):
            if time.time() > deadline - 3:
                break
            
            S.remove(remove_v)
            added = 0
            for add_v in sorted(available - S, reverse=True):
                can_add = True
                for tidx in vertex_triples[add_v]:
                    a, b, c = triples[tidx]
                    others = [x for x in (a, b, c) if x != add_v]
                    if others[0] in S and others[1] in S:
                        can_add = False
                        break
                
                if can_add:
                    S.add(add_v)
                    added += 1
                    if added >= 2:
                        break
            
            if added < 1:
                S.add(remove_v)
            elif added == 1:
                # Keep the improvement (removed 1, added 1)
                improved = True
            # If added >= 2, net gain
    
    return S
