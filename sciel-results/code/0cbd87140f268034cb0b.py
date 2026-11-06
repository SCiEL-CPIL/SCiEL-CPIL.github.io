import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    N = 180
    
    # Precompute conflicts
    triples = []
    
    for a in range(1, N + 1):
        a2 = a * a
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
                
    # Build conflict map
    conflicts = {i: [] for i in range(1, N + 1)}
    for a, b, c in triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))
        
    def can_add(x, S):
        for u, v in conflicts[x]:
            if u in S and v in S:
                return False
        return True

    def local_search(S, time_limit):
        S = set(S)
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            candidates = [x for x in range(1, N + 1) if x not in S]
            random.shuffle(candidates)
            for x in candidates:
                if time.time() >= time_limit:
                    break
                if can_add(x, S):
                    S.add(x)
                    changed = True
        return S

    best_set = set()
    
    # Try MILP with optimized formulation
    milp_success = False
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        c_obj = -np.ones(N)
        
        # Build triple constraints efficiently using sparse representation
        # For each triple (a,b,c): x_a + x_b + x_c <= 2
        row_indices = []
        col_indices = []
        row_values = []
        constraint_ubs = []
        
        for (a, b, c) in triples:
            row_idx = len(constraint_ubs)
            for var in [a-1, b-1, c-1]:
                row_indices.append(row_idx)
                col_indices.append(var)
                row_values.append(1.0)
            constraint_ubs.append(2.0)
            
        # Add pair-based strengthening constraints
        # If two elements appear together in many triples, they're likely to conflict
        # For pairs that appear in >= 2 triples: x_u + x_v <= 1 (if they can't both be in any valid set with a third)
        # Actually, this is too strong. Instead, let's add constraints for "almost-cliques"
        
        # Better: For each element x, if it forms triples with pairs (u1,v1), (u2,v2), etc.
        # Then we can add: x + u_i + v_i <= 2 for each such triple (already done)
        
        # Let's add constraints for small cliques more efficiently
        # Build a set of all triples for O(1) lookup
        triple_set = set()
        for a, b, c in triples:
            triple_set.add((min(a,b,c), min(max(a,b),max(c)), max(a,b,c)))
        
        # Find 4-cliques: sets of 4 where all 4 choose 3 = 4 triples exist
        # For efficiency, only check promising pairs
        pair_in_triple_count = {}
        for a, b, c in triples:
            for p in [(a,b), (a,c), (b,c)]:
                ps = (min(p), max(p))
                pair_in_triple_count[ps] = pair_in_triple_count.get(ps, 0) + 1
        
        # Only consider pairs that appear in at least 2 triples
        candidate_pairs = [p for p, cnt in pair_in_triple_count.items() if cnt >= 2]
        
        clique_constraints_added = set()
        for (u, v) in candidate_pairs:
            if time.time() > deadline - 30:
                break
            # Find w such that (u,v,w) is a triple
            w_candidates = []
            for w in range(1, N+1):
                if w == u or w == v:
                    continue
                t = (min(u,v,w), min(max(u,v),max(w)), max(u,v,w))
                if t in triple_set:
                    w_candidates.append(w)
            
            # Try to form 4-cliques
            for i in range(len(w_candidates)):
                for j in range(i+1, len(w_candidates)):
                    w1, w2 = w_candidates[i], w_candidates[j]
                    # Check all 4 triples
                    clique = True
                    for triple_combo in [(u,v,w1), (u,v,w2), (u,w1,w2), (v,w1,w2)]:
                        t = (min(triple_combo), min(max(triple_combo[0],triple_combo[1]),max(triple_combo[1],triple_combo[2])), max(triple_combo))
                        if t not in triple_set:
                            clique = False
                            break
                    if clique:
                        key = tuple(sorted([u, v, w1, w2]))
                        if key not in clique_constraints_added:
                            clique_constraints_added.add(key)
                            row_idx = len(constraint_ubs)
                            for var in [u-1, v-1, w1-1, w2-1]:
                                row_indices.append(row_idx)
                                col_indices.append(var)
                                row_values.append(1.0)
                            constraint_ubs.append(2.0)
        
        # Build sparse matrix
        from scipy.sparse import csr_matrix
        A_ub = csr_matrix((row_values, (row_indices, col_indices)), shape=(len(constraint_ubs), N))
        b_ub = np.array(constraint_ubs)
        
        bounds = Bounds(lb=np.zeros(N), ub=np.ones(N))
        integrality = np.ones(N)
        
        remaining_for_milp = max(1, deadline - time.time() - 5)
        
        res = milp(
            c=c_obj,
            constraints=LinearConstraint(A_ub, ub=b_ub),
            bounds=bounds,
            integrality=integrality,
            options={"time_limit": remaining_for_milp}
        )
        
        if res.status in [0, 2]:
            S = set()
            for i in range(N):
                if res.x[i] > 0.5:
                    S.add(i+1)
            
            valid = True
            for a, b, c in triples:
                if a in S and b in S and c in S:
                    valid = False
                    break
            
            if valid and len(S) > 0:
                best_set = S
                milp_success = True
                if res.status == 0:
                    return sorted(S)
                
    except Exception:
        pass
        
    # Fallback: heuristic search
    remaining_time = deadline - time.time()
    if remaining_time < 1:
        S = set()
        for a in range(N, 0, -1):
            if can_add(a, S):
                S.add(a)
        if S:
            return sorted(S)
        return [1]

    S_base = set()
    for a in range(1, N + 1):
        if can_add(a, S_base):
            S_base.add(a)
    S_base = local_search(S_base, deadline - 1)
    best_set = S_base
    best_size = len(S_base)
    
    while time.time() < deadline - 3:
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S_curr = set()
        for a in order:
            if time.time() >= deadline - 3:
                break
            if can_add(a, S_curr):
                S_curr.add(a)
                
        remaining = deadline - 3 - time.time()
        if remaining > 0.5:
            S_curr = local_search(S_curr, deadline - 3)
        
        if len(S_curr) > best_size:
            best_size = len(S_curr)
            best_set = S_curr
            
    return sorted(best_set)

