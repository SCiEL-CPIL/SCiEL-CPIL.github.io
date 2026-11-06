import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    N = 180
    
    # Precompute conflicts
    # triples contains (a, b, c) such that 1/a = 1/b + 1/c with a < b < c
    # We need to know for each element x, which pairs (u, v) form a triple with x.
    # Let's build a list of all triples first.
    triples = []
    
    for a in range(1, N + 1):
        a2 = a * a
        # Find factor pairs d, e of a^2 such that d < e
        # b = a + d, c = a + e
        # We need b, c <= N
        # Also b != c, which is guaranteed by d != e
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d != 0:
                continue
            e = a2 // d
            if d >= e:
                continue
            b = a + d
            c = a + e
            if b <= N and c <= N:
                # Ensure distinctness: a, b, c are distinct because d>0 => b>a, e>d => c>b?
                # c - b = e - d > 0. So a < b < c.
                triples.append((a, b, c))
                
    # Build conflict map for efficient checking
    # conflicts[x] = list of (u, v) such that {x, u, v} is a triple
    conflicts = {i: [] for i in range(1, N + 1)}
    for a, b, c in triples:
        # Triple is {a, b, c}
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))
        
    # Helper to check if x can be added to S
    def can_add(x, S):
        for u, v in conflicts[x]:
            if u in S and v in S:
                return False
        return True

    # Helper to perform local search on a set S
    # Returns a potentially larger set
    def local_search(S, time_limit):
        S = set(S)
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            # Try to add any element not in S, in random order for diversity
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
    
    # Attempt to solve using MILP if available
    milp_success = False
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Objective: maximize sum(x_i) -> minimize -sum(x_i)
        c_obj = -np.ones(N)
        
        # Build constraint matrix with triple constraints
        # For each triple, x_a + x_b + x_c <= 2
        constraint_rows = []
        constraint_ubs = []
        
        for (a, b, c) in triples:
            row = np.zeros(N)
            row[a-1] = 1
            row[b-1] = 1
            row[c-1] = 1
            constraint_rows.append(row)
            constraint_ubs.append(2)
        
        # Add clique constraints: find sets of vertices where every 3-subset is a triple
        # For such a set of size k, we can have at most 2 vertices
        # This gives stronger constraints than individual triple constraints
        
        # First, build a lookup for which pairs are in triples
        pair_in_triple = set()
        for (a, b, c) in triples:
            pair_in_triple.add((min(a,b), max(a,b)))
            pair_in_triple.add((min(a,c), max(a,c)))
            pair_in_triple.add((min(b,c), max(b,c)))
        
        # Try to find small cliques (sets where all 3-subsets are triples)
        # Start with pairs that are in many triples, then extend
        # For efficiency, only look for cliques of size 4 and 5
        
        # Count how many triples each pair is in
        pair_count = {}
        for (a, b, c) in triples:
            for p in [(a,b), (a,c), (b,c)]:
                p_sorted = (min(p), max(p))
                pair_count[p_sorted] = pair_count.get(p_sorted, 0) + 1
        
        # Find pairs that appear in at least 2 triples - these are candidates for clique extension
        candidate_pairs = [p for p, cnt in pair_count.items() if cnt >= 2]
        
        clique_constraints = []
        for (u, v) in candidate_pairs:
            # Try to extend {u, v} to a larger clique
            # A vertex w can be added if (u,v,w), (u,w,?) and (v,w,?) form triples with all pairs
            # Actually, for {u,v,w} to be a clique, we need triples (u,v,w) and all other pairs in the set
            # For a 4-clique {u,v,w,x}, we need all 4 choose 3 = 4 triples to exist
            
            # Find common neighbors that form triples with both u and v
            common_candidates = []
            for w in range(1, N+1):
                if w == u or w == v:
                    continue
                # Check if (u,v,w) is a triple
                if (min(u,v), max(u,v), w) in pair_in_triple or \
                   (min(u,w), max(u,w), v) in pair_in_triple or \
                   (min(v,w), max(v,w), u) in pair_in_triple:
                    # More precise check: is there a triple containing u, v, w?
                    uvw_triple = False
                    for (a, b, c) in triples:
                        if a in (u,v,w) and b in (u,v,w) and c in (u,v,w):
                            uvw_triple = True
                            break
                    if uvw_triple:
                        common_candidates.append(w)
            
            # Try to form 4-cliques from pairs of common candidates
            for i in range(len(common_candidates)):
                for j in range(i+1, len(common_candidates)):
                    w1 = common_candidates[i]
                    w2 = common_candidates[j]
                    # Check if {u, v, w1, w2} is a 4-clique
                    # Need triples: (u,v,w1), (u,v,w2), (u,w1,w2), (v,w1,w2)
                    is_clique = True
                    for triple_set in [(u,v,w1), (u,v,w2), (u,w1,w2), (v,w1,w2)]:
                        found = False
                        for (a, b, c) in triples:
                            if a in triple_set and b in triple_set and c in triple_set:
                                found = True
                                break
                        if not found:
                            is_clique = False
                            break
                    
                    if is_clique:
                        # Add constraint: x_u + x_v + x_w1 + x_w2 <= 2
                        row = np.zeros(N)
                        row[u-1] = 1
                        row[v-1] = 1
                        row[w1-1] = 1
                        row[w2-1] = 1
                        clique_constraints.append((row, 2))
        
        # Add clique constraints
        for row, ub in clique_constraints:
            constraint_rows.append(row)
            constraint_ubs.append(ub)
        
        if constraint_rows:
            A_ub = np.array(constraint_rows)
            b_ub = np.array(constraint_ubs)
        else:
            A_ub = np.zeros((0, N))
            b_ub = np.zeros(0)
        
        # Bounds: 0 <= x_i <= 1
        bounds = Bounds(lb=np.zeros(N), ub=np.ones(N))
        
        # Integrality: all variables are integer
        integrality = np.ones(N)
        
        # Solve with time limit
        remaining_for_milp = max(1, deadline - time.time() - 5)
        
        res = milp(
            c=c_obj,
            constraints=LinearConstraint(A_ub, ub=b_ub),
            bounds=bounds,
            integrality=integrality,
            options={"time_limit": remaining_for_milp}
        )
        
        if res.status in [0, 2]:  # 0 = optimal, 2 = time limit but feasible
            S = set()
            for i in range(N):
                if res.x[i] > 0.5:
                    S.add(i+1)
            
            # Verify validity
            valid = True
            for a, b, c in triples:
                if a in S and b in S and c in S:
                    valid = False
                    break
            
            if valid and len(S) > 0:
                best_set = S
                milp_success = True
                if res.status == 0:
                    # Optimal found, return immediately
                    return sorted(S)
                
    except Exception:
        pass
        
    # If MILP failed or wasn't used, use Heuristic Search
    # Strategy: Randomized Greedy Restarts + Local Search
    
    remaining_time = deadline - time.time()
    if remaining_time < 1:
        # Not enough time for heuristic, return empty or best found (which is empty)
        # But we must return a nonempty subset.
        # Fallback to a single fast greedy if no time.
        S = set()
        for a in range(N, 0, -1):
            if can_add(a, S):
                S.add(a)
        if S:
            return sorted(S)
        return [1] # Guaranteed non-empty, valid? 1 has no conflicts with any pair from [2..N] because 1/1 = 1, 1/b+1/c < 1.
        # Actually, is {1} valid? Yes.

    # Run multiple randomized greedy restarts
    # We have 'remaining_time'. Let's divide it into chunks.
    # Each chunk: Generate random order, greedy construct, local search.
    
    # To be safe, let's do a deterministic greedy first as a baseline.
    S_base = set()
    for a in range(1, N + 1): # Ascending
        if can_add(a, S_base):
            S_base.add(a)
    S_base = local_search(S_base, deadline - 1) # Leave 1s for return
    best_set = S_base
    best_size = len(S_base)
    
    # Now try randomized restarts if time permits
    num_restarts = 0
    while time.time() < deadline - 3: # Leave 3s safety
        # Random order
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S_curr = set()
        for a in order:
            if time.time() >= deadline - 3:
                break
            if can_add(a, S_curr):
                S_curr.add(a)
                
        # Local search with remaining time
        remaining = deadline - 3 - time.time()
        if remaining > 0.5:
            S_curr = local_search(S_curr, deadline - 3)
        
        if len(S_curr) > best_size:
            best_size = len(S_curr)
            best_set = S_curr
            
        num_restarts += 1
        
    return sorted(best_set)
