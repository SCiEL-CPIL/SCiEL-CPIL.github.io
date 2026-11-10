"""Direct construction for dense odd-order graphs avoiding equal-degree path endpoints."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    def is_valid(edge_set, n):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        for (u, v) in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True

    def fast_is_valid(edge_set, n):
        """Faster validity check using degree sets for each edge's endpoint neighborhoods."""
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        for (u, v) in edge_set:
            # Get degree sets of neighbors (excluding each other)
            degs_u = set()
            for s in adj[u]:
                if s != v:
                    degs_u.add(deg[s])
            for t in adj[v]:
                if t != u:
                    if deg[t] in degs_u:
                        # Need to check if there exists s in adj[u]\{v} with deg[s]==deg[t] and s != t
                        # If t is not in adj[u], then any s with deg[s]==deg[t] works
                        if t not in adj[u]:
                            return False
                        else:
                            # t is in adj[u], so we need another s with same degree
                            # Check if there's an s in adj[u]\{v,t} with deg[s]==deg[t]
                            for s in adj[u]:
                                if s != v and s != t and deg[s] == deg[t]:
                                    return False
            # Also need symmetric check: for s in adj[u]\{v}, check if deg[s] matches any t in adj[v]\{u}
            # The above loop covers this partially. Let's do it more carefully.
            pass
        return True

    # Actually, let's use a cleaner fast validity check
    def is_valid_fast(edge_set, n):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        for (u, v) in edge_set:
            # Count degree frequencies in N(u)\{v} and N(v)\{u}
            from collections import Counter
            deg_u = Counter(deg[s] for s in adj[u] if s != v)
            deg_v = Counter(deg[t] for t in adj[v] if t != u)
            
            # For each degree d, if count_u[d] + count_v[d] > 1, we need the extra
            # to be a shared vertex. A shared vertex x in N(u)\{v} ∩ N(v)\{u} with deg[x]=d
            # can serve as the "matching" pair (s=t=x).
            common = (adj[u] - {v}) & (adj[v] - {u})
            common_deg = Counter(deg[x] for x in common)
            
            for d in set(deg_u.keys()) | set(deg_v.keys()):
                cu = deg_u.get(d, 0)
                cv = deg_v.get(d, 0)
                if cu + cv > 1:
                    # Need at least (cu + cv - 1) shared vertices of degree d
                    # Actually: we can pair min(cu, cv) shared vertices, but shared vertices
                    # appear in both counts. Let's think differently.
                    # Total "slots" of degree d on side u: cu
                    # Total "slots" of degree d on side v: cv
                    # Shared vertices of degree d: cs
                    # A shared vertex x can only pair with itself (s=t=x is allowed).
                    # Non-shared vertices on side u with degree d: cu - cs
                    # Non-shared vertices on side v with degree d: cv - cs
                    # Each non-shared u-vertex must pair only with shared v-vertices or vice versa?
                    # No: s in N(u)\{v}, t in N(v)\{u}, s!=t. If s is shared (also in N(v)\{u}),
                    # it can pair with t=s (allowed since s==t is skipped). But if s is NOT shared,
                    # it can only pair with t where t != s, and if t has same degree, conflict.
                    # 
                    # Simpler: conflict iff there exist s in N(u)\{v}, t in N(v)\{u} with s!=t and deg[s]==deg[t].
                    # This is equivalent to: for some d, (number of vertices of degree d in N(u)\{v}) 
                    # + (number of vertices of degree d in N(v)\{u}) > (number of shared vertices of degree d) + 1?
                    # No. Let's just check directly with the counter approach.
                    # 
                    # Actually the simplest correct check: for degree d, let U_d = {s in N(u)\{v} : deg[s]=d}, V_d = {t in N(v)\{u} : deg[t]=d}.
                    # Conflict iff there exist s in U_d, t in V_d with s != t.
                    # This is false iff U_d ⊆ V_d and |U_d| <= |V_d|... no.
                    # It's false iff every s in U_d equals every t in V_d, i.e., U_d ∩ V_d covers all and |U_d|=|V_d|=1 or they're the same single element.
                    # Actually: no conflict iff for all s in U_d and t in V_d, s==t. This means U_d and V_d are both subsets of the same singleton, i.e., U_d = V_d = {x} for some x, or one is empty.
                    # So conflict iff |U_d| >= 1 and |V_d| >= 1 and (|U_d| > 1 or |V_d| > 1 or U_d != V_d).
                    # Equivalently: conflict iff min(cu, cv) >= 1 and (cu > 1 or cv > 1 or the sets differ).
                    # But we only have counts, not sets. Let me just use the sets directly.
                    pass
            
            # Direct set-based check for correctness
            U_d = {}
            V_d = {}
            for s in adj[u]:
                if s != v:
                    d = deg[s]
                    U_d.setdefault(d, set()).add(s)
            for t in adj[v]:
                if t != u:
                    d = deg[t]
                    V_d.setdefault(d, set()).add(t)
            
            for d in U_d:
                if d in V_d:
                    su = U_d[d]
                    sv = V_d[d]
                    # Conflict iff exists s in su, t in sv with s != t
                    # This is true iff su - sv is non-empty OR sv - su is non-empty OR |su|>1 OR |sv|>1
                    # More precisely: no conflict iff su and sv are both exactly {x} for the same x
                    if len(su) == 1 and len(sv) == 1:
                        if su != sv:
                            return False
                    elif len(su) > 1 or len(sv) > 1:
                        # Check if all elements are the same
                        if su != sv:
                            return False
                        # If su == sv and len > 1, then pick s1, s2 from su, t1=s1, t2=s2: s1!=t2 so conflict
                        if len(su) > 1:
                            return False
        return True

    # Start with K_{7,8} as baseline (56 edges, proven valid)
    A = list(range(7))
    B = list(range(7, 15))
    base_edges = set()
    for a in A:
        for b in B:
            base_edges.add((a, b))
    
    best_edges = set(base_edges)
    best_count = len(base_edges)

    # Precompute all possible edges
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Phase 1: Greedy edge addition with multiple restarts
    random.seed(42)
    
    # Try multiple orderings
    for restart in range(5):
        if time.time() > deadline:
            break
        
        current = set(base_edges)
        missing = [e for e in all_possible if e not in current]
        random.shuffle(missing)
        
        for e in missing:
            if time.time() > deadline:
                break
            if e in current:
                continue
            trial = current | {e}
            if is_valid_fast(trial, n):
                current = trial
                if len(current) > best_count:
                    best_edges = set(current)
                    best_count = len(current)
        
        # Also try reverse order
        random.shuffle(missing)
        current2 = set(base_edges)
        for e in missing:
            if time.time() > deadline:
                break
            if e in current2:
                continue
            trial = current2 | {e}
            if is_valid_fast(trial, n):
                current2 = trial
                if len(current2) > best_count:
                    best_edges = set(current2)
                    best_count = len(current2)
    
    # Phase 2: Local search with swaps - more aggressive
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        improved = False
        current_list = list(current)
        random.shuffle(current_list)
        
        for rem in current_list:
            if time.time() > deadline:
                break
            
            trial = current - {rem}
            
            # Try adding each missing edge
            for add in all_possible:
                if add in trial:
                    continue
                if add == rem:
                    continue
                
                trial2 = trial | {add}
                if len(trial2) <= len(current):
                    continue
                
                if is_valid_fast(trial2, n):
                    current = trial2
                    improved = True
                    if len(current) > best_count:
                        best_edges = set(current)
                        best_count = len(current)
                    break
            
            if improved:
                break
        
        if not improved:
            # Try 2-edge swaps: remove 2, add 2
            found = False
            for _ in range(100):
                if time.time() > deadline:
                    break
                rem1, rem2 = random.sample(list(current), 2)
                trial = current - {rem1, rem2}
                
                for add1 in all_possible:
                    if add1 in trial or add1 == rem1 or add1 == rem2:
                        continue
                    trial1 = trial | {add1}
                    for add2 in all_possible:
                        if add2 in trial1 or add2 == rem1 or add2 == rem2 or add2 == add1:
                            continue
                        trial2 = trial1 | {add2}
                        if len(trial2) <= len(current):
                            continue
                        if is_valid_fast(trial2, n):
                            current = trial2
                            found = True
                            if len(current) > best_count:
                                best_edges = set(current)
                                best_count = len(current)
                            break
                    if found:
                        break
                if found:
                    break
            
            if not found:
                # Try 1-remove, 2-add (net +1)
                found = False
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    rem = random.choice(list(current))
                    trial = current - {rem}
                    
                    for add1 in all_possible:
                        if add1 in trial or add1 == rem:
                            continue
                        trial1 = trial | {add1}
                        for add2 in all_possible:
                            if add2 in trial1 or add2 == rem or add2 == add1:
                                continue
                            trial2 = trial1 | {add2}
                            if len(trial2) <= len(current):
                                continue
                            if is_valid_fast(trial2, n):
                                current = trial2
                                found = True
                                if len(current) > best_count:
                                    best_edges = set(current)
                                    best_count = len(current)
                                break
                        if found:
                            break
                    if found:
                        break
                
                if not found:
                    break
    
    # Phase 3: Try different initial partitions
    for split_a in [4, 5, 6, 7, 8, 9, 10]:
        if time.time() > deadline:
            break
        
        A2 = list(range(split_a))
        B2 = list(range(split_a, n))
        base2 = set()
        for a in A2:
            for b in B2:
                base2.add((a, b))
        
        if len(base2) <= best_count:
            continue
        
        # Verify base is valid
        if not is_valid_fast(base2, n):
            continue
        
        # Try multiple greedy orderings
        for restart in range(3):
            if time.time() > deadline:
                break
            cur = set(base2)
            miss2 = [e for e in all_possible if e not in cur]
            random.shuffle(miss2)
            
            for e in miss2:
                if time.time() > deadline:
                    break
                if e in cur:
                    continue
                trial = cur | {e}
                if is_valid_fast(trial, n):
                    cur = trial
            
            if len(cur) > best_count:
                best_edges = set(cur)
                best_count = len(cur)
        
        # Local search from best result of this partition
        if best_count > len(base2) and time.time() < deadline:
            for iteration in range(100):
                if time.time() > deadline:
                    break
                improved = False
                cur_list = list(best_edges)
                random.shuffle(cur_list)
                
                for rem in cur_list:
                    if time.time() > deadline:
                        break
                    trial = set(best_edges) - {rem}
                    for add in all_possible:
                        if add in trial or add == rem:
                            continue
                        trial2 = trial | {add}
                        if len(trial2) <= len(best_edges):
                            continue
                        if is_valid_fast(trial2, n):
                            best_edges = trial2
                            best_count = len(best_edges)
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
    
    # Final verification
    if not is_valid_fast(best_edges, n):
        # Fallback to K_{7,8}
        best_edges = set(base_edges)
        best_count = len(base_edges)
    
    result = [list(e) for e in sorted(best_edges)]
    final = []
    for e in result:
        final.append([int(e[0]), int(e[1])])
    return final

# EVOLVE-BLOCK-END
