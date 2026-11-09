import time, math, random

def _triples(n):
    # all triples (x,y,z) with 1<=x<y<z<=n and (y-x)(z-x)==x*x
    T=[]
    for x in range(1,n+1):
        x2=x*x
        d=1
        while d*d<=x2:
            if x2%d==0:
                e=x2//d
                if d<e:
                    y=x+d; z=x+e
                    if y<=n and z<=n and y!=z:
                        T.append((x,y,z))
            d+=1
    return T

def construct():
    t0=time.time()
    DEADLINE=85.0
    n=180
    T=_triples(n)
    m=len(T)
    
    # Precompute for each variable which clauses it belongs to and its position in each
    var_clauses = [[] for _ in range(n+1)]
    for ci, (a,b,c) in enumerate(T):
        var_clauses[a].append((ci, 0))
        var_clauses[b].append((ci, 1))
        var_clauses[c].append((ci, 2))
    
    # Precompute: for each variable x, list of (other1, other2) pairs from its triples
    # This speeds up delta computation
    var_others = [[] for _ in range(n+1)]
    for ci, (a,b,c) in enumerate(T):
        var_others[a].append((b,c))
        var_others[b].append((a,c))
        var_others[c].append((a,b))
    
    def violation_count(S):
        cnt = 0
        for (a,b,c) in T:
            if S[a] and S[b] and S[c]:
                cnt += 1
        return cnt

    def objective(S):
        size = sum(S)
        viol = violation_count(S)
        return size - 100 * viol
    
    # Helper to check if x can be added to S without creating violations
    def can_add(x, S):
        for (o1, o2) in var_others[x]:
            if S[o1] and S[o2]:
                return False
        return True
    
    def repair(S):
        """Remove elements to make S feasible, preferring to remove elements in fewest triples"""
        changed = True
        while changed:
            changed = False
            for (a,b,c) in T:
                if S[a] and S[b] and S[c]:
                    # Remove the element that appears in the fewest other violated triples
                    # Or just remove 'a' (smallest) as a heuristic
                    # Better: remove the one with highest index to keep smaller numbers?
                    # Actually, we want to maximize size, so any removal works
                    # Let's remove 'a' consistently
                    S[a] = False
                    changed = True
        return S
    
    def sa_pass(initial_S, T_start, T_end, max_iter, deadline):
        """Run simulated annealing from initial_S"""
        cur_S = initial_S[:]
        cur_obj = objective(cur_S)
        best_S = cur_S[:]
        best_obj = cur_obj
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            x = random.randint(1, n)
            
            if cur_S[x]:
                delta_size = -1
                delta_viol = 0
                for (o1, o2) in var_others[x]:
                    if cur_S[o1] and cur_S[o2]:
                        delta_viol -= 1
                delta_obj = delta_size - 100 * delta_viol
            else:
                delta_size = 1
                delta_viol = 0
                for (o1, o2) in var_others[x]:
                    if cur_S[o1] and cur_S[o2]:
                        delta_viol += 1
                delta_obj = delta_size - 100 * delta_viol
                
            progress = it / max_iter
            T_cur = T_start * (T_end / T_start) ** progress
            
            if delta_obj >= 0 or random.random() < math.exp(delta_obj / T_cur):
                cur_S[x] = not cur_S[x]
                cur_obj += delta_obj
                
            if cur_obj > best_obj:
                best_obj = cur_obj
                best_S = cur_S[:]
                
        return best_S, best_obj
    
    def local_search(S):
        """Final polish: try to add elements, then remove violating ones minimally"""
        S = S[:]
        # Try to add each element if it doesn't create violations
        for x in range(1, n+1):
            if not S[x] and can_add(x, S):
                S[x] = True
        # Repair if needed
        S = repair(S)
        return S
    
    random.seed(42)
    
    # Strategy 1: Start with high numbers [90,180], greedily add lows
    S1 = [False]*(n+1)
    for i in range(90, n+1):
        S1[i] = True
    for a in range(1, 90):
        if can_add(a, S1):
            S1[a] = True
    
    # Strategy 2: Start with all elements, repair
    S2 = [False]*(n+1)
    for i in range(1, n+1):
        S2[i] = True
    S2 = repair(S2)
    
    # Strategy 3: Random dense start
    S3 = [False]*(n+1)
    for i in range(1, n+1):
        if random.random() < 0.7:
            S3[i] = True
    S3 = repair(S3)
    
    # Strategy 4: Greedy by degree (add elements with fewest constraints first)
    S4 = [False]*(n+1)
    degrees = [len(var_others[i]) for i in range(n+1)]
    order = sorted(range(1, n+1), key=lambda x: degrees[x])
    for x in order:
        if can_add(x, S4):
            S4[x] = True
    
    best_S = None
    best_obj = -float('inf')
    
    # Run SA from multiple initial solutions
    initial_solutions = [S1, S2, S3, S4]
    
    for init in initial_solutions:
        # Quick local search first
        polished = local_search(init)
        obj = objective(polished)
        if obj > best_obj:
            best_obj = obj
            best_S = polished[:]
        
        # SA from this initial solution
        remaining = DEADLINE - 3 - (time.time() - t0)
        if remaining < 5:
            break
        
        # Use fraction of remaining time for each SA
        sa_deadline = time.time() + min(remaining * 0.25, 15.0)
        sa_result, sa_obj = sa_pass(init, 5.0, 0.05, 200000, sa_deadline)
        
        if sa_obj > best_obj:
            best_obj = sa_obj
            best_S = sa_result[:]
        
        # Local search on SA result
        polished_sa = local_search(sa_result)
        obj_sa = objective(polished_sa)
        if obj_sa > best_obj:
            best_obj = obj_sa
            best_S = polished_sa[:]
    
    # If time remains, do another round of SA from the best solution
    if time.time() < DEADLINE - 5:
        sa_deadline = DEADLINE - 3
        sa_result, sa_obj = sa_pass(best_S, 2.0, 0.01, 300000, sa_deadline)
        if sa_obj > best_obj:
            best_obj = sa_obj
            best_S = sa_result[:]
        
        polished_final = local_search(best_S)
        obj_final = objective(polished_final)
        if obj_final > best_obj:
            best_obj = obj_final
            best_S = polished_final[:]
    
    # Final repair to ensure feasibility
    best_S = repair(best_S)
    
    out = [i for i in range(1, n+1) if best_S[i]]
    if not out:
        out = [180]
    return out
