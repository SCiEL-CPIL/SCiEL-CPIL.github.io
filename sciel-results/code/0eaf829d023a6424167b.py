import time, math, random

def _triples(n):
    """Find all triples (x,y,z) with 1<=x<y<z<=n satisfying 1/x=1/y+1/z"""
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
    
    # Precompute clause membership
    var_clauses = [[] for _ in range(n+1)]
    for ci, (a,b,c) in enumerate(T):
        var_clauses[a].append((ci, 0))
        var_clauses[b].append((ci, 1))
        var_clauses[c].append((ci, 2))
    
    def violation_count(S):
        cnt = 0
        for (a,b,c) in T:
            if S[a] and S[b] and S[c]:
                cnt += 1
        return cnt
    
    def feasible_check(S):
        return violation_count(S) == 0
    
    # Greedy construction: add elements in descending order
    S = [False]*(n+1)
    for x in range(n, 0, -1):
        ok = True
        for (ci, pos) in var_clauses[x]:
            a, b, c = T[ci]
            others = [(a,b) if pos==2 else ((a,c) if pos==1 else (b,c))]
            o1, o2 = others[0]
            if S[o1] and S[o2]:
                ok = False
                break
        if ok:
            S[x] = True
    
    best_S = S[:]
    best_size = sum(S)
    
    # Simulated Annealing
    random.seed(42)
    
    def sa_pass(S, T_init, T_final, max_iter):
        """Run SA for max_iter steps, cooling from T_init to T_final"""
        S = S[:]
        cur_size = sum(S)
        cur_viol = violation_count(S)
        best_S = S[:]
        best_size = cur_size
        best_viol = cur_viol
        
        for it in range(max_iter):
            if time.time() - t0 > DEADLINE - 3:
                break
            
            # Choose a random variable to flip
            x = random.randint(1, n)
            
            if S[x]:
                # Removing x
                delta_size = -1
                delta_viol = 0
                for (ci, pos) in var_clauses[x]:
                    a, b, c = T[ci]
                    others = [(a,b) if pos==2 else ((a,c) if pos==1 else (b,c))]
                    o1, o2 = others[0]
                    if S[o1] and S[o2]:
                        delta_viol -= 1
            else:
                # Adding x
                delta_size = 1
                delta_viol = 0
                for (ci, pos) in var_clauses[x]:
                    a, b, c = T[ci]
                    others = [(a,b) if pos==2 else ((a,c) if pos==1 else (b,c))]
                    o1, o2 = others[0]
                    if S[o1] and S[o2]:
                        delta_viol += 1
            
            # Objective: maximize size, minimize violations
            # Use a penalty-based approach
            penalty = 50
            delta_obj = delta_size - penalty * delta_viol
            
            if delta_obj >= 0:
                S[x] = not S[x]
                cur_size += delta_size
                cur_viol += delta_viol
            else:
                # Compute temperature at this iteration
                progress = it / max_iter
                T_cur = T_init * (T_final / T_init) ** progress
                # Boltzmann acceptance
                if random.random() < math.exp(delta_obj / T_cur):
                    S[x] = not S[x]
                    cur_size += delta_size
                    cur_viol += delta_viol
            
            # Track best: prefer feasible sets, then larger size
            if (cur_viol < best_viol) or (cur_viol == best_viol and cur_size > best_size):
                best_viol = cur_viol
                best_size = cur_size
                best_S = S[:]
        
        return best_S, best_size, best_viol
    
    # Multiple SA passes with different initial temperatures
    remaining = DEADLINE - 3 - (time.time() - t0)
    if remaining > 5:
        max_iter = min(200000, int(remaining * 8000))
        best_S, best_size, best_viol = sa_pass(best_S, 5.0, 0.5, max_iter)
        
        remaining = DEADLINE - 3 - (time.time() - t0)
        if remaining > 3:
            max_iter = min(150000, int(remaining * 10000))
            best_S, best_size, best_viol = sa_pass(best_S, 1.0, 0.05, max_iter)
        
        remaining = DEADLINE - 3 - (time.time() - t0)
        if remaining > 2:
            max_iter = min(100000, int(remaining * 12000))
            best_S, best_size, best_viol = sa_pass(best_S, 0.3, 0.01, max_iter)
    
    # Final local improvement: try to add any element that doesn't create violations
    for x in range(1, n+1):
        if not best_S[x]:
            ok = True
            for (ci, pos) in var_clauses[x]:
                a, b, c = T[ci]
                others = [(a,b) if pos==2 else ((a,c) if pos==1 else (b,c))]
                o1, o2 = others[0]
                if best_S[o1] and best_S[o2]:
                    ok = False
                    break
            if ok:
                best_S[x] = True
    
    # Try swap improvements
    if time.time() - t0 < DEADLINE - 2:
        improved = True
        while improved and time.time() - t0 < DEADLINE - 2:
            improved = False
            in_list = [i for i in range(1, n+1) if best_S[i]]
            for out in in_list:
                if time.time() - t0 > DEADLINE - 2:
                    break
                for add in range(1, n+1):
                    if best_S[add]:
                        continue
                    # Check if removing out and adding add is beneficial
                    ok = True
                    for (ci, pos) in var_clauses[add]:
                        a, b, c = T[ci]
                        others = [(a,b) if pos==2 else ((a,c) if pos==1 else (b,c))]
                        o1, o2 = others[0]
                        if o1 == out:
                            if best_S[o2]:
                                ok = False
                                break
                        elif o2 == out:
                            if best_S[o1]:
                                ok = False
                                break
                        else:
                            if best_S[o1] and best_S[o2]:
                                ok = False
                                break
                    if ok:
                        best_S[out] = False
                        best_S[add] = True
                        improved = True
                        break
    
    out = [i for i in range(1, n+1) if best_S[i]]
    if not out:
        out = [180]
    return out
