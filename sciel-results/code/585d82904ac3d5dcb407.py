import time, math, random

def _triples(n):
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
    
    # Precompute for each clause the three vars
    clause_vars = [(a,b,c) for (a,b,c) in T]
    
    def violation_count(S):
        cnt = 0
        for (a,b,c) in T:
            if S[a] and S[b] and S[c]:
                cnt += 1
        return cnt
    
    def can_add(x, S):
        for (ci, pos) in var_clauses[x]:
            a, b, c = T[ci]
            if pos == 0:
                if S[b] and S[c]:
                    return False
            elif pos == 1:
                if S[a] and S[c]:
                    return False
            else:
                if S[a] and S[b]:
                    return False
        return True
    
    # Greedy initialization
    S = [False]*(n+1)
    for i in range(90, n+1):
        S[i] = True
    for a in range(1, 90):
        if can_add(a, S):
            S[a] = True
    
    best_S = S[:]
    best_size = sum(S)
    best_viol = violation_count(best_S)
    
    # SA + local search
    def sa_and_local(start_S, time_budget):
        cur_S = start_S[:]
        cur_viol = violation_count(cur_S)
        cur_size = sum(cur_S)
        cur_obj = cur_size - 500 * cur_viol
        best_local_S = cur_S[:]
        best_local_obj = cur_obj
        best_local_size = cur_size
        best_local_viol = cur_viol
        
        # SA phase
        T_start = 10.0
        T_end = 0.05
        max_iter = 200000
        t_sa_start = time.time()
        
        for it in range(max_iter):
            if time.time() - t_sa_start > time_budget * 0.6:
                break
            
            x = random.randint(1, n)
            
            if cur_S[x]:
                delta_size = -1
                delta_viol = 0
                for (ci, pos) in var_clauses[x]:
                    a, b, c = T[ci]
                    if pos == 0:
                        if cur_S[b] and cur_S[c]:
                            delta_viol -= 1
                    elif pos == 1:
                        if cur_S[a] and cur_S[c]:
                            delta_viol -= 1
                    else:
                        if cur_S[a] and cur_S[b]:
                            delta_viol -= 1
                delta_obj = delta_size - 500 * delta_viol
            else:
                delta_size = 1
                delta_viol = 0
                for (ci, pos) in var_clauses[x]:
                    a, b, c = T[ci]
                    if pos == 0:
                        if cur_S[b] and cur_S[c]:
                            delta_viol += 1
                    elif pos == 1:
                        if cur_S[a] and cur_S[c]:
                            delta_viol += 1
                    else:
                        if cur_S[a] and cur_S[b]:
                            delta_viol += 1
                delta_obj = delta_size - 500 * delta_viol
            
            progress = it / max_iter
            T_cur = T_start * (T_end / T_start) ** progress
            
            if delta_obj >= 0 or random.random() < math.exp(delta_obj / T_cur):
                cur_S[x] = not cur_S[x]
                cur_obj += delta_obj
                cur_size += delta_size
                cur_viol += delta_viol
                
                if cur_obj > best_local_obj:
                    best_local_obj = cur_obj
                    best_local_size = cur_size
                    best_local_viol = cur_viol
                    best_local_S = cur_S[:]
        
        # Local search polish: systematic 1-flip and 2-flip
        t_ls_start = time.time()
        improved = True
        while improved and time.time() - t_ls_start < time_budget * 0.4:
            improved = False
            # 1-flip local search
            for x in range(1, n+1):
                if cur_S[x]:
                    # Try removing x
                    delta_size = -1
                    delta_viol = 0
                    for (ci, pos) in var_clauses[x]:
                        a, b, c = T[ci]
                        if pos == 0:
                            if cur_S[b] and cur_S[c]:
                                delta_viol -= 1
                        elif pos == 1:
                            if cur_S[a] and cur_S[c]:
                                delta_viol -= 1
                        else:
                            if cur_S[a] and cur_S[b]:
                                delta_viol -= 1
                    delta_obj = delta_size - 500 * delta_viol
                    if delta_obj > 0:
                        cur_S[x] = False
                        cur_obj += delta_obj
                        cur_size += delta_size
                        cur_viol += delta_viol
                        improved = True
                        if cur_obj > best_local_obj:
                            best_local_obj = cur_obj
                            best_local_size = cur_size
                            best_local_viol = cur_viol
                            best_local_S = cur_S[:]
                else:
                    # Try adding x
                    delta_size = 1
                    delta_viol = 0
                    for (ci, pos) in var_clauses[x]:
                        a, b, c = T[ci]
                        if pos == 0:
                            if cur_S[b] and cur_S[c]:
                                delta_viol += 1
                        elif pos == 1:
                            if cur_S[a] and cur_S[c]:
                                delta_viol += 1
                        else:
                            if cur_S[a] and cur_S[b]:
                                delta_viol += 1
                    delta_obj = delta_size - 500 * delta_viol
                    if delta_obj > 0:
                        cur_S[x] = True
                        cur_obj += delta_obj
                        cur_size += delta_size
                        cur_viol += delta_viol
                        improved = True
                        if cur_obj > best_local_obj:
                            best_local_obj = cur_obj
                            best_local_size = cur_size
                            best_local_viol = cur_viol
                            best_local_S = cur_S[:]
            
            # 2-flip local search (add two, or remove two, or swap)
            if not improved:
                # Try adding one element that's blocked, see if removing another helps
                for x in range(1, n+1):
                    if not cur_S[x]:
                        # Check if adding x creates violations
                        new_viol = 0
                        for (ci, pos) in var_clauses[x]:
                            a, b, c = T[ci]
                            if pos == 0:
                                if cur_S[b] and cur_S[c]:
                                    new_viol += 1
                            elif pos == 1:
                                if cur_S[a] and cur_S[c]:
                                    new_viol += 1
                            else:
                                if cur_S[a] and cur_S[b]:
                                    new_viol += 1
                        if new_viol == 0:
                            # Can add freely
                            cur_S[x] = True
                            cur_obj += 1
                            cur_size += 1
                            improved = True
                            if cur_obj > best_local_obj:
                                best_local_obj = cur_obj
                                best_local_size = cur_size
                                best_local_viol = cur_viol
                                best_local_S = cur_S[:]
                        elif new_viol == 1:
                            # Try removing one element from the violated triple
                            for (ci, pos) in var_clauses[x]:
                                a, b, c = T[ci]
                                if pos == 0:
                                    if cur_S[b] and cur_S[c]:
                                        # Try removing b or c
                                        for y in [b, c]:
                                            # Check if removing y and adding x is net positive
                                            # Removing y: lose 1 size, gain some violations removed
                                            rem_viol = 0
                                            for (ci2, pos2) in var_clauses[y]:
                                                a2, b2, c2 = T[ci2]
                                                if pos2 == 0:
                                                    if cur_S[b2] and cur_S[c2]:
                                                        rem_viol += 1
                                                elif pos2 == 1:
                                                    if cur_S[a2] and cur_S[c2]:
                                                        rem_viol += 1
                                                else:
                                                    if cur_S[a2] and cur_S[b2]:
                                                        rem_viol += 1
                                            # Net: add x (+1 size, +new_viol viol), remove y (-1 size, -rem_viol viol)
                                            # But removing y might also affect the violation from x
                                            # This is complex; skip for now
                                        break
            if time.time() - t_ls_start > time_budget * 0.4:
                break
        
        return best_local_S, best_local_size, best_local_viol
    
    # Run multiple SA passes with different seeds
    random.seed(42)
    passes = 5
    pass_time = (DEADLINE - 3 - (time.time() - t0)) / passes
    
    for p in range(passes):
        if time.time() - t0 > DEADLINE - 3:
            break
        
        # Create a perturbed start
        start_S = best_S[:]
        # Randomly flip a few elements
        for _ in range(random.randint(0, 10)):
            x = random.randint(1, n)
            start_S[x] = not start_S[x]
        
        res_S, res_size, res_viol = sa_and_local(start_S, pass_time)
        
        if res_viol == 0 and res_size > best_size:
            best_S = res_S[:]
            best_size = res_size
            best_viol = res_viol
        elif res_viol == 0 and res_size == best_size:
            # Tie, keep current
            pass
    
    # Final repair
    for (a,b,c) in T:
        if best_S[a] and best_S[b] and best_S[c]:
            best_S[a] = False
    
    out = [i for i in range(1, n+1) if best_S[i]]
    if not out:
        out = [180]
    return out
