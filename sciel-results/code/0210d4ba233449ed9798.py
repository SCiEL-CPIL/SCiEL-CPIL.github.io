import time, random, math

def _precompute(n=180):
    triples=[]
    for x in range(1,n+1):
        x2=x*x
        d=1
        while d*d<=x2:
            if x2%d==0:
                e=x2//d
                if d<e:
                    y=x+d
                    z=x+e
                    if y<=n and z<=n and y!=z:
                        triples.append((x-1,y-1,z-1))
            d+=1
    pairs=[[] for _ in range(n)]
    for (i,j,k) in triples:
        pairs[i].append((j,k)); pairs[j].append((i,k)); pairs[k].append((i,j))
    return triples,pairs

def _min_degree_greedy(n, pairs, random_order):
    """Select vertices in minimum-conflict-degree order."""
    S=[0]*n
    in_set=set(random_order)
    # Compute initial conflict degrees
    deg=[0]*n
    for v in in_set:
        for (u,w) in pairs[v]:
            if u in in_set and w in in_set:
                # This pair (u,w) contributes to v's conflict if v is added
                pass
    # Actually, let's think differently: greedily add vertices that create fewest conflicts
    S=[0]*n
    remaining=set(range(n))
    # Sort by a heuristic: prefer vertices with fewer potential conflicts
    # Use the random_order as a tiebreaker
    order=list(remaining)
    random.shuffle(order)
    
    while order:
        # For each candidate, count how many new violations it would create
        best_v=-1
        best_conflicts=float('inf')
        best_idx=-1
        for idx,v in enumerate(order):
            conflicts=0
            for (u,w) in pairs[v]:
                if S[u] and S[w]:
                    conflicts+=1
            if conflicts < best_conflicts:
                best_conflicts=conflicts
                best_v=v
                best_idx=idx
            elif conflicts == best_conflicts and random.random()<0.5:
                best_v=v
                best_idx=idx
        if best_v>=0:
            S[best_v]=1
            order.pop(best_idx)
    return S

def _repair(S, triples, pairs, n):
    """Remove one element per violation until feasible."""
    S=S[:]
    changed=True
    while changed:
        changed=False
        for (i,j,k) in triples:
            if S[i] and S[j] and S[k]:
                # Remove the one with highest degree in remaining set
                deg_i=sum(1 for (u,w) in pairs[i] if S[u] and S[w])
                deg_j=sum(1 for (u,w) in pairs[j] if S[u] and S[w])
                deg_k=sum(1 for (u,w) in pairs[k] if S[u] and S[w])
                if deg_i>=deg_j and deg_i>=deg_k:
                    S[i]=0
                elif deg_j>=deg_i and deg_j>=deg_k:
                    S[j]=0
                else:
                    S[k]=0
                changed=True
                break
    return S

def _greedy_add(S, pairs, n):
    """Add any vertex that doesn't create violations."""
    improved=True
    while improved:
        improved=False
        for v in range(n):
            if not S[v]:
                ok=True
                for (u,w) in pairs[v]:
                    if S[u] and S[w]:
                        ok=False; break
                if ok:
                    S[v]=1; improved=True
    return S

def construct():
    t0=time.time()
    DEADLINE=55.0
    n=180
    triples,pairs=_precompute(n)
    
    def feasible(S):
        return all(not (S[i] and S[j] and S[k]) for (i,j,k) in triples)
    
    def count_violations(S):
        c=0
        for (i,j,k) in triples:
            if S[i] and S[j] and S[k]: c+=1
        return c
    
    best=None
    best_size=0
    
    # Try multiple strategies
    strategies=[]
    
    # Strategy 1: Suffix [91,180]
    S1=[0]*n
    for i in range(90,n): S1[i]=1
    strategies.append(S1[:])
    
    # Strategy 2: Minimum-degree greedy with random ordering
    for seed in [1,2,3,4,5]:
        random.seed(seed)
        order=list(range(n))
        random.shuffle(order)
        S2=_min_degree_greedy(n, pairs, order)
        strategies.append(S2)
    
    # Strategy 3: Random subsets of varying density
    for seed in [10,11,12]:
        random.seed(seed)
        S3=[1 if random.random()<0.5 else 0 for _ in range(n)]
        strategies.append(S3)
    
    for strat in strategies:
        if time.time()-t0 > DEADLINE-5:
            break
        
        S=strat[:]
        cur_v=count_violations(S)
        cur_size=sum(S)
        
        # Repair if needed
        if cur_v>0:
            S=_repair(S, triples, pairs, n)
            cur_v=0
            cur_size=sum(S)
        
        # SA phase
        T=2.0
        step=0
        while time.time()-t0 < DEADLINE-5:
            step+=1
            v=random.randrange(n)
            old=S[v]
            if old==0:
                addc=0
                for (u,w) in pairs[v]:
                    if S[u] and S[w]: addc+=1
                new_v=cur_v+addc
                new_size=cur_size+1
            else:
                remc=0
                for (u,w) in pairs[v]:
                    if S[u] and S[w]: remc+=1
                new_v=cur_v-remc
                new_size=cur_size-1
            new_score=new_size - 1000.0*new_v
            cur_score=cur_size - 1000.0*cur_v
            dE=new_score-cur_score
            if dE>=0 or random.random()<math.exp(dE/T):
                S[v]=1-old
                cur_v=new_v; cur_size=new_size
                if new_v==0 and cur_size>best_size:
                    best=S[:]; best_size=cur_size
            T*=0.99999
            if T<0.01: T=0.01
        
        # Greedy add
        S=_greedy_add(S, pairs, n)
        cur_size=sum(S)
        if feasible(S) and cur_size>best_size:
            best=S[:]; best_size=cur_size
    
    # Final greedy add on best
    if best is not None:
        best=_greedy_add(best, pairs, n)
        best_size=sum(best)
    
    if best is None or not feasible(best):
        # Fallback: just use [180]
        return [180]
    
    out=[i+1 for i in range(n) if best[i]]
    if not out: out=[180]
    return out
