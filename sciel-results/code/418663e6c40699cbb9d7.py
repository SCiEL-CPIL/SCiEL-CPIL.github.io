import time, random, math

def _precompute(n=180):
    # triples (x,y,z) 0-indexed, x<y<z, (y-x)*(z-x)==x^2
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
    # for each vertex, list of (other1,other2)
    pairs=[[] for _ in range(n)]
    for (i,j,k) in triples:
        pairs[i].append((j,k)); pairs[j].append((i,k)); pairs[k].append((i,j))
    return triples,pairs

def _greedy_add(S, pairs, n):
    """Try to add elements greedily. Returns modified S and size."""
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
    return S, sum(S)

def _repair(S, triples, pairs, n):
    """Remove elements to fix violations. Returns modified S and size."""
    S = S[:]
    changed = True
    while changed:
        changed = False
        for (i,j,k) in triples:
            if S[i] and S[j] and S[k]:
                # remove the element with highest degree (most conflicts)
                di = len(pairs[i]); dj = len(pairs[j]); dk = len(pairs[k])
                if di >= dj and di >= dk:
                    S[i] = 0
                elif dj >= di and dj >= dk:
                    S[j] = 0
                else:
                    S[k] = 0
                changed = True
                break
    return S, sum(S)

def construct():
    t0=time.time()
    DEADLINE=55.0
    n=180
    triples,pairs=_precompute(n)
    random.seed(12345)
    
    best = [0]*n
    best_size = 0
    
    # Multiple restarts with different initial states
    num_restarts = 5
    restart_time = (DEADLINE - (time.time()-t0)) / num_restarts
    
    for restart in range(num_restarts):
        if time.time()-t0 > DEADLINE:
            break
        
        # Different initial seeds for each restart
        if restart == 0:
            S=[0]*n
            for i in range(90,180):
                S[i]=1
        elif restart == 1:
            S=[0]*n
            for i in range(60,180):
                S[i]=1
        elif restart == 2:
            S=[0]*n
            for i in range(120,180):
                S[i]=1
        elif restart == 3:
            S=[0]*n
            random.seed(12345 + restart)
            for i in range(n):
                if random.random() < 0.5:
                    S[i]=1
        else:
            S=[0]*n
            random.seed(99999 + restart)
            for i in range(n):
                if random.random() < 0.4:
                    S[i]=1
        
        # Repair initial state
        S, cur_size = _repair(S, triples, pairs, n)
        # Compute violations
        cur_v = 0
        for (i,j,k) in triples:
            if S[i] and S[j] and S[k]: cur_v += 1
        
        # SA phase
        T=1.0
        sa_deadline = min(time.time()-t0 + restart_time, DEADLINE)
        while time.time()-t0 < sa_deadline:
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
        
        # Greedy add from current S
        S, cur_size = _greedy_add(S, pairs, n)
        if cur_size > best_size:
            # Check feasibility
            feasible = True
            for (i,j,k) in triples:
                if S[i] and S[j] and S[k]:
                    feasible = False; break
            if feasible:
                best = S[:]; best_size = cur_size
        
        # Also try greedy from best
        Sb = best[:]
        Sb, sz = _greedy_add(Sb, pairs, n)
        if sz > best_size:
            feasible = True
            for (i,j,k) in triples:
                if Sb[i] and Sb[j] and Sb[k]:
                    feasible = False; break
            if feasible:
                best = Sb; best_size = sz
    
    out=[i+1 for i in range(n) if best[i]]
    if not out: out=[180]
    return out
