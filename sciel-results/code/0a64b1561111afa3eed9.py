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
    DEADLINE=88.0
    n=180
    T=_triples(n)
    m=len(T)
    
    # Precompute adjacency for fast conflict checking
    # For each x, list of (y, z) such that (x, y, z) is a triple
    elem_triples=[[] for _ in range(n+1)]
    for (a,b,c) in T:
        elem_triples[a].append((b,c))
        elem_triples[b].append((a,c))
        elem_triples[c].append((a,b))
        
    def can_add(S, x):
        # Check if adding x to S creates a forbidden triple
        for (y,z) in elem_triples[x]:
            if S[y] and S[z]:
                return False
        return True

    def greedy_construct(order):
        S=[False]*(n+1)
        for x in order:
            if can_add(S, x):
                S[x]=True
        return S

    best_S = None
    best_size = 0
    
    # 1. Deterministic Greedy (Descending)
    S_init = greedy_construct(range(n, 0, -1))
    best_S = S_init
    best_size = sum(S_init)
    
    # 2. Multi-Start Random Greedy
    start_time = time.time()
    max_time = DEADLINE - 5.0
    
    # Calculate degree for heuristic orders
    degrees = [0]*(n+1)
    for x in range(1, n+1):
        degrees[x] = len(elem_triples[x])
    
    # Try sorting by degree
    order_deg_asc = sorted(range(1, n+1), key=lambda x: degrees[x])
    order_deg_desc = sorted(range(1, n+1), key=lambda x: -degrees[x])
    
    for order in [order_deg_asc, order_deg_desc, list(reversed(range(1, n+1)))]:
        S_try = greedy_construct(order)
        size_try = sum(S_try)
        if size_try > best_size:
            best_size = size_try
            best_S = S_try
    
    # Random restarts
    while time.time() - start_time < max_time:
        order = list(range(1, n+1))
        random.shuffle(order)
        
        S_try = greedy_construct(order)
        size_try = sum(S_try)
        if size_try > best_size:
            best_size = size_try
            best_S = S_try
            
    # Final Polish on best_S
    # Try to add any element that fits
    improved = True
    while improved and time.time() < DEADLINE - 2.0:
        improved = False
        for x in range(1, n+1):
            if not best_S[x]:
                if can_add(best_S, x):
                    best_S[x] = True
                    best_size += 1
                    improved = True
                    
    out=[i for i in range(1,n+1) if best_S[i]]
    if not out:
        out=[180]
    return out
