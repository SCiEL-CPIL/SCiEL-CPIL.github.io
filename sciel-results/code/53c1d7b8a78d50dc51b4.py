import time
from collections import deque

P = {'n': 24, 'edges': [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,16],[16,17],[17,18],[18,19],[19,20],[20,21],[21,22],[22,23]], 'diameter': 4}

_N = P['n']
_BASE = frozenset((min(a,b),max(a,b)) for a,b in P['edges'])
_BASE_ADJ = None
_CANDS = None
_CACHE = None

def _base_adj():
    global _BASE_ADJ
    if _BASE_ADJ is None:
        a=[set() for _ in range(_N)]
        for u,v in _BASE:
            a[u].add(v); a[v].add(u)
        _BASE_ADJ=a
    return _BASE_ADJ

def _all_cands():
    global _CANDS
    if _CANDS is None:
        cs=[]
        for i in range(12):
            u=2*i
            for j in range(12):
                v=2*j+1
                e=(min(u,v),max(u,v))
                if e not in _BASE:
                    cs.append(e)
        _CANDS=cs
    return _CANDS

def _distances(adj):
    n=_N
    dist=[None]*n
    for s in range(n):
        d=[-1]*n; d[s]=0; q=deque([s])
        while q:
            u=q.popleft()
            for v in adj[u]:
                if d[v]==-1:
                    d[v]=d[u]+1; q.append(v)
        dist[s]=d
    return dist

def _diam(dist):
    n=_N; m=0
    for u in range(n):
        du=dist[u]
        for v in range(n):
            if du[v]>m: m=du[v]
    return m

def _viol(dist):
    n=_N; s=set()
    for u in range(n):
        for v in range(u+1,n):
            if dist[u][v]>4: s.add((u,v))
    return s

def _rebuild(base,S):
    a=[set(s) for s in base]
    for u,v in S:
        a[u].add(v); a[v].add(u)
    return a

def _valid(base,S):
    return _diam(_distances(_rebuild(base,S)))<=4

def _solve():
    t0=time.time()
    DEADLINE=t0+65.0
    base=_base_adj()
    cands=_all_cands()
    base_dist=_distances(base)
    viol=_viol(base_dist)

    cover={}
    for e in cands:
        a=_rebuild(base,{e})
        d=_distances(a)
        cov=frozenset(p for p in viol if d[p[0]][p[1]]<=4)
        if cov:
            cover[e]=cov

    best=None
    try:
        import numpy as np
        from scipy.optimize import milp, LinearConstraint, Bounds
        ekeys=list(cover.keys())
        pkeys=sorted(viol)
        pidx={p:i for i,p in enumerate(pkeys)}
        m=len(ekeys); npairs=len(pkeys)
        if m>0 and npairs>0:
            A=np.zeros((npairs,m),dtype=int)
            for j,e in enumerate(ekeys):
                for p in cover[e]:
                    A[pidx[p],j]=1
            c=np.ones(m)
            res=milp(c=c, constraints=[LinearConstraint(A,1,np.inf)],
                     integrality=np.ones(m), bounds=Bounds(0,1),
                     options={"time_limit":20})
            if res.status in (0,1,2) and res.x is not None:
                sel=frozenset(ekeys[j] for j in range(m) if res.x[j]>0.5)
                if _valid(base,sel):
                    best=sel
    except Exception:
        best=None

    if best is not None:
        cur=set(best)
    else:
        cur=set()
        a=_rebuild(base,cur)
        d=_distances(a)
        while _diam(d)>4 and time.time()<DEADLINE:
            cv=_viol(d)
            best_e=None; best_gain=-1
            for e in cands:
                if e in cur: continue
                a2=_rebuild(base,cur|{e})
                d2=_distances(a2)
                dv=_viol(d2)
                gain=len(cv)-len(dv)
                if gain>best_gain:
                    best_gain=gain; best_e=e
            if best_e is None or best_gain<=0:
                break
            cur.add(best_e)
            a=_rebuild(base,cur)
            d=_distances(a)
        if _diam(_distances(_rebuild(base,cur)))>4:
            cur=set(cands)

    improved=True
    while improved and time.time()<DEADLINE:
        improved=False
        if not _valid(base,cur):
            break
        for e in list(cur):
            if time.time()>=DEADLINE: break
            if _valid(base,cur-{e}):
                cur=cur-{e}; improved=True; break
        if not improved:
            for e in list(cur):
                if time.time()>=DEADLINE: break
                for f in cands:
                    if f in cur: continue
                    if _valid(base,(cur-{e})|{f}):
                        cur=(cur-{e})|{f}; improved=True; break
                if improved: break

    if not _valid(base,cur):
        cur=set(cands)
    return cur

def construct():
    global _CACHE
    if _CACHE is None:
        S=_solve()
        _CACHE=[[int(u),int(v)] for u,v in sorted(S)]
    return [list(x) for x in _CACHE]
