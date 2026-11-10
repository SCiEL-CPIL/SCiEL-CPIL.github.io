import time, random, itertools

_N=14; _M=35; _MAXV=10000; _DEADLINE=75.0

def _evaluate(values, edges):
    s=set(); p=set()
    for i,j in edges:
        s.add(values[i]+values[j]); p.add(values[i]*values[j])
    return max(len(s),len(p))

def _greedy_edges(values, m=_M):
    all_edges=list(itertools.combinations(range(_N),2))
    sp={(i,j):(values[i]+values[j],values[i]*values[j]) for i,j in all_edges}
    chosen=[]; cset=set(); ss=set(); ps=set()
    for _ in range(m):
        best=None; bs=-1
        for e in all_edges:
            if e in cset: continue
            s,p=sp[e]; sc=0
            if s in ss: sc+=3
            if p in ps: sc+=3
            if sc>bs: bs=sc; best=e
        if best is None: break
        chosen.append(best); cset.add(best); ss.add(sp[best][0]); ps.add(sp[best][1])
    if len(chosen)<m:
        for e in all_edges:
            if len(chosen)>=m: break
            if e not in cset: chosen.append(e); cset.add(e)
    return [list(e) for e in chosen]

def _structured(rng):
    vals=[]
    for a in range(4):
        for b in range(4):
            v=(2**a)*(3**b)
            if v<=_MAXV: vals.append(v)
    rng.shuffle(vals); return vals[:_N]

def _random(rng):
    vals=set()
    while len(vals)<_N: vals.add(rng.randint(1,_MAXV))
    return list(vals)

def _pv(values,rng):
    v=list(values)
    if rng.random()<0.5:
        i=rng.randrange(_N); nv=rng.randint(1,_MAXV)
        if nv not in v: v[i]=nv
    else:
        i,j=rng.sample(range(_N),2); v[i],v[j]=v[j],v[i]
    return v

def _pe(edges,rng):
    all_edges=list(itertools.combinations(range(_N),2))
    e=[tuple(x) for x in edges]; es=set(e)
    if e:
        rem=rng.choice(e); e.remove(rem); es.discard(rem)
    while True:
        c=rng.choice(all_edges)
        if c not in es: e.append(c); es.add(c); break
    return [list(x) for x in e]

def construct():
    t0=time.time(); deadline=t0+_DEADLINE
    rng=random.Random(987654)
    best=None; bs=None
    inits=[_structured(rng) for _ in range(4)]+[_random(rng) for _ in range(4)]
    for vals in inits:
        if time.time()>deadline: break
        ed=_greedy_edges(vals); sc=_evaluate(vals,ed)
        if bs is None or sc<bs: bs=sc; best=(list(vals),ed)
    cv,ce=best; cs=bs; ni=0
    while time.time()<deadline:
        if rng.random()<0.5:
            nv=_pv(cv,rng); ne=_greedy_edges(nv); sc=_evaluate(nv,ne)
        else:
            nv=cv; ne=_pe(ce,rng); sc=_evaluate(nv,ne)
        if sc<=cs:
            cv,ce,cs=nv,ne,sc; ni=0
            if bs is None or cs<bs: bs=cs; best=(list(cv),list(ce))
        else:
            ni+=1
            if ni>80:
                ni=0
                if time.time()>deadline-5: break
                cv=_pv(best[0],rng); ce=_greedy_edges(cv); cs=_evaluate(cv,ce)
                if cs<bs: bs=cs; best=(list(cv),list(ce))
    vals,ed=best
    return {'values':[int(x) for x in vals],'edges':[[int(a),int(b)] for a,b in ed]}
