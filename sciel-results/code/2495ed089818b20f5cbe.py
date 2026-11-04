import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    DEADLINE = t0 + 110.0

    nm, mp, np_ = n*m, m*p, p*n

    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i*m+j, j*p+k, k*n+i))
    triple_set = set(triples)
    T = len(triples)

    # Schoolbook
    def schoolbook():
        r = T
        U = [[0]*r for _ in range(nm)]
        V = [[0]*r for _ in range(mp)]
        W = [[0]*r for _ in range(np_)]
        for c,(a,b,cc) in enumerate(triples):
            U[a][c]=1; V[b][c]=1; W[cc][c]=1
        return U,V,W

    best = schoolbook()

    # Fast verification using sparse residual
    def verify_terms(terms):
        # terms: list of (u,v,w) with Fraction or float entries
        # residual dict
        res = {}
        for u,v,w in terms:
            su = [i for i,x in enumerate(u) if x != 0]
            sv = [i for i,x in enumerate(v) if x != 0]
            sw = [i for i,x in enumerate(w) if x != 0]
            for a in su:
                for b in sv:
                    for c in sw:
                        val = u[a]*v[b]*w[c]
                        key = (a,b,c)
                        res[key] = res.get(key, 0) + val
        # check: for each triple, res == 1; for others, res == 0
        for (a,b,c) in triples:
            if res.get((a,b,c), 0) != 1:
                return False
        for key, val in res.items():
            if key not in triple_set and val != 0:
                return False
        return True

    # Dictionary of candidates (as lists of Fraction)
    def F(x): return Fraction(x)
    cands = []
    # schoolbook
    for (a,b,c) in triples:
        u=[0]*nm; v=[0]*mp; w=[0]*np_
        u[a]=F(1); v[b]=F(1); w[c]=F(1)
        cands.append((u,v,w))
    # lines
    for i in range(n):
        for j in range(m):
            u=[0]*nm; v=[0]*mp; w=[0]*np_
            u[i*m+j]=F(1)
            for k in range(p):
                v[j*p+k]=F(1); w[k*n+i]=F(1)
            cands.append((u,v,w))
    for j in range(m):
        for k in range(p):
            u=[0]*nm; v=[0]*mp; w=[0]*np_
            v[j*p+k]=F(1)
            for i in range(n):
                u[i*m+j]=F(1); w[k*n+i]=F(1)
            cands.append((u,v,w))
    for i in range(n):
        for k in range(p):
            u=[0]*nm; v=[0]*mp; w=[0]*np_
            w[k*n+i]=F(1)
            for j in range(m):
                u[i*m+j]=F(1); v[j*p+k]=F(1)
            cands.append((u,v,w))

    # Greedy exact cover with backtracking (limited)
    # residual starts as target: 1 at triples
    # We'll do a randomized greedy: repeatedly pick a candidate that best reduces residual
    # Use exact Fraction arithmetic
    def greedy_search(max_terms, seed):
        random.seed(seed)
        res = {}
        for (a,b,c) in triples:
            res[(a,b,c)] = F(1)
        chosen = []
        for step in range(max_terms):
            # find candidate that minimizes residual norm
            best_cand = None; best_norm = None
            order = list(range(len(cands)))
            random.shuffle(order)
            for d in order:
                u,v,w = cands[d]
                # compute new residual norm if we add this candidate (with coeff +1 or -1)
                # try coeff in {1, -1, 1/2, -1/2}
                for coeff in [F(1), F(-1), F(1,2), F(-1,2)]:
                    # apply
                    su=[i for i,x in enumerate(u) if x!=0]
                    sv=[i for i,x in enumerate(v) if x!=0]
                    sw=[i for i,x in enumerate(w) if x!=0]
                    newres = dict(res)
                    for a in su:
                        for b in sv:
                            for c in sw:
                                key=(a,b,c)
                                newres[key]=newres.get(key,0)+coeff*u[a]*v[b]*w[c]
                    # norm: sum of squares of (residual - target)
                    # target is 1 at triples, 0 else
                    norm = 0
                    for key,val in newres.items():
                        tgt = F(1) if key in triple_set else F(0)
                        diff = val - tgt
                        norm += diff*diff
                    # also account for missing triples (target 1 but res 0)
                    for (a,b,c) in triples:
                        if (a,b,c) not in newres:
                            norm += F(1)
                    if best_norm is None or norm < best_norm:
                        best_norm = norm; best_cand = (d, coeff)
                if time.time() > DEADLINE:
                    return None
            if best_cand is None:
                return None
            d, coeff = best_cand
            u,v,w = cands[d]
            su=[i for i,x in enumerate(u) if x!=0]
            sv=[i for i,x in enumerate(v) if x!=0]
            sw=[i for i,x in enumerate(w) if x!=0]
            for a in su:
                for b in sv:
                    for c in sw:
                        key=(a,b,c)
                        res[key]=res.get(key,0)+coeff*u[a]*v[b]*w[c]
            chosen.append((d, coeff))
            if best_norm == 0:
                break
        # verify
        terms = []
        for d, coeff in chosen:
            u,v,w = cands[d]
            terms.append(( [x*coeff for x in u], v, w ))
        if verify_terms(terms):
            r = len(terms)
            U=[[0]*r for _ in range(nm)]
            V=[[0]*r for _ in range(mp)]
            W=[[0]*r for _ in range(np_)]
            for c,(u,v,w) in enumerate(terms):
                for i in range(nm): U[i][c]=u[i]
                for i in range(mp): V[i][c]=v[i]
                for i in range(np_): W[i][c]=w[i]
            return U,V,W
        return None

    # Try greedy with various seeds and max_terms
    for max_terms in [T, T-1, T-2, T-5, T-10, T-20, T-40]:
        if time.time() > DEADLINE:
            break
        for seed in range(20):
            if time.time() > DEADLINE:
                break
            res = greedy_search(max_terms, seed)
            if res is not None:
                U,V,W = res
                r = len(U[0])
                if r < best[0][0].__len__():
                    best = (U,V,W)
                    if r <= T-1:
                        return {"U":U,"V":V,"W":W}
    return {"U":best[0],"V":best[1],"W":best[2]}
# EVOLVE-BLOCK-END
