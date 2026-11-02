"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2():
    """Return 7 Strassen 2x2x2 rank-1 terms as sparse dicts."""
    return [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]

def _build_2x2x3():
    """Return 11 rank-1 terms for 2x2 x 2x3 -> 2x3."""
    terms = []
    for s in range(7):
        u, v, w = _strassen_2x2x2()[s]
        terms.append((u, v, w))
    terms.append(({(0,0):1}, {(0,2):1}, {(2,0):1}))
    terms.append(({(0,1):1}, {(1,2):1}, {(2,0):1}))
    terms.append(({(1,0):1}, {(0,2):1}, {(2,1):1}))
    terms.append(({(1,1):1}, {(1,2):1}, {(2,1):1}))
    return terms

def _tensor_product_term(u1, v1, w1, u2, v2, w2):
    """Compute tensor product of two rank-1 terms for 2x2x2 x 2x2x3 -> 4x4x6."""
    U_g, V_g, W_g = {}, {}, {}
    for (i1,j1),c1 in u1.items():
        for (i2,j2),c2 in u2.items():
            k=(2*i1+i2, 2*j1+j2)
            U_g[k]=U_g.get(k,0)+c1*c2
    for (j1,k1),c1 in v1.items():
        for (j2,k2),c2 in v2.items():
            k=(2*j1+j2, 3*k1+k2)
            V_g[k]=V_g.get(k,0)+c1*c2
    for (k1,i1),c1 in w1.items():
        for (k2,i2),c2 in w2.items():
            k=(3*k1+k2, 2*i1+i2)
            W_g[k]=W_g.get(k,0)+c1*c2
    return U_g, V_g, W_g

def _build_rank77():
    n,m,p = 4,4,6
    s22 = _strassen_2x2x2()
    t223 = _build_2x2x3()
    terms = []
    for s in range(7):
        for t in range(11):
            terms.append(_tensor_product_term(s22[s][0],s22[s][1],s22[s][2],
                                              t223[t][0],t223[t][1],t223[t][2]))
    rank = len(terms)
    U=np.zeros((n*m,rank)); V=np.zeros((m*p,rank)); W=np.zeros((p*n,rank))
    for r,(ud,vd,wd) in enumerate(terms):
        for (i,j),c in ud.items(): U[i*m+j,r]=c
        for (j,k),c in vd.items(): V[j*p+k,r]=c
        for (k,i),c in wd.items(): W[k*n+i,r]=c
    return U,V,W

def _verify(U,V,W,n,m,p,atol=1e-10):
    np.random.seed(42)
    A=np.random.randn(n,m); B=np.random.randn(m,p)
    C_exp=A@B
    C=np.zeros((n,p))
    for r in range(U.shape[1]):
        a=np.sum(A*U[:,r].reshape(n,m))
        b=np.sum(B*V[:,r].reshape(m,p))
        C+=a*b*W[:,r].reshape(p,n).T
    return np.allclose(C,C_exp,atol=atol)

def _try_reduce_rank(U,V,W,n,m,p,max_time=100.0):
    """Try to reduce rank by eliminating redundant terms via linear algebra."""
    start=time.time()
    rank=U.shape[1]
    best_rank=rank
    best_U,best_V,best_W=U.copy(),V.copy(),W.copy()
    
    # Check for linear dependencies in U, V, W columns
    # If columns are linearly dependent, we can reduce rank
    for attempt in range(5):
        if time.time()-start>max_time: break
        # Find null space of U (as a matrix)
        try:
            _,s,vt=np.linalg.svd(U,full_matrices=False)
            # Find singular values close to zero
            tol=1e-12
            null_idx=[i for i in range(len(s)) if s[i]<tol]
            if null_idx:
                # Rank of U is less than rank, try to reduce
                r_u=np.sum(s>tol)
                if r_u<rank:
                    # Use SVD to get reduced basis
                    U_red=U[:,s>tol]
                    # Similarly for V and W - but they must be consistent
                    # This approach only works if all three have same null space
                    pass
        except:
            pass
        break  # This is complex; fall through to pair-merge
    
    # Try pair-merge: for each pair (i,j), check if U_i+U_j, V_i+V_j, W_i+W_j can be
    # absorbed. Actually, we check if removing term i and adjusting others works.
    # Simpler: check if any two terms are "proportional" (can be merged)
    for i in range(rank-1):
        if time.time()-start>max_time: break
        for j in range(i+1,rank):
            # Check if (U_i,V_i,W_i) and (U_j,V_j,W_j) are proportional
            # i.e., U_i = a*U_j, V_i = a*V_j, W_i = a*W_j for some scalar a
            # This is very restrictive; unlikely to find
            Ui=U[:,i]; Vj=V[:,j]
            if np.linalg.norm(Vj)>1e-15:
                a=np.dot(Ui,Vj)/np.dot(Vj,Vj)
                if np.allclose(Ui,a*U[:,j],atol=1e-10) and np.allclose(V[:,i],a*Vj,atol=1e-10) and np.allclose(W[:,i],a*W[:,j],atol=1e-10):
                    # Merge: keep term j, scale it
                    U[:,j]*=(1+a)
                    V[:,j]*=(1+a)
                    W[:,j]*=(1+a)
                    U=np.delete(U,i,axis=1)
                    V=np.delete(V,i,axis=1)
                    W=np.delete(W,i,axis=1)
                    rank-=1
                    best_rank=rank
                    best_U,best_V,best_W=U.copy(),V.copy(),W.copy()
                    break
        if rank<best_rank: break
    
    return best_U,best_V,best_W,best_rank

def construct():
    start_time=time.time()
    n,m,p=DIMENSIONS
    deadline=start_time+110.0
    
    U,V,W=_build_rank77()
    if not _verify(U,V,W,n,m,p):
        rank=n*m*p
        U=np.zeros((n*m,rank));V=np.zeros((m*p,rank));W=np.zeros((p*n,rank))
        col=0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j,col]=1;V[j*p+k,col]=1;W[k*n+i,col]=1;col+=1
        return {"U":U.tolist(),"V":V.tolist(),"W":W.tolist()}
    
    # Try to reduce rank
    if time.time()<deadline-20:
        U2,V2,W2,rank2=_try_reduce_rank(U,V,W,n,m,p,max_time=min(80,deadline-time.time()-10))
        if rank2<U.shape[1] and _verify(U2,V2,W2,n,m,p):
            U,V,W=U2,V2,W2
    
    return {"U":U.tolist(),"V":V.tolist(),"W":W.tolist()}
# EVOLVE-BLOCK-END
