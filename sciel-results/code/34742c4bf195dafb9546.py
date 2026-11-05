"""Tensor decomposition for 3x4 * 4x7 matrix multiplication.
Strategy: randomized sparse support search + linear-system refinement,
combined with the deterministic block-combination baseline.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_222():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1; W[1][0]=-1; W[2][0]=-1
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[2][1]=1; W[3][1]=1
    U[0][2]=1
    V[2][2]=1; V[3][2]=-1
    W[0][2]=1; W[1][2]=1
    U[3][3]=1
    V[0][3]=1; V[1][3]=-1
    W[2][3]=1; W[3][3]=1
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[0][4]=1; W[1][4]=1
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[2][5]=1; W[3][5]=-1
    U[1][6]=1; U[3][6]=-1
    V[2][6]=1; V[3][6]=1
    W[0][6]=1; W[1][6]=-1
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col]=1; V[j*p+k][col]=1; W[k*n+i][col]=1
                col+=1
    return U, V, W

def _block_combine_n(U1,V1,W1,U2,V2,W2,n1,n2,m,p):
    n=n1+n2; r1=len(W1[0]); r2=len(W2[0]); rank=r1+r2
    U=[[0]*rank for _ in range(n*m)]; V=[[0]*rank for _ in range(m*p)]; W=[[0]*rank for _ in range(p*n)]
    for i in range(n1):
        for j in range(m):
            for c in range(r1): U[i*m+j][c]=U1[i*m+j][c]
    for j in range(m):
        for k in range(p):
            for c in range(r1): V[j*p+k][c]=V1[j*p+k][c]
    for k in range(p):
        for i in range(n1):
            for c in range(r1): W[k*n+i][c]=W1[k*n1+i][c]
    for i in range(n2):
        for j in range(m):
            for c in range(r2): U[(n1+i)*m+j][r1+c]=U2[i*m+j][c]
    for j in range(m):
        for k in range(p):
            for c in range(r2): V[j*p+k][r1+c]=V2[j*p+k][c]
    for k in range(p):
        for i in range(n2):
            for c in range(r2): W[k*n+(n1+i)][r1+c]=W2[k*n2+i][c]
    return U,V,W

def _block_combine_m(U1,V1,W1,U2,V2,W2,n,m1,m2,p):
    m=m1+m2; r1=len(W1[0]); r2=len(W2[0]); rank=r1+r2
    U=[[0]*rank for _ in range(n*m)]; V=[[0]*rank for _ in range(m*p)]; W=[[0]*rank for _ in range(p*n)]
    for i in range(n):
        for j in range(m1):
            for c in range(r1): U[i*m+j][c]=U1[i*m1+j][c]
    for j in range(m1):
        for k in range(p):
            for c in range(r1): V[j*p+k][c]=V1[j*p+k][c]
    for k in range(p):
        for i in range(n):
            for c in range(r1): W[k*n+i][c]=W1[k*n+i][c]
    for i in range(n):
        for j in range(m2):
            for c in range(r2): U[i*m+(m1+j)][r1+c]=U2[i*m2+j][c]
    for j in range(m2):
        for k in range(p):
            for c in range(r2): V[(m1+j)*p+k][r1+c]=V2[j*p+k][c]
    for k in range(p):
        for i in range(n):
            for c in range(r2): W[k*n+i][r1+c]=W2[k*n+i][c]
    return U,V,W

def _block_combine_p(U1,V1,W1,U2,V2,W2,n,m,p1,p2):
    p=p1+p2; r1=len(W1[0]); r2=len(W2[0]); rank=r1+r2
    U=[[0]*rank for _ in range(n*m)]; V=[[0]*rank for _ in range(m*p)]; W=[[0]*rank for _ in range(p*n)]
    for i in range(n):
        for j in range(m):
            for c in range(r1): U[i*m+j][c]=U1[i*m+j][c]
    for j in range(m):
        for k in range(p1):
            for c in range(r1): V[j*p+k][c]=V1[j*p1+k][c]
    for k in range(p1):
        for i in range(n):
            for c in range(r1): W[k*n+i][c]=W1[k*n+i][c]
    for i in range(n):
        for j in range(m):
            for c in range(r2): U[i*m+j][r1+c]=U2[i*m+j][c]
    for j in range(m):
        for k in range(p2):
            for c in range(r2): V[j*p+(p1+k)][r1+c]=V2[j*p2+k][c]
    for k in range(p2):
        for i in range(n):
            for c in range(r2): W[(p1+k)*n+i][r1+c]=W2[k*n+i][c]
    return U,V,W

def _decomp_m347():
    def m32p(p):
        if p==1: return _schoolbook_nm_p(2,2,1)
        if p==2: return _strassen_222()
        U1,V1,W1=m32p(2); U2,V2,W2=m32p(p-2)
        return _block_combine_p(U1,V1,W1,U2,V2,W2,2,2,2,p-2)
    def m12p(p): return _schoolbook_nm_p(1,2,p)
    def m32p_full(p):
        U1,V1,W1=m32p(p); U2,V2,W2=m12p(p)
        return _block_combine_n(U1,V1,W1,U2,V2,W2,2,1,2,p)
    def m34p(p):
        U1,V1,W1=m32p_full(p); U2,V2,W2=m32p_full(p)
        return _block_combine_m(U1,V1,W1,U2,V2,W2,3,2,2,p)
    U1,V1,W1=m34p(2); U2,V2,W2=m34p(2); U3,V3,W3=m34p(2); U4,V4,W4=m34p(1)
    Ua,Va,Wa=_block_combine_p(U1,V1,W1,U2,V2,W2,3,4,2,2)
    Ub,Vb,Wb=_block_combine_p(U3,V3,W3,U4,V4,W4,3,4,2,1)
    return _block_combine_p(Ua,Va,Wa,Ub,Vb,Wb,3,4,4,3)

def _verify(U,V,W,n,m,p,tol=1e-7):
    rank=len(W[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s=0
                for c in range(rank):
                    s+=U[i*m+j][c]*V[j*p+k][c]*W[k*n+i][c]
                if abs(s-1)>tol: return False
    return True

def _check_half_integer(U,V,W):
    for mat in (U,V,W):
        for row in mat:
            for v in row:
                if v==0: continue
                if abs(v*2-round(v*2))>1e-9: return False
    return True

def _refine_support(U,V,W,n,m,p,rank,tol=1e-8):
    """Given a valid sparse support, solve for exact coefficients."""
    eqs=[]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row=[]
                for c in range(rank):
                    if U[i*m+j][c]!=0 and V[j*p+k][c]!=0 and W[k*n+i][c]!=0:
                        row.append(c)
                eqs.append((row,1.0))
    A=np.zeros((len(eqs),rank))
    b=np.zeros(len(eqs))
    for idx,(row,target) in enumerate(eqs):
        for c in row:
            A[idx,c]=1.0
        b[idx]=target
    # Solve least-squares (system is overdetermined but consistent if support is valid)
    x,_,_,_=np.linalg.lstsq(A,b,rcond=None)
    # Check residual
    if np.max(np.abs(A@x-b))>tol: return None
    # Check sparsity: non-support entries should be ~0
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                if U[i*m+j][c]==0 and abs(x[c])>tol: return None
        for j in range(m):
            for k in range(p):
                if V[j*p+k][c]==0 and abs(x[c])>tol: return None
        for k in range(p):
            for i in range(n):
                if W[k*n+i][c]==0 and abs(x[c])>tol: return None
    # Round to half-integers
    x_rounded=np.round(x*2)/2
    if np.max(np.abs(A@x_rounded-b))>1e-6: return None
    # Build result
    Ur=[[0]*rank for _ in range(n*m)]
    Vr=[[0]*rank for _ in range(m*p)]
    Wr=[[0]*rank for _ in range(p*n)]
    for c in range(rank):
        val=x_rounded[c]
        if abs(val)<1e-9: continue
        for i in range(n):
            for j in range(m):
                if U[i*m+j][c]!=0: Ur[i*m+j][c]=val
        for j in range(m):
            for k in range(p):
                if V[j*p+k][c]!=0: Vr[j*p+k][c]=val
        for k in range(p):
            for i in range(n):
                if W[k*n+i][c]!=0: Wr[k*n+i][c]=val
    if not _verify(Ur,Vr,Wr,n,m,p): return None
    return Ur,Vr,Wr

def _random_sparse_search(n,m,p,max_rank,iterations,time_budget,seed=42):
    """Random sparse support search for rank < max_rank."""
    rng=random.Random(seed)
    start=time.time()
    best=None
    best_rank=max_rank
    for it in range(iterations):
        if time.time()-start>time_budget: break
        if time.time()>140: break
        rank=rng.randint(max(20,max_rank-6),max_rank-1)
        # Generate random sparse support
        U=np.zeros((n*m,rank))
        V=np.zeros((m*p,rank))
        W=np.zeros((p*n,rank))
        for c in range(rank):
            # Each term touches a few indices
            nu=rng.randint(1,3)
            nv=rng.randint(1,3)
            nw=rng.randint(1,3)
            u_idx=rng.sample(range(n*m),nu)
            v_idx=rng.sample(range(m*p),nv)
            w_idx=rng.sample(range(p*n),nw)
            for idx in u_idx: U[idx,c]=rng.choice([1,-1,0.5,-0.5])
            for idx in v_idx: V[idx,c]=rng.choice([1,-1,0.5,-0.5])
            for idx in w_idx: W[idx,c]=rng.choice([1,-1,0.5,-0.5])
        # Quick check: does the tensor equation hold?
        # T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1
        # Check a random subset of equations first
        eqs=rng.sample(range(n*m*p),min(50,n*m*p))
        valid=True
        for eq in eqs:
            k=eq//p; rem=eq%p; i=rem//m; j=rem%m
            s=0
            for c in range(rank):
                uv=U[i*m+j,c]*V[j*p+k,c]
                if uv!=0: s+=uv*W[k*n+i,c]
            if abs(s-1)>1e-6: valid=False; break
        if not valid: continue
        # Full verification
        full_ok=True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s=0
                    for c in range(rank):
                        s+=U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                    if abs(s-1)>1e-6: full_ok=False; break
                if not full_ok: break
            if not full_ok: break
        if not full_ok: continue
        # Convert to list and try refinement
        Ul=U.tolist(); Vl=V.tolist(); Wl=W.tolist()
        if rank<best_rank:
            best=(Ul,Vl,Wl); best_rank=rank
    return best,best_rank

def _structured_overlap_search(n,m,p,base_rank,time_budget,seed=123):
    """Try to find overlapping Strassen-like structures by perturbing the base decomposition."""
    rng=random.Random(seed)
    start=time.time()
    # Start from the deterministic decomposition
    U,V,W=_decomp_m347()
    base_r=len(W[0])
    # Try removing columns and re-solving
    for trial in range(50):
        if time.time()-start>time_budget: break
        if time.time()>140: break
        # Try rank = base_r - 1 - (trial%3)
        target_rank=base_r-1-(trial%3)
        if target_rank<20: continue
        # Randomly select which columns to keep (drop some)
        cols=list(range(base_r))
        rng.shuffle(cols)
        kept=cols[:target_rank]
        Uk=np.array(U)[:,kept]; Vk=np.array(V)[:,kept]; Wk=np.array(W)[:,kept]
        # Now solve: for each (i,j,k), sum_c Uk[i*m+j,c]*Vk[j*p+k,c]*Wk[k*n+i,c] = 1
        # This is a system in the W values (treating U,V as fixed)
        # Actually all three are unknowns... but if we fix U,V we can solve for W
        # Let's fix U and V, solve for W
        # Equation: for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # Unknowns: W[k*n+i,c] for all k,i,c
        # This is a linear system!
        A=np.zeros((n*m*p, p*n*target_rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx=i*m*p+j*p+k
                    for c in range(target_rank):
                        coeff=Uk[i*m+j,c]*Vk[j*p+k,c]
                        if coeff!=0:
                            var_idx=(k*n+i)*target_rank+c
                            A[eq_idx,var_idx]=coeff
        b=np.ones(n*m*p)
        try:
            x,_,_,_=np.linalg.lstsq(A,b,rcond=None)
        except:
            continue
        if np.max(np.abs(A@x-b))>1e-6: continue
        # Check half-integer
        x_r=np.round(x*2)/2
        if np.max(np.abs(A@x_r-b))>1e-5: continue
        # Build W
        Wr=np.zeros((p*n,target_rank))
        for k in range(p):
            for i in range(n):
                for c in range(target_rank):
                    Wr[k*n+i,c]=x_r[(k*n+i)*target_rank+c]
        # Verify full decomposition
        Ul=Uk.tolist(); Vl=Vk.tolist(); Wrl=Wr.tolist()
        if _verify(Ul,Vl,Wrl,n,m,p):
            return Ul,Vl,Wrl,target_rank
    return None,None

def construct():
    start_time=time.time()
    deadline=start_time+140
    n,m,p=DIMENSIONS
    
    # Baseline: schoolbook
    U_sb,V_sb,W_sb=_schoolbook_nm_p(n,m,p)
    best={"U":U_sb,"V":V_sb,"W":W_sb}
    best_rank=n*m*p
    
    # Deterministic decomposition
    try:
        U,V,W=_decomp_m347()
        r=len(W[0])
        if r<best_rank and _verify(U,V,W,n,m,p):
            best={"U":U,"V":V,"W":W}; best_rank=r
    except: pass
    
    if time.time()>deadline-30: return best
    
    # Structured overlap search (fix U,V, solve for W)
    try:
        Ur,Vr,Wr,rr=_structured_overlap_search(n,m,p,best_rank,30)
        if Ur is not None and rr<best_rank:
            best={"U":Ur,"V":Vr,"W":Wr}; best_rank=rr
    except: pass
    
    if time.time()>deadline-20: return best
    
    # Random sparse search
    try:
        result,rr=_random_sparse_search(n,m,p,best_rank,200,20,seed=42)
        if result is not None and rr<best_rank:
            U,V,W=result
            if _verify(U,V,W,n,m,p):
                best={"U":U,"V":V,"W":W}; best_rank=rr
    except: pass
    
    if time.time()>deadline-10: return best
    
    # Second round random search with different seed
    try:
        result,rr=_random_sparse_search(n,m,p,best_rank,100,10,seed=999)
        if result is not None and rr<best_rank:
            U,V,W=result
            if _verify(U,V,W,n,m,p):
                best={"U":U,"V":V,"W":W}; best_rank=rr
    except: pass
    
    return best

# EVOLVE-BLOCK-END
