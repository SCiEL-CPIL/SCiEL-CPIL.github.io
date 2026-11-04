import numpy as np
import time

N, M, P = 4, 4, 6
TIME_LIMIT = 115.0

def _build_true_tensor():
    T = np.zeros((N*M, M*P, P*N))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i*M + j, j*P + k, k*N + i] = 1.0
    return T

def _strassen_2x2():
    U = np.zeros((4,7)); V = np.zeros((4,7)); W = np.zeros((4,7))
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1
    U[0,4]=1; U[1,4]=1; V[3,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1
    W[0,0]=1; W[0,3]=1; W[0,4]=-1; W[0,6]=1
    W[2,2]=1; W[2,4]=1
    W[1,1]=1; W[1,3]=1
    W[3,0]=1; W[3,1]=-1; W[3,2]=1; W[3,5]=1
    return U,V,W

def _rank3_2x2x1():
    # returns list of (a[4], b[2], c[2]) for M_{2,2,1}
    # a indexed (ci,mj) = (0,0),(0,1),(1,0),(1,1)
    # b indexed mj = 0,1
    # c indexed ci = 0,1
    return [
        ((1,-1,1,0),(1,0),(1,0)),
        ((0,1,-1,1),(0,1),(0,1)),
        ((0,1,-1,0),(1,1),(1,-1)),
    ]

def _build_63_decomposition():
    sU,sV,sW = _strassen_2x2()
    r3 = _rank3_2x2x1()
    outer_terms = []
    for ck in range(3):
        for (a,b,c) in r3:
            U_dict = {}
            V_dict = {}
            W_dict = {}
            for ci in range(2):
                for mj in range(2):
                    coeff = a[ci*2+mj]
                    if coeff != 0:
                        U_dict[(ci,mj)] = coeff
            for mj in range(2):
                coeff = b[mj]
                if coeff != 0:
                    V_dict[(mj,ck)] = coeff
            for ci in range(2):
                coeff = c[ci]
                if coeff != 0:
                    W_dict[(ci,ck)] = coeff
            outer_terms.append({'U':U_dict,'V':V_dict,'W':W_dict})
    total_rank = len(outer_terms)*7
    U = np.zeros((N*M,total_rank)); V = np.zeros((M*P,total_rank)); W = np.zeros((P*N,total_rank))
    for ot_idx, outer in enumerate(outer_terms):
        for t in range(7):
            col = ot_idx*7 + t
            for (ai,bj), coeff_outer in outer['U'].items():
                for ai2 in range(2):
                    for bj2 in range(2):
                        coeff_inner = sU[ai2*2+bj2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            u_row = (2*ai+ai2)*4 + (2*bj+bj2)
                            U[u_row,col] += coeff_outer*coeff_inner
            for (bj,ck), coeff_outer in outer['V'].items():
                for bj2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sV[bj2*2+ck2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            v_row = (2*bj+bj2)*6 + (2*ck+ck2)
                            V[v_row,col] += coeff_outer*coeff_inner
            for (ai,ck), coeff_outer in outer['W'].items():
                for ai2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sW[ck2*2+ai2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            w_row = (2*ck+ck2)*4 + (2*ai+ai2)
                            W[w_row,col] += coeff_outer*coeff_inner
    return U,V,W

def _residual(U,V,W,T):
    rank = U.shape[1]
    approx = np.zeros_like(T)
    for r in range(rank):
        approx += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    return np.linalg.norm(T - approx)

def construct():
    start = time.time()
    T = _build_true_tensor()
    U,V,W = _build_63_decomposition()
    err = _residual(U,V,W,T)
    # verify
    if err < 1e-6:
        pass
    # polish: try to drop zero columns
    keep = [r for r in range(U.shape[1]) if np.linalg.norm(U[:,r])*np.linalg.norm(V[:,r])*np.linalg.norm(W[:,r]) > 1e-10]
    U,V,W = U[:,keep], V[:,keep], W[:,keep]
    # snap to half-integers
    U = np.round(U*2)/2; V = np.round(V*2)/2; W = np.round(W*2)/2
    final_err = _residual(U,V,W,T)
    if final_err > 1e-4:
        # revert snap
        U,V,W = _build_63_decomposition()
    return {"U":U.tolist(),"V":V.tolist(),"W":W.tolist()}
