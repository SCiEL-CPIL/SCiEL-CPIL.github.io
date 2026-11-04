import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook
    rs = n*m*p
    Us = np.zeros((n*m, rs)); Vs = np.zeros((m*p, rs)); Ws = np.zeros((p*n, rs))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Us[i*m+j,c]=1; Vs[j*p+k,c]=1; Ws[k*n+i,c]=1; c+=1
    bestU, bestV, bestW, bestR = Us, Vs, Ws, rs
    
    for tr in range(55, rs+1):
        if time.time() > deadline - 25: break
        found = False
        for trial in range(30):
            if time.time() > deadline - 8: break
            np.random.seed(tr*1000+trial)
            U = np.random.randn(n*m,tr)*0.5; V = np.random.randn(m*p,tr)*0.5; W = np.random.randn(p*n,tr)*0.5
            for it in range(800):
                if time.time() > deadline - 5: break
                VW = (V[:,:,None]*W[:,None,:]).transpose(2,0,1).reshape(tr,-1)
                U = np.linalg.lstsq(VW.T, T.reshape(n*m,-1).T, rcond=None)[0].T
                UW = (U[:,:,None]*W[:,None,:]).transpose(2,0,1).reshape(tr,-1)
                V = np.linalg.lstsq(UW.T, T.transpose(1,0,2).reshape(m*p,-1).T, rcond=None)[0].T
                UV = (U[:,:,None]*V[:,None,:]).transpose(2,0,1).reshape(tr,-1)
                W = np.linalg.lstsq(UV.T, T.transpose(2,0,1).reshape(p*n,-1).T, rcond=None)[0].T
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(recon-T))
            if err < 1e-6:
                U2 = np.round(U*2)/2; V2 = np.round(V*2)/2; W2 = np.round(W*2)/2
                r2 = np.einsum('ar,br,cr->abc', U2, V2, W2)
                if np.max(np.abs(r2-T)) < 1e-9:
                    bestU, bestV, bestW, bestR = U2, V2, W2, tr
                    found = True
                    break
                elif err < 1e-9:
                    bestU, bestV, bestW, bestR = U, V, W, tr
                    found = True
                    break
        if found: break
    
    return {"U": bestU.tolist(), "V": bestV.tolist(), "W": bestW.tolist()}
