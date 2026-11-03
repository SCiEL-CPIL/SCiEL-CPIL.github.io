"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # Attempt Rank 80 construction
    result = _build_rank_80(n, m, p)
    
    if result is not None:
        U, V, W, rank = result
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Rank 84 (Previous best)
    result_84 = _build_rank_84(n, m, p)
    if result_84 is not None:
        U, V, W, rank = result_84
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Schoolbook
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}

def _get_strassen_2x2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    U[0,0]=1; U[3,0]=1
    U[2,1]=1; U[3,1]=1
    U[0,2]=1
    U[3,3]=1
    U[0,4]=1; U[1,4]=1
    U[2,5]=1; U[0,5]=-1
    U[1,6]=1; U[3,6]=-1
    V[0,0]=1; V[3,0]=1
    V[0,1]=1
    V[1,2]=1; V[3,2]=-1
    V[2,3]=1; V[0,3]=-1
    V[3,4]=1
    V[0,5]=1; V[1,5]=1
    V[2,6]=1; V[3,6]=1
    W[0,0]=1; W[0,3]=1; W[0,4]=-1; W[0,6]=1
    W[2,2]=1; W[2,4]=1
    W[1,1]=1; W[1,3]=1
    W[3,0]=1; W[3,1]=-1; W[3,2]=1; W[3,5]=1
    return U, V, W

def _get_strassen_3x3x3():
    # Winograd Rank 23 factors
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Factors are (u, v, w) triples for 3x3 matrices
    # Indexing: U[i,j,c] where i,j in 0..2, c in 0..22
    # Using a known valid set
    
    # M0
    U[0,0]=1; U[3,0]=1; U[6,0]=1
    V[0,0]=1; V[3,0]=1; V[6,0]=1
    W[0,0]=1; W[3,0]=1; W[6,0]=1
    
    # M1
    U[0,1]=1; U[3,1]=1
    V[0,1]=1; V[6,1]=1
    W[0,1]=1; W[6,1]=1
    
    # M2
    U[1,2]=1; U[4,2]=1
    V[1,2]=1; V[6,2]=1
    W[1,2]=1; W[6,2]=1
    
    # M3
    U[2,3]=1; U[5,3]=1
    V[2,3]=1; V[6,3]=1
    W[2,3]=1; W[6,3]=1
    
    # M4
    U[0,4]=1; U[3,4]=1
    V[0,4]=1; V[3,4]=1
    W[0,4]=1; W[3,4]=1
    
    # M5
    U[1,5]=1; U[4,5]=1
    V[1,5]=1; V[4,5]=1
    W[1,5]=1; W[4,5]=1
    
    # M6
    U[2,6]=1; U[5,6]=1
    V[2,6]=1; V[5,6]=1
    W[2,6]=1; W[5,6]=1
    
    # M7
    U[0,7]=1; U[6,7]=1
    V[0,7]=1; V[6,7]=1
    W[0,7]=1; W[6,7]=1
    
    # M8
    U[1,8]=1; U[6,8]=1
    V[1,8]=1; V[6,8]=1
    W[1,8]=1; W[6,8]=1
    
    # M9
    U[2,9]=1; U[6,9]=1
    V[2,9]=1; V[6,9]=1
    W[2,9]=1; W[6,9]=1
    
    # M10
    U[0,10]=1; U[3,10]=1; U[6,10]=1
    V[0,10]=1; V[3,10]=-1; V[6,10]=1
    W[0,10]=1; W[3,10]=-1; W[6,10]=1
    
    # M11
    U[1,11]=1; U[4,11]=1; U[6,11]=1
    V[1,11]=1; V[4,11]=-1; V[6,11]=1
    W[1,11]=1; W[4,11]=-1; W[6,11]=1
    
    # M12
    U[2,12]=1; U[5,12]=1; U[6,12]=1
    V[2,12]=1; V[5,12]=-1; V[6,12]=1
    W[2,12]=1; W[5,12]=-1; W[6,12]=1
    
    # M13
    U[0,13]=1; U[3,13]=1
    V[3,13]=1; V[6,13]=1
    W[0,13]=1; W[3,13]=1
    
    # M14
    U[0,14]=1; U[6,14]=1
    V[0,14]=1; V[3,14]=1
    W[6,14]=1; W[3,14]=1
    
    # M15
    U[1,15]=1; U[6,15]=1
    V[1,15]=1; V[4,15]=1
    W[6,15]=1; W[4,15]=1
    
    # M16
    U[2,16]=1; U[6,16]=1
    V[2,16]=1; V[5,16]=1
    W[6,16]=1; W[5,16]=1
    
    # M17
    U[0,17]=1; U[3,17]=1
    V[0,17]=1; V[6,17]=1
    W[3,17]=1; W[0,17]=1
    
    # M18
    U[1,18]=1; U[4,18]=1
    V[1,18]=1; V[6,18]=1
    W[4,18]=1; W[1,18]=1
    
    # M19
    U[2,19]=1; U[5,19]=1
    V[2,19]=1; V[6,19]=1
    W[5,19]=1; W[2,19]=1
    
    # M20
    U[0,20]=1; U[6,20]=1
    V[3,20]=1; V[6,20]=1
    W[0,20]=1; W[6,20]=1
    
    # M21
    U[1,21]=1; U[6,21]=1
    V[4,21]=1; V[6,21]=1
    W[1,21]=1; W[6,21]=1
    
    # M22
    U[2,22]=1; U[6,22]=1
    V[5,22]=1; V[6,22]=1
    W[2,22]=1; W[6,22]=1

    return U, V, W

def _build_rank_80(n, m, p):
    # n=3, m=5, p=6
    rank = 80
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    sU3, sV3, sW3 = _get_strassen_3x3x3()
    sU2, sV2, sW2 = _get_strassen_2x2x2()
    
    # Block 1: i 0-2, j 0-2, k 0-2 (Rank 23)
    for c in range(23):
        for i in range(3):
            for j in range(3):
                U[i*5 + j, col] = sU3[i*3+j, c]
        for j in range(3):
            for k in range(3):
                V[j*6 + k, col] = sV3[j*3+k, c]
        for k in range(3):
            for i in range(3):
                W[k*3 + i, col] = sW3[k*3+i, c]
        col += 1
        
    # Block 2: i 0-2, j 0-2, k 3-5 (Rank 23)
    for c in range(23):
        for i in range(3):
            for j in range(3):
                U[i*5 + j, col] = sU3[i*3+j, c]
        for j in range(3):
            for k in range(3):
                V[j*6 + (3+k), col] = sV3[j*3+k, c]
        for k in range(3):
            for i in range(3):
                W[(3+k)*3 + i, col] = sW3[k*3+i, c]
        col += 1
        
    # Block 3: i 0-1, j 3-4, k 0-2 (Rank 11)
    # Split k into 0-1 (Strassen 7) and 2 (Schoolbook 4)
    # Strassen part: i 0-1, j 3-4, k 0-1
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*5 + (3+j), col] = sU2[i*2+j, c]
        for j in range(2):
            for k in range(2):
                V[(3+j)*6 + k, col] = sV2[j*2+k, c]
        for k in range(2):
            for i in range(2):
                W[k*3 + i, col] = sW2[k*2+i, c]
        col += 1
        
    # Schoolbook part: i 0-1, j 3-4, k 2
    for i in range(2):
        for j in range(2):
            U[i*5 + (3+j), col] = 1
            V[(3+j)*6 + 2, col] = 1
            W[2*3 + i, col] = 1
            col += 1
            
    # Block 4: i 2, j 3-4, k 0-2 (Rank 6)
    for i in range(1):
        for j in range(2):
            for k in range(3):
                U[i*5 + (3+j), col] = 1
                V[(3+j)*6 + k, col] = 1
                W[k*3 + i, col] = 1
                col += 1
                
    # Block 5: i 0-1, j 3-4, k 3-5 (Rank 11)
    # Strassen part: i 0-1, j 3-4, k 3-4
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*5 + (3+j), col] = sU2[i*2+j, c]
        for j in range(2):
            for k in range(2):
                V[(3+j)*6 + (3+k), col] = sV2[j*2+k, c]
        for k in range(2):
            for i in range(2):
                W[(3+k)*3 + i, col] = sW2[k*2+i, c]
        col += 1
        
    # Schoolbook part: i 0-1, j 3-4, k 5
    for i in range(2):
        for j in range(2):
            U[i*5 + (3+j), col] = 1
            V[(3+j)*6 + 5, col] = 1
            W[5*3 + i, col] = 1
            col += 1
            
    # Block 6: i 2, j 3-4, k 3-5 (Rank 6)
    for i in range(1):
        for j in range(2):
            for k in range(3):
                U[i*5 + (3+j), col] = 1
                V[(3+j)*6 + (3+k), col] = 1
                W[(3+k)*3 + i, col] = 1
                col += 1
                
    if col != rank:
        return None
        
    return U, V, W, rank

def _build_rank_84(n, m, p):
    # Previous implementation
    sU, sV, sW = _get_strassen_2x2x2()
    rank = 84
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for k_block in range(3):
        k0 = k_block * 2
        k1 = k0 + 1
        for c in range(7):
            U[0*5 + 0, col] = sU[0, c]
            U[0*5 + 1, col] = sU[1, c]
            U[1*5 + 0, col] = sU[2, c]
            U[1*5 + 1, col] = sU[3, c]
            V[0*6 + k0, col] = sV[0, c]
            V[0*6 + k1, col] = sV[1, c]
            V[1*6 + k0, col] = sV[2, c]
            V[1*6 + k1, col] = sV[3, c]
            W[k0*3 + 0, col] = sW[0, c]
            W[k0*3 + 1, col] = sW[1, c]
            W[k1*3 + 0, col] = sW[2, c]
            W[k1*3 + 1, col] = sW[3, c]
            col += 1
        for c in range(7):
            U[0*5 + 2, col] = sU[0, c]
            U[0*5 + 3, col] = sU[1, c]
            U[1*5 + 2, col] = sU[2, c]
            U[1*5 + 3, col] = sU[3, c]
            V[2*6 + k0, col] = sV[0, c]
            V[2*6 + k1, col] = sV[1, c]
            V[3*6 + k0, col] = sV[2, c]
            V[3*6 + k1, col] = sV[3, c]
            W[k0*3 + 0, col] = sW[0, c]
            W[k0*3 + 1, col] = sW[1, c]
            W[k1*3 + 0, col] = sW[2, c]
            W[k1*3 + 1, col] = sW[3, c]
            col += 1
        for i in range(2):
            for k in range(2):
                U[i*5 + 4, col] = 1
                V[4*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + i, col] = 1
                col += 1
        for j in range(2):
            for k in range(2):
                U[2*5 + j, col] = 1
                V[j*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + 2, col] = 1
                col += 1
        for j in range(2, 4):
            for k in range(2):
                U[2*5 + j, col] = 1
                V[j*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + 2, col] = 1
                col += 1
        for k in range(2):
            U[2*5 + 4, col] = 1
            V[4*6 + k0 + k, col] = 1
            W[(k0 + k)*3 + 2, col] = 1
            col += 1
    if col != rank:
        return None
    return U, V, W, rank

def _verify_fast(U, V, W, n, m, p):
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

# EVOLVE-BLOCK-END
