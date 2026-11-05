import time
import numpy as np
import itertools

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    # The task is to find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x8 matrix.
    # The output factor W follows the official trace convention W[k*n+i, r].
    # The tensor T is defined such that T[i*m+j, j*p+k, k*n+i] = 1.
    
    # Schoolbook rank is n*m*p = 3*4*8 = 96.
    # We will try to find a better decomposition using known small-block decompositions.
    
    # Strategy:
    # 1. Define a library of known optimal decompositions for small rectangular matrix multiplications.
    #    Specifically, we will use:
    #    - 1x1x1: rank 1 (trivial)
    #    - 2x2x2: rank 7 (Strassen)
    #    - 2x2x1, 2x1x2, 1x2x2: rank 3 (Laderman-like or simple optimization)
    #    - 2x2x3, 2x3x2, 3x2x2: rank 11 (known optimal for these small cases)
    #    - 3x3x3: rank 23 (Winograd) or 21 (Bini et al)? 23 is safe.
    #    
    # 2. We will search over partitions of the dimensions n, m, p into blocks.
    #    However, as noted in the previous attempt, simply summing the ranks of independent blocks
    #    is not straightforward because the tensor is not a direct sum unless the blocks are disjoint
    #    in the "inner" dimension j.
    #    
    #    Actually, the matrix multiplication tensor M_{n,m,p} can be decomposed by partitioning the
    #    middle dimension m into m1, m2, ...
    #    If we partition m into m1, m2, then:
    #    C = A B = A1 B1 + A2 B2
    #    This is a direct sum of two MM problems: (n, m1, p) and (n, m2, p).
    #    The rank is rank(n, m1, p) + rank(n, m2, p).
    #    
    #    Similarly, we can partition n or p.
    #    If we partition n into n1, n2:
    #    C = A B. A is (n1+n2) x m.
    #    C1 = A1 B (n1 x p)
    #    C2 = A2 B (n2 x p)
    #    This is a direct sum of (n1, m, p) and (n2, m, p).
    #    
    #    If we partition p into p1, p2:
    #    C = A B. B is m x (p1+p2).
    #    C1 = A B1 (n x p1)
    #    C2 = A B2 (n x p2)
    #    This is a direct sum of (n, m, p1) and (n, m, p2).
    #    
    #    So, we can recursively decompose the problem by splitting any of the three dimensions.
    #    We can use dynamic programming to find the minimal rank for each sub-problem (ni, mj, pk).
    #    
    #    Let R[ni, mj, pk] be the minimal rank for multiplying an ni x mj matrix by a mj x pk matrix.
    #    R[1,1,1] = 1.
    #    R[ni, mj, pk] = min(
    #        R[ni, mj, pk] (schoolbook: ni*mj*pk),
    #        min_{mj1+ mj2 = mj} (R[ni, mj1, pk] + R[ni, mj2, pk]),
    #        min_{ni1+ ni2 = ni} (R[ni1, mj, pk] + R[ni2, mj, pk]),
    #        min_{pk1+ pk2 = pk} (R[ni, mj, pk1] + R[ni, mj, pk2])
    #    )
    #    
    #    We will compute R for all ni <= 3, mj <= 4, pk <= 8.
    #    Then, we will construct the actual U, V, W matrices for the optimal decomposition.
    
    def get_partitions(n):
        parts = []
        def gen(rem, curr):
            if rem == 0:
                parts.append(tuple(curr))
                return
            for x in range(1, rem + 1):
                gen(rem - x, curr + [x])
        gen(n, [])
        return parts

    # We will use a dictionary to store the minimal rank for each (ni, mj, pk).
    R = {}
    
    # Base cases
    for ni in range(1, n + 1):
        for mj in range(1, m + 1):
            for pk in range(1, p + 1):
                # Initialize with schoolbook
                R[(ni, mj, pk)] = ni * mj * pk
                
                # Try splitting mj
                for mj1 in range(1, mj):
                    mj2 = mj - mj1
                    if (ni, mj1, pk) in R and (ni, mj2, pk) in R:
                        cost = R[(ni, mj1, pk)] + R[(ni, mj2, pk)]
                        if cost < R[(ni, mj, pk)]:
                            R[(ni, mj, pk)] = cost
                            
                # Try splitting ni
                for ni1 in range(1, ni):
                    ni2 = ni - ni1
                    if (ni1, mj, pk) in R and (ni2, mj, pk) in R:
                        cost = R[(ni1, mj, pk)] + R[(ni2, mj, pk)]
                        if cost < R[(ni, mj, pk)]:
                            R[(ni, mj, pk)] = cost
                            
                # Try splitting pk
                for pk1 in range(1, pk):
                    pk2 = pk - pk1
                    if (ni, mj, pk1) in R and (ni, mj, pk2) in R:
                        cost = R[(ni, mj, pk1)] + R[(ni, mj, pk2)]
                        if cost < R[(ni, mj, pk)]:
                            R[(ni, mj, pk)] = cost
    
    # Now, we have the minimal rank for each sub-problem.
    # We need to construct the actual U, V, W for the top-level problem (n, m, p).
    # We will recursively build the decomposition.
    
    def build_decomp(ni, mj, pk, i_off, j_off, k_off):
        """
        Returns (U, V, W) for the sub-problem (ni, mj, pk) with offsets.
        U has shape (ni*mj, rank), V has shape (mj*pk, rank), W has shape (pk*ni, rank).
        The rows are indexed relative to the sub-problem.
        """
        rank = R[(ni, mj, pk)]
        
        # Check if we should split
        best_split = None
        best_cost = rank
        
        # Check mj split
        for mj1 in range(1, mj):
            mj2 = mj - mj1
            cost = R[(ni, mj1, pk)] + R[(ni, mj2, pk)]
            if cost < best_cost:
                best_cost = cost
                best_split = ('mj', mj1, mj2)
                
        # Check ni split
        for ni1 in range(1, ni):
            ni2 = ni - ni1
            cost = R[(ni1, mj, pk)] + R[(ni2, mj, pk)]
            if cost < best_cost:
                best_cost = cost
                best_split = ('ni', ni1, ni2)
                
        # Check pk split
        for pk1 in range(1, pk):
            pk2 = pk - pk1
            cost = R[(ni, mj, pk1)] + R[(ni, mj, pk2)]
            if cost < best_cost:
                best_cost = cost
                best_split = ('pk', pk1, pk2)
                
        if best_split is not None:
            # Recursive case
            U_list = []
            V_list = []
            W_list = []
            
            if best_split[0] == 'mj':
                _, mj1, mj2 = best_split
                U1, V1, W1 = build_decomp(ni, mj1, pk, i_off, j_off, k_off)
                U2, V2, W2 = build_decomp(ni, mj2, pk, i_off, j_off + mj1, k_off)
                U_list.extend([U1, U2])
                V_list.extend([V1, V2])
                W_list.extend([W1, W2])
                
            elif best_split[0] == 'ni':
                _, ni1, ni2 = best_split
                U1, V1, W1 = build_decomp(ni1, mj, pk, i_off, j_off, k_off)
                U2, V2, W2 = build_decomp(ni2, mj, pk, i_off + ni1, j_off, k_off)
                U_list.extend([U1, U2])
                V_list.extend([V1, V2])
                W_list.extend([W1, W2])
                
            elif best_split[0] == 'pk':
                _, pk1, pk2 = best_split
                U1, V1, W1 = build_decomp(ni, mj, pk1, i_off, j_off, k_off)
                U2, V2, W2 = build_decomp(ni, mj, pk2, i_off, j_off, k_off + pk1)
                U_list.extend([U1, U2])
                V_list.extend([V1, V2])
                W_list.extend([W1, W2])
                
            U = np.hstack(U_list)
            V = np.hstack(V_list)
            W = np.hstack(W_list)
            return U, V, W
        else:
            # Base case: use schoolbook
            U = np.zeros((ni * mj, rank))
            V = np.zeros((mj * pk, rank))
            W = np.zeros((pk * ni, rank))
            col = 0
            for i in range(ni):
                for j in range(mj):
                    for k in range(pk):
                        U[i * mj + j, col] = 1
                        V[j * pk + k, col] = 1
                        W[k * ni + i, col] = 1
                        col += 1
            return U, V, W

    # Build the top-level decomposition
    U, V, W = build_decomp(n, m, p, 0, 0, 0)
    
    # Verify
    S = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Create target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
                
    if not np.allclose(S, T, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
