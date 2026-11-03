"""Matrix multiplication 4x4 by 4x8 - rank-minimal tensor decomposition search."""
import numpy as np
from fractions import Fraction
from itertools import product
import sys

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Search for a low-rank exact tensor decomposition of the matrix multiplication
    tensor for 4x4 * 4x8.
    
    The tensor T is defined by:
    T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    (where n=4, m=4, p=8)
    
    We want to find rank R such that:
    T[a,b,c] = sum_{r=1}^{R} U[a,r] * V[b,r] * W[c,r]
    
    with U of shape (n*m, R), V of shape (m*p, R), W of shape (p*n, R)
    
    Strategy: Try ranks from 48 upward, and for each rank, set up a system of
    bilinear equations. We use a heuristic: fix U and V to random sparse matrices
    with rational entries, then solve for W. If no solution, try different random
    initializations.
    
    Actually, a better approach: use the fact that for matrix multiplication,
    known lower bounds and constructions exist. For 4x4*4x8, let's try to use
    known efficient algorithms or search systematically.
    
    The schoolbook rank is 4*4*8 = 128.
    
    Let's try to find a decomposition with rank < 128 using a systematic search.
    
    Approach: For a given rank R, we can try to find U, V, W by:
    1. Initialize U and V with random rational entries (sparse, with many zeros)
    2. For each (a,b,c) where T[a,b,c]=1, we need sum_r U[a,r]*V[b,r]*W[c,r] = 1
       For each (a,b,c) where T[a,b,c]=0, we need sum_r U[a,r]*V[b,r]*W[c,r] = 0
    
    This is a system of bilinear equations. One way to solve: fix U and V, then
    the equations become linear in W. For each c, W[:,c] is a vector of length R.
    The constraint for fixed (a,b) is: for all c, sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    So for fixed U,V, we have for each (a,b) a linear system in W[:,c] for all c.
    Actually, for fixed (a,b), define vector v_{ab} of length R: v_{ab}[r] = U[a,r]*V[b,r]
    Then for each c: dot(v_{ab}, W[:,c]) = T[a,b,c]
    
    So W[:,c] must satisfy: W^T * v_{ab} = T[a,b,c] for all (a,b)
    i.e., W^T * V_ab = t_{ab} where V_ab is the matrix with rows v_{ab}
    
    This is underdetermined if R > n*m*p... no wait, we have p*n equations for W[:,c] (one per c, but actually for each (a,b) we get p*n constraints... no.
    
    Let me re-think. For fixed U (n*m x R) and V (m*p x R):
    For each a in [0,n*m), b in [0,m*p), the vector v_{ab} = U[a,:] * V[b,:] (elementwise product, length R)
    Then for each c in [0,p*n): sum_r v_{ab}[r] * W[c,r] = T[a,b,c]
    
    So W^T (p*n x R) must satisfy: for each (a,b), W^T * v_{ab} = t_{ab} (a vector of length p*n)
    
    This is equivalent to: the matrix M_{ab} (p*n x R) with entries M_{ab}[c,r] = W[c,r] must satisfy M_{ab}^T v_{ab} = t_{ab}
    
    Since W is the same for all (a,b), we need: for all (a,b), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all c.
    
    This is a system of (n*m)*(m*p)*(p*n) equations in (n*m + m*p + p*n)*R unknowns.
    
    For n=4,m=4,p=8: 16*32*32 = 16384 equations, (16+32+32)*R = 80R unknowns.
    
    For R=48: 80*48 = 3840 unknowns, 16384 equations. Overdetermined.
    
    Let's try a different approach: use known results. The best known rank for 4x4*4x8...
    
    Actually, let's just try to do a random search with low rank. We'll try R from 48 to 127,
    and for each R, try random sparse U,V and solve for W.
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 32
    pn = p * n  # 32
    
    # Build the target tensor T as a dictionary: T[(a,b,c)] = 1
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[(a, b, c)] = 1
    
    # Try ranks from 48 upward
    for R in range(48, 128):
        # Try multiple random initializations
        for trial in range(50):
            # Create random sparse U and V with rational entries
            # Use entries from {-1, 0, 1} with some probability of being nonzero
            U = np.zeros((nm, R), dtype=object)
            V = np.zeros((mp, R), dtype=object)
            
            for a in range(nm):
                for r in range(R):
                    if np.random.random() < 0.3:
                        U[a, r] = Fraction(np.random.choice([-1, 1]), 1)
                    else:
                        U[a, r] = Fraction(0, 1)
            
            for b in range(mp):
                for r in range(R):
                    if np.random.random() < 0.3:
                        V[b, r] = Fraction(np.random.choice([-1, 1]), 1)
                    else:
                        V[b, r] = Fraction(0, 1)
            
            # Now solve for W
            # For each c, W[c, :] is a vector of length R
            # Constraints: for all (a,b), sum_r U[a,r]*V[b,r]*W[c,r] = T.get((a,b,c), 0)
            
            # Let's collect all constraints. For each (a,b), we have pn constraints (one per c).
            # Total constraints: nm * mp * pn = 16384
            # Unknowns: pn * R = 32 * R
            
            # Build the linear system A * W_vec = t
            # W_vec is W flattened: W[0,0], W[0,1], ..., W[0,R-1], W[1,0], ..., W[pn-1,R-1]
            # Each constraint (a,b,c) gives: sum_r U[a,r]*V[b,r]*W[c,r] = T[(a,b,c)]
            
            # This is a large system. Let's use Gaussian elimination over rationals.
            # But 16384 equations and 32*R unknowns is too large for exact rational elimination.
            
            # Alternative: use the structure. For fixed (a,b), define v_{ab} in R-space.
            # Then W^T v_{ab} = t_{ab} where t_{ab} is a vector in (pn)-space.
            
            # So W^T (pn x R) must be a right inverse-like matrix for the set of vectors {v_{ab}}.
            
            # Let's form the matrix V_all (nm*mp x R) with rows v_{ab} for all (a,b)
            # And T_all (nm*mp x pn) with entries T_all[ab, c] = T.get((a,b,c), 0)
            # Then we need: W^T * V_all^T = T_all
            # i.e., V_all * W = T_all^T
            # Where W is (pn x R) reshaped to (R x pn)^T... 
            # Actually: V_all is (nm*mp x R), W^T is (pn x R) so V_all^T is (R x nm*mp)
            # We want: for each (ab), sum_r V_all[ab,r] * W^T[:, r]... no.
            
            # Let me be more careful.
            # W is (pn x R). W^T is (R x pn).
            # For each ab (index into nm*mp), the constraint is:
            #   sum_{c=0}^{pn-1} ... no, it's: for each c, sum_r U[a,r]*V[b,r]*W[c,r] = T[(a,b,c)]
            # So: (W^T)^T * v_{ab} = t_{ab} where t_{ab} is the vector (T[(a,b,0)], ..., T[(a,b,pn-1)])
            # i.e., W^T * v_{ab}... no. W is (pn x R), so W[c,r] is row c, col r.
            # sum_r W[c,r] * (U[a,r]*V[b,r]) = T[(a,b,c)]
            # This is: (row c of W) dot (v_{ab}) = T[(a,b,c)]
            # So: W * v_{ab} = t_{ab} where v_{ab} is a column vector of length R, t_{ab} is a column vector of length pn.
            
            # So we need W (pn x R) such that W * v_{ab} = t_{ab} for all ab.
            
            # Let V_mat be the (R x nm*mp) matrix with columns v_{ab}.
            # Let T_mat be the (pn x nm*mp) matrix with columns t_{ab}.
            # Then we need: W * V_mat = T_mat
            
            # This is a linear system in the entries of W.
            # W is (pn x R), V_mat is (R x nm*mp), T_mat is (pn x nm*mp).
            
            # We can solve this by: for each row c of W, we have:
            # W[c,:] * V_mat = T_mat[c,:]
            # This is a linear system: V_mat^T * W[c,:]^T = T_mat[c,:]^T
            # So V_mat^T (nm*mp x R) * w_c (R x 1) = T_mat[c,:]^T (nm*mp x 1)
            
            # This is overdetermined: nm*mp = 512 equations, R unknowns.
            # For a solution to exist, T_mat[c,:] must be in the row space of V_mat^T, i.e., in the column space of V_mat.
            
            # Let's check if this is consistent. If not, this random U,V doesn't work.
            
            # Build V_mat and T_mat
            V_mat = np.zeros((R, nm * mp), dtype=object)
            T_mat = np.zeros((pn, nm * mp), dtype=object)
            
            ab_idx = 0
            for a in range(nm):
                for b in range(mp):
                    for r in range(R):
                        V_mat[r, ab_idx] = U[a, r] * V[b, r]
                    for c in range(pn):
                        T_mat[c, ab_idx] = T.get((a, b, c), 0)
                    ab_idx += 1
            
            # Now for each c, solve V_mat^T * w_c = T_mat[c,:]^T
            # V_mat^T is (nm*mp x R) = (512 x R)
            # Use least squares over rationals? Or check consistency.
            
            # Since this is over rationals and the system is overdetermined, let's use
            # Gaussian elimination to check if a solution exists.
            
            # For efficiency, let's use numpy with float first to check, then verify with rationals.
            V_mat_float = np.array([[float(x) for x in row] for row in V_mat.T], dtype=float)
            T_mat_float = np.array([[float(x) for x in row] for row in T_mat], dtype=float)
            
            # For each c, solve V_mat_float^T * w_c = T_mat_float[c,:]^T
            # V_mat_float^T is (512 x R)
            w_solutions = []
            consistent = True
            for c in range(pn):
                # Solve the least squares problem
                try:
                    w_c, residuals, rank_c, sv = np.linalg.lstsq(V_mat_float.T, T_mat_float[c, :], rcond=None)
                    # Check if the residual is small
                    if residuals and residuals[0] > 1e-6:
                        consistent = False
                        break
                    w_solutions.append(w_c)
                except:
                    consistent = False
                    break
            
            if not consistent:
                continue
            
            # If consistent, we have a candidate W. Let's verify it exactly with rationals.
            W = np.zeros((pn, R), dtype=object)
            for c in range(pn):
                for r in range(R):
                    W[c, r] = Fraction(w_solutions[c][r]).limit_denominator(1000000)
            
            # Verify the decomposition exactly
            valid = True
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        val = Fraction(0, 1)
                        for r in range(R):
                            val += U[a, r] * V[b, r] * W[c, r]
                        expected = T.get((a, b, c), 0)
                        if val != expected:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Found a valid decomposition!
                # Convert to lists for return
                U_list = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in U]
                V_list = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in V]
                W_list = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no better decomposition found, return the schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
