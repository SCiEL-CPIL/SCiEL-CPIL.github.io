"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the multiplication tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0
    
    # Schoolbook rank as fallback
    best_result = _schoolbook(n, m, p)
    
    deadline = time.time() + 130  # 130s budget, leave margin for return
    
    # Try ranks from schoolbook-1 down
    schoolbook_rank = n * m * p
    for rank in range(schoolbook_rank - 1, 6, -1):
        if time.time() > deadline:
            break
        result = _try_rank_enumerate(T, nm, mp, pn, rank, deadline)
        if result is not None:
            best_result = result
            break  # Found lower rank, stop
    
    return best_result

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_rank_enumerate(T, nm, mp, pn, rank, deadline):
    """Try to find a rank-`rank` decomposition using enumeration of small rational vectors."""
    # The tensor T has shape (nm, mp, pn) = (6, 12, 8)
    # We need U (6 x rank), V (12 x rank), W (8 x rank) such that
    # sum_c U[:,c] outer V[:,c] outer W[:,c] == T
    
    # Strategy: enumerate candidate rank-1 tensors with small integer/half-integer entries
    # and use linear algebra to find which subset spans T
    
    # Flatten T
    T_flat = T.flatten()
    dim = T_flat.shape[0]  # 6*12*8 = 576
    
    # Generate candidate vectors with entries in {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}
    # But we need to be smart about this - the space is 576 dimensional
    
    # Better approach: use the structure of the problem
    # For matrix multiplication, we can use known algorithms or search more intelligently
    
    # Let's try a structured search:
    # The rank of 2x3 * 3x4 matrix multiplication
    # Known: rank(2x2) = 7, rank(2x3) = 11, rank(3x3) = 23
    # For 2x3 * 3x4, we can try various known constructions
    
    # First, let's try to find if rank < 24 is possible
    # Use a more targeted approach: try specific known decompositions
    
    # For 2x3 * 3x4:
    # We can decompose 3x4 as (3x2)*(2x4) or use other splits
    # Or use the fact that 2x3 * 3x4 can be related to other known ranks
    
    # Let's try a direct optimization approach with better initialization
    result = _try_rank_optimize(T, nm, mp, pn, rank, deadline)
    if result is not None:
        return result
    
    # Also try structured enumeration for small ranks
    result = _try_rank_structured(T, nm, mp, pn, rank, deadline)
    return result

def _try_rank_optimize(T, nm, mp, pn, rank, deadline):
    """Try to find a rank-`rank` decomposition via optimization."""
    np.random.seed(42)
    T_flat = T.flatten()
    
    num_tries = 30
    for trial in range(num_tries):
        if time.time() > deadline:
            return None
        
        # Random initialization with various scales
        scale = 0.1 * (trial % 3 + 1)
        U = np.random.randn(nm, rank) * scale
        V = np.random.randn(mp, rank) * scale
        W = np.random.randn(pn, rank) * scale
        
        # Use scipy least_squares if available
        try:
            from scipy.optimize import least_squares
        except ImportError:
            least_squares = None
        
        if least_squares is not None:
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def fun(x):
                Ux = x[:nm*rank].reshape(nm, rank)
                Vx = x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                Wx = x[nm*rank+mp*rank:].reshape(pn, rank)
                tensor = np.einsum('ac,bc,dc->abd', Ux, Vx, Wx)
                return (tensor - T).flatten()
            
            try:
                res = least_squares(fun, x0, max_nfev=300, xtol=1e-14, ftol=1e-14, gtol=1e-14)
            except Exception:
                continue
            
            if res.cost < 1e-10:
                U_sol = res.x[:nm*rank].reshape(nm, rank)
                V_sol = res.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                W_sol = res.x[nm*rank+mp*rank:].reshape(pn, rank)
                
                # Snap to half-integers
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                # Verify
                if _verify(T, U_snap, V_snap, W_snap):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    return None

def _try_rank_structured(T, nm, mp, pn, rank, deadline):
    """Try structured decompositions based on known matrix multiplication algorithms."""
    n, m, p = 2, 3, 4
    
    # For 2x3 * 3x4, try various block decompositions
    # Method 1: Split the 3x4 into two 3x2 blocks
    # C = A * B where A is 2x3, B is 3x4
    # Split B = [B1 | B2] where B1, B2 are 3x2
    # Then C = [A*B1 | A*B2]
    # Each A*B1 and A*B2 is 2x2, rank 7 each
    # Total rank: 7 + 7 = 14
    
    if rank >= 14:
        result = _decompose_split_3x4(n, m, p)
        if result is not None:
            return result
    
    # Method 2: Try other splits
    if rank >= 16:
        result = _decompose_alternative(n, m, p)
        if result is not None:
            return result
    
    return None

def _decompose_split_3x4(n, m, p):
    """Decompose 2x3 * 3x4 by splitting 3x4 into two 3x2 blocks."""
    # B is 3x4, split into B1 (3x2) and B2 (3x2)
    # C = A * B = [A*B1 | A*B2]
    # Each A*B1 and A*B2 is 2x2 matrix multiplication (rank 7)
    
    # Get the 2x2 decomposition
    def _rank2x2():
        # Strassen-like or known rank-7 decomposition for 2x2
        # Use a known rank-7 decomposition
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Known rank-7 decomposition for 2x2
        # This is a standard construction
        factors = [
            # (U_row, V_row, W_row) for each rank-1 term
            # Using a known rank-7 decomposition
        ]
        
        # Let's use a verified rank-7 decomposition
        # The entries should be -1, 0, 1
        U[0,0] = 1; V[0,0] = 1; W[0,0] = 1  # a11*b11
        U[1,1] = 1; V[1,1] = 1; W[1,1] = 1  # a22*b22
        U[0,2] = 1; V[2,2] = 1; W[2,2] = 1  # a11*b22
        U[1,3] = 1; V[3,3] = 1; W[3,3] = 1  # a22*b11
        U[0,4] = 1; V[4,4] = 1; W[4,4] = 1  # a12*b11
        U[1,5] = 1; V[5,5] = 1; W[5,5] = 1  # a21*b22
        U[0,6] = 1; V[6,6] = 1; W[6,6] = 1  # a12*b22 + a21*b11 - a11*b22 - a22*b11
        
        # Actually, let me use a proper rank-7 decomposition
        # The standard one has specific patterns
        return None  # Placeholder - need to implement properly
    
    # Instead, let's build the full decomposition directly
    # For 2x3 * 3x4 with rank 14:
    # We need U (6x14), V (12x14), W (8x14)
    
    # The tensor T[a,b,d] where a=i*3+j (i=0,1; j=0,1,2), b=j'*4+k (j'=0,1,2; k=0,1,2,3), d=k'*2+i' (k'=0,1; i'=0,1)
    # T[a,b,d] = 1 if j==j' and i==i'
    
    # For the split approach:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Split k into k1 (0,1) and k2 (2,3)
    # C1[i,k1] = sum_j A[i,j] * B[j,k1]  (2x2 result)
    # C2[i,k2] = sum_j A[i,j] * B[j,k2]  (2x2 result)
    
    # Each 2x2 multiplication has rank 7
    # Total rank: 14
    
    # Let's construct this properly
    # For 2x2 * 2x2 -> 2x2, use a rank-7 decomposition
    
    # Known rank-7 decomposition for 2x2:
    # M = A*B where A, B, M are 2x2
    # Use the following rank-1 terms (U, V, W) where each is 4x7:
    
    U22 = np.zeros((4, 7))
    V22 = np.zeros((4, 7))
    W22 = np.zeros((4, 7))
    
    # Standard rank-7 decomposition
    # Term 1: a11*b11
    U22[0,0] = 1; V22[0,0] = 1; W22[0,0] = 1
    # Term 2: a22*b22
    U22[3,1] = 1; V22[3,1] = 1; W22[3,1] = 1
    # Term 3: (a11+a12)*b11
    U22[0,2] = 1; U22[1,2] = 1; V22[0,2] = 1; W22[0,2] = 1; W22[1,2] = 1
    # Term 4: (a22+a21)*b22
    U22[3,3] = 1; U22[2,3] = 1; V22[3,3] = 1; W22[3,3] = 1; W22[2,3] = 1
    # Term 5: (a11+a21)*b12
    U22[0,4] = 1; U22[2,4] = 1; V22[1,4] = 1; W22[0,4] = 1; W22[1,4] = 1
    # Term 6: (a22+a12)*b21
    U22[3,5] = 1; U22[1,5] = 1; V22[2,5] = 1; W22[3,5] = 1; W22[2,5] = 1
    # Term 7: (a12-a21)*(b12-b21) + correction
    # Actually, let me use a verified decomposition
    
    # Let me use a different approach - just verify if rank 14 works
    # by constructing it explicitly
    
    n, m, p = 2, 3, 4
    nm, mp, pn = n*m, m*p, p*n  # 6, 12, 8
    
    U = np.zeros((nm, 14))
    V = np.zeros((mp, 14))
    W = np.zeros((pn, 14))
    
    # For each of the two 2x2 multiplications (k1=0,1 and k2=2,3):
    for block in range(2):
        # k indices for this block
        k_start = block * 2
        
        # For this block, we do 2x2 * 2x2 where:
        # A is 2x3 (rows i=0,1; cols j=0,1,2)
        # B is 3x2 (rows j=0,1,2; cols k=k_start, k_start+1)
        # Result is 2x2
        
        # This is still not 2x2 * 2x2, it's 2x3 * 3x2
        # Rank of 2x3 * 3x2 is at most 2*3*2 = 12, but can be less
        
        # Actually, 2x3 * 3x2 has rank at most 2*2=4 if we use the middle dimension efficiently
        # No, the rank is bounded by min(n*p, m^2) = min(4, 9) = 4? No...
        
        # The tensor rank of 2x3 * 3x2 is actually 12 (schoolbook) or less
        # Let me just use schoolbook for each block: rank 2*3*2 = 12 each
        # Total: 24, which doesn't help
        
        # Better: use the fact that 2x3 * 3x2 can be done with rank 8 or so
        
        # For now, let's just return None and rely on optimization
        pass
    
    return None

def _decompose_alternative(n, m, p):
    """Try alternative decompositions."""
    return None

def _verify(T, U, V, W):
    """Verify that sum_c U[:,c]*V[:,c]*W[:,c] == T exactly (within tolerance)."""
    tensor = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.allclose(tensor, T, atol=1e-10)
# EVOLVE-BLOCK-END
