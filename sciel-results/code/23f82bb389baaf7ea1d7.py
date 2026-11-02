"""Matrix multiplication 3x4x5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 5]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def _build_expected_tensor(n, m, p):
    """Build the expected matmul tensor T[x, y, z] where:
    x = i*m+j (A index)
    y = j*p+k (B index)
    z = k*n+i (C index, column-major convention)
    T[x,y,z] = 1 if indices are consistent, else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[x,r]*V[y,r]*W[z,r] equals the matmul tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Compute decomposed tensor
    D = np.einsum('xr,yr,zr->xyz', U, V, W)
    
    # Build expected
    expected = _build_expected_tensor(n, m, p)
    
    return np.allclose(D, expected, atol=tol)


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W


def _try_block_decomposition(n, m, p, deadline):
    """
    Try to exploit block structure. The idea from the inspiration is to see if
    we can decompose using the fact that certain sub-structures have known ranks.
    
    For 3x4x5: We can view A as 3 blocks of 1x4, B as 4 blocks of 1x5 (or other splits).
    The rank of 1xk times kx1 is k. But combining these doesn't obviously reduce rank below nmp.
    
    However, let's try a structured search: use the schoolbook as a starting point and
    attempt to merge columns using linear algebra. If two columns c1, c2 of (U,V,W) satisfy
    that their outer products sum to zero in some subspace, we might merge them.
    
    A more practical approach: try to find a basis for the space of valid decompositions
    and then minimize rank. This is hard, so we'll use a heuristic:
    
    1. Start with schoolbook.
    2. Try to find linear dependencies among the rank-1 tensors.
    3. If dependencies exist, reduce rank.
    
    For matmul, the rank-1 tensors in schoolbook are linearly independent (each has a unique
    non-zero pattern), so no reduction is possible via this method.
    
    Let's instead try a random search with structured initialization for lower ranks.
    """
    n, m, p = n, m, p
    schoolbook_rank = n * m * p
    
    # For 3x4x5, the known lower bound is not well-established.
    # Let's try a few lower ranks with structured random search.
    
    # Strategy: Use the fact that the tensor has a specific structure.
    # We'll try to solve for U, V, W with a given rank using a least-squares approach
    # on a subset of the tensor entries.
    
    for target_rank in range(schoolbook_rank - 1, max(schoolbook_rank - 15, 40), -1):
        if time.time() > deadline - 30:
            break
        
        # Try a few random initializations
        for seed in range(5):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(seed + target_rank)
            
            # Initialize with small random values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Use ALS (Alternating Least Squares) to try to fit the tensor
            # We'll fix one matrix at a time and solve for the others.
            
            expected = _build_expected_tensor(n, m, p)
            
            # Unroll the tensor: T[x,y,z] = sum_r U[x,r] V[y,r] W[z,r]
            # For fixed V and W, this is linear in U.
            # For fixed U and W, this is linear in V.
            # For fixed U and V, this is linear in W.
            
            # We'll do a few iterations of ALS.
            for it in range(10):
                if time.time() > deadline - 30:
                    break
                
                # Solve for U: For each x, U[x,:] * (V ⊙ W) = T[x,:,:].ravel()
                # where (V ⊙ W)[y,z,r] = V[y,r] * W[z,r]
                # This is a linear system: U[x,:] @ (V ⊙ W)[:, :, :].reshape(target_rank, -1) = T[x,:,:].ravel()
                
                # Build the design matrix for U
                # For each x, we need to solve: sum_r U[x,r] * V[y,r] * W[z,r] = T[x,y,z] for all y,z
                # This is equivalent to: U[x,:] @ M_x = T[x,:,:].ravel()
                # where M_x is a (m*p*p*n) x target_rank matrix... but this is huge.
                
                # Alternative: Use the fact that for fixed V and W, the equation is:
                # U[x,:] @ (V ⊙ W) = T[x,:,:]
                # We can solve this as a least squares problem.
                
                # Build V ⊙ W as a 3D array: (m*p, p*n, target_rank)
                VW = np.einsum('yr,zr->yzr', V, W)  # (m*p, p*n, target_rank)
                
                # For each x, solve U[x,:] @ VW.reshape(m*p*p*n, target_rank).T = T[x,:,:].ravel()
                # Actually, T[x,:,:].ravel() is a vector of length m*p*p*n
                # And we want U[x,:] such that U[x,:] @ VW_r = T_x
                # where VW_r is (m*p*p*n, target_rank) and U[x,:] is (1, target_rank)
                
                VW_flat = VW.reshape(-1, target_rank)  # (m*p*p*n, target_rank)
                
                # Solve for U[x,:] for each x
                for x in range(n * m):
                    T_x = expected[x, :, :].ravel()
                    # Solve: U[x,:] @ VW_flat.T = T_x
                    # This is: (VW_flat @ U[x,:].T).T = T_x
                    # Or: VW_flat.T @ U[x,:] = T_x (if we think of it as linear system)
                    # Actually, we want: sum_r U[x,r] * VW_flat[idx, r] = T_x[idx] for all idx
                    # This is: U[x,:] @ VW_flat.T = T_x
                    # Which is: (VW_flat @ U[x,:].T).T = T_x
                    # So: VW_flat @ U[x,:].T = T_x.T = T_x (since T_x is 1D)
                    # Wait, dimensions: VW_flat is (N, R), U[x,:].T is (R, 1), so VW_flat @ U[x,:].T is (N, 1)
                    # And T_x is (N,). So we want VW_flat @ U[x,:].T = T_x.reshape(-1, 1)
                    # This is a least squares problem: solve for U[x,:].T
                    try:
                        sol, _, _, _ = np.linalg.lstsq(VW_flat, T_x.reshape(-1, 1), rcond=None)
                        U[x, :] = sol.flatten()
                    except:
                        pass
                
                # Similarly solve for V and W
                UW = np.einsum('xr,zr->x zr', U, W)  # (n*m, p*n, target_rank)
                UW_flat = UW.reshape(-1, target_rank)
                for y in range(m * p):
                    # For fixed U and W, V[y,:] should satisfy:
                    # sum_r V[y,r] * U[x,r] * W[z,r] = T[x,y,z] for all x,z
                    # This is: V[y,:] @ (U ⊙ W)[x,z,:].ravel()... 
                    # Let's build U ⊙ W: (n*m, p*n, target_rank)
                    UW_3d = np.einsum('xr,zr->x zr', U, W)
                    UW_flat_y = UW_3d.reshape(-1, target_rank)
                    T_y = expected[:, y, :].ravel()
                    try:
                        sol, _, _, _ = np.linalg.lstsq(UW_flat_y, T_y.reshape(-1, 1), rcond=None)
                        V[y, :] = sol.flatten()
                    except:
                        pass
                
                UV = np.einsum('xr,yr->x yr', U, V)  # (n*m, m*p, target_rank)
                UV_flat = UV.reshape(-1, target_rank)
                for z in range(p * n):
                    UV_3d = np.einsum('xr,yr->x yr', U, V)
                    UV_flat_z = UV_3d.reshape(-1, target_rank)
                    T_z = expected[:, :, z].ravel()
                    try:
                        sol, _, _, _ = np.linalg.lstsq(UV_flat_z, T_z.reshape(-1, 1), rcond=None)
                        W[z, :] = sol.flatten()
                    except:
                        pass
            
            # Check if the solution is valid
            if _verify_decomposition(U, V, W, n, m, p):
                # Round to nearest half-integer if possible
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                    return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
                else:
                    return (U.tolist(), V.tolist(), W.tolist())
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    
    # Verify schoolbook is correct
    if not _verify_decomposition(U, V, W, n, m, p):
        # This should never happen, but if it does, we have a problem
        pass
    
    # Try to find a lower rank decomposition
    result = _try_block_decomposition(n, m, p, deadline)
    if result is not None:
        U, V, W = result
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
