import numpy as np
import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Target: Rec[i, j, k] = 1 for all i, j, k
    # Rec[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # 1. Generate schoolbook solution as fallback (Rank 45)
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    schoolbook_sol, schoolbook_rank = get_schoolbook()
    best_solution = schoolbook_sol
    best_rank = schoolbook_rank

    def is_valid(U, V, W, rank):
        # Compute the reconstructed tensor
        # Rec[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # Rec[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
        # Let's transpose W_3d to (n, p, rank) so it is W_new[i, k, r] = W_3d[k, i, r]
        W_3d_T = W_3d.transpose(1, 0, 2) # Shape (n, p, rank)
        
        # Now Rec[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d_T[i, k, r]
        # einsum: 'ijr,jkr,ikr->ijk'
        Rec = np.einsum('ijr,jkr,ikr->ijk', U_3d, V_3d, W_3d_T, optimize=True)
        
        return np.allclose(Rec, 1.0, atol=1e-8)

    def solve_rank(rank):
        """
        Attempt to find a rank-`rank` decomposition using Integer Linear Programming.
        Variables are scaled by 2 to handle half-integers.
        x_u[i, r] = 2 * U[i, r]
        x_v[j, r] = 2 * V[j, r]
        x_w[k, r] = 2 * W[k, r]
        
        Constraint: sum_r (x_u[i,j,r] * x_v[j,k,r] * x_w[k,i,r]) / 8 = 1
        => sum_r (x_u * x_v * x_w) = 8
        
        This is a non-linear constraint (cubic). MILP solvers in scipy only handle linear constraints.
        However, for small dimensions, we can use a different strategy:
        We know that for matrix multiplication, the tensor is bilinear in U and V if W is fixed, etc.
        Actually, standard ILP cannot solve cubic constraints directly.
        
        Alternative Strategy:
        Since we suspect the rank might be close to 45, or we can use a heuristic search for small ranks.
        But wait, the error said "coefficients must be integers or half-integers".
        The previous code failed because it returned floats.
        
        Let's try a simpler approach:
        If we can't find a better rank via ILP (which is hard for cubic), we can try to verify if rank 45 is minimal or if there's a known lower rank.
        For 3x3x5, the minimal rank is likely 45 (schoolbook) or slightly less.
        Actually, for M(n,n,p), the rank is often n^2 p.
        
        Let's try a random search with snapping to half-integers for ranks < 45.
        This is more robust than gradient descent for lattice constraints.
        """
        if time.time() > deadline:
            return None
            
        # We will use a random search with local refinement (snapping)
        # This is slower but avoids the complexity of non-linear ILP.
        # We will try ranks from best_rank - 1 down to 40.
        
        for r in range(rank, 39, -1):
            if time.time() > deadline:
                return None
                
            # Try a few random restarts for this rank
            for restart in range(20):
                if time.time() > deadline:
                    return None
                
                U = np.random.randint(-2, 3, size=(n * m, r)) / 2.0
                V = np.random.randint(-2, 3, size=(m * p, r)) / 2.0
                W = np.random.randint(-2, 3, size=(p * n, r)) / 2.0
                
                # Check if this random guess is valid (unlikely)
                if is_valid(U, V, W, r):
                    return {"U": U, "V": V, "W": W}
                    
                # Local search: Snap to nearest half-integer and check
                # This is just the initial step. We need a way to move.
                # Since gradient descent failed, let's try a simple "coordinate descent" on the lattice.
                
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    
                    # Try perturbing each variable slightly
                    # To keep it fast, we only try small steps (-1, 0, 1) in the scaled integer space
                    # and check validity. This is expensive.
                    
                    # Better: Use the fact that the tensor is linear in each factor if the others are fixed.
                    # If we fix V and W, the constraint on U is linear?
                    # Rec[i,j,k] = sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r])
                    # For a fixed j,k, Rec[i,j,k] depends on U[i,j,r].
                    # This is not linear in U because U[i,j,r] multiplies V[j,k,r] which depends on r.
                    # But for a specific output element Rec[i,j,k], the term is sum_r U[i,j,r] * C_r where C_r = V[j,k,r]W[k,i,r].
                    # So for fixed V,W, the problem is linear in U?
                    # No, because U[i,j,r] appears in Rec[i,j,k] for all k.
                    # Rec[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                    # Let A_{r} = V[j,k,r] * W[k,i,r]. Then Rec[i,j,k] = dot(U[i,j,:], A).
                    # So for fixed V,W, the constraints on U are linear!
                    # We can solve for U given V,W using Linear Programming (or just check if a solution exists).
                    # Similarly for V and W.
                    
                    # Algorithm:
                    # 1. Random V, W (half-integers)
                    # 2. Solve for U such that Rec=1.
                    #    For each i,j:
                    #    sum_k Rec[i,j,k] * error? No, we want Rec[i,j,k]=1 for all k.
                    #    We have p equations (for k=0..p-1) and r unknowns (U[i,j,0..r-1]).
                    #    If r >= p, we can solve the linear system.
                    #    Let M[i,j] be a matrix where M[i,j][k, r] = V[j,k,r] * W[k,i,r].
                    #    We want M[i,j] @ U[i,j,:] = [1, 1, ..., 1] (length p).
                    #    This is a linear system. We can use least squares or exact integer solution search.
                    #    Since we need half-integers, we can search for integer solutions to 2*M @ (2*U) = 8?
                    #    Actually, let's just use least squares and snap.
                    
                    # Let's try to solve for U given V,W
                    U_new = np.zeros((n * m, r))
                    valid_U = True
                    for i in range(n):
                        for j in range(m):
                            # Build matrix A of shape (p, r)
                            A = np.zeros((p, r))
                            for k in range(p):
                                for rr in range(r):
                                    A[k, rr] = V[j * p + k, rr] * W[k * n + i, rr]
                            
                            # Solve A @ u = 1
                            try:
                                # Use least squares
                                u_sol, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(p), rcond=None)
                                # Snap to nearest half-integer
                                u_snap = np.round(u_sol * 2) / 2.0
                                # Check if this snap is close enough
                                if not np.allclose(A @ u_snap, np.ones(p), atol=1e-6):
                                    # Try to find an exact integer solution?
                                    # For now, mark as invalid if snap doesn't work
                                    valid_U = False
                                    break
                                U_new[i * m + j, :] = u_snap
                            except:
                                valid_U = False
                                break
                        if not valid_U:
                            break
                    
                    if valid_U:
                        # We found U. Now check if the whole tensor is valid.
                        if is_valid(U_new, V, W, r):
                            return {"U": U_new, "V": V, "W": W}
                    
                    # If U didn't work, maybe we should have solved for V or W?
                    # For simplicity, we just restart the random V,W.
        
        return None

    # Search for lower ranks
    # Try ranks 44 down to 40
    found = None
    for r in range(44, 39, -1):
        if time.time() > deadline:
            break
        found = solve_rank(r)
        if found:
            best_solution = found
            best_rank = r
            break

    # Ensure the returned values are lists of lists of floats
    def to_lists(mat):
        return mat.tolist()

    return {
        "U": to_lists(best_solution["U"]),
        "V": to_lists(best_solution["V"]),
        "W": to_lists(best_solution["W"])
    }
# EVOLVE-BLOCK-END
