"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    np_ = n * p
    schoolbook_rank = nm * mp // m * p  # n*m*p = 36
    
    # Build target tensor T as a 3D array T[i,j,k]
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The bilinear form: C[i,k] = sum_j A[i,j] B[j,k]
    # In flattened form: c[i*p+k] = sum_j a[i*m+j] * b[j*p+k]
    
    # T is a 3D tensor of shape (np_, nm, mp)
    # T[l, r, s] = 1 if l=i*p+k, r=i*m+j, s=j*p+k for some i,j,k
    T = np.zeros((np_, nm, mp), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                l = i * p + k
                r = i * m + j
                s = j * p + k
                T[l, r, s] = 1.0
    
    # Schoolbook decomposition (fallback)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(np_)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1  # Wait, W index is k*n+i? Let me check.
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # The W matrix has shape (p*n, rank) with index k*n+i
    # But in the tensor, the output index is l = i*p + k
    # So W[k*n + i, c] should correspond to output index... 
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k, i) where k is column of B/C, i is row of A/C
    # The output C[i,k] is at position k*n + i in W's rows
    
    # Let me verify: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For schoolbook: U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # So C[i,k] = sum_{col} ... = 1 for the term where i,j,k match
    # Summing over j: C[i,k] = sum_j A[i,j] B[j,k] * (contribution)
    # Yes, this works.
    
    # Now let's try to find a lower-rank decomposition
    # We'll use a local search approach
    
    deadline = time.time() + 130  # Leave 20s margin
    
    best_rank = n * m * p
    best_decomp = None
    
    # Try to find decompositions with rank < best_rank
    # Use a simple approach: random initialization + local search
    
    def verify_decomp(U, V, W, rank):
        """Check if U, V, W with given rank exactly decompose T."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # Reconstruct tensor
        T_recon = np.zeros((np_, nm, mp), dtype=float)
        for c in range(rank):
            T_recon += np.outer(np.outer(U[:, c], V[:, c]), W[:, c])
        # Check if T_recon equals T
        return np.allclose(T_recon, T, atol=1e-9)
    
    # Try ranks from best_rank-1 down
    for rank in range(best_rank - 1, 20, -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Initialize with random half-integer values
            U = np.round(np.random.randn(nm, rank) * 2) / 2
            V = np.round(np.random.randn(mp, rank) * 2) / 2
            W = np.round(np.random.randn(np_, rank) * 2) / 2
            
            # Local search: try to adjust entries to reduce error
            error = np.linalg.norm(T - np.einsum('ic,jc,kc->ijk', U, V, W))
            
            # Simple gradient-free local search
            for step in range(200):
                if time.time() > deadline:
                    break
                
                # Pick a random entry to modify
                mat = random.choice(['U', 'V', 'W'])
                row = random.randint(0, (nm, mp, np_)[0] if mat=='U' else (nm, mp, np_)[1] if mat=='V' else (nm, mp, np_)[2])
                col = random.randint(0, rank - 1)
                
                if mat == 'U':
                    row = random.randint(0, nm - 1)
                    old_val = U[row, col]
                elif mat == 'V':
                    row = random.randint(0, mp - 1)
                    old_val = V[row, col]
                else:
                    row = random.randint(0, np_ - 1)
                    old_val = W[row, col]
                
                # Try a few perturbed values
                best_val = old_val
                best_err = error
                
                for delta in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0, -1.0]:
                    new_val = round((old_val + delta) * 2) / 2
                    if new_val == old_val:
                        continue
                    
                    if mat == 'U':
                        U[row, col] = new_val
                    elif mat == 'V':
                        V[row, col] = new_val
                    else:
                        W[row, col] = new_val
                    
                    new_err = np.linalg.norm(T - np.einsum('ic,jc,kc->ijk', U, V, W))
                    
                    if new_err < best_err:
                        best_err = new_err
                        best_val = new_val
                    else:
                        # Revert
                        if mat == 'U':
                            U[row, col] = old_val
                        elif mat == 'V':
                            V[row, col] = old_val
                        else:
                            W[row, col] = old_val
                
                if mat == 'U':
                    U[row, col] = best_val
                elif mat == 'V':
                    V[row, col] = best_val
                else:
                    W[row, col] = best_val
                
                error = best_err
                
                if error < 1e-9:
                    break
            
            if error < 1e-9:
                # Found a valid decomposition!
                if rank < best_rank:
                    best_rank = rank
                    best_decomp = (U.copy(), V.copy(), W.copy())
                break
        
        if best_rank <= rank:
            break
    
    if best_decomp is not None:
        U, V, W = best_decomp
        rank = best_rank
        # Convert to list of lists with integer/half-integer entries
        U_list = [[round(U[i, c] * 2) / 2 for c in range(rank)] for i in range(nm)]
        V_list = [[round(V[i, c] * 2) / 2 for c in range(rank)] for i in range(mp)]
        W_list = [[round(W[i, c] * 2) / 2 for c in range(rank)] for i in range(np_)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
