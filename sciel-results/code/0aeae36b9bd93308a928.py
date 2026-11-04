import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    Nm = n * m  # 20
    Mp = m * p  # 25
    Pn = p * n  # 20
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Build target indices for 4x5x5
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
    target_set = set(target_indices)
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p  # 100
    U_sb = np.zeros((Nm, rank_sb))
    V_sb = np.zeros((Mp, rank_sb))
    W_sb = np.zeros((Pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1.0
                V_sb[b, col] = 1.0
                W_sb[c, col] = 1.0
                col += 1
    
    def verify(U_mat, V_mat, W_mat, tol=1e-6, full_check=False):
        """Check if U, V, W decompose the target tensor exactly"""
        R = U_mat.shape[1]
        max_err = 0.0
        
        # Check all target positions (should be 1)
        for (a, b, c) in target_indices:
            val = np.dot(U_mat[a, :], V_mat[b, :] * W_mat[c, :])
            err = abs(val - 1.0)
            if err > max_err:
                max_err = err
            if max_err > tol:
                return False, max_err
        
        # Check non-target positions (should be 0)
        if full_check:
            # Check ALL non-target positions
            for a in range(Nm):
                for b in range(Mp):
                    for c in range(Pn):
                        if (a, b, c) in target_set:
                            continue
                        val = np.dot(U_mat[a, :], V_mat[b, :] * W_mat[c, :])
                        err = abs(val)
                        if err > max_err:
                            max_err = err
                        if max_err > tol:
                            return False, max_err
        else:
            # Sample non-target positions
            np.random.seed(12345)
            for _ in range(500):
                a = np.random.randint(Nm)
                b = np.random.randint(Mp)
                c = np.random.randint(Pn)
                if (a, b, c) in target_set:
                    continue
                val = np.dot(U_mat[a, :], V_mat[b, :] * W_mat[c, :])
                err = abs(val)
                if err > max_err:
                    max_err = err
                if max_err > tol:
                    return False, max_err
        
        return max_err < tol, max_err
    
    # Verify schoolbook
    is_valid, err = verify(U_sb, V_sb, W_sb, full_check=True)
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    if not is_valid:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Strategy: Embed BCLR 5x5x5 rank-19 tensor into 4x5x5
    # The BCLR decomposition of 5x5x5 has 19 rank-1 terms.
    # Each term is (u_i, v_j, w_k) where the term contributes to T[i,j,k] = u_i * v_j * w_k
    # For 5x5x5: T[i,j,k] = sum_r u_r[i] * v_r[j] * w_r[k] = delta_{i,j,k} pattern
    # Actually for matrix mult: T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B 
    # contributes to (i,k) entry of C. In tensor form: T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    
    # The BCLR 5x5x5 tensor decomposition:
    # This is the famous rank-19 decomposition by Bini, Capovani, Lotti, Romani (1980)
    # Let me construct it based on the known structure.
    
    # Actually, let me try a different approach: use the known fact that
    # 4x5x5 can potentially be done with rank < 100 by exploiting structure.
    
    # Let me try gradient-based optimization to find lower rank decompositions.
    
    def objective_and_grad(U, V, W, sample_size=200):
        """Compute objective (sum of squared errors) and gradients efficiently"""
        R = U.shape[1]
        obj = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Target positions: should be 1
        for (a, b, c) in target_indices:
            u = U[a, :]
            v = V[b, :]
            w = W[c, :]
            val = np.dot(u, v * w)
            diff = val - 1.0
            obj += diff * diff
            grad_U[a, :] += 2.0 * diff * (v * w)
            grad_V[b, :] += 2.0 * diff * (u * w)
            grad_W[c, :] += 2.0 * diff * (u * v)
        
        # Non-target positions: should be 0 (sample)
        np.random.seed(999)
        for _ in range(sample_size):
            a = np.random.randint(Nm)
            b = np.random.randint(Mp)
            c = np.random.randint(Pn)
            if (a, b, c) in target_set:
                continue
            u = U[a, :]
            v = V[b, :]
            w = W[c, :]
            val = np.dot(u, v * w)
            obj += val * val
            grad_U[a, :] += 2.0 * val * (v * w)
            grad_V[b, :] += 2.0 * val * (u * w)
            grad_W[c, :] += 2.0 * val * (u * v)
        
        return obj, grad_U, grad_V, grad_W
    
    def try_rank(target_rank, max_iters=300, lr_init=0.003, init_method='perturb'):
        """Try to find a valid decomposition of given rank"""
        nonlocal best_rank, best_U, best_V, best_W
        if time.time() > deadline:
            return False
        
        np.random.seed(42 + target_rank)
        
        if init_method == 'perturb':
            # Initialize near schoolbook: use first target_rank columns with small perturbation
            indices = np.random.choice(rank_sb, target_rank, replace=False)
            U_curr = U_sb[:, indices].copy()
            V_curr = V_sb[:, indices].copy()
            W_curr = W_sb[:, indices].copy()
            
            # Add small random perturbation
            U_curr += np.random.randn(Nm, target_rank) * 0.01
            V_curr += np.random.randn(Mp, target_rank) * 0.01
            W_curr += np.random.randn(Pn, target_rank) * 0.01
        else:
            # Random initialization
            U_curr = np.random.randn(Nm, target_rank) * 0.1
            V_curr = np.random.randn(Mp, target_rank) * 0.1
            W_curr = np.random.randn(Pn, target_rank) * 0.1
        
        lr = lr_init
        best_obj = np.inf
        no_improve = 0
        
        for it in range(max_iters):
            if time.time() > deadline:
                return False
            
            obj, gU, gV, gW = objective_and_grad(U_curr, V_curr, W_curr)
            
            if obj < best_obj:
                best_obj = obj
                no_improve = 0
            else:
                no_improve += 1
            
            # Adaptive learning rate
            if no_improve > 15:
                lr *= 0.5
                no_improve = 0
                if lr < 1e-7:
                    break
            
            # Gradient step with momentum-like damping
            U_curr -= lr * gU
            V_curr -= lr * gV
            W_curr -= lr * gW
            
            # Check if we're close to a solution
            if obj < 1e-8:
                # Round to half-integers
                U_rounded = np.round(U_curr * 2) / 2
                V_rounded = np.round(V_curr * 2) / 2
                W_rounded = np.round(W_curr * 2) / 2
                
                is_valid, err = verify(U_rounded, V_rounded, W_rounded)
                if is_valid:
                    best_rank = target_rank
                    best_U = U_rounded.copy()
                    best_V = V_rounded.copy()
                    best_W = W_rounded.copy()
                    return True
        
        # Final attempt: round and verify
        U_rounded = np.round(U_curr * 2) / 2
        V_rounded = np.round(V_curr * 2) / 2
        W_rounded = np.round(W_curr * 2) / 2
        is_valid, err = verify(U_rounded, V_rounded, W_rounded)
        if is_valid:
            best_rank = target_rank
            best_U = U_rounded.copy()
            best_V = V_rounded.copy()
            best_W = W_rounded.copy()
            return True
        
        return False
    
    # Try to reduce rank from 99 down to 60
    # Focus on the most promising ranges first
    for r in range(99, 59, -1):
        if time.time() > deadline:
            break
        
        # Skip some ranks to save time
        if r % 3 == 0 and r > 70:
            continue
            
        success = try_rank(r)
        if success:
            # Found a valid decomposition at this rank
            # Try to go lower
            continue
    
    # If we found something better, do a polish pass
    if best_rank < rank_sb and time.time() < deadline - 15:
        # Try to polish with more iterations
        for r in range(best_rank - 1, max(best_rank - 10, 59), -1):
            if time.time() > deadline:
                break
            success = try_rank(r, max_iters=100, lr_init=0.001)
            if success:
                continue
    
    # Final verification with full check
    if best_rank < rank_sb:
        is_valid, err = verify(best_U, best_V, best_W, full_check=True)
        if not is_valid:
            # Fall back to schoolbook
            best_rank = rank_sb
            best_U = U_sb.copy()
            best_V = V_sb.copy()
            best_W = W_sb.copy()
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
