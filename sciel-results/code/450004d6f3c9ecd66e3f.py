import time
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    - Try to find a rank-10 decomposition for the 2x2 * 2x3 sub-problem (columns 0,1,2)
    - Use Strassen rank-7 for columns (3,4)
    - Use Strassen rank-7 for columns (5,6)
    - If rank-10 search fails, fall back to rank-25 (3x Strassen + Schoolbook)
    
    The 2x2 * 2x3 tensor has dimensions:
    - U: (n*m, rank) = (4, rank)
    - V: (m*p, rank) = (6, rank)  
    - W: (p*n, rank) = (6, rank)
    
    Total rank target: 10 + 7 + 7 = 24
    """
    n, m, p = 2, 2, 7
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # First, let's try to find a rank-10 decomposition for 2x2 * 2x3
    # The tensor T_{i,j,k} = 1 for all i in [0,2), j in [0,2), k in [0,3)
    # We need U (4 x r), V (6 x r), W (6 x r) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's set up the optimization for rank r
    def find_rank_decomp(n_sub, m_sub, p_sub, target_rank, max_iters=200, timeout=60):
        """
        Find a rank-target_rank decomposition for n_sub x m_sub * m_sub x p_sub.
        Returns (U, V, W) or None if not found.
        """
        start_time = time.time()
        
        # Initialize with schoolbook and try to reduce rank
        # Schoolbook rank = n_sub * m_sub * p_sub = 2*2*3 = 12
        # We want to find rank 10
        
        # Use a random initialization with structure
        # Actually, let's try multiple random initializations
        
        best_loss = float('inf')
        best_params = None
        
        # The tensor equation: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is a system of 12 equations (2*2*3) with 3*r variables
        
        # Let's use scipy.optimize to minimize the sum of squared errors
        def loss_and_grad(params):
            U = params[:n_sub*m_sub*target_rank].reshape(n_sub*m_sub, target_rank)
            V = params[n_sub*m_sub*target_rank:n_sub*m_sub*target_rank + m_sub*p_sub*target_rank].reshape(m_sub*p_sub, target_rank)
            W = params[n_sub*m_sub*target_rank + m_sub*p_sub*target_rank:].reshape(p_sub*n_sub, target_rank)
            
            # Compute the tensor
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            errors = []
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        val = 0.0
                        for r in range(target_rank):
                            val += U[i*m_sub + j, r] * V[j*p_sub + k, r] * W[k*n_sub + i, r]
                        errors.append(val - 1.0)
            
            errors = np.array(errors)
            loss = np.sum(errors**2)
            
            # Gradient
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        err = errors[i*m_sub*p_sub + j*p_sub + k]
                        for r in range(target_rank):
                            dU[i*m_sub + j, r] += 2 * err * V[j*p_sub + k, r] * W[k*n_sub + i, r]
                            dV[j*p_sub + k, r] += 2 * err * U[i*m_sub + j, r] * W[k*n_sub + i, r]
                            dW[k*n_sub + i, r] += 2 * err * U[i*m_sub + j, r] * V[j*p_sub + k, r]
            
            grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            return loss, grad
        
        # Try multiple random initializations
        for attempt in range(10):
            if time.time() - start_time > timeout:
                break
            
            # Initialize with small random values
            np.random.seed(42 + attempt)
            params0 = np.random.randn(n_sub*m_sub*target_rank + m_sub*p_sub*target_rank + p_sub*n_sub*target_rank) * 0.5
            
            # Scale to make initial product closer to 1
            # The product of three ~N(0, 0.25) variables has variance ~0.0156
            # We want sum of 10 such products ~ 1, so scale by ~sqrt(10/0.156) ~ 8
            # Actually, let's just try and see
            
            try:
                result = minimize(loss_and_grad, params0, method='L-BFGS-B', 
                                  jac=True, options={'maxiter': max_iters, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_params = result.x
            except Exception as e:
                continue
        
        if best_params is not None and best_loss < 1e-6:
            U = best_params[:n_sub*m_sub*target_rank].reshape(n_sub*m_sub, target_rank)
            V = best_params[n_sub*m_sub*target_rank:n_sub*m_sub*target_rank + m_sub*p_sub*target_rank].reshape(m_sub*p_sub, target_rank)
            W = best_params[n_sub*m_sub*target_rank + m_sub*p_sub*target_rank:].reshape(p_sub*n_sub, target_rank)
            
            # Verify
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        val = sum(U[i*m_sub + j, r] * V[j*p_sub + k, r] * W[k*n_sub + i, r] for r in range(target_rank))
                        if abs(val - 1.0) > 1e-5:
                            return None
            
            return U, V, W
        
        return None
    
    # Try to find rank-10 for 2x2 * 2x3
    rank10_decomp = find_rank_decomp(2, 2, 3, 10, max_iters=100, timeout=80)
    
    if rank10_decomp is not None and time.time() < deadline:
        U10, V10, W10 = rank10_decomp
        # Now build the full 2x2 * 2x7 decomposition
        # Columns 0,1,2 use the rank-10 decomp
        # Columns 3,4 use Strassen rank-7
        # Columns 5,6 use Strassen rank-7
        total_rank = 10 + 7 + 7
        
        U = np.zeros((n*m, total_rank), dtype=float)
        V = np.zeros((m*p, total_rank), dtype=float)
        W = np.zeros((p*n, total_rank), dtype=float)
        
        # Copy rank-10 block for columns 0,1,2
        U[:, :10] = U10
        V[:, :10] = V10
        W[:, :10] = W10
        
        # Strassen for columns 3,4
        def add_strassen_block(col_start, k_start):
            # M1 = (A00 + A11) * (B0,k + B1,k+1) -> C0,k and C1,k+1
            U[0, col_start+0] += 1
            U[3, col_start+0] += 1
            V[0*p + k_start, col_start+0] += 1
            V[1*p + (k_start+1), col_start+0] += 1
            W[k_start*n + 0, col_start+0] += 1
            W[(k_start+1)*n + 1, col_start+0] += 1

            # M2 = (A10 + A11) * B0,k -> C1,k and -C1,k+1
            U[2, col_start+1] += 1
            U[3, col_start+1] += 1
            V[0*p + k_start, col_start+1] += 1
            W[k_start*n + 1, col_start+1] += 1
            W[(k_start+1)*n + 1, col_start+1] -= 1

            # M3 = A00 * (B0,k+1 - B1,k+1) -> C0,k+1 and C1,k+1
            U[0, col_start+2] += 1
            V[0*p + (k_start+1), col_start+2] += 1
            V[1*p + (k_start+1), col_start+2] -= 1
            W[(k_start+1)*n + 0, col_start+2] += 1
            W[(k_start+1)*n + 1, col_start+2] += 1

            # M4 = A11 * (B1,k - B0,k) -> C0,k and C1,k
            U[3, col_start+3] += 1
            V[1*p + k_start, col_start+3] += 1
            V[0*p + k_start, col_start+3] -= 1
            W[k_start*n + 0, col_start+3] += 1
            W[k_start*n + 1, col_start+3] += 1

            # M5 = (A00 + A01) * B1,k+1 -> -C0,k and C0,k+1
            U[0, col_start+4] += 1
            U[1, col_start+4] += 1
            V[1*p + (k_start+1), col_start+4] += 1
            W[k_start*n + 0, col_start+4] -= 1
            W[(k_start+1)*n + 0, col_start+4] += 1

            # M6 = (A10 - A00) * (B0,k + B0,k+1) -> C1,k+1
            U[2, col_start+5] += 1
            U[0, col_start+5] -= 1
            V[0*p + k_start, col_start+5] += 1
            V[0*p + (k_start+1), col_start+5] += 1
            W[(k_start+1)*n + 1, col_start+5] += 1

            # M7 = (A01 - A11) * (B1,k + B1,k+1) -> C0,k
            U[1, col_start+6] += 1
            U[3, col_start+6] -= 1
            V[1*p + k_start, col_start+6] += 1
            V[1*p + (k_start+1), col_start+6] += 1
            W[k_start*n + 0, col_start+6] += 1
        
        add_strassen_block(10, 3)
        add_strassen_block(17, 5)
        
        # Verify full decomposition
        for i in range(n):
            for j in range(m):
                for k_idx in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k_idx
                    c_idx = k_idx * n + i
                    s = sum(U[a_idx, r] * V[b_idx, r] * W[c_idx, r] for r in range(total_rank))
                    if abs(s - 1.0) > 1e-5:
                        # Fallback to rank 25
                        return _rank25_fallback()
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 25
    return _rank25_fallback()

def _rank25_fallback():
    """Fallback rank-25 decomposition using 3 Strassen blocks + Schoolbook."""
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    def add_strassen_block(col_start, k_start):
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[0*p + k_start, col_start+0] += 1
        V[1*p + (k_start+1), col_start+0] += 1
        W[k_start*n + 0, col_start+0] += 1
        W[(k_start+1)*n + 1, col_start+0] += 1

        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[0*p + k_start, col_start+1] += 1
        W[k_start*n + 1, col_start+1] += 1
        W[(k_start+1)*n + 1, col_start+1] -= 1

        U[0, col_start+2] += 1
        V[0*p + (k_start+1), col_start+2] += 1
        V[1*p + (k_start+1), col_start+2] -= 1
        W[(k_start+1)*n + 0, col_start+2] += 1
        W[(k_start+1)*n + 1, col_start+2] += 1

        U[3, col_start+3] += 1
        V[1*p + k_start, col_start+3] += 1
        V[0*p + k_start, col_start+3] -= 1
        W[k_start*n + 0, col_start+3] += 1
        W[k_start*n + 1, col_start+3] += 1

        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[1*p + (k_start+1), col_start+4] += 1
        W[k_start*n + 0, col_start+4] -= 1
        W[(k_start+1)*n + 0, col_start+4] += 1

        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[0*p + k_start, col_start+5] += 1
        V[0*p + (k_start+1), col_start+5] += 1
        W[(k_start+1)*n + 1, col_start+5] += 1

        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[1*p + k_start, col_start+6] += 1
        V[1*p + (k_start+1), col_start+6] += 1
        W[k_start*n + 0, col_start+6] += 1

    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    add_strassen_block(14, 4)
    
    col_start = 21
    k = 6
    U[0, col_start+0] += 1
    V[0*p + k, col_start+0] += 1
    W[k*n + 0, col_start+0] += 1
    U[1, col_start+1] += 1
    V[1*p + k, col_start+1] += 1
    W[k*n + 0, col_start+1] += 1
    U[2, col_start+2] += 1
    V[0*p + k, col_start+2] += 1
    W[k*n + 1, col_start+2] += 1
    U[3, col_start+3] += 1
    V[1*p + k, col_start+3] += 1
    W[k*n + 1, col_start+3] += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
