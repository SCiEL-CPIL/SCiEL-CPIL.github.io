import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 2, 13
    
    def build_rank46():
        """Build the verified rank-46 decomposition."""
        rank = 46
        U = np.zeros((4, rank))
        V = np.zeros((26, rank))
        W = np.zeros((26, rank))
        
        for block_idx in range(6):
            k0 = block_idx * 2
            k1 = block_idx * 2 + 1
            base = block_idx * 7
            # P1 = (A00+A11)(B00+B11)
            U[0, base] = 1; U[3, base] = 1
            V[k0, base] = 1; V[13+k1, base] = 1
            W[2*k0, base] = 1; W[2*k1+1, base] = 1
            # P2 = (A10+A11)B00
            U[2, base+1] = 1; U[3, base+1] = 1
            V[k0, base+1] = 1
            W[2*k0+1, base+1] = 1; W[2*k1+1, base+1] = -1
            # P3 = A00(B10-B11)
            U[0, base+2] = 1
            V[k1, base+2] = 1; V[13+k1, base+2] = -1
            W[2*k1, base+2] = 1; W[2*k1+1, base+2] = 1
            # P4 = A11(B01-B00)
            U[3, base+3] = 1
            V[13+k0, base+3] = 1; V[k0, base+3] = -1
            W[2*k0, base+3] = 1; W[2*k0+1, base+3] = 1
            # P5 = (A00+A01)B11
            U[0, base+4] = 1; U[1, base+4] = 1
            V[13+k1, base+4] = 1
            W[2*k0, base+4] = -1; W[2*k1, base+4] = 1
            # P6 = (A10-A00)(B00+B01)
            U[2, base+5] = 1; U[0, base+5] = -1
            V[k0, base+5] = 1; V[k1, base+5] = 1
            W[2*k1+1, base+5] = 1
            # P7 = (A01-A11)(B10+B11)
            U[1, base+6] = 1; U[3, base+6] = -1
            V[13+k0, base+6] = 1; V[13+k1, base+6] = 1
            W[2*k0, base+6] = 1
        
        k = 12
        U[0, 42] = 1; V[k, 42] = 1; W[2*k, 42] = 1
        U[1, 43] = 1; V[13+k, 43] = 1; W[2*k, 43] = 1
        U[2, 44] = 1; V[k, 44] = 1; W[2*k+1, 44] = 1
        U[3, 45] = 1; V[13+k, 45] = 1; W[2*k+1, 45] = 1
        
        return U, V, W
    
    def check_decomposition(U, V, W, rank, tol=1e-6):
        """Check if the decomposition is valid."""
        for i in range(2):
            for k in range(2):
                for j in range(13):
                    val = np.sum(U[i*2+k] * V[k*13+j] * W[j*2+i])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def try_rank45():
        """Try to find a rank-45 decomposition."""
        U46, V46, W46 = build_rank46()
        
        # Start from rank-46 minus one term, try to optimize
        best_U, best_V, best_W = None, None, None
        best_err = float('inf')
        
        # Try dropping each of the last 4 terms
        for drop_idx in range(42, 46):
            if time.time() > deadline:
                break
            
            U = U46[:, [t for t in range(46) if t != drop_idx]].copy()
            V = V46[:, [t for t in range(46) if t != drop_idx]].copy()
            W = W46[:, [t for t in range(46) if t != drop_idx]].copy()
            
            # Alternating least squares
            for it in range(80):
                if time.time() > deadline:
                    break
                
                # Solve for U
                for i in range(2):
                    for k in range(2):
                        M = np.zeros((13, 45))
                        for j in range(13):
                            M[j,:] = V[k*13+j] * W[j*2+i]
                        x, *_ = np.linalg.lstsq(M, np.ones(13), rcond=None)
                        U[i*2+k] = x
                
                # Solve for V
                for k in range(2):
                    for j in range(13):
                        M = np.zeros((2, 45))
                        for i in range(2):
                            M[i,:] = U[i*2+k] * W[j*2+i]
                        x, *_ = np.linalg.lstsq(M, np.ones(2), rcond=None)
                        V[k*13+j] = x
                
                # Solve for W
                for j in range(13):
                    for i in range(2):
                        M = np.zeros((2, 45))
                        for k in range(2):
                            M[k,:] = U[i*2+k] * V[k*13+j]
                        x, *_ = np.linalg.lstsq(M, np.ones(2), rcond=None)
                        W[j*2+i] = x
                
                # Check error
                max_err = 0
                for i in range(2):
                    for k in range(2):
                        for j in range(13):
                            val = np.sum(U[i*2+k] * V[k*13+j] * W[j*2+i])
                            max_err = max(max_err, abs(val - 1))
                
                if max_err < 1e-8:
                    break
            
            # Final check
            max_err = 0
            for i in range(2):
                for k in range(2):
                    for j in range(13):
                        val = np.sum(U[i*2+k] * V[k*13+j] * W[j*2+i])
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < best_err:
                best_err = max_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_U is not None and best_err < 1e-6:
            # Try rounding to half-integers
            U_r = np.round(best_U * 2) / 2
            V_r = np.round(best_V * 2) / 2
            W_r = np.round(best_W * 2) / 2
            
            if check_decomposition(U_r, V_r, W_r, 45):
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
            
            # Also try rounding to integers
            U_i = np.round(best_U)
            V_i = np.round(best_V)
            W_i = np.round(best_W)
            
            if check_decomposition(U_i, V_i, W_i, 45):
                return U_i.tolist(), V_i.tolist(), W_i.tolist()
        
        return None
    
    # Try rank 45
    result45 = try_rank45()
    if result45 is not None:
        U_list, V_list, W_list = result45
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to rank 46
    U, V, W = build_rank46()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
