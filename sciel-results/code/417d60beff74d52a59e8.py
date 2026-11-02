import numpy as np
import time
from scipy.optimize import least_squares

_start_time = time.time()
_DEADLINE = _start_time + 115.0

def construct():
    n, m, p = 2, 2, 15
    
    # Fallback: Rank 53
    def get_rank53():
        rank = 53
        U = np.zeros((4, rank))
        V = np.zeros((30, rank))
        W = np.zeros((30, rank))
        
        def add_strassen(base_r, k0, k1):
            # P1 = (a + d) * (e + h)
            U[0, base_r] = 1.0; U[3, base_r] = 1.0
            V[k0, base_r] = 1.0; V[15+k1, base_r] = 1.0
            W[2*k0, base_r] = 1.0; W[2*k1+1, base_r] = 1.0
            
            # P2 = (c + d) * e
            U[2, base_r+1] = 1.0; U[3, base_r+1] = 1.0
            V[k0, base_r+1] = 1.0
            W[2*k0+1, base_r+1] = 1.0; W[2*k1+1, base_r+1] = -1.0
            
            # P3 = a * (f - h)
            U[0, base_r+2] = 1.0
            V[15+k1, base_r+2] = 1.0; V[15+k1, base_r+2] = -1.0 # Wait, f is B[0,k1], h is B[1,k1]
            # V index for B[0,k1] is k1. V index for B[1,k1] is 15+k1.
            V[k1, base_r+2] = 1.0
            V[15+k1, base_r+2] = -1.0
            W[2*k1, base_r+2] = 1.0; W[2*k1+1, base_r+2] = 1.0
            
            # P4 = d * (g - e)
            U[3, base_r+3] = 1.0
            V[15+k0, base_r+3] = 1.0
            V[k0, base_r+3] = -1.0
            W[2*k0, base_r+3] = 1.0; W[2*k0+1, base_r+3] = 1.0
            
            # P5 = (a + b) * h
            U[0, base_r+4] = 1.0; U[1, base_r+4] = 1.0
            V[15+k1, base_r+4] = 1.0
            W[2*k0, base_r+4] = -1.0; W[2*k1, base_r+4] = 1.0
            
            # P6 = (c - a) * (e + f)
            U[2, base_r+5] = 1.0; U[0, base_r+5] = -1.0
            V[k0, base_r+5] = 1.0
            V[k1, base_r+5] = 1.0
            W[2*k1+1, base_r+5] = 1.0
            
            # P7 = (b - d) * (g + h)
            U[1, base_r+6] = 1.0; U[3, base_r+6] = -1.0
            V[15+k0, base_r+6] = 1.0
            V[15+k1, base_r+6] = 1.0
            W[2*k0, base_r+6] = 1.0

        for i in range(7):
            add_strassen(7*i, 2*i, 2*i+1)
            
        # Last col 14
        base_r = 49
        U[0, base_r] = 1.0; V[14, base_r] = 1.0; W[28, base_r] = 1.0
        U[1, base_r+1] = 1.0; V[29, base_r+1] = 1.0; W[28, base_r+1] = 1.0
        U[2, base_r+2] = 1.0; V[14, base_r+2] = 1.0; W[29, base_r+2] = 1.0
        U[3, base_r+3] = 1.0; V[29, base_r+3] = 1.0; W[29, base_r+3] = 1.0
        
        return {"U": U, "V": V, "W": W}

    # Try to find Rank 50 (5 x Rank 10 for 2x2x3)
    def try_rank50():
        # Target tensor for 2x2x3
        # A: 4, B: 6, C: 6
        # C[0,0] = a*e + b*h
        # C[0,1] = a*f + b*i
        # C[0,2] = a*g + b*j
        # C[1,0] = c*e + d*h
        # C[1,1] = c*f + d*i
        # C[1,2] = c*g + d*j
        
        # Flatten indices
        # A: 0:a, 1:b, 2:c, 3:d
        # B: 0:e, 1:f, 2:g, 3:h, 4:i, 5:j
        # C: 0:C00, 1:C01, 2:C02, 3:C10, 4:C11, 5:C12
        
        target = np.zeros((4, 6, 6))
        # C00 = a*e + b*h -> T[0,0,0]=1, T[1,3,0]=1
        target[0, 0, 0] = 1.0
        target[1, 3, 0] = 1.0
        # C01 = a*f + b*i -> T[0,1,1]=1, T[1,4,1]=1
        target[0, 1, 1] = 1.0
        target[1, 4, 1] = 1.0
        # C02 = a*g + b*j -> T[0,2,2]=1, T[1,5,2]=1
        target[0, 2, 2] = 1.0
        target[1, 5, 2] = 1.0
        # C10 = c*e + d*h -> T[2,0,3]=1, T[3,3,3]=1
        target[2, 0, 3] = 1.0
        target[3, 3, 3] = 1.0
        # C11 = c*f + d*i -> T[2,1,4]=1, T[3,4,4]=1
        target[2, 1, 4] = 1.0
        target[3, 4, 4] = 1.0
        # C12 = c*g + d*j -> T[2,2,5]=1, T[3,5,5]=1
        target[2, 2, 5] = 1.0
        target[3, 5, 5] = 1.0
        
        # We want U(4x10), V(6x10), W(6x10)
        # Sum_r U[:,r] V[:,r] W[:,r] = target
        
        # Use least squares on the flattened tensor
        # Reshape target to (4*6*6) = 144
        target_flat = target.flatten()
        
        # Random initialization
        np.random.seed(42)
        best_err = np.inf
        best_params = None
        
        # Time limit for search
        search_deadline = time.time() + 10.0
        
        for _ in range(50):
            if time.time() > search_deadline:
                break
            
            # Initialize
            U = np.random.randn(4, 10)
            V = np.random.randn(6, 10)
            W = np.random.randn(6, 10)
            
            # Scale to avoid overflow/underflow in optimization
            U /= np.linalg.norm(U) * 10
            V /= np.linalg.norm(V) * 10
            W /= np.linalg.norm(W) * 10
            
            def objective(params):
                U_ = params[0:40].reshape(4, 10)
                V_ = params[40:100].reshape(6, 10)
                W_ = params[100:160].reshape(6, 10)
                
                # Compute tensor
                # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
                # Efficiently:
                # Temp[i,j,k]
                # Use einsum
                T = np.einsum('ir,jr,kr->ijk', U_, V_, W_)
                return (T.flatten() - target_flat).flatten()
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = least_squares(objective, x0, method='lm', max_nfev=1000)
                if res.cost < best_err:
                    best_err = res.cost
                    best_params = res.x
                    # Check if solution is close to integer/half-integer
                    if best_err < 1e-6:
                        break
            except Exception:
                pass
        
        if best_err < 1e-6 and best_params is not None:
            U_sol = best_params[0:40].reshape(4, 10)
            V_sol = best_params[40:100].reshape(6, 10)
            W_sol = best_params[100:160].reshape(6, 10)
            
            # Construct full 2x2x15
            rank = 50
            U_full = np.zeros((4, 50))
            V_full = np.zeros((30, 50))
            W_full = np.zeros((30, 50))
            
            for i in range(5):
                offset = i * 10
                # A is same for all blocks
                U_full[:, offset:offset+10] = U_sol
                
                # B block i: cols 3i, 3i+1, 3i+2
                # V_sol indices: 0:e, 1:f, 2:g, 3:h, 4:i, 5:j
                # V_full indices for block i:
                # e -> 3i, f -> 3i+1, g -> 3i+2
                # h -> 15+3i, i -> 15+3i+1, j -> 15+3i+2
                
                V_full[3*i, offset:offset+10] = V_sol[0]
                V_full[3*i+1, offset:offset+10] = V_sol[1]
                V_full[3*i+2, offset:offset+10] = V_sol[2]
                V_full[15+3*i, offset:offset+10] = V_sol[3]
                V_full[15+3*i+1, offset:offset+10] = V_sol[4]
                V_full[15+3*i+2, offset:offset+10] = V_sol[5]
                
                # W block i: cols 3i, 3i+1, 3i+2
                # W_sol indices: 0:C00, 1:C01, 2:C02, 3:C10, 4:C11, 5:C12
                # W_full indices:
                # C00 -> 2*(3i) = 6i
                # C01 -> 2*(3i+1) = 6i+2
                # C02 -> 2*(3i+2) = 6i+4
                # C10 -> 2*(3i)+1 = 6i+1
                # C11 -> 2*(3i+1)+1 = 6i+3
                # C12 -> 2*(3i+2)+1 = 6i+5
                
                W_full[6*i, offset:offset+10] = W_sol[0]
                W_full[6*i+2, offset:offset+10] = W_sol[1]
                W_full[6*i+4, offset:offset+10] = W_sol[2]
                W_full[6*i+1, offset:offset+10] = W_sol[3]
                W_full[6*i+3, offset:offset+10] = W_sol[4]
                W_full[6*i+5, offset:offset+10] = W_sol[5]
                
            return {"U": U_full, "V": V_full, "W": W_full}
            
        return None

    # Try rank 50
    if time.time() < _DEADLINE - 20.0:
        res50 = try_rank50()
        if res50 is not None:
            return res50
            
    return get_rank53()
