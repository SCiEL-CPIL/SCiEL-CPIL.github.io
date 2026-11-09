import numpy as np
import time

def priority(adjacency):
        start_time = time.time()
        A = np.asarray(adjacency, dtype=np.float64)
        n = A.shape[0]
        
        # 1. Local Features
        d = A.sum(axis=1)
        S = A @ d
        # E = 0.5 * diag(A^3)
        A2 = A @ A
        # diag(A @ A2)
        E = 0.5 * np.einsum('ij,ji->i', A, A2)
        
        # 2. Global Feature (Fiedler)
        F_score = np.zeros(n)
        try:
            if time.time() - start_time < 100: # Plenty of time
                D = np.diag(d)
                L = D - A
                # Symmetric matrix, use eigh
                eigvals, eigvecs = np.linalg.eigh(L)
                # eigvals are sorted ascending
                # Check if connected (eigvals[1] > 0)
                if eigvals[1] > 1e-6:
                    fiedler_vec = eigvecs[:, 1]
                    max_abs = np.max(np.abs(fiedler_vec))
                    if max_abs > 0:
                        fiedler_vec = fiedler_vec / max_abs
                        F_score = np.abs(fiedler_vec)
        except Exception:
            pass # If eigen fails, F_score remains 0
            
        # 3. Combine
        # Normalize features to Z-scores
        def z_score(x):
            std = np.std(x)
            if std < 1e-9:
                return np.zeros_like(x)
            return (x - np.mean(x)) / std
            
        Z_d = z_score(d)
        Z_S = z_score(S)
        Z_E = z_score(E)
        
        # Weights
        # Degree is primary (minimize)
        # S and E are secondary (maximize)
        # Fiedler is tertiary (maximize abs)
        
        # Tune weights
        # If Z_d is 2 (high degree), penalty is -20.
        # If Z_S is 2 (high neighbor deg), gain is +6.
        # Net -14. So high degree still loses.
        
        P = -10.0 * Z_d + 3.0 * Z_S + 3.0 * Z_E + 2.0 * F_score
        
        if time.time() - start_time > 110:
             return P.astype(float)
             
        return P.astype(float)
