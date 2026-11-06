<<<<<<< SEARCH
            # Generalized eigenvalue problem: D a = omega^2 M a
            # Use scipy.linalg.eigh
            try:
                # M is Hermitian positive definite
                eigvals, eigvecs = sla.eigh(D_vals, M)
                eigvals = np.sort(eigvals)
                
                if len(eigvals) >= 5:
                    w4_sq = eigvals[3]
                    w5_sq = eigvals[4]
                    
                    if w4_sq > 0 and w5_sq > 0:
                        w4 = np.sqrt(w4_sq)
                        w5 = np.sqrt(w5_sq)
                        
                        max_omega_4 = max(max_omega_4, w4)
                        min_omega_5 = min(min_omega_5, w5)
            except Exception:
                pass
=======
            # Generalized eigenvalue problem: D a = omega^2 M a
            # Use np.linalg.eigh (pure NumPy)
            try:
                # M is Hermitian positive definite
                # eigh(A, B) solves A v = lambda B v
                eigvals, _ = np.linalg.eigh(D_vals, M)
                eigvals = np.sort(eigvals)
                
                if len(eigvals) >= 5:
                    w4_sq = eigvals[3]
                    w5_sq = eigvals[4]
                    
                    if w4_sq > 0 and w5_sq > 0:
                        w4 = np.sqrt(w4_sq)
                        w5 = np.sqrt(w5_sq)
                        
                        max_omega_4 = max(max_omega_4, w4)
                        min_omega_5 = min(min_omega_5, w5)
            except Exception:
                pass
>>>>>>> REPLACE

