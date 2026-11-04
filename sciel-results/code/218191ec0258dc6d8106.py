<<<<<<< SEARCH
            recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
            residual = np.linalg.norm(recon - T)
            
            if residual < 1e-6:
                # Polish
                for _ in range(30):
                    if time.time() > deadline - 8:
                        break
                    U_sol, V_sol, W_sol = als_solve(T1, T2, T3, U_sol, V_sol, W_sol, max_iters=10, tol=1e-14)
                recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
                residual = np.linalg.norm(recon - T)

            if verify_decomp(U_sol, V_sol, W_sol):
=======
            recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
            residual = np.linalg.norm(recon - T)
            
            if residual < 1e-6:
                # Polish with tighter tolerance
                for _ in range(40):
                    if time.time() > deadline - 8:
                        break
                    U_sol, V_sol, W_sol = als_solve(T1, T2, T3, U_sol, V_sol, W_sol, max_iters=15, tol=1e-15)
                    recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
                    residual = np.linalg.norm(recon - T)
                    if residual < 1e-13:
                        break

            if verify_decomp(U_sol, V_sol, W_sol):
>>>>>>> REPLACE
