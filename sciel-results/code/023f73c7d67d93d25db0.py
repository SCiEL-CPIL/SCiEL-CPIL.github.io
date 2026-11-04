<<<<<<< SEARCH
        # Phase 4: Final Polish if we have a good candidate
        if best_result is not None and best_residual < 1e-4:
            if time_left() > 10:
                x_polished, residual_polished = polish(best_result, max_iter=1000)
                if residual_polished < best_residual:
                    best_result = x_polished
                    best_residual = residual_polished
                
                U_opt = best_result[:144].reshape(9, 16)
                V_opt = best_result[144:240].reshape(6, 16)
                W_opt = best_result[240:336].reshape(6, 16)
                rounded = check_rounded_solution(U_opt, V_opt, W_opt)
                if rounded is not None:
                    return rounded
        
        return None
=======
        # Phase 4: L1-based sparse recovery
        if best_result is not None and best_residual < 1e-2:
            if time_left() > 15:
                sparse_solution, sparse_residual = lasso_sparse_recovery(best_result)
                if sparse_solution is not None and sparse_residual < best_residual:
                    best_result = sparse_solution
                    best_residual = sparse_residual
        
        # Phase 5: Final Polish if we have a good candidate
        if best_result is not None and best_residual < 1e-4:
            if time_left() > 10:
                x_polished, residual_polished = polish(best_result, max_iter=1000)
                if residual_polished < best_residual:
                    best_result = x_polished
                    best_residual = residual_polished
                
                U_opt = best_result[:144].reshape(9, 16)
                V_opt = best_result[144:240].reshape(6, 16)
                W_opt = best_result[240:336].reshape(6, 16)
                rounded = check_rounded_solution(U_opt, V_opt, W_opt)
                if rounded is not None:
                    return rounded
        
        return None
>>>>>>> REPLACE
