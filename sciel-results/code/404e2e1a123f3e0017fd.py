<<<<<<< SEARCH
            U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs, [rem_j])
            
            if not verify_fast(U, V, W, rank):
                continue
            
            # Try to eliminate terms
            U_list, V_list, W_list, new_rank = try_eliminate_terms(U, V, W, rank)
            
            # Try pairwise elimination if time permits
            if time_left() > 15:
                U_list, V_list, W_list, new_rank = try_pairwise_elimination(U_list, V_list, W_list, new_rank)
                
            if new_rank < best_rank:
                best_rank = new_rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
=======
            U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs, [rem_j])
            
            if not verify_fast(U, V, W, rank):
                continue
            
            # Try to eliminate terms
            U_list, V_list, W_list, new_rank = try_eliminate_terms(U, V, W, rank)
            
            # Try linear (laser method) elimination if time permits
            if time_left() > 20:
                U_list, V_list, W_list, new_rank = try_linear_elimination(U_list, V_list, W_list, new_rank)
            
            # Try pairwise elimination if time permits
            if time_left() > 15:
                U_list, V_list, W_list, new_rank = try_pairwise_elimination(U_list, V_list, W_list, new_rank)
                
            if new_rank < best_rank:
                best_rank = new_rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
>>>>>>> REPLACE

