    def make_cell_from_params(params):
        R0, R1, R2, bg_flag = params
        R_sq = np.zeros(len(centers))
        for k, t in enumerate(center_types):
            if t == 0: R_sq[k] = R0*R0
            elif t == 1: R_sq[k] = R1*R1
            else: R_sq[k] = R2*R2
            
        covered = np.any(dist_sq < R_sq[None, None, :], axis=2)
        
        if bg_flag < 0.5:
            cell_arr = covered.astype(np.float64)
        else:
            cell_arr = (1.0 - covered.astype(np.float64))
            
        # Vectorized Symmetrization
        # 1. Identity
        # 2. Transpose
        # 3. Flip 0
        # 4. Flip 1
        # 5. Flip 0 + Flip 1 (180 rot)
        # 6. Flip 1 + Transpose
        # 7. Flip 0 + Transpose
        # 8. Transpose + Flip 0 + Flip 1 (Rotation 90? No, C4v group)
        
        # C4v elements:
        # E: cell
        # C4: rot90 (transpose + flip 0)
        # C2: rot180 (flip 0 + flip 1)
        # C4^3: rot270 (transpose + flip 1)
        # Mx: flip 1 (reflection across y-axis? No, x-axis is row index i. Flip 1 is j->31-j)
        # My: flip 0 (i->31-i)
        # Mxy: transpose (i<->j)
        # Mx+y: rot90 + Mx?
        
        # Let's just average the 8 unique transformations:
        sym_arr = cell_arr.copy()
        sym_arr += cell_arr.T
        sym_arr += np.flip(cell_arr, axis=0)
        sym_arr += np.flip(cell_arr, axis=1)
        sym_arr += np.flip(np.flip(cell_arr, axis=0), axis=1) # 180
        sym_arr += np.flip(cell_arr.T, axis=0) # rot90
        sym_arr += np.flip(cell_arr.T, axis=1) # rot270
        sym_arr += np.flip(np.flip(cell_arr.T, axis=0), axis=1) # rot180 (same as above)
        
        # Wait, the 8 elements of C4v are:
        # 1. Identity
        # 2. Transpose (Reflection across diagonal)
        # 3. Flip 0 (Reflection across horizontal axis)
        # 4. Flip 1 (Reflection across vertical axis)
        # 5. Flip 0 + Flip 1 (180 rotation)
        # 6. Transpose + Flip 0 (90 rotation)
        # 7. Transpose + Flip 1 (270 rotation)
        # 8. Transpose + Flip 0 + Flip 1 (Reflection across anti-diagonal)
        
        # Let's just sum all 8 and divide by 8.
        
        sym_arr = (cell_arr + 
                   cell_arr.T + 
                   np.flip(cell_arr, 0) + 
                   np.flip(cell_arr, 1) + 
                   np.flip(np.flip(cell_arr, 0), 1) + 
                   np.flip(cell_arr.T, 0) + 
                   np.flip(cell_arr.T, 1) + 
                   np.flip(np.flip(cell_arr.T, 0), 1)) / 8.0
                   
        cell = (sym_arr >= 0.5).astype(int)
        return cell.tolist()
