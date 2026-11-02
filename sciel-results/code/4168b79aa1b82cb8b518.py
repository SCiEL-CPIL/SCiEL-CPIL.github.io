                        # Constraint: s * (A_coeff x_i + B_coeff y_i + C_const) >= 2t
                        # Rearranging to form: Row dot [x,y,t] <= Val
                        # s*A*x + s*B*y + s*C - 2t >= 0
                        # 2t - s*A*x - s*B*y - s*C <= 0
                        # Row: [-s*A, -s*B, 2]
                        # Val: s*C
                        
                        row = np.array([-s * A_coeff, -s * B_coeff, 2])
                        val = s * C_const
