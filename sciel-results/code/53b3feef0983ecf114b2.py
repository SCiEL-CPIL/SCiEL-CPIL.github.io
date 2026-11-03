    def add_rect_2x1x2(a_rows, c_cols, d_cols):
        """2x1x2: A is 2x1, B is 1x2, C is 2x2. Rank 1."""
        a0, a1 = a_rows
        c0 = c_cols[0]
        d0, d1 = d_cols
        
        def U(la, lc): return a_rows[la]*m + c_cols[0]
        def V(lc, ld): return c_cols[0]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        
        # T_{i,k} = 1.
        # U_i = 1, V_k = 1, W_{ki} = 1.
        terms.append((
            [(U(0,0), 1), (U(1,0), 1)],
            [(V(0,0), 1), (V(0,1), 1)],
            [(W(0,0), 1), (W(1,0), 1), (W(0,1), 1), (W(1,1), 1)]
        ))
