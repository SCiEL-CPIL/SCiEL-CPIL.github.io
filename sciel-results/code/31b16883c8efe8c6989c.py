    def add_strassen_2x2(x_entries, y_entries, c_entries):
        nonlocal col
        # x_entries: [ [list of (idx, coeff)], [list], [list], [list] ]
        # y_entries: same
        # c_entries: same
        
        def merge_entries(e1, e2, sign1=1, sign2=1):
            res = []
            for idx, c in e1:
                res.append((idx, c * sign1))
            for idx, c in e2:
                res.append((idx, c * sign2))
            return res

        def add_w_entries(e1, e2, sign1=1, sign2=1):
            res = []
            for idx, c in e1:
                res.append((idx, c * sign1))
            for idx, c in e2:
                res.append((idx, c * sign2))
            return res

        # M1
        u_m1 = merge_entries(x_entries[0][0], x_entries[1][1], 1, 1)
        v_m1 = merge_entries(y_entries[0][0], y_entries[1][1], 1, 1)
        w_m1 = add_w_entries(c_entries[0][0], c_entries[1][1], 1, 1)
        
        # Wait, x_entries[0][0] is x[0,0].
        # I need to structure x_entries as x[0][0], x[0][1], x[1][0], x[1][1]
        
        # Let's assume x_entries is a list of 4 lists.
        # x_entries[0] = x[0,0] coeffs
        # x_entries[1] = x[0,1] coeffs
        # x_entries[2] = x[1,0] coeffs
        # x_entries[3] = x[1,1] coeffs
        
        # M1 = (x00 + x11)(y00 + y11)
        u_m1 = merge_entries(x_entries[0], x_entries[3], 1, 1)
        v_m1 = merge_entries(y_entries[0], y_entries[3], 1, 1)
        w_m1 = add_w_entries(c_entries[0], c_entries[3], 1, 1)
        
        for ui, uc in u_m1: U[ui][col] += uc
        for vi, vc in v_m1: V[vi][col] += vc
        for wi, wc in w_m1: W[wi][col] += wc
        col += 1
        
        # M2 = (x10 + x11)y00
        u_m2 = merge_entries(x_entries[2], x_entries[3], 1, 1)
        v_m2 = merge_entries(y_entries[0], [], 1, 1)
        w_m2 = add_w_entries(c_entries[2], c_entries[3], 1, 1)
        
        for ui, uc in u_m2: U[ui][col] += uc
        for vi, vc in v_m2: V[vi][col] += vc
        for wi, wc in w_m2: W[wi][col] += wc
        col += 1
        
        # M3 = x00(y01 - y11)
        u_m3 = merge_entries(x_entries[0], [], 1, 1)
        v_m3 = merge_entries(y_entries[1], y_entries[3], 1, -1)
        w_m3 = add_w_entries(c_entries[1], c_entries[3], 1, -1)
        
        for ui, uc in u_m3: U[ui][col] += uc
        for vi, vc in v_m3: V[vi][col] += vc
        for wi, wc in w_m3: W[wi][col] += wc
        col += 1
        
        # M4 = x11(y10 - y00)
        u_m4 = merge_entries(x_entries[3], [], 1, 1)
        v_m4 = merge_entries(y_entries[2], y_entries[0], 1, -1)
        w_m4 = add_w_entries(c_entries[0], c_entries[2], -1, 1)
        
        for ui, uc in u_m4: U[ui][col] += uc
        for vi, vc in v_m4: V[vi][col] += vc
        for wi, wc in w_m4: W[wi][col] += wc
        col += 1
        
        # M5 = (x00 + x01)y11
        u_m5 = merge_entries(x_entries[0], x_entries[1], 1, 1)
        v_m5 = merge_entries(y_entries[3], [], 1, 1)
        w_m5 = add_w_entries(c_entries[0], c_entries[1], 1, 1)
        
        for ui, uc in u_m5: U[ui][col] += uc
        for vi, vc in v_m5: V[vi][col] += vc
        for wi, wc in w_m5: W[wi][col] += wc
        col += 1
        
        # M6 = (x10 - x00)(y00 + y01)
        u_m6 = merge_entries(x_entries[2], x_entries[0], 1, -1)
        v_m6 = merge_entries(y_entries[0], y_entries[1], 1, 1)
        w_m6 = add_w_entries(c_entries[3], [], -1, 1)
        
        for ui, uc in u_m6: U[ui][col] += uc
        for vi, vc in v_m6: V[vi][col] += vc
        for wi, wc in w_m6: W[wi][col] += wc
        col += 1
        
        # M7 = (x01 - x11)(y10 + y11)
        u_m7 = merge_entries(x_entries[1], x_entries[3], 1, -1)
        v_m7 = merge_entries(y_entries[2], y_entries[3], 1, 1)
        w_m7 = add_w_entries(c_entries[0], [], -1, 1)
        
        for ui, uc in u_m7: U[ui][col] += uc
        for vi, vc in v_m7: V[vi][col] += vc
        for wi, wc in w_m7: W[wi][col] += wc
        col += 1
