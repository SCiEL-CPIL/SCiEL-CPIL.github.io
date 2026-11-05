        # 3. m3 = a00*(b01-b11)
        add_strassen_term(
            [(i0*m+j0, 1)],
            [(j0*p+k1, 1), (j1*p+k1, -1)],
            [(k1*n+i0, 1)]
        )
