<<<<<<< SEARCH
    # Laderman 3x3x3 algorithm (rank 23)
    def laderman_3x3x3():
        """Return 23 (u, v, w) vectors for a 3x3x3 Laderman block."""
        terms = []
        def mk_u(idx):
            u = np.zeros(nm)
            for i, j in idx:
                u[i*m+j] = 1.0
            return u
        def mk_v(idx):
            v = np.zeros(mp)
            for j, k in idx:
                v[j*p+k] = 1.0
            return v
        def mk_w(idx):
            w = np.zeros(pn)
            for k, i in idx:
                w[k*n+i] = 1.0
            return w
        def mk_u2(d):
            u = np.zeros(nm)
            for (i,j), val in d.items():
                u[i*m+j] = val
            return u
        def mk_v2(d):
            v = np.zeros(mp)
            for (j,k), val in d.items():
                v[j*p+k] = val
            return v
        def mk_w2(d):
            w = np.zeros(pn)
            for (k,i), val in d.items():
                w[k*n+i] = val
            return w
        
        # Laderman's 23 multiplications for 3x3x3
        # Using the standard Laderman scheme
        # M1 = A00 * B00
        terms.append((mk_u({(0,0)}), mk_v({(0,0)}), mk_w({(0,0)})))
        # M2 = A00 * B01
        terms.append((mk_u({(0,0)}), mk_v({(0,1)}), mk_w({(1,0)})))
        # M3 = A01 * B00
        terms.append((mk_u({(0,1)}), mk_v({(0,0)}), mk_w({(0,1)})))
        # M4 = A00 * B02
        terms.append((mk_u({(0,0)}), mk_v({(0,2)}), mk_w({(2,0)})))
        # M5 = A02 * B00
        terms.append((mk_u({(0,2)}), mk_v({(0,0)}), mk_w({(0,2)})))
        # M6 = A00 * B10
        terms.append((mk_u({(0,0)}), mk_v({(1,0)}), mk_w({(0,1)})))
        # M7 = A10 * B00
        terms.append((mk_u({(1,0)}), mk_v({(0,0)}), mk_w({(1,0)})))
        # M8 = A00 * B11
        terms.append((mk_u({(0,0)}), mk_v({(1,1)}), mk_w({(1,1)})))
        # M9 = A00 * B12
        terms.append((mk_u({(0,0)}), mk_v({(1,2)}), mk_w({(2,1)})))
        # M10 = A02 * B10
        terms.append((mk_u({(0,2)}), mk_v({(1,0)}), mk_w({(0,2)})))
        # M11 = A12 * B00
        terms.append((mk_u({(1,2)}), mk_v({(0,0)}), mk_w({(1,2)})))
        # M12 = A00 * B20
        terms.append((mk_u({(0,0)}), mk_v({(2,0)}), mk_w({(0,2)})))
        # M13 = A20 * B00
        terms.append((mk_u({(2,0)}), mk_v({(0,0)}), mk_w({(2,0)})))
        # M14 = A00 * B21
        terms.append((mk_u({(0,0)}), mk_v({(2,1)}), mk_w({(1,2)})))
        # M15 = A01 * B20
        terms.append((mk_u({(0,1)}), mk_v({(2,0)}), mk_w({(0,1)})))
        # M16 = A21 * B00
        terms.append((mk_u({(2,1)}), mk_v({(0,0)}), mk_w({(2,1)})))
        # M17 = A00 * B22
        terms.append((mk_u({(0,0)}), mk_v({(2,2)}), mk_w({(2,2)})))
        # M18 = A02 * B20
        terms.append((mk_u({(0,2)}), mk_v({(2,0)}), mk_w({(0,2)})))
        # M19 = A20 * B02
        terms.append((mk_u({(2,0)}), mk_v({(0,2)}), mk_w({(2,0)})))
        # M20 = A02 * B22
        terms.append((mk_u({(0,2)}), mk_v({(2,2)}), mk_w({(2,2)})))
        # M21 = A22 * B00
        terms.append((mk_u({(2,2)}), mk_v({(0,0)}), mk_w({(2,2)})))
        # M22 = A01 * B02 - A02 * B01
        u22 = mk_u2({(0,1): 1.0, (0,2): -1.0})
        v22 = mk_v2({(0,2): 1.0, (0,1): -1.0})
        w22 = mk_w2({(2,1): 1.0, (1,2): -1.0})
        terms.append((u22, v22, w22))
        # M23 = A01 * B12 - A12 * B01 + A02 * B11 - A11 * B02
        u23 = mk_u2({(0,1): 1.0, (1,2): -1.0, (0,2): 1.0, (1,1): -1.0})
        v23 = mk_v2({(1,2): 1.0, (0,1): -1.0, (1,1): 1.0, (0,2): -1.0})
        w23 = mk_w2({(2,1): 1.0, (1,2): -1.0, (1,1): 1.0, (2,2): -1.0})
        terms.append((u23, v23, w23))
        
        return terms

    # Try each block configuration
=======
    # Try each block configuration
>>>>>>> REPLACE
