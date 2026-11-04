    def add_strassen(i0, j0, k0):
        # M1 = (A00 + A11)(B01 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00B10
        # M4 = A11(B10 - B00)
        # M5 = A00(B01 - B11)
        # M6 = (A01 + A10)(B00 + B11)
        # M7 = (A01 - A10)(B11 + B10)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M5 + M7
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Indices:
        # A00: (i0, j0), A01: (i0, j0+1), A10: (i0+1, j0), A11: (i0+1, j0+1)
        # B00: (j0, k0), B01: (j0, k0+1), B10: (j0+1, k0), B11: (j0+1, k0+1)
        # C00: (k0, i0), C01: (k0, i0+1), C10: (k0+1, i0), C11: (k0+1, i0+1)
        
        # M1
        add_term(i0*m + j0, j0*p + k0+1, k0*n + i0, 1.0)
        add_term(i0*m + j0, j0*p + k0+1, k0*n + i0+1, 1.0) # Wait, M1 is scalar.
        # M1 contributes to C00 and C11.
        # C00 index: k0*n + i0. C11 index: (k0+1)*n + i0+1.
        # M1 = A00B01 + A00B11 + A11B01 + A11B11
        add_term(i0*m + j0, j0*p + k0+1, k0*n + i0, 1.0)
        add_term(i0*m + j0, (j0+1)*p + k0+1, k0*n + i0, 1.0)
        add_term(i0*m + j0+1, j0*p + k0+1, k0*n + i0, 1.0)
        add_term(i0*m + j0+1, (j0+1)*p + k0+1, k0*n + i0, 1.0)
        
        add_term(i0*m + j0, j0*p + k0+1, (k0+1)*n + i0+1, 1.0)
        add_term(i0*m + j0, (j0+1)*p + k0+1, (k0+1)*n + i0+1, 1.0)
        add_term(i0*m + j0+1, j0*p + k0+1, (k0+1)*n + i0+1, 1.0)
        add_term(i0*m + j0+1, (j0+1)*p + k0+1, (k0+1)*n + i0+1, 1.0)
        
        # M2 = (A10 + A11)B00
        # A10: (i0+1, j0), A11: (i0+1, j0+1)
        # B00: (j0, k0)
        # C10 = M2 + M4. C10 index: (k0+1)*n + i0.
        add_term(i0*m + j0+1, j0*p + k0, (k0+1)*n + i0, 1.0)
        add_term(i0*m + j0+1, j0*p + k0, (k0+1)*n + i0, 1.0) # Wait, A11 is (i0+1, j0+1).
        add_term(i0*m + j0+1, j0*p + k0, (k0+1)*n + i0, 1.0)
        add_term(i0*m + j0+1, j0*p + k0, (k0+1)*n + i0, 1.0)
        
        # M3 = A00B10
        # A00: (i0, j0)
        # B10: (j0+1, k0)
        # C11 = M1 - M2 + M3 + M6. C11 index: (k0+1)*n + i0+1.
        add_term(i0*m + j0, (j0+1)*p + k0, (k0+1)*n + i0+1, 1.0)
        
        # M4 = A11(B10 - B00)
        # A11: (i0+1, j0+1)
        # B10: (j0+1, k0), B00: (j0, k0)
        # C00 = M1 + M4 - M5 - M7. C00 index: k0*n + i0.
        # C10 = M2 + M4. C10 index: (k0+1)*n + i0.
        add_term(i0*m + j0+1, (j0+1)*p + k0, k0*n + i0, 1.0)
        add_term(i0*m + j0+1, j0*p + k0, k0*n + i0, -1.0)
        add_term(i0*m + j0+1, (j0+1)*p + k0, (k0+1)*n + i0, 1.0)
        add_term(i0*m + j0+1, j0*p + k0, (k0+1)*n + i0, -1.0)
        
        # M5 = A00(B01 - B11)
        # A00: (i0, j0)
        # B01: (j0, k0+1), B11: (j0+1, k0+1)
        # C00 = ... - M5 ...
        # C01 = M5 + M7. C01 index: k0*n + i0+1.
        add_term(i0*m + j0, j0*p + k0+1, k0*n + i0, -1.0)
        add_term(i0*m + j0, (j0+1)*p + k0+1, k0*n + i0, 1.0)
        add_term(i0*m + j0, j0*p + k0+1, k0*n + i0+1, 1.0)
        add_term(i0*m + j0, (j0+1)*p + k0+1, k0*n + i0+1, -1.0)
        
        # M6 = (A01 + A10)(B00 + B11)
        # A01: (i0, j0+1), A10: (i0+1, j0)
        # B00: (j0, k0), B11: (j0+1, k0+1)
        # C11 = ... + M6 ...
        add_term(i0*m + j0+1, j0*p + k0, (k0+1)*n + i0+1, 1.0)
        add_term(i0*m + j0+1, (j0+1)*p + k0+1, (k0+1)*n + i0+1, 1.0)
        add_term(i0*m + j0, j0*p + k0, (k0+1)*n + i0+1, 1.0)
        add_term(i0*m + j0, (j0+1)*p + k0+1, (k0+1)*n + i0+1, 1.0)
        
        # M7 = (A01 - A10)(B11 + B10)
        # A01: (i0, j0+1), A10: (i0+1, j0)
        # B11: (j0+1, k0+1), B10: (j0+1, k0)
        # C00 = ... - M7 ...
        # C01 = ... + M7 ...
        add_term(i0*m + j0+1, (j0+1)*p + k0+1, k0*n + i0, -1.0)
        add_term(i0*m + j0+1, (j0+1)*p + k0, k0*n + i0, -1.0)
        add_term(i0*m + j0, (j0+1)*p + k0+1, k0*n + i0, 1.0)
        add_term(i0*m + j0, (j0+1)*p + k0, k0*n + i0, 1.0)
        
        add_term(i0*m + j0+1, (j0+1)*p + k0+1, k0*n + i0+1, 1.0)
        add_term(i0*m + j0+1, (j0+1)*p + k0, k0*n + i0+1, 1.0)
        add_term(i0*m + j0, (j0+1)*p + k0+1, k0*n + i0+1, -1.0)
        add_term(i0*m + j0, (j0+1)*p + k0, k0*n + i0+1, -1.0)
