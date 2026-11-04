def _get_strassen_terms(n):
    # Returns list of (u, v, w) where u, v, w are dicts mapping (i,j) to coeff
    # For n=1, return [({(0,0):1}, {(0,0):1}, {(0,0):1})]
    if n == 1:
        return [({(0,0):1}, {(0,0):1}, {(0,0):1})]
    
    # Split into 2x2 blocks
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B10], [B01, B11]] # Note: B is column major in some contexts?
    # Standard matrix mult: C = AB.
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to express these as combinations of scalar multiplications.
    # Each Mk is a matrix product of size n/2.
    # We recursively get terms for Mk.
    # Each term in Mk is (u_mk, v_mk, w_mk).
    # u_mk is a linear form in A-blocks.
    # v_mk is a linear form in B-blocks.
    # w_mk is a linear form in C-blocks.
    
    # We need to map block indices to global indices.
