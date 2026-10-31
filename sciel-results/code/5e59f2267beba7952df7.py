import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 60.0  # Safety margin

    n, m, p = DIMENSIONS

    # Strassen coefficients for 2x2 matrix multiplication
    A_coeff = np.zeros((7, 2, 2), dtype=int)
    B_coeff = np.zeros((7, 2, 2), dtype=int)
    C_coeff = np.zeros((7, 2, 2), dtype=int)

    # s=0: M0 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    A_coeff[0, 0, 0] = 1; A_coeff[0, 1, 1] = 1
    B_coeff[0, 0, 0] = 1; B_coeff[0, 1, 1] = 1
    C_coeff[0, 0, 0] = 1; C_coeff[0, 1, 1] = 1

    # s=1: M1 = (A10+A11)B00 -> C10 += 1, C11 -= 1
    A_coeff[1, 1, 0] = 1; A_coeff[1, 1, 1] = 1
    B_coeff[1, 0, 0] = 1
    C_coeff[1, 1, 0] = 1; C_coeff[1, 1, 1] = -1

    # s=2: M2 = A00(B01-B11) -> C01 += 1, C11 += 1
    A_coeff[2, 0, 0] = 1
    B_coeff[2, 0, 1] = 1; B_coeff[2, 1, 1] = -1
    C_coeff[2, 0, 1] = 1; C_coeff[2, 1, 1] = 1

    # s=3: M3 = A11(B10-B00) -> C00 += 1, C10 += 1
    A_coeff[3, 1, 1] = 1
    B_coeff[3, 1, 0] = 1; B_coeff[3, 0, 0] = -1
    C_coeff[3, 0, 0] = 1; C_coeff[3, 1, 0] = 1

    # s=4: M4 = (A00+A01)B11 -> C00 -= 1, C01 += 1
    A_coeff[4, 0, 0] = 1; A_coeff[4, 0, 1] = 1
    B_coeff[4, 1, 1] = 1
    C_coeff[4, 0, 0] = -1; C_coeff[4, 0, 1] = 1

    # s=5: M5 = (A10-A00)(B00+B01) -> C11 += 1
    A_coeff[5, 1, 0] = 1; A_coeff[5, 0, 0] = -1
    B_coeff[5, 0, 0] = 1; B_coeff[5, 0, 1] = 1
    C_coeff[5, 1, 1] = 1

    # s=6: M6 = (A01-A11)(B10+B11) -> C00 += 1
    A_coeff[6, 0, 1] = 1; A_coeff[6, 1, 1] = -1
    B_coeff[6, 1, 0] = 1; B_coeff[6, 1, 1] = 1
    C_coeff[6, 0, 0] = 1

    # Build all 49 Strassen⊗Strassen terms
    U_full = np.zeros((16, 49), dtype=float)
    V_full = np.zeros((16, 49), dtype=float)
    W_full = np.zeros((16, 49), dtype=float)

    for s in range(7):
        for t in range(7):
            r = 7 * s + t

            for I in range(2):
                for J in range(2):
                    ca_out = A_coeff[s, I, J]
                    if ca_out == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            ca_in = A_coeff[t, i2, j2]
                            if ca_in == 0: continue
                            i = 2 * I + i2
                            j = 2 * J + j2
                            U_full[i * 4 + j, r] = ca_out * ca_in

            for J in range(2):
                for K in range(2):
                    cb_out = B_coeff[s, J, K]
                    if cb_out == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            cb_in = B_coeff[t, j2, k2]
                            if cb_in == 0: continue
                            j = 2 * J + j2
                            k = 2 * K + k2
                            V_full[j * 4 + k, r] = cb_out * cb_in

            for I in range(2):
                for K in range(2):
                    cc_out = C_coeff[s, I, K]
                    if cc_out == 0: continue
                    for i2 in range(2):
                        for k2 in range(2):
                            cc_in = C_coeff[t, i2, k2]
                            if cc_in == 0: continue
                            i = 2 * I + i2
                            k = 2 * K + k2
                            W_full[k * 4 + i, r] = cc_out * cc_in

    # Try to find a linear dependency to reduce rank from 49 to 48
    # Check if any term can be expressed as linear combination of others
    # such that the tensor identity still holds
    
    # For each candidate term to remove, check if the tensor identity
    # sum_r U[i,r]*V[j,r]*W[k,r] = delta_{i+k,j} (in appropriate form)
    # still holds with 48 terms
    
    # The tensor identity for matrix multiplication:
    # For each (i, k) in output: C[i,k] = sum_j A[i,j]*B[j,k]
    # In tensor form: sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] = delta condition
    
    # Actually the correct identity: for all (i,j,k) with the constraint that
    # the sum over r of U[i,j,r]*V[j,k,r]*W[k,i,r] gives the product
    
    # Let's verify: the full 49-term decomposition satisfies:
    # sum_r U_full[i*4+j, r] * V_full[j*4+k, r] * W_full[k*4+i, r]
    # should equal 1 if this corresponds to a valid matrix mult term, 0 otherwise
    
    # For matrix mult (AB)_{ik} = sum_j A_{ij} B_{jk}
    # So the tensor T[i,j,k] where the decomposition computes sum_j T[i,j,k] = (AB)_{ik}
    # Actually: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] summed over j gives (AB)_{ik}
    
    # Let's define the target: for each (i,k), sum_j sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = (AB)_{ik}
    # For random A,B: (AB)_{ik} = sum_j A[i,j]*B[j,k]
    
    # Test the 49-term version
    np.random.seed(123)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 4)
    C_test = A_test @ B_test
    
    # Verify 49-term works
    C_49 = np.zeros((4, 4))
    for i in range(4):
        for k in range(4):
            total = 0.0
            for j in range(4):
                for r in range(49):
                    total += U_full[i*4+j, r] * V_full[j*4+k, r] * W_full[k*4+i, r]
            C_49[i, k] = total
    
    if not np.allclose(C_49, C_test, atol=1e-9):
        # 49-term is broken, return it anyway (shouldn't happen)
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Now try removing each term and checking if we can compensate
    # For term r to remove, we need: for all (i,j,k):
    # U_full[i,j,r]*V_full[j,k,r]*W_full[k,i,r] = sum_{r' != r} alpha_{r,r'} * U_full[i,j,r']*V_full[j,k,r']*W_full[k,i,r']
    # This is very restrictive. Instead, try: can the r-th summand be written as
    # sum of other summands with the SAME U,V,W structure?
    
    # Simpler: just try dropping each term and see if the identity still holds
    # (unlikely, but worth checking for the special case)
    
    for skip_r in range(49):
        if time.time() > deadline:
            break
        
        C_48 = np.zeros((4, 4))
        for i in range(4):
            for k in range(4):
                total = 0.0
                for j in range(4):
                    for r in range(49):
                        if r == skip_r:
                            continue
                        total += U_full[i*4+j, r] * V_full[j*4+k, r] * W_full[k*4+i, r]
                C_48[i, k] = total
        
        if np.allclose(C_48, C_test, atol=1e-9):
            # Found a removable term!
            rank = 48
            U = np.zeros((16, 48), dtype=float)
            V = np.zeros((16, 48), dtype=float)
            W = np.zeros((16, 48), dtype=float)
            r_idx = 0
            for r in range(49):
                if r == skip_r:
                    continue
                U[:, r_idx] = U_full[:, r]
                V[:, r_idx] = V_full[:, r]
                W[:, r_idx] = W_full[:, r]
                r_idx += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # No single term is removable; return rank 49
    rank = 49
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
