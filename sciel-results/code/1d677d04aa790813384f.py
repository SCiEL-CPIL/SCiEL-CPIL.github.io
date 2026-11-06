import numpy as np

def coefficients(stages):
    """
    Implements a 4-substep Strang splitting scheme using 12 stages.
    Each Strang step uses 3 slots: B, A, B.
    4 steps * 3 slots = 12 slots.
    """
    a = [0.0] * stages
    b = [0.0] * stages
    
    # 4 substeps
    num_substeps = 4
    dt = 1.0 / num_substeps
    
    # Strang coefficients for one step of size dt:
    # B(dt/2), A(dt), B(dt/2)
    b_coeff = dt / 2.0
    a_coeff = dt
    
    # Map to slots
    # Step 1: slots 0, 1, 2
    b[0] = b_coeff
    a[1] = a_coeff
    b[2] = b_coeff
    
    # Step 2: slots 3, 4, 5
    b[3] = b_coeff
    a[4] = a_coeff
    b[5] = b_coeff
    
    # Step 3: slots 6, 7, 8
    b[6] = b_coeff
    a[7] = a_coeff
    b[8] = b_coeff
    
    # Step 4: slots 9, 10, 11
    b[9] = b_coeff
    a[10] = a_coeff
    b[11] = b_coeff
    
    return a, b
