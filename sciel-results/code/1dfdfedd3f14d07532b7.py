import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Estimate A and b for y' = A*y + b
    # b = f(t0, 0)
    b = f(t0, np.zeros(n))
    
    # A[:, j] = f(t0, e_j) - b
    A_sys = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A_sys[:, j] = f(t0, e_j) - b
        
    # Radau IIA Order 5 Coefficients
    c = np.array([
        0.048396337553358806803427969821,
        0.169649022298599010611965614378,
        0.341614705547710487081856534110,
        0.573793064377103830102542813802,
        0.872551870223227866151074951680
    ])
    b_w = np.array([
        0.012752109364978262372193943113,
        0.049813498387742152986325141585,
        0.108176240942924507728203982452,
        0.189580107165178289626707946550,
        0.339678045079176787286568986200
    ])
    A_radau = np.array([
        [0.002342851384782599015470724796, 0.0, 0.0, 0.0, 0.0],
        [0.008899150939734246472758226277, 0.006054871358864755584166879467, 0.0, 0.0, 0.0],
        [0.024808553126434741512492837540, 0.032713422694441132276628482722, 0.024092730396834672549055955000, 0.0, 0.0],
        [0.060181733018407135358231409884, 0.093807382127626740276748194537, 0.106796062760081429576681985554, 0.072997936470988636764873412000, 0.0],
        [0.128203509648830299186014813568, 0.229259498253848895834890921352, 0.284270156150187624479316169322, 0.271358762919711296957592172480, 0.236248129776772133848925048320]
    ])
    
    s = 5
    dt = t1 - t0
    if abs(dt) < 1e-15:
        return y0.copy()
        
    # Choose number of steps
    # For high accuracy, use many steps.
    # h = dt / num_steps
    num_steps = 100
    h = dt / num_steps
    
    # Pre-compute the block matrix K for the stage equations
    # Y_i = y_n + h * sum_j A_radau[i,j] * (A_sys * Y_j + b)
    # sum_j (delta_ij I - h A_radau[i,j] A_sys) Y_j = y_n + h sum_j A_radau[i,j] b
    
    # Construct K (size 5n x 5n)
    K = np.zeros((s*n, s*n))
    # Construct RHS constant part (size 5n)
    # RHS_i = y_n + h * (A_radau[i,:] @ b)
    # Note: b is a vector. A_radau[i,:] is a row of scalars.
    # So h * (A_radau[i,:] @ b) is a vector.
    
    # We will solve K * Y_flat = RHS_flat
    # Where Y_flat = [Y_1, Y_2, Y_3, Y_4, Y_5]
    
    I_n = np.eye(n)
    for i in range(s):
        for j in range(s):
            block = -h * A_radau[i, j] * A_sys
            if i == j:
                block = I_n + block
            K[i*n:(i+1)*n, j*n:(j+1)*n] = block
            
    # We can factorize K once if h is constant
    # But RHS changes with y_n.
    # We can use np.linalg.solve for each step.
    
    y = y0.copy()
    for _ in range(num_steps):
        # Construct RHS
        RHS = np.zeros(s*n)
        for i in range(s):
            # term = h * sum_j A_radau[i,j] * b
            # A_radau[i, :] is shape (5,). b is shape (n,).
            # We need dot product of scalars and vector?
            # No, sum_j A_radau[i,j] * b.
            # This is a linear combination of b? No, b is a vector.
            # The term is h * (sum_j A_radau[i,j]) * b?
            # Wait. f(t, Y) = A_sys Y + b.
            # sum_j A_radau[i,j] f(t, Y_j) = sum_j A_radau[i,j] (A_sys Y_j + b)
            # = (sum_j A_radau[i,j] A_sys) Y_j + (sum_j A_radau[i,j]) b
            # So the constant part is h * (sum_j A_radau[i,j]) * b.
            coeff_b = np.sum(A_radau[i, :])
            RHS[i*n:(i+1)*n] = y + h * coeff_b * b
            
        # Solve for Y
        Y_flat = np.linalg.solve(K, RHS)
        
        # Extract Y stages
        Y_stages = [Y_flat[i*n:(i+1)*n] for i in range(s)]
        
        # Update y
        # y_new = y + h * sum_j b_w[j] * f(t, Y_j)
        # = y + h * sum_j b_w[j] * (A_sys Y_j + b)
        # = y + h * (sum_j b_w[j] A_sys Y_j) + h * (sum_j b_w[j]) * b
        
        y_new = y.copy()
        for j in range(s):
            y_new += h * b_w[j] * (A_sys @ Y_stages[j] + b)
            
        y = y_new
        
    return y
