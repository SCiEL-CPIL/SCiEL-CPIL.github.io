from scipy.optimize import linprog

def polish_radii_linprog(v):
    x = v[0::3]
    y = v[1::3]
    
    c = -np.ones(N) # Minimize -sum(r)
    
    # Inequality constraints A_ub @ r <= b_ub
    A_ub = []
    b_ub = []
    
    # Boundary constraints:
    # r_i <= x_i
    # r_i <= 1 - x_i
    # r_i <= y_i
    # r_i <= 1 - y_i
    for i in range(N):
        # r_i <= x_i
        row = np.zeros(N)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(x[i])
        
        # r_i <= 1 - x_i
        row = np.zeros(N)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(1 - x[i])
        
        # r_i <= y_i
        row = np.zeros(N)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(y[i])
        
        # r_i <= 1 - y_i
        row = np.zeros(N)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(1 - y[i])
        
    # Overlap constraints:
    # r_i + r_j <= d_ij
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            row = np.zeros(N)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(d)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(EPS, None)] * N
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        v_new = v.copy()
        v_new[2::3] = res.x
        return v_new
    else:
        return v
