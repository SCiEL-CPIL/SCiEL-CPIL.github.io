import math

def coefficients(stages):
    # Calculate Yoshida 12th order coefficients
    # Order 2j, j=6
    j = 6
    c = [0.0] * (j + 1)
    c[1] = 1.0
    for k in range(2, j + 1):
        c[k] = c[k-1] / (2.0 - 2.0 ** (1.0 / (2*k - 1)))
    
    # Alpha and Beta coefficients
    # Number of steps is 2j - 1 = 11
    num_steps = 2 * j - 1
    alpha = [0.0] * num_steps
    beta = [0.0] * (num_steps - 1)
    
    # alpha_1
    alpha[0] = 0.5 * (1.0 - 1.0 / c[2])
    
    # alpha_k for 1 < k < num_steps
    # Formula: alpha_k = 0.5 * (1/c_{k-1} - 1/c_k) ? 
    # Let's check indices.
    # c indices 1..j.
    # alpha indices 1..2j-1.
    # The mapping is usually:
    # alpha_1 = 0.5 * (1 - 1/c_2)
    # alpha_k = 0.5 * (1/c_{k-1} - 1/c_k) for k=2..j? No.
    
    # Correct recurrence from Yoshida:
    # alpha_1 = 0.5 * (1 - 1/c_2)
    # alpha_k = 0.5 * (1/c_{k-1} - 1/c_k) for 1 < k <= j
    # alpha_{2j-k} = alpha_k
    # beta_k = alpha_k + alpha_{k+1} for 1 <= k < j
    # beta_{2j-1-k} = beta_k
    
    # Let's implement this carefully.
    for k in range(2, j + 1):
        alpha[k-1] = 0.5 * (1.0 / c[k-1] - 1.0 / c[k])
        
    for k in range(j, 0, -1):
        alpha[2*j - k - 1] = alpha[k-1]
        
    for k in range(1, j):
        beta[k-1] = alpha[k-1] + alpha[k]
        
    for k in range(j-1, 0, -1):
        beta[2*j - k - 1] = beta[k-1]
        
    # Verify sums
    assert abs(sum(alpha) - 1.0) < 1e-10, f"alpha sum {sum(alpha)}"
    assert abs(sum(beta) - 1.0) < 1e-10, f"beta sum {sum(beta)}"
    
    # Construct a and b lists of length 12
    a_list = [0.0] * 12
    b_list = [0.0] * 12
    
    # Map 11 steps (11 A, 10 B) to 12 pairs
    # k=1: B=0, A=alpha[0]
    a_list[0] = alpha[0]
    b_list[0] = 0.0
    
    # k=2..11: B=beta[i-1], A=alpha[i]
    for i in range(1, num_steps):
        idx = i # 1-based index in lists? 
        # a_list index i-1? No.
        # Pair k corresponds to index k-1 in lists.
        # Pair 1 (idx 0): B=0, A=alpha[0]
        # Pair 2 (idx 1): B=beta[0], A=alpha[1]
        # ...
        # Pair 11 (idx 10): B=beta[9], A=alpha[10]
        # Pair 12 (idx 11): B=0, A=0
        
        b_list[i] = beta[i-1]
        a_list[i] = alpha[i]
        
    # Last pair is 0
    a_list[11] = 0.0
    b_list[11] = 0.0
    
    # Final checks
    assert abs(sum(a_list) - 1.0) < 1e-10
    assert abs(sum(b_list) - 1.0) < 1e-10
    for val in a_list + b_list:
        assert math.isfinite(val) and -50 <= val <= 50
        
    return a_list, b_list
