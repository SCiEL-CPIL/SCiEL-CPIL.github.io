<<<<<<< SEARCH
    # Use K=5000 to balance accuracy and feasibility constraints.
    # Smaller K reduces the risk of M(x) exceeding 1.0001 due to truncation errors
    # and ensures the evaluator's O(m*K) step is fast.
    K = 5000
    
    mu = _compute_mobius(K)
=======
    # Use K=2000 to reduce the impact of the linear error term from f(1) adjustment.
    # Smaller K reduces the range of x, making the constraint M(x) <= 1.0001 easier to satisfy
    # if the sum S = sum(mu(k)/k) is not perfectly zero.
    K = 2000
    
    mu = _compute_mobius(K)
>>>>>>> REPLACE
