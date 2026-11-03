    # Attempt to reduce rank by removing dependent columns
    # Flatten the tensor contributions
    # T = sum_r u_r v_r w_r
    # We want to find if any u_r v_r w_r is in the span of others
    
    # This is equivalent to checking if the set of rank-1 tensors is linearly independent.
    # Flatten each rank-1 tensor into a vector of length n*m*p*n? No, n*m * m*p * p*n.
    # This is too large.
    
    # Instead, use the property that if U[:, r] is in span of U[:, others], 
    # AND V[:, r] is in span of V[:, others], AND W[:, r] is in span of W[:, others],
    # it's a necessary condition.
    
    # Let's try a more aggressive but safe reduction:
    # If we can find a column r such that U[:,r] is a linear combination of others,
    # we can't just remove it. We need to adjust the others.
    
    # However, for Strassen, the columns are often linearly independent.
    
    # Let's just ensure the code is clean and fast.
    # The current score is 1.1429.
    # If I cannot improve the rank, I should at least ensure no bugs.
