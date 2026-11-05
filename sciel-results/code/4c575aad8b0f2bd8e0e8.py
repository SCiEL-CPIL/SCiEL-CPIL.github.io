        R_trial = R.copy()
        
        # Update for k >= 1
        # Part 1: k from 1 to N-1-i
        if N - i > 1:
            R_trial[1 : N-i] -= 2 * s[i] * s[i+1 : N]
            
        # Part 2: k from 1 to i
        if i > 0:
            R_trial[1 : i+1] -= 2 * s[i] * s[i-1 :: -1]
            
        # R[0] remains unchanged
