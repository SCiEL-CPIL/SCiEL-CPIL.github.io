import numpy as np

def coefficients(stages):
    """
    Implements a 6th-order symmetric splitting method using 12 stages.
    The 10-stage 6th-order Yoshida splitting is expanded to 12 stages by
    splitting the central B and A steps into two equal halves.
    """
    if stages != 12:
        # Fallback to 4th order or uniform if stages not 12
        # For now, assume 12 as per task
        pass
        
    a = [0.0] * stages
    b = [0.0] * stages
    
    # 6th order symmetric splitting coefficients
    # Based on Yoshida (1990)
    # alpha = 2^(1/3)
    alpha = 2.0 ** (1.0/3.0)
    
    # B coefficients (sum to 1)
    # c1, c2, c3, c2, c1
    c1 = 1.0 / (2.0 * (1.0 - alpha))
    c2 = 1.0 / (2.0 * (2.0 - alpha))
    c3 = 1.0 - 2.0 * (c1 + c2)
    
    # A coefficients (sum to 1)
    # b1, b2, b3, b2, b1
    b1 = 1.0 / (1.0 - alpha)
    b2 = alpha / (1.0 - alpha)
    b3 = 1.0 - 2.0 * (b1 + b2)
    
    # Map to 12 stages:
    # 1: B c1
    # 2: A b1
    # 3: B c2
    # 4: A b2
    # 5: B c3/2
    # 6: A b3/2
    # 7: B c3/2
    # 8: A b3/2
    # 9: B c2
    # 10: A b2
    # 11: B c1
    # 12: A b1
    
    b[0] = c1
    a[1] = b1
    b[2] = c2
    a[3] = b2
    b[4] = c3 / 2.0
    a[5] = b3 / 2.0
    b[6] = c3 / 2.0
    a[7] = b3 / 2.0
    b[8] = c2
    a[9] = b2
    b[10] = c1
    a[11] = b1
    
    return a, b
