import numpy as np

def coefficients(stages):
    # 4th order symmetric scheme parameter
    theta = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    
    # Coefficients for one 4-stage block
    b_block = [theta / 2.0, (1.0 - theta) / 2.0, (1.0 - theta) / 2.0, theta / 2.0]
    a_block = [theta, 1.0 - 2.0 * theta, theta, 0.0]
    
    # Repeat 3 times to fill 12 stages
    num_repeats = stages // 4
    if num_repeats == 0:
        # Fallback for small stages
        return [1.0] * stages, [0.0] * stages
        
    b = []
    a = []
    for _ in range(num_repeats):
        b.extend([x / num_repeats for x in b_block])
        a.extend([x / num_repeats for x in a_block])
        
    # Pad if necessary
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
        
    return a, b
