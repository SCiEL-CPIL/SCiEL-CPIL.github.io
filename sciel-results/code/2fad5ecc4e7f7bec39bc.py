import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k) * log(k) / k subject to
    sum_k f(k)/k = 0 and M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for all x in [1, 10K].
    """
    start_time = time.time()
    deadline = 590.0  # Leave safety margin
    
    # We will construct a function with small K to ensure M(x) is easily controlled.
    # Let's try K=10.
    K = 10
    
    # Initialize values
    # We want v(k) negative for small k (high log k / k) to boost score.
    # But this increases f_new(1) = -sum_{k=2} v(k)/k.
    # We need f_new(1) <= 1.
    # Let's set v(2) = -10, v(3) = -10.
    # Sum_{k=2,3} v(k)/k = -10/2 - 10/3 = -5 - 3.33 = -8.33.
    # f_new(1) = 8.33. Too high.
    # We need to add positive v(k) to reduce f_new(1).
    # Adding v(k)=10 reduces f_new(1) by 10/k.
    # We need to reduce 8.33 to 1. Delta = 7.33.
    # v(4)=10 -> reduce 2.5. Rem 4.83.
    # v(5)=10 -> reduce 2.0. Rem 2.83.
    # v(6)=10 -> reduce 1.66. Rem 1.17.
    # v(7)=10 -> reduce 1.42. Rem -0.25. (f_new(1) = -0.25).
    # This is feasible (f_new(1) <= 1).
    
    # Let's verify M(x) for this configuration.
    # v = [?, -10, -10, 10, 10, 10, 10, 0, 0, 0]
    # v(1) is set by evaluator. We output v(1)=10.
    # S = 10 + (-5) + (-3.33) + 2.5 + 2 + 1.66 + 1.42 = 9.25.
    # f_new(1) = 10 - 9.25 = 0.75.
    # f_new(k) = v(k) for k>1.
    
    # M(1) = 0.75. OK.
    # M(2) = 0.75*2 + (-10)*1 = 1.5 - 10 = -8.5. OK.
    # M(3) = 0.75*3 + (-10)*1 + (-10)*1 = 2.25 - 20 = -17.75. OK.
    # M(4) = 0.75*4 + (-10)*1 + (-10)*1 + 10*1 = 3 - 20 + 10 = -7. OK.
    # M(5) = 0.75*5 + (-10)*1 + (-10)*1 + 10*1 + 10*1 = 3.75 - 20 + 20 = 3.75. 
    # Wait, M(5) = 3.75 > 1.0001. Violation!
    
    # We need to check M(x) carefully.
    # M(x) = x * f_new(1) + sum_{k=2}^K v(k) floor(x/k).
    
    # Let's try to make f_new(1) smaller.
    # If f_new(1) = 0.1.
    # Sum v(k)/k = -0.1.
    # v(2)=-10, v(3)=-10 -> Sum = -8.33.
    # Need +8.23 from v(4)..
    # v(4)=10 (2.5), v(5)=10 (2.0), v(6)=10 (1.66), v(7)=10 (1.42), v(8)=10 (1.25).
    # Sum added = 8.83.
    # Total Sum = -8.33 + 8.83 = 0.5.
    # f_new(1) = -0.5.
    # M(1) = -0.5. OK.
    # M(5) = -0.5*5 + (-10) + (-10) + 10 + 10 = -2.5 - 20 + 20 = -2.5. OK.
    # M(10) = -0.5*10 + (-10)*5 + (-10)*3 + 10*2 + 10*2 + 10*1 + 10*1 + 10*1 + 10*1
    # = -5 - 50 - 30 + 20 + 20 + 10 + 10 + 10 + 10 = -5 - 80 + 80 = -5. OK.
    
    # Score a = - sum v(k) log(k)/k.
    # v(2)=-10: 10 * 0.346 = 3.46
    # v(3)=-10: 10 * 0.366 = 3.66
    # v(4)=10: -10 * 0.346 = -3.46
    # v(5)=10: -10 * 0.322 = -3.22
    # v(6)=10: -10 * 0.299 = -2.99
    # v(7)=10: -10 * 0.278 = -2.78
    # v(8)=10: -10 * 0.259 = -2.59
    # Sum = 3.46 + 3.66 - 3.46 - 3.22 - 2.99 - 2.78 - 2.59 = 7.12 - 15.04 = -7.92.
    # Negative score.
    
    # We need to be more selective.
    # Only use v(2) and v(3) negative.
    # Offset with v(4) and v(5) positive.
    # v(2)=-10, v(3)=-10. Sum = -8.33.
    # v(4)=10, v(5)=10. Sum = 4.5.
    # Total Sum = -3.83.
    # f_new(1) = 3.83.
    # M(1) = 3.83 > 1. Violation.
    
    # We must have f_new(1) <= 1.
    # So Sum v(k)/k >= -1.
    # Maximize - sum v(k) w_k.
    # Let v(2) = -x, v(3) = -y.
    # Offset with v(4) = z.
    # -x/2 - y/3 + z/4 >= -1.
    # Maximize x w_2 + y w_3 - z w_4.
    # w_2=0.346, w_3=0.366, w_4=0.346.
    # To maximize, set x,y as large as possible (10).
    # -5 - 3.33 + z/4 >= -1 => z/4 >= 7.33 => z >= 29.3.
    # But z <= 10.
    # So we cannot use both v(2) and v(3) at -10.
    
    # Try v(2)=-10.
    # -5 + z/4 >= -1 => z/4 >= 4 => z >= 16. Impossible.
    # So v(2) cannot be -10 if we only use v(4) to offset.
    
    # Use v(4), v(5) to offset.
    # -5 + 2.5 + 2 = -0.5 >= -1. OK.
    # v(2)=-10, v(4)=10, v(5)=10.
    # f_new(1) = 0.5.
    # M(1) = 0.5. OK.
    # M(2) = 0.5*2 - 10 = -9. OK.
    # M(4) = 0.5*4 - 10 + 10 = 2. OK.
    # M(5) = 0.5*5 - 10 + 10 + 10 = 12.5 > 1. Violation.
    
    # M(x) grows with x if f_new(1) > 0.
    # We need M(x) <= 1 for all x.
    # The worst case is usually small x or x where floor terms don't cancel.
    # If f_new(1) is small, M(x) is small.
    # If f_new(1) = 0, M(x) = - sum v(k) floor(x/k).
    # If v(k) are mixed, M(x) can be positive.
    
    # Safe bet: Set all v(k) = 0 except v(1)=0.
    # Score 0.
    
    # To get positive score, we need f_new(1) close to 0 and negative v(k) for small k.
    # Let's try v(2) = -1, v(3) = -1.
    # Sum = -0.5 - 0.33 = -0.83.
    # f_new(1) = 0.83.
    # M(1) = 0.83. OK.
    # M(2) = 1.66 - 1 = 0.66. OK.
    # M(3) = 2.49 - 1 - 1 = 0.49. OK.
    # M(4) = 3.32 - 1 - 1 = 1.32 > 1. Violation.
    
    # Reduce v(2), v(3).
    # v(2) = -0.5, v(3) = -0.5.
    # Sum = -0.25 - 0.16 = -0.41.
    # f_new(1) = 0.41.
    # M(4) = 1.64 - 0.5 - 0.5 = 0.64. OK.
    # M(5) = 2.05 - 0.5 - 0.5 = 1.05 > 1. Violation.
    
    # v(2) = -0.4, v(3) = -0.4.
    # Sum = -0.2 - 0.13 = -0.33.
    # f_new(1) = 0.33.
    # M(5) = 1.65 - 0.4 - 0.4 = 0.85. OK.
    # M(6) = 1.98 - 0.4 - 0.4 - 0.4 = 0.78. OK.
    
    # Score a = 0.4 * 0.346 + 0.4 * 0.366 = 0.138 + 0.146 = 0.284.
    
    # This is better than 0.
    
    domain = np.array([1, 2, 3], dtype=np.int64)
    values = np.array([10.0, -0.4, -0.4], dtype=np.float64)
    
    return domain, values
