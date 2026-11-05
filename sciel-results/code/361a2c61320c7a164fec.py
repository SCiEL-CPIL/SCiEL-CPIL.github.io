import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 55  # Safety margin for 60s limit
    
    n = 30
    
    # Compute L = lcm(1..30)
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # a[i] = L // (i+1) for 0-indexed i, corresponding to 1/(i+1)
    # We need sum(c[i] * a[i]) = +/- 1
    a = [L // (i + 1) for i in range(n)]
    
    mid = 15
    left_A = a[:mid]
    right_A = a[mid:]
    
    # Precompute powers of 3 for encoding
    pow3_left = [1] * mid
    for i in range(1, mid):
        pow3_left[i] = pow3_left[i-1] * 3
        
    pow3_right = [1] * (n - mid)
    for i in range(1, n - mid):
        pow3_right[i] = pow3_right[i-1] * 3

    # Function to generate all sums and their encoded patterns
    # Returns list of (sum_value, code)
    def generate_sums(A_part, pow3):
        current = [(0, 0)]
        for i, val in enumerate(A_part):
            p = pow3[i]
            new_current = []
            # Optimization: use local variables for speed
            append = new_current.append
            for s, c in current:
                # c = -1
                append((s - val, c - p))
                # c = 0
                append((s, c))
                # c = 1
                append((s + val, c + p))
            current = new_current
            if time.time() - start_time > deadline:
                return None
        return current

    # Generate left half sums
    left_data = generate_sums(left_A, pow3_left)
    if left_data is None:
        return [0] * 28 + [1, -1]
        
    # Build dictionary: sum -> code
    # We only need one code per sum
    left_dict = {}
    for s, c in left_data:
        if s not in left_dict:
            left_dict[s] = c
            
    if time.time() - start_time > deadline:
        return [0] * 28 + [1, -1]

    # Generate right half sums
    right_data = generate_sums(right_A, pow3_right)
    if right_data is None:
        return [0] * 28 + [1, -1]
        
    # Search for match
    # We want sum_left + sum_right = +/- 1
    # So sum_left = +/- 1 - sum_right
    
    # Check time before starting the search loop
    if time.time() - start_time > deadline:
        return [0] * 28 + [1, -1]

    for s_r, c_r in right_data:
        if time.time() - start_time > deadline:
            break
            
        # Target 1: sum_left = 1 - s_r
        # Target 2: sum_left = -1 - s_r
        
        target1 = 1 - s_r
        if target1 in left_dict:
            c_l = left_dict[target1]
            # Reconstruct coefficients
            # Decode c_l
            coefs_l = [0] * mid
            temp_c = c_l
            for i in range(mid):
                # Since we used signed integers for code, we need to be careful
                # Actually, standard base-3 decoding assumes digits 0,1,2.
                # Here we used -1, 0, 1 directly in the code accumulation.
                # So c_l is a sum of +/- pow3[i].
                # We can recover digits by checking modulo or simply iterating.
                # However, since we accumulated c = c +/- p, the code is just an integer.
                # To decode:
                # digit_i = (c_l // pow3_left[i]) % 3 ... this is tricky with negative numbers.
                
                # Alternative: store the actual list of coefficients? No, memory.
                # Better: decode properly.
                # Let's just extract the coefficients from the code.
                # The code is sum(d_i * 3^i) where d_i in {-1, 0, 1}.
                # We can recover d_i by:
                # d_i = round( (c_l - sum_{j<i} d_j 3^j) / 3^i )
                # Or simpler:
                # Since 3^i > 2 * sum_{j<i} 3^j, the digit is uniquely determined by the magnitude.
                # Actually, let's just re-calculate the coefficients from the code.
                
                pass # We'll do this after finding the match to save time inside the loop if possible, 
                     # but we need the code to reconstruct.
            
            # Let's reconstruct
            res = [0] * n
            
            # Decode left
            temp_l = c_l
            for i in range(mid):
                p = pow3_left[i]
                # Determine digit
                # The contribution of higher digits is multiples of 3^(i+1).
                # So we can look at temp_l % (3*p) ? No.
                # Let's use the property that d_i * p is the term.
                # Since max sum of lower terms is sum_{j=0}^{i-1} 3^j = (3^i - 1)/2.
                # So d_i * p + lower_sum.
                # We can find d_i by checking which of -p, 0, p brings temp_l closest to a multiple of 3*p?
                # Actually, simpler:
                # d_i = (temp_l // p) % 3 ... no, negative division is tricky.
                
                # Let's just extract by trial for the 3 possibilities?
                # Or better, store the coefficients as a tuple in the dict? 
                # 14M tuples of 15 ints is too much memory.
                
                # Let's use a standard balanced ternary decoding.
                # d_i = (temp_l + (3^i // 2)) // (3^i) % 3 ... 
                # Let's just do:
                # q = temp_l // p
                # r = temp_l % p
                # This is not straightforward for negative numbers in Python (floor division).
                
                # Let's stick to the simplest robust method:
                # Since we have the code, we can just re-derive the coefficients by checking
                # if (c_l - d*p) is divisible by 3^(i+1)? No, that's for standard base.
                
                # Actually, the most robust way given the code is just to check the 3 possibilities
                # and see which one makes the remaining value divisible by the next power of 3?
                # No, the code is just an integer sum.
                
                # Let's decode by extracting digits from least significant to most.
                # In balanced ternary, the digit at position i is determined by:
                # d_i = (code // 3^i) % 3 ... adjusted for negative?
                
                # Let's just calculate the coefficients directly from the code using a simple loop
                # that checks -1, 0, 1.
                # Since mid=15, this is fast.
                
                # We'll do the full reconstruction after breaking the loop.
                break # Found match, break outer loop
            
        target2 = -1 - s_r
        if target2 in left_dict:
            c_l = left_dict[target2]
            break # Found match
            
    else:
        # No match found in right_data loop? 
        # This should not happen if a solution exists.
        # If it does, fall back.
        return [0] * 28 + [1, -1]

    # Reconstruct coefficients
    # We need to identify which target was hit and get the codes.
    # The loop above broke, but we lost c_l and s_r context.
    # Let's re-structure to keep the found codes.
    
    # Refactored search to keep codes
    found = False
    final_c_l = 0
    final_c_r = 0
    
    for s_r, c_r in right_data:
        if time.time() - start_time > deadline:
            return [0] * 28 + [1, -1]
            
        target1 = 1 - s_r
        if target1 in left_dict:
            final_c_l = left_dict[target1]
            final_c_r = c_r
            found = True
            break
            
        target2 = -1 - s_r
        if target2 in left_dict:
            final_c_l = left_dict[target2]
            final_c_r = c_r
            found = True
            break
            
    if not found:
        return [0] * 28 + [1, -1]

    # Decode final_c_l
    res = [0] * n
    temp = final_c_l
    for i in range(mid):
        p = pow3_left[i]
        # Determine digit d in {-1, 0, 1}
        # We know temp = d*p + lower_part
        # lower_part is divisible by 1, but the structure is sum_{j<i} d_j 3^j.
        # The magnitude of lower_part is at most (3^i - 1)/2.
        # So d*p is the dominant term if we look at temp // p.
        
        # Let's just try the 3 values
        for d in [-1, 0, 1]:
            # Check if (temp - d*p) is consistent with lower digits
            # Actually, we can just extract the digit by:
            # d = (temp + (3^i)) // (3^(i+1)) ... no.
            
            # Simplest reliable decode for balanced ternary integer:
            # d_i = (temp // p) % 3
            # But Python's // is floor division.
            # Example: temp = -1, p=1. -1 // 1 = -1. -1 % 3 = 2. (Digit 2 is not in {-1,0,1})
            
            # Correct balanced ternary decoding:
            # d = (temp % 3)
            # if d == 2: d = -1
            # temp = (temp - d) // 3
            
            # Let's use this standard method.
            d = temp % 3
            if d == 2:
                d = -1
            res[i] = d
            temp = (temp - d) // 3
            
    # Decode final_c_r
    temp = final_c_r
    for i in range(n - mid):
        d = temp % 3
        if d == 2:
            d = -1
        res[mid + i] = d
        temp = (temp - d) // 3

    # Verify the solution
    # sum(res[i] / (i+1))
    val = 0
    for i in range(n):
        if res[i] != 0:
            val += res[i] / (i + 1)
            
    if val != 0:
        return res
    else:
        # Should not happen if we targeted +/- 1
        return [0] * 28 + [1, -1]
